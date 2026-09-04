import json
import math
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import torch

from lerobot_async_inference.configs import cosine_ramp
from lerobot_async_inference.diagnostic_logger import AsyncJSONLWriter
from lerobot_async_inference.helpers import TimedAction
from lerobot_async_inference.offline_trajectory_replay import replay_profile
from lerobot_async_inference.robot_client import RobotClient
from lerobot_async_inference.trajectory import GripperPostprocessor, JerkLimitedTrajectory
from lerobot_async_inference.urdf_limits import (
    ARM_JOINT_NAMES,
    build_operational_profile,
    load_active_urdf_limits,
    validate_arm_action_map,
)


JOINTS = ["left_arm_0", "left_arm_1"]


def generator() -> JerkLimitedTrajectory:
    return JerkLimitedTrajectory(
        JOINTS,
        velocity_limits={name: 1.0 for name in JOINTS},
        acceleration_limits={name: 2.0 for name in JOINTS},
        jerk_limits={name: 10.0 for name in JOINTS},
        position_limits={name: (-2.0, 2.0) for name in JOINTS},
    )


def assert_limits(samples, dts) -> None:
    for sample, _dt in zip(samples, dts, strict=True):
        assert np.max(np.abs(sample.velocity)) <= 1.0 + 1e-8
        assert np.max(np.abs(sample.acceleration)) <= 2.0 + 1e-8
        assert np.max(np.abs(sample.jerk)) <= 10.0 + 1e-8


def test_cosine_ramp_full_action_regression_including_grippers() -> None:
    old = torch.arange(16, dtype=torch.float64)
    new = old + 16
    count = 5
    index = 2
    alpha = (1 - math.cos(math.pi * (index + 1) / (count + 1))) / 2
    expected = (1 - alpha) * old + alpha * new
    torch.testing.assert_close(
        cosine_ramp(old, new, overlap_index=index, overlap_count=count), expected
    )
    assert expected[-2:].tolist() == [22.0, 23.0]


def test_constant_input_does_not_move() -> None:
    trajectory = generator()
    trajectory.reset([0.3, -0.2])
    trajectory.set_target([0.3, -0.2])
    for _ in range(20):
        sample = trajectory.step(0.01)
        np.testing.assert_array_equal(sample.position, [0.3, -0.2])
        np.testing.assert_array_equal(sample.velocity, [0.0, 0.0])


def test_missing_joint_limits_are_rejected() -> None:
    try:
        JerkLimitedTrajectory(
            JOINTS,
            velocity_limits={"left_arm_0": 1.0},
            acceleration_limits={name: 2.0 for name in JOINTS},
            jerk_limits={name: 10.0 for name in JOINTS},
            position_limits={name: (-2.0, 2.0) for name in JOINTS},
        )
    except ValueError as exc:
        assert "left_arm_1" in str(exc)
    else:
        raise AssertionError("incomplete live limits must not be accepted")


def test_step_input_respects_velocity_acceleration_and_jerk() -> None:
    trajectory = generator()
    trajectory.reset([0.0, 0.0])
    trajectory.set_target([1.0, -1.0])
    dts = [0.01] * 300
    samples = [trajectory.step(dt) for dt in dts]
    assert_limits(samples, dts)
    assert max(sample.position[0] for sample in samples) <= 1.0 + 1e-9
    assert min(sample.position[1] for sample in samples) >= -1.0 - 1e-9


def test_irregular_dt_respects_all_limits() -> None:
    trajectory = generator()
    trajectory.reset([0.0, 0.0])
    trajectory.set_target([1.0, -0.7])
    dts = [0.004, 0.013, 0.007, 0.021, 0.009] * 80
    samples = [trajectory.step(dt) for dt in dts]
    assert_limits(samples, dts)


def test_new_target_has_no_position_jump() -> None:
    trajectory = generator()
    trajectory.reset([0.0, 0.0])
    trajectory.set_target([1.0, 1.0])
    before = trajectory.step(0.01)
    trajectory.set_target([-1.0, -1.0])
    after = trajectory.step(0.01)
    assert np.max(np.abs(after.position - before.position)) <= 1.0 * 0.01 + 1e-9


def test_gripper_passthrough_is_immediate() -> None:
    gripper = GripperPostprocessor(["left_gripper_0"], "passthrough", {})
    gripper.reset([1.0])
    np.testing.assert_array_equal(gripper.update([0.0], 0.001), [0.0])


def test_joint_position_limit_and_nan_fallback() -> None:
    trajectory = generator()
    trajectory.reset([0.0, 0.0])
    trajectory.set_target([99.0, -99.0])
    samples = [trajectory.step(0.01) for _ in range(500)]
    assert np.max(samples[-1].position) <= 2.0
    assert np.min(samples[-1].position) >= -2.0
    last = samples[-1].position.copy()
    try:
        trajectory.set_target([float("nan"), 0.0])
    except ValueError:
        sample = trajectory.hold_last_valid("non-finite target")
    np.testing.assert_array_equal(sample.position, last)
    assert not sample.valid


def make_scheduling_client(actions: list[TimedAction]) -> RobotClient:
    client = RobotClient.__new__(RobotClient)
    client.action_queue = Queue()
    for action in actions:
        client.action_queue.put(action)
    client.action_queue_lock = threading.Lock()
    client.latest_action_lock = threading.Lock()
    client.latest_action = -1
    client.latest_action_tensor = None
    client._chunk_counter = 0
    client._action_diagnostic_writer = None
    client._diagnostic_sample_counter = 0
    client.config = SimpleNamespace(
        trajectory_postprocess=SimpleNamespace(logging=SimpleNamespace(downsample=1))
    )
    client.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None, debug=lambda *_args: None)
    return client


def test_stale_actions_are_dropped_by_scheduled_time() -> None:
    actions = [
        TimedAction(timestamp=99.0, timestep=0, action=torch.tensor([0.0])),
        TimedAction(timestamp=101.0, timestep=1, action=torch.tensor([1.0])),
    ]
    client = make_scheduling_client([])
    record = client._ingest_action_chunk(
        actions, None, receive_wall_time=100.0, receive_monotonic_time=10.0
    )
    assert record["stale_actions_dropped"] == 1
    assert [item.get_timestep() for item in client.action_queue.queue] == [1]
    assert client.action_queue.queue[0].metadata["scheduled_execution_time"] == 11.0


def test_control_delay_drops_old_waypoints_without_burst() -> None:
    actions = [
        TimedAction(
            timestamp=float(i),
            timestep=i,
            action=torch.tensor([float(i)]),
            metadata={"scheduled_execution_time": float(i)},
        )
        for i in range(4)
    ]
    client = make_scheduling_client(actions)
    newest, dropped = client._pop_due_waypoint(2.5)
    assert newest.get_timestep() == 2
    assert dropped == 2
    assert [item.get_timestep() for item in client.action_queue.queue] == [3]


def test_async_logger_flushes_and_sanitizes_nan() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/trace.jsonl"
        writer = AsyncJSONLWriter(path, max_queue_size=4)
        assert writer.submit({"sample": 1, "bad": float("nan")})
        writer.close()
        assert json.loads(Path(path).read_text(encoding="utf-8")) == {
            "sample": 1,
            "bad": None,
        }


def test_active_urdf_parser_extracts_all_arm_limits() -> None:
    limits = load_active_urdf_limits(
        Path("/home/nvidia/rby1-sdk/models"), "a", "1.2"
    )
    assert limits.robot_name == "RBY1_A_v1.2"
    assert limits.joint_names == ARM_JOINT_NAMES
    assert len(limits.position_limits) == 14
    assert limits.velocity_limits["right_arm_4"] == 6.283185308
    assert limits.acceleration_limits["left_arm_6"] == 10.0
    assert limits.units["position"] == "rad"


def test_arm_joint_map_must_be_exactly_complete() -> None:
    validate_arm_action_map(list(ARM_JOINT_NAMES) + list(("right_gripper_0", "left_gripper_0")))
    try:
        validate_arm_action_map(list(ARM_JOINT_NAMES[:-1]) + ["left_arm_7"])
    except ValueError as exc:
        assert "left_arm_6" in str(exc)
        assert "left_arm_7" in str(exc)
    else:
        raise AssertionError("an incomplete/mismatched arm map must fail closed")


def test_wrong_model_or_urdf_fails_closed() -> None:
    try:
        load_active_urdf_limits(Path("/home/nvidia/rby1-sdk/models"), "x", "1.2")
    except ValueError as exc:
        assert "unsupported RB-Y1 model" in str(exc)
    else:
        raise AssertionError("an unknown RB-Y1 model must fail closed")
    try:
        load_active_urdf_limits(Path("/home/nvidia/rby1-sdk/models"), "m", "9.9")
    except FileNotFoundError as exc:
        assert "exact RB-Y1 M v9.9 URDF not found" in str(exc)
    else:
        raise AssertionError("a missing exact URDF must not fall back to model.urdf")


def test_urdf_without_explicit_si_acceleration_units_fails_closed() -> None:
    source = Path("/home/nvidia/rby1-sdk/models/rby1a/urdf/model_v1.2.urdf")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "rby1a/urdf/model_v1.2.urdf"
        target.parent.mkdir(parents=True)
        target.write_text(
            source.read_text(encoding="utf-8").replace(
                "Nm, rad/s^2, rad/s, (min) rad, (max) rad", "units unspecified"
            ),
            encoding="utf-8",
        )
        try:
            load_active_urdf_limits(Path(directory), "a", "1.2")
        except ValueError as exc:
            assert "rad/s^2" in str(exc)
        else:
            raise AssertionError("ambiguous URDF acceleration units must fail closed")


def test_profile_above_manufacturer_limit_is_rejected() -> None:
    limits = load_active_urdf_limits(
        Path("/home/nvidia/rby1-sdk/models"), "a", "1.2"
    )
    velocity_limits = dict(limits.velocity_limits)
    velocity_limits["right_arm_0"] = 0.1
    reduced_limits = replace(limits, velocity_limits=velocity_limits)
    try:
        build_operational_profile("mild", reduced_limits)
    except ValueError as exc:
        assert "right_arm_0 velocity" in str(exc)
    else:
        raise AssertionError("a profile above the manufacturer limit must be rejected")


def test_offline_enabled_replay_uses_real_limiter_at_500hz() -> None:
    limits = load_active_urdf_limits(
        Path("/home/nvidia/rby1-sdk/models"), "a", "1.2"
    )
    profile = build_operational_profile("balanced", limits)
    names = list(ARM_JOINT_NAMES) + ["right_gripper_0", "left_gripper_0"]
    actions = np.zeros((3, len(names)), dtype=np.float64)
    actions[1:, :14] = 0.01
    actions[:, 14:] = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    stream = replay_profile(
        np.asarray([10.0, 10.071, 10.151]),
        names,
        actions,
        "irregular_log_dt",
        profile,
    )
    assert np.allclose(np.diff(stream.times), 0.002)
    assert not np.array_equal(stream.command, stream.reference)
    for index, name in enumerate(ARM_JOINT_NAMES):
        assert np.max(np.abs(stream.velocity[:, index])) <= profile.velocity_limits[name] + 1e-8
        assert np.max(np.abs(stream.acceleration[:, index])) <= profile.acceleration_limits[name] + 1e-8
        assert np.max(np.abs(stream.jerk[:, index])) <= profile.jerk_limits[name] + 1e-8
    assert stream.invalid_samples == 0


def load_tests(_loader, _tests, _pattern):
    functions = [
        test_cosine_ramp_full_action_regression_including_grippers,
        test_constant_input_does_not_move,
        test_missing_joint_limits_are_rejected,
        test_step_input_respects_velocity_acceleration_and_jerk,
        test_irregular_dt_respects_all_limits,
        test_new_target_has_no_position_jump,
        test_gripper_passthrough_is_immediate,
        test_joint_position_limit_and_nan_fallback,
        test_stale_actions_are_dropped_by_scheduled_time,
        test_control_delay_drops_old_waypoints_without_burst,
        test_async_logger_flushes_and_sanitizes_nan,
        test_active_urdf_parser_extracts_all_arm_limits,
        test_arm_joint_map_must_be_exactly_complete,
        test_wrong_model_or_urdf_fails_closed,
        test_urdf_without_explicit_si_acceleration_units_fails_closed,
        test_profile_above_manufacturer_limit_is_rejected,
        test_offline_enabled_replay_uses_real_limiter_at_500hz,
    ]
    return unittest.TestSuite(unittest.FunctionTestCase(function) for function in functions)
