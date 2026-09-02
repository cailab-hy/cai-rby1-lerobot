#!/usr/bin/env python3
"""Trace SmolVLA state shapes without loading weights or changing runtime code.

The diagnostic exercises the local LeRobotDataset, the checkpoint's saved
preprocessor (including its normalizer state), and SmolVLAPolicy.prepare_state.
It deliberately does not instantiate the VLM or modify the checkpoint.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader


DEFAULT_CHECKPOINT = Path(
    "/home/cai/rby1-lerobot/cai-rby1-lerobot/outputs/"
    "smolVLA_bs32_ViT_VLM_expert_left_only/checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path(
    "/home/cai/rby1-lerobot/dataset/datasets_table_bushing_v3_left_only"
)
DEFAULT_FROZEN_BATCH = Path("/home/cai/frozen_smolvla_batch.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default="local/rby1-table-bussing-v3")
    parser.add_argument("--frozen-batch", type=Path, default=DEFAULT_FROZEN_BATCH)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def tensor_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor, got {type(value).__name__}")
    return tuple(value.shape)


def transition_state(transition: dict[Any, Any]) -> torch.Tensor:
    from lerobot.types import TransitionKey

    observation = transition[TransitionKey.OBSERVATION]
    if not isinstance(observation, dict):
        raise TypeError("Processor transition has no observation dictionary")
    state = observation.get("observation.state")
    if not isinstance(state, torch.Tensor):
        raise TypeError("Processor transition has no tensor observation.state")
    return state


def run_preprocessor_with_trace(
    preprocessor: Any, batch: dict[str, Any]
) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...], Any]:
    from lerobot.processor import NormalizerProcessorStep

    transition = preprocessor.to_transition(batch)
    normalizer_input_shape: tuple[int, ...] | None = None
    normalizer_output_shape: tuple[int, ...] | None = None
    normalizer = None

    for step in preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            normalizer = step
            normalizer_input_shape = tensor_shape(transition_state(transition))
        transition = step(transition)
        if isinstance(step, NormalizerProcessorStep):
            normalizer_output_shape = tensor_shape(transition_state(transition))

    if normalizer is None or normalizer_input_shape is None or normalizer_output_shape is None:
        raise RuntimeError("Checkpoint preprocessor has no normalizer_processor step")
    return (
        preprocessor.to_output(transition),
        normalizer_input_shape,
        normalizer_output_shape,
        normalizer,
    )


def source_location(callable_obj: Any) -> str:
    path = inspect.getsourcefile(callable_obj)
    _, line = inspect.getsourcelines(callable_obj)
    return f"{path}:{line}"


def main() -> int:
    args = parse_args()

    # Deferred imports make --help and syntax checks independent of optional
    # SmolVLA dependencies.
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = "cpu"

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(config, metadata)

    # Plain __getitem__ shows the stored/raw per-frame feature. The training
    # dataset uses SmolVLA's delta timestamps, adding a one-step time axis.
    raw_dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )
    raw_item = raw_dataset[args.sample_index]

    training_dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    training_item = training_dataset[args.sample_index]
    training_batch = next(iter(DataLoader(training_dataset, batch_size=1, num_workers=0)))

    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    processed_single, single_norm_in, single_norm_out, _ = run_preprocessor_with_trace(
        preprocessor, raw_item
    )
    processed_training, norm_in, norm_out, normalizer = run_preprocessor_with_trace(
        preprocessor, training_batch
    )

    frozen_batch = torch.load(args.frozen_batch, map_location="cpu", weights_only=False)
    if not isinstance(frozen_batch, dict):
        raise TypeError(f"Frozen batch must be a dict, got {type(frozen_batch).__name__}")
    frozen_state = frozen_batch.get("observation.state")
    if not isinstance(frozen_state, torch.Tensor):
        raise TypeError("Frozen batch has no tensor observation.state")

    # Call the exact policy method without constructing the 500M VLM. The
    # method only reads self.config and the supplied batch.
    policy_probe = SimpleNamespace(config=config)
    prepared_training = SmolVLAPolicy.prepare_state(policy_probe, processed_training)
    prepared_frozen = SmolVLAPolicy.prepare_state(policy_probe, frozen_batch)

    training_state = processed_training["observation.state"]
    training_before_padding = (
        training_state[:, -1, :] if training_state.ndim > 2 else training_state
    )
    frozen_before_padding = frozen_state[:, -1, :] if frozen_state.ndim > 2 else frozen_state

    declared_state_shape = tuple(config.input_features["observation.state"].shape)
    state_stats = normalizer._tensor_stats["observation.state"]
    stats_shapes = {name: tensor_shape(value) for name, value in state_stats.items()}

    training_prefix_preserved = torch.equal(
        prepared_training[..., : training_before_padding.shape[-1]], training_before_padding
    )
    frozen_prefix_preserved = torch.equal(
        prepared_frozen[..., : frozen_before_padding.shape[-1]], frozen_before_padding
    )
    padded_tail_is_zero = bool(
        torch.count_nonzero(prepared_frozen[..., frozen_before_padding.shape[-1] :]) == 0
    )
    config_shape_causes_truncation = not (
        single_norm_in[-1] == 16
        and single_norm_out[-1] == 16
        and norm_in[-1] == 16
        and norm_out[-1] == 16
        and processed_single["observation.state"].shape[-1] == 16
        and training_state.shape[-1] == 16
        and frozen_state.shape[-1] == 16
        and prepared_training.shape[-1] == config.max_state_dim
        and prepared_frozen.shape[-1] == config.max_state_dim
        and training_prefix_preserved
        and frozen_prefix_preserved
        and padded_tail_is_zero
    )

    print("SmolVLA state-shape trace")
    print(f"Dataset metadata state shape       : {tuple(metadata.features['observation.state']['shape'])}")
    print(f"Dataset/raw __getitem__ state      : {tensor_shape(raw_item['observation.state'])}")
    print(f"Single-observation preproc input   : {tensor_shape(raw_item['observation.state'])}")
    print(f"Single-observation normalizer in   : {single_norm_in}")
    print(f"Single-observation normalizer out  : {single_norm_out}")
    print(
        f"Single-observation policy state    : "
        f"{tensor_shape(processed_single['observation.state'])}"
    )
    print(f"Training __getitem__ state         : {tensor_shape(training_item['observation.state'])}")
    print(f"Training DataLoader state          : {tensor_shape(training_batch['observation.state'])}")
    print(f"Policy preprocessor input state    : {tensor_shape(training_batch['observation.state'])}")
    print(f"Normalizer input state             : {norm_in}")
    print(f"Normalizer output state            : {norm_out}")
    print(f"SmolVLAPolicy.forward batch state  : {tensor_shape(training_state)}")
    print(f"Before training state padding      : {tensor_shape(training_before_padding)}")
    print(f"After training state padding       : {tensor_shape(prepared_training)}")
    print(f"Frozen policy-ready batch state    : {tensor_shape(frozen_state)}")
    print(f"Before frozen state padding        : {tensor_shape(frozen_before_padding)}")
    print(f"After frozen state padding         : {tensor_shape(prepared_frozen)}")
    print(f"Config declared state shape        : {declared_state_shape}")
    print(f"Config max_state_dim               : {config.max_state_dim}")
    print(f"Normalizer state stats shapes      : {stats_shapes}")
    print(f"Training state prefix preserved    : {training_prefix_preserved}")
    print(f"Frozen state prefix preserved      : {frozen_prefix_preserved}")
    print(f"Padded frozen tail is zero         : {padded_tail_is_zero}")
    print(f"Config shape causes truncation     : {config_shape_causes_truncation}")
    print("\nExecuted source")
    print(f"LeRobotDataset.__getitem__         : {source_location(LeRobotDataset.__getitem__)}")
    print(f"NormalizerProcessorStep.__call__   : {source_location(type(normalizer).__call__)}")
    print(f"SmolVLAPolicy.prepare_state        : {source_location(SmolVLAPolicy.prepare_state)}")
    print(f"pad_vector                         : {source_location(pad_vector)}")

    if config_shape_causes_truncation:
        raise RuntimeError("Observed a state truncation or failed prefix-preservation check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
