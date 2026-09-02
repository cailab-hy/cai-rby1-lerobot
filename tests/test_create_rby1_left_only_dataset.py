from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "create_rby1_left_only_dataset.py"
SPEC = importlib.util.spec_from_file_location("create_rby1_left_only_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


ARM_INDICES = module.ArmIndices(right=tuple(range(7)), left=tuple(range(7, 14)))


def trajectory(right_scale: float, left_scale: float, frames: int = 8) -> np.ndarray:
    action = np.zeros((frames, 16), dtype=np.float32)
    ramp = np.arange(frames, dtype=np.float32)[:, None]
    action[:, :7] = ramp * right_scale
    action[:, 7:14] = ramp * left_scale
    return action


def classify(action: np.ndarray) -> str:
    return module.classify_actions(action, ARM_INDICES, 3.0)[-1]


class CreateLeftOnlyDatasetTests(unittest.TestCase):
    def test_right_stationary_left_moving_is_left(self) -> None:
        self.assertEqual(classify(trajectory(0.0, 1.0)), "left")

    def test_left_stationary_right_moving_is_right(self) -> None:
        self.assertEqual(classify(trajectory(1.0, 0.0)), "right")

    def test_similar_arm_movement_is_ambiguous(self) -> None:
        self.assertEqual(classify(trajectory(1.0, 1.1)), "ambiguous")

    def test_zero_movement_is_ambiguous(self) -> None:
        self.assertEqual(classify(trajectory(0.0, 0.0)), "ambiguous")

    def test_episode_reindex_is_contiguous(self) -> None:
        self.assertEqual(module.build_episode_mapping([2, 8, 15]), {2: 0, 8: 1, 15: 2})

    def test_frame_index_resets_per_episode(self) -> None:
        first = module.reindex_frame_metadata(3, 0, 0, 0)
        second = module.reindex_frame_metadata(2, 1, 3, 1)
        self.assertEqual(first["frame_index"].tolist(), [0, 1, 2])
        self.assertEqual(second["frame_index"].tolist(), [0, 1])
        self.assertEqual(second["index"].tolist(), [3, 4])
        self.assertEqual(second["episode_index"].tolist(), [1, 1])

    def test_diagnostics_do_not_modify_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            (source / "meta").mkdir(parents=True)
            (source / "data" / "chunk-000").mkdir(parents=True)
            (source / "videos" / "camera" / "chunk-000").mkdir(parents=True)
            info = {"total_episodes": 400, "total_frames": 119419}
            (source / "meta" / "info.json").write_text(
                '{"total_episodes": 400, "total_frames": 119419}', encoding="utf-8"
            )
            (source / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"data")
            (source / "videos" / "camera" / "chunk-000" / "file-000.mp4").write_bytes(b"video")
            before = module.capture_source_snapshot(source, info)
            module.write_diagnostics(root / "outputs", [], [])
            after = module.assert_source_unchanged(before, source)
            self.assertEqual(after, before)

    def test_task_count_and_order_preservation(self) -> None:
        module.validate_task_mapping({0: module.CUP_TASK, 1: module.BOWL_TASK})

    def test_stale_episode_metadata_locations_are_repaired_in_memory(self) -> None:
        import datasets
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.table({"episode_index": [0, 1, 2]}), path)

            class FakeMeta:
                total_episodes = 3
                episodes = datasets.Dataset.from_dict(
                    {
                        "episode_index": [0, 1, 2],
                        "meta/episodes/chunk_index": [0, 0, 0],
                        "meta/episodes/file_index": [0, 1, 2],
                    }
                )

            class FakeDataset:
                pass

            fake_dataset = FakeDataset()
            fake_dataset.root = root
            fake_dataset.meta = FakeMeta()
            corrections = module.repair_in_memory_episode_metadata_locations(fake_dataset)
            self.assertEqual(corrections, 2)
            self.assertEqual(fake_dataset.meta.episodes["meta/episodes/file_index"], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
