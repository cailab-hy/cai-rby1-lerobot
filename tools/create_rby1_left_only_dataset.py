#!/usr/bin/env python3
"""Select confident left-arm RB-Y1 episodes and build a standalone LeRobot dataset.

The default invocation is intentionally non-destructive: pass ``--dry-run`` to
classify episodes and write diagnostic CSV files.  Dataset creation always uses
a separate staging directory and never writes to the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CUP_TASK = "Pick up the cup and place it in the box."
BOWL_TASK = "Pick up the bowl and place it in the box."
TASK_LABELS = {CUP_TASK: "cup", BOWL_TASK: "bowl"}
EXPECTED_TASKS = (CUP_TASK, BOWL_TASK)
EPSILON = 1e-12


class SelectionError(RuntimeError):
    """Raised when the source or classification does not meet safety checks."""


@dataclass(frozen=True)
class ArmIndices:
    right: tuple[int, ...]
    left: tuple[int, ...]


@dataclass(frozen=True)
class MotionScores:
    right_l1: float
    left_l1: float
    right_l2: float
    left_l2: float


@dataclass(frozen=True)
class EpisodeClassification:
    episode_index: int
    task: str
    right_motion_score: float
    left_motion_score: float
    left_over_right_ratio: float
    right_over_left_ratio: float
    right_motion_score_l2: float
    left_motion_score_l2: float
    left_over_right_ratio_l2: float
    right_over_left_ratio_l2: float
    classified_arm_l1: str
    classified_arm_l2: str
    classified_arm: str
    num_frames: int


@dataclass(frozen=True)
class SourceSnapshot:
    total_episodes: int
    total_frames: int
    data_file_count: int
    video_file_count: int
    data_total_bytes: int
    video_total_bytes: int
    metadata_sha256: str


def finite_ratio(numerator: float, denominator: float) -> float:
    if denominator <= EPSILON:
        return math.inf if numerator > EPSILON else 1.0
    return numerator / denominator


def classify_metric(left: float, right: float, ratio_threshold: float) -> str:
    if left <= EPSILON and right <= EPSILON:
        return "ambiguous"
    if left > right and finite_ratio(left, right) >= ratio_threshold:
        return "left"
    if right > left and finite_ratio(right, left) >= ratio_threshold:
        return "right"
    return "ambiguous"


def compute_motion_scores(actions: np.ndarray, arm_indices: ArmIndices) -> MotionScores:
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise SelectionError(f"Expected a 2-D action trajectory, got shape {actions.shape}.")
    if actions.shape[0] < 2:
        return MotionScores(0.0, 0.0, 0.0, 0.0)
    delta_right = np.diff(actions[:, arm_indices.right], axis=0)
    delta_left = np.diff(actions[:, arm_indices.left], axis=0)
    return MotionScores(
        right_l1=float(np.abs(delta_right).sum(dtype=np.float64)),
        left_l1=float(np.abs(delta_left).sum(dtype=np.float64)),
        right_l2=float(np.linalg.norm(delta_right, axis=1).sum(dtype=np.float64)),
        left_l2=float(np.linalg.norm(delta_left, axis=1).sum(dtype=np.float64)),
    )


def classify_actions(
    actions: np.ndarray,
    arm_indices: ArmIndices,
    ratio_threshold: float,
) -> tuple[MotionScores, str, str, str]:
    scores = compute_motion_scores(actions, arm_indices)
    l1_class = classify_metric(scores.left_l1, scores.right_l1, ratio_threshold)
    l2_class = classify_metric(scores.left_l2, scores.right_l2, ratio_threshold)
    combined = l1_class if l1_class == l2_class else "ambiguous"
    return scores, l1_class, l2_class, combined


def infer_arm_indices(action_feature: dict[str, Any]) -> ArmIndices:
    names = action_feature.get("names")
    shape = action_feature.get("shape")
    if not isinstance(names, list) or not isinstance(shape, list) or len(shape) != 1:
        raise SelectionError("The action feature must have one-dimensional shape and named dimensions.")
    if shape[0] != len(names):
        raise SelectionError(f"Action shape {shape} does not match {len(names)} action names.")

    def collect(prefix: str) -> tuple[int, ...]:
        found: list[tuple[int, int]] = []
        for index, name in enumerate(names):
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            suffix = name.removeprefix(prefix)
            if suffix.isdigit():
                found.append((int(suffix), index))
        found.sort()
        dofs = [dof for dof, _ in found]
        if dofs != list(range(7)):
            raise SelectionError(
                f"Expected exactly {prefix}0..{prefix}6 in action names, found DoFs {dofs}."
            )
        return tuple(index for _, index in found)

    indices = ArmIndices(right=collect("right_arm_"), left=collect("left_arm_"))
    if set(indices.right) & set(indices.left):
        raise SelectionError("Left and right arm action dimensions overlap.")
    return indices


def build_episode_mapping(source_episode_indices: Sequence[int]) -> dict[int, int]:
    if len(source_episode_indices) != len(set(source_episode_indices)):
        raise SelectionError("Selected source episode indices contain duplicates.")
    return {source_index: new_index for new_index, source_index in enumerate(source_episode_indices)}


def reindex_frame_metadata(
    num_frames: int,
    new_episode_index: int,
    global_start_index: int,
    task_index: int,
) -> dict[str, np.ndarray]:
    return {
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "episode_index": np.full(num_frames, new_episode_index, dtype=np.int64),
        "index": np.arange(global_start_index, global_start_index + num_frames, dtype=np.int64),
        "task_index": np.full(num_frames, task_index, dtype=np.int64),
    }


def load_info(source_root: Path) -> dict[str, Any]:
    info_path = source_root / "meta" / "info.json"
    if not info_path.is_file():
        raise SelectionError(f"Missing source metadata: {info_path}")
    with info_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_source_info(
    info: dict[str, Any],
    expected_total_episodes: int,
    expected_total_frames: int,
    expected_fps: int,
) -> ArmIndices:
    actual = (info.get("total_episodes"), info.get("total_frames"), info.get("fps"))
    expected = (expected_total_episodes, expected_total_frames, expected_fps)
    if actual != expected:
        raise SelectionError(
            "Source sanity check failed: "
            f"expected episodes/frames/fps={expected}, got {actual}. No selection was performed."
        )
    if info.get("total_tasks") != 2:
        raise SelectionError(f"Expected total_tasks=2, got {info.get('total_tasks')}.")
    features = info.get("features", {})
    if "action" not in features or "observation.state" not in features:
        raise SelectionError("Source features are missing action or observation.state.")
    if features["observation.state"].get("shape") != [16]:
        raise SelectionError(
            f"Expected observation.state shape [16], got {features['observation.state'].get('shape')}."
        )
    return infer_arm_indices(features["action"])


def _hash_metadata(source_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((source_root / "meta").rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(source_root).as_posix().encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def capture_source_snapshot(source_root: Path, info: dict[str, Any] | None = None) -> SourceSnapshot:
    info = info if info is not None else load_info(source_root)
    data_files = [path for path in (source_root / "data").rglob("*.parquet") if path.is_file()]
    video_files = [path for path in (source_root / "videos").rglob("*.mp4") if path.is_file()]
    return SourceSnapshot(
        total_episodes=int(info["total_episodes"]),
        total_frames=int(info["total_frames"]),
        data_file_count=len(data_files),
        video_file_count=len(video_files),
        data_total_bytes=sum(path.stat().st_size for path in data_files),
        video_total_bytes=sum(path.stat().st_size for path in video_files),
        metadata_sha256=_hash_metadata(source_root),
    )


def assert_source_unchanged(before: SourceSnapshot, source_root: Path) -> SourceSnapshot:
    after = capture_source_snapshot(source_root)
    if after != before:
        raise SelectionError(f"Source dataset changed during the operation. Before={before}; after={after}")
    return after


def validate_task_mapping(task_index_to_name: dict[int, str]) -> None:
    if set(task_index_to_name) != {0, 1}:
        raise SelectionError(f"Expected task indices {{0, 1}}, got {set(task_index_to_name)}.")
    if tuple(task_index_to_name[index] for index in (0, 1)) != EXPECTED_TASKS:
        raise SelectionError(
            "Task mapping differs from the required source mapping: "
            f"{task_index_to_name}."
        )


def _numpy_column(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist())


def read_episode_arrays(source_dataset: Any, episode_index: int) -> dict[str, np.ndarray]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    relative_path = source_dataset.meta.get_data_file_path(episode_index)
    table = pq.read_table(
        source_dataset.root / relative_path,
        columns=[
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    )
    mask = pc.equal(table["episode_index"], episode_index)
    table = table.filter(mask)
    return {name: _numpy_column(table, name) for name in table.column_names}


def classify_source_dataset(
    source_dataset: Any,
    arm_indices: ArmIndices,
    ratio_threshold: float,
) -> list[EpisodeClassification]:
    task_index_to_name = {
        int(row.task_index): str(task)
        for task, row in source_dataset.meta.tasks.iterrows()
    }
    validate_task_mapping(task_index_to_name)
    results: list[EpisodeClassification] = []
    for episode_index in range(source_dataset.meta.total_episodes):
        arrays = read_episode_arrays(source_dataset, episode_index)
        num_frames = len(arrays["action"])
        metadata_length = int(source_dataset.meta.episodes[episode_index]["length"])
        if num_frames != metadata_length:
            raise SelectionError(
                f"Episode {episode_index}: parquet frames={num_frames}, metadata length={metadata_length}."
            )
        if not np.array_equal(arrays["frame_index"], np.arange(num_frames)):
            raise SelectionError(f"Episode {episode_index}: source frame_index is not episode-local contiguous.")
        expected_timestamps = np.arange(num_frames, dtype=np.float64) / source_dataset.meta.fps
        if not np.allclose(arrays["timestamp"], expected_timestamps, atol=1e-5, rtol=0):
            raise SelectionError(f"Episode {episode_index}: timestamp != frame_index/fps.")
        task_indices = np.unique(arrays["task_index"])
        if len(task_indices) != 1 or int(task_indices[0]) not in task_index_to_name:
            raise SelectionError(f"Episode {episode_index}: invalid task indices {task_indices.tolist()}.")
        task = task_index_to_name[int(task_indices[0])]
        scores, l1_class, l2_class, combined = classify_actions(
            arrays["action"], arm_indices, ratio_threshold
        )
        results.append(
            EpisodeClassification(
                episode_index=episode_index,
                task=task,
                right_motion_score=scores.right_l1,
                left_motion_score=scores.left_l1,
                left_over_right_ratio=finite_ratio(scores.left_l1, scores.right_l1),
                right_over_left_ratio=finite_ratio(scores.right_l1, scores.left_l1),
                right_motion_score_l2=scores.right_l2,
                left_motion_score_l2=scores.left_l2,
                left_over_right_ratio_l2=finite_ratio(scores.left_l2, scores.right_l2),
                right_over_left_ratio_l2=finite_ratio(scores.right_l2, scores.left_l2),
                classified_arm_l1=l1_class,
                classified_arm_l2=l2_class,
                classified_arm=combined,
                num_frames=num_frames,
            )
        )
    return results


def classification_counts(results: Sequence[EpisodeClassification]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        label = TASK_LABELS.get(result.task, result.task)
        counts[label][result.classified_arm] += 1
        counts["total"][result.classified_arm] += 1
    return counts


def validate_selection(
    results: Sequence[EpisodeClassification],
    expected_counts: dict[tuple[str, str], int],
) -> list[EpisodeClassification]:
    counts = classification_counts(results)
    ambiguous = [result for result in results if result.classified_arm == "ambiguous"]
    mismatches = []
    for (task_label, arm), expected in expected_counts.items():
        actual = counts[task_label][arm]
        if actual != expected:
            mismatches.append(f"{task_label}-{arm}: expected {expected}, got {actual}")
    if ambiguous:
        details = ", ".join(
            f"ep={item.episode_index} L1(L/R)={item.left_over_right_ratio:.6g} "
            f"L2(L/R)={item.left_over_right_ratio_l2:.6g}"
            for item in ambiguous
        )
        mismatches.append(f"ambiguous episodes ({len(ambiguous)}): {details}")
    if mismatches:
        raise SelectionError("Classification validation failed; do not create destination. " + "; ".join(mismatches))
    return [result for result in results if result.classified_arm == "left"]


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def write_diagnostics(
    output_dir: Path,
    results: Sequence[EpisodeClassification],
    selected: Sequence[EpisodeClassification],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    classification_path = output_dir / "episode_arm_classification.csv"
    classification_fields = list(asdict(results[0]).keys()) if results else []
    with classification_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=classification_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: _csv_value(value) for key, value in asdict(result).items()})

    with (output_dir / "selected_left_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["source_episode_index", "task", "classified_arm", "source_num_frames"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in selected:
            writer.writerow(
                {
                    "source_episode_index": result.episode_index,
                    "task": result.task,
                    "classified_arm": result.classified_arm,
                    "source_num_frames": result.num_frames,
                }
            )

    mapping = build_episode_mapping([result.episode_index for result in selected])
    with (output_dir / "episode_index_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["new_episode_index", "source_episode_index", "task", "classified_arm"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in selected:
            writer.writerow(
                {
                    "new_episode_index": mapping[result.episode_index],
                    "source_episode_index": result.episode_index,
                    "task": result.task,
                    "classified_arm": result.classified_arm,
                }
            )


def _format_indices(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def print_source_summary(info: dict[str, Any], snapshot: SourceSnapshot, arm_indices: ArmIndices) -> None:
    features = info["features"]
    image_features = [name for name, feature in features.items() if feature.get("dtype") == "video"]
    print("SOURCE DATASET")
    print(f"  total_episodes = {info['total_episodes']}")
    print(f"  total_frames = {info['total_frames']}")
    print(f"  fps = {info['fps']}")
    print(f"  total_tasks = {info['total_tasks']}")
    print(f"  feature_names = {list(features)}")
    print(f"  action_shape = {features['action']['shape']}")
    print(f"  observation.state_shape = {features['observation.state']['shape']}")
    print(f"  image_features = {image_features}")
    print(f"  action_names = {features['action']['names']}")
    print(f"  right_arm_action_indices = {list(arm_indices.right)}")
    print(f"  left_arm_action_indices = {list(arm_indices.left)}")
    print(f"  data_path = {info['data_path']}")
    print(f"  video_path = {info['video_path']}")
    print(f"  data_file_count = {snapshot.data_file_count}")
    print(f"  video_file_count = {snapshot.video_file_count}")
    print(f"  metadata_sha256 = {snapshot.metadata_sha256}")


def validate_source_file_counts(info: dict[str, Any], snapshot: SourceSnapshot) -> None:
    video_features = [
        name for name, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    expected_data_files = int(info["total_episodes"])
    expected_video_files = expected_data_files * len(video_features)
    if snapshot.data_file_count != expected_data_files:
        raise SelectionError(
            f"Expected one source parquet per episode ({expected_data_files}), "
            f"found {snapshot.data_file_count}."
        )
    if snapshot.video_file_count != expected_video_files:
        raise SelectionError(
            f"Expected one source video per episode/camera ({expected_video_files}), "
            f"found {snapshot.video_file_count}."
        )


def print_classification_summary(results: Sequence[EpisodeClassification]) -> None:
    counts = classification_counts(results)
    print("CLASSIFICATION")
    for task_label in ("cup", "bowl", "total"):
        print(
            f"  {task_label}: left={counts[task_label]['left']} "
            f"right={counts[task_label]['right']} ambiguous={counts[task_label]['ambiguous']}"
        )
    left_results = [result for result in results if result.classified_arm == "left"]
    ratios = np.asarray([result.left_over_right_ratio for result in left_results], dtype=np.float64)
    if len(ratios):
        print("LEFT L1 MOVEMENT RATIO DISTRIBUTION")
        print(f"  median = {np.median(ratios):.9g}")
        print(f"  p05 = {np.quantile(ratios, 0.05):.9g}")
        print(f"  minimum = {np.min(ratios):.9g}")
    for task in EXPECTED_TASKS:
        indices = [
            result.episode_index
            for result in left_results
            if result.task == task
        ]
        print(f"  {TASK_LABELS[task]}-left source episode indices = {_format_indices(indices)}")
    ambiguous = [result.episode_index for result in results if result.classified_arm == "ambiguous"]
    print(f"  ambiguous source episode indices = {_format_indices(ambiguous) if ambiguous else 'none'}")


def ensure_destination_available(destination_root: Path, overwrite: bool) -> None:
    if destination_root.exists() and not overwrite:
        raise SelectionError("ERROR: destination already exists. Use --overwrite only if explicitly requested.")
    if destination_root.resolve() == destination_root.parent.resolve():
        raise SelectionError(f"Unsafe destination path: {destination_root}")


def repair_in_memory_episode_metadata_locations(source_dataset: Any) -> int:
    """Align logical episode-metadata locations with physical parquet files.

    LeRobot 0.6.1's ``delete_episodes`` resolves per-episode statistics through
    ``meta/episodes/chunk_index`` and ``file_index``.  Some valid v3 datasets
    contain stale values in those two columns even though all episode rows are
    physically stored in a consolidated parquet.  Correct only the loaded
    Hugging Face Dataset object; never rewrite source metadata on disk.

    Returns the number of in-memory rows whose location was corrected.
    """
    import pyarrow.parquet as pq

    physical_locations: dict[int, tuple[int, int]] = {}
    episodes_root = source_dataset.root / "meta" / "episodes"
    for path in sorted(episodes_root.glob("chunk-*/*.parquet")):
        try:
            chunk_index = int(path.parent.name.removeprefix("chunk-"))
            file_index = int(path.stem.removeprefix("file-"))
        except ValueError as error:
            raise SelectionError(f"Invalid episode metadata path convention: {path}") from error
        episode_indices = pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist()
        for episode_index in episode_indices:
            episode_index = int(episode_index)
            if episode_index in physical_locations:
                raise SelectionError(
                    f"Episode {episode_index} occurs in multiple physical metadata parquet files."
                )
            physical_locations[episode_index] = (chunk_index, file_index)

    expected_indices = set(range(source_dataset.meta.total_episodes))
    if set(physical_locations) != expected_indices:
        missing = sorted(expected_indices - set(physical_locations))
        unexpected = sorted(set(physical_locations) - expected_indices)
        raise SelectionError(
            f"Physical episode metadata coverage mismatch: missing={missing}, unexpected={unexpected}."
        )

    episodes = source_dataset.meta.episodes
    chunk_column = "meta/episodes/chunk_index"
    file_column = "meta/episodes/file_index"
    corrected_chunks: list[int] = []
    corrected_files: list[int] = []
    corrections = 0
    for row in episodes:
        episode_index = int(row["episode_index"])
        chunk_index, file_index = physical_locations[episode_index]
        corrected_chunks.append(chunk_index)
        corrected_files.append(file_index)
        if int(row[chunk_column]) != chunk_index or int(row[file_column]) != file_index:
            corrections += 1

    if corrections:
        source_dataset.meta.episodes = (
            episodes.remove_columns([chunk_column, file_column])
            .add_column(chunk_column, corrected_chunks)
            .add_column(file_column, corrected_files)
        )
    return corrections


def _replace_arrow_column(table: Any, name: str, values: np.ndarray) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(name)
    if index < 0:
        raise SelectionError(f"Destination parquet is missing column {name}.")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def _normalized_file_location(index: int, chunks_size: int) -> tuple[int, int]:
    return divmod(index, chunks_size)


def normalize_destination_layout(
    destination_root: Path,
    source_task_to_index: dict[str, int],
) -> None:
    """Make every destination data/video filename match its new episode index."""
    import datasets
    import pyarrow.parquet as pq
    from lerobot.datasets.io_utils import load_nested_dataset, write_episodes

    info = load_info(destination_root)
    chunks_size = int(info["chunks_size"])
    episodes_dir = destination_root / "meta" / "episodes"
    episodes = load_nested_dataset(episodes_dir).sort("episode_index")
    expected_indices = list(range(len(episodes)))
    if episodes["episode_index"] != expected_indices:
        raise SelectionError("Official subset output did not produce contiguous episode indices.")

    normalized_data = destination_root / f".normalized-data-{uuid.uuid4().hex}"
    normalized_videos = destination_root / f".normalized-videos-{uuid.uuid4().hex}"
    rows: list[dict[str, Any]] = []
    global_index = 0
    seen_data_paths: set[Path] = set()
    seen_video_paths: set[Path] = set()

    for row in episodes:
        row = dict(row)
        new_episode_index = int(row["episode_index"])
        task_values = row["tasks"]
        if len(task_values) != 1 or task_values[0] not in source_task_to_index:
            raise SelectionError(f"Episode {new_episode_index}: invalid task metadata {task_values}.")
        task_index = source_task_to_index[task_values[0]]
        new_chunk, new_file = _normalized_file_location(new_episode_index, chunks_size)

        old_data = destination_root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        if old_data in seen_data_paths:
            raise SelectionError("Expected one parquet file per episode, but a destination data file is shared.")
        seen_data_paths.add(old_data)
        table = pq.read_table(old_data)
        num_frames = table.num_rows
        frame_metadata = reindex_frame_metadata(num_frames, new_episode_index, global_index, task_index)
        for column, values in frame_metadata.items():
            table = _replace_arrow_column(table, column, values)
        new_data = normalized_data / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.parquet"
        new_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_data, compression="snappy", use_dictionary=True)

        row["data/chunk_index"] = new_chunk
        row["data/file_index"] = new_file
        row["dataset_from_index"] = global_index
        row["dataset_to_index"] = global_index + num_frames
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        global_index += num_frames

        for video_key in [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]:
            old_video = destination_root / info["video_path"].format(
                video_key=video_key,
                chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                file_index=int(row[f"videos/{video_key}/file_index"]),
            )
            if old_video in seen_video_paths:
                raise SelectionError("Expected one video file per episode and camera, but a file is shared.")
            seen_video_paths.add(old_video)
            new_video = (
                normalized_videos
                / video_key
                / f"chunk-{new_chunk:03d}"
                / f"file-{new_file:03d}.mp4"
            )
            new_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(old_video, new_video)
            row[f"videos/{video_key}/chunk_index"] = new_chunk
            row[f"videos/{video_key}/file_index"] = new_file
        rows.append(row)

    normalized_meta_root = destination_root / f".normalized-meta-{uuid.uuid4().hex}"
    normalized_episodes = datasets.Dataset.from_list(rows, features=episodes.features)
    write_episodes(normalized_episodes, normalized_meta_root)

    shutil.rmtree(destination_root / "data")
    os.replace(normalized_data, destination_root / "data")
    shutil.rmtree(destination_root / "videos")
    os.replace(normalized_videos, destination_root / "videos")
    shutil.rmtree(episodes_dir)
    os.replace(normalized_meta_root / "meta" / "episodes", episodes_dir)
    shutil.rmtree(normalized_meta_root)


def _task_maps(dataset: Any) -> tuple[dict[int, str], dict[str, int]]:
    index_to_name = {int(row.task_index): str(task) for task, row in dataset.meta.tasks.iterrows()}
    validate_task_mapping(index_to_name)
    return index_to_name, {name: index for index, name in index_to_name.items()}


def validate_destination(
    source_dataset: Any,
    destination_root: Path,
    destination_repo_id: str,
    selected: Sequence[EpisodeClassification],
    arm_indices: ArmIndices,
    ratio_threshold: float,
    random_seed: int,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    destination = LeRobotDataset(
        repo_id=destination_repo_id,
        root=destination_root,
        download_videos=False,
    )
    if destination.meta.total_episodes != len(selected):
        raise SelectionError(
            f"Destination episodes={destination.meta.total_episodes}, expected {len(selected)}."
        )
    expected_frames = sum(item.num_frames for item in selected)
    if destination.meta.total_frames != expected_frames:
        raise SelectionError(f"Destination frames={destination.meta.total_frames}, expected {expected_frames}.")
    if destination.meta.total_tasks != 2:
        raise SelectionError(f"Destination total_tasks={destination.meta.total_tasks}, expected 2.")
    destination_indices = sorted(set(int(value) for value in destination.hf_dataset["episode_index"]))
    if destination_indices != list(range(len(selected))):
        raise SelectionError("Destination episode indices are not exactly 0..N-1.")
    global_indices = np.asarray(destination.hf_dataset["index"])
    if not np.array_equal(global_indices, np.arange(expected_frames)):
        raise SelectionError("Destination global index is not contiguous.")

    video_keys = destination.meta.video_keys
    for video_key in video_keys:
        files = list((destination_root / "videos" / video_key).rglob("*.mp4"))
        if len(files) != len(selected):
            raise SelectionError(f"{video_key}: found {len(files)} videos, expected {len(selected)}.")

    destination_results = classify_source_dataset(destination, arm_indices, ratio_threshold)
    destination_counts = classification_counts(destination_results)
    if destination_counts["total"] != Counter({"left": len(selected)}):
        raise SelectionError(f"Destination is not left-only: {destination_counts['total']}.")
    for task_label in ("cup", "bowl"):
        expected = sum(1 for item in selected if TASK_LABELS[item.task] == task_label)
        if destination_counts[task_label]["left"] != expected:
            raise SelectionError(f"Destination {task_label}-left count is incorrect.")

    rng = random.Random(random_seed)
    sample_size = min(10, len(selected))
    sampled_new_indices = rng.sample(range(len(selected)), sample_size)
    mapping = {new: item.episode_index for new, item in enumerate(selected)}
    max_action_diff = 0.0
    max_state_diff = 0.0
    for new_index in sampled_new_indices:
        source_arrays = read_episode_arrays(source_dataset, mapping[new_index])
        destination_arrays = read_episode_arrays(destination, new_index)
        max_action_diff = max(
            max_action_diff,
            float(np.max(np.abs(source_arrays["action"] - destination_arrays["action"]))),
        )
        max_state_diff = max(
            max_state_diff,
            float(
                np.max(
                    np.abs(source_arrays["observation.state"] - destination_arrays["observation.state"])
                )
            ),
        )
    if max_action_diff != 0.0 or max_state_diff != 0.0:
        raise SelectionError(
            f"Numerical identity failed: action diff={max_action_diff}, state diff={max_state_diff}."
        )
    ratios = np.asarray(
        [item.left_over_right_ratio for item in destination_results], dtype=np.float64
    )
    print("DESTINATION VALIDATION")
    print(f"  total_episodes = {destination.meta.total_episodes}")
    print(f"  total_frames = {destination.meta.total_frames}")
    print(f"  total_tasks = {destination.meta.total_tasks}")
    print(f"  camera_video_counts = {dict.fromkeys(video_keys, len(selected))}")
    print(f"  left/right ratio median={np.median(ratios):.9g} p05={np.quantile(ratios, .05):.9g} min={np.min(ratios):.9g}")
    print(f"  sampled_source_episode_indices = {[mapping[index] for index in sampled_new_indices]}")
    print(f"  max_abs_action_diff = {max_action_diff}")
    print(f"  max_abs_state_diff = {max_state_diff}")


def create_destination_dataset(
    source_dataset: Any,
    destination_root: Path,
    destination_repo_id: str,
    selected: Sequence[EpisodeClassification],
    arm_indices: ArmIndices,
    ratio_threshold: float,
    overwrite: bool,
    random_seed: int,
) -> None:
    from lerobot.datasets.dataset_tools import delete_episodes

    ensure_destination_available(destination_root, overwrite)
    staging_root = destination_root.parent / f".{destination_root.name}.building-{uuid.uuid4().hex}"
    all_indices = set(range(source_dataset.meta.total_episodes))
    selected_indices = {item.episode_index for item in selected}
    deleted_indices = sorted(all_indices - selected_indices)
    _, source_task_to_index = _task_maps(source_dataset)
    try:
        corrected_locations = repair_in_memory_episode_metadata_locations(source_dataset)
        if corrected_locations:
            print(
                "SOURCE METADATA COMPATIBILITY: corrected "
                f"{corrected_locations} in-memory episode metadata file references; "
                "source files were not modified."
            )
        delete_episodes(
            source_dataset,
            episode_indices=deleted_indices,
            output_dir=staging_root,
            repo_id=destination_repo_id,
        )
        normalize_destination_layout(staging_root, source_task_to_index)
        shutil.copy2(source_dataset.root / "meta" / "tasks.parquet", staging_root / "meta" / "tasks.parquet")
        validate_destination(
            source_dataset,
            staging_root,
            destination_repo_id,
            selected,
            arm_indices,
            ratio_threshold,
            random_seed,
        )
        if destination_root.exists():
            shutil.rmtree(destination_root)
        os.replace(staging_root, destination_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def make_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--destination-repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "rby1_left_only_dataset_selection")
    parser.add_argument("--arm-motion-ratio-threshold", type=float, default=3.0)
    parser.add_argument("--expected-total-episodes", type=int, default=400)
    parser.add_argument("--expected-total-frames", type=int, default=119419)
    parser.add_argument("--expected-fps", type=int, default=15)
    parser.add_argument("--expected-cup-left", type=int, default=100)
    parser.add_argument("--expected-cup-right", type=int, default=100)
    parser.add_argument("--expected-bowl-left", type=int, default=100)
    parser.add_argument("--expected-bowl-right", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260901)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.arm_motion_ratio_threshold <= 1.0:
        raise SelectionError("--arm-motion-ratio-threshold must be greater than 1.0.")
    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    if source_root == destination_root or source_root in destination_root.parents:
        raise SelectionError("Destination must not be the source or a child of the source directory.")

    info = load_info(source_root)
    arm_indices = validate_source_info(
        info,
        args.expected_total_episodes,
        args.expected_total_frames,
        args.expected_fps,
    )
    before = capture_source_snapshot(source_root, info)
    validate_source_file_counts(info, before)
    print(f"installed_lerobot_version = {importlib.metadata.version('lerobot')}")
    print_source_summary(info, before, arm_indices)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        source_dataset = LeRobotDataset(
            repo_id=args.source_repo_id,
            root=source_root,
            download_videos=False,
        )
        task_index_to_name, _ = _task_maps(source_dataset)
        print(f"  tasks = {[task_index_to_name[index] for index in sorted(task_index_to_name)]}")
        results = classify_source_dataset(source_dataset, arm_indices, args.arm_motion_ratio_threshold)
        print_classification_summary(results)
        preliminary_selected = [result for result in results if result.classified_arm == "left"]
        write_diagnostics(args.output_dir.resolve(), results, preliminary_selected)
        print(f"diagnostics_dir = {args.output_dir.resolve()}")
        print(f"selected_left_episodes = {len(preliminary_selected)}")
        expected_counts = {
            ("cup", "left"): args.expected_cup_left,
            ("cup", "right"): args.expected_cup_right,
            ("bowl", "left"): args.expected_bowl_left,
            ("bowl", "right"): args.expected_bowl_right,
        }
        selected = validate_selection(results, expected_counts)

        if args.dry_run:
            print("DRY RUN COMPLETE: destination dataset was not created.")
        else:
            create_destination_dataset(
                source_dataset,
                destination_root,
                args.destination_repo_id,
                selected,
                arm_indices,
                args.arm_motion_ratio_threshold,
                args.overwrite,
                args.random_seed,
            )
            print(f"destination_root = {destination_root}")
    finally:
        after = assert_source_unchanged(before, source_root)
        print("SOURCE AFTER")
        print(f"  episodes={after.total_episodes} frames={after.total_frames}")
        print(f"  data_files={after.data_file_count} video_files={after.video_file_count}")
        print("  source_unchanged=True")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SelectionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
