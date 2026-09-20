"""Deterministic, parent-grouped development partitions for falcon-x data."""

from __future__ import annotations

import random
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data import DATASET_FORMAT, PathSource
from .dataset import FalconXDataset, NativeId

PARTITION_FORMAT = "falcon-partitions"
PARTITION_ROLES = ("train", "validation", "calibration")


def _id_key(value: NativeId) -> tuple[int, str]:
    return (0, str(value)) if isinstance(value, int) else (1, value)


def _identifier(value: Any, context: str) -> NativeId:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{context} must be an integer or non-empty string")
    if isinstance(value, int) and value < 1:
        raise ValueError(f"{context} integer IDs must be positive")
    if isinstance(value, str) and not value:
        raise ValueError(f"{context} string IDs must be non-empty")
    return value


@dataclass(frozen=True)
class ParentGroup:
    source_image_id: NativeId
    role: str
    image_ids: tuple[NativeId, ...]
    source_instance_counts: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image_id": self.source_image_id,
            "role": self.role,
            "image_ids": list(self.image_ids),
            "source_instance_counts": list(self.source_instance_counts),
        }


@dataclass(frozen=True)
class ParentPartitions:
    seed: int
    validation_percent: int
    calibration_percent: int
    groups: tuple[ParentGroup, ...]
    report: Mapping[str, Any]
    _dataset_root: Path = field(repr=False, compare=False)

    def image_ids(self, role: str) -> frozenset[NativeId]:
        _role(role)
        return frozenset(
            image_id
            for group in self.groups
            if group.role == role
            for image_id in group.image_ids
        )

    def source_image_ids(self, role: str) -> frozenset[NativeId]:
        _role(role)
        return frozenset(
            group.source_image_id for group in self.groups if group.role == role
        )

    def to_dict(self) -> dict[str, Any]:
        document = {
            "format": PARTITION_FORMAT,
            "source_split": "train",
            "seed": self.seed,
            "validation_percent": self.validation_percent,
            "calibration_percent": self.calibration_percent,
            "groups": [group.to_dict() for group in self.groups],
            "report": dict(self.report),
        }
        return document


@dataclass(frozen=True)
class PartitionSelection:
    role: str
    image_ids: frozenset[NativeId]
    source_image_ids: frozenset[NativeId]
    report: Mapping[str, Any]
    source_path: Path = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "image_ids": sorted(self.image_ids, key=_id_key),
            "source_image_ids": sorted(self.source_image_ids, key=_id_key),
            **dict(self.report),
        }


def _role(value: str) -> str:
    if value not in PARTITION_ROLES:
        raise ValueError(f"partition role must be one of {PARTITION_ROLES!r}")
    return value


def _percentage(value: int, name: str, *, allow_zero: bool) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= 99
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and 99")
    return value


def _target_count(total: int, percent: int) -> int:
    if percent == 0:
        return 0
    return max(1, (total * percent + 50) // 100)


def _stratified_selection(
    groups: Mapping[tuple[int, int, int], Sequence[NativeId]],
    count: int,
    *,
    seed: int,
    namespace: str,
    excluded: frozenset[NativeId] = frozenset(),
) -> frozenset[NativeId]:
    available = {
        stratum: [item for item in values if item not in excluded]
        for stratum, values in groups.items()
    }
    available = {key: values for key, values in available.items() if values}
    total = sum(len(values) for values in available.values())
    if not 0 <= count <= total:
        raise ValueError(f"cannot select {count} {namespace} groups from {total}")
    if count == 0:
        return frozenset()

    give_each_stratum = count >= len(available)
    minimum = 1 if give_each_stratum else 0
    quotas = {stratum: minimum for stratum in available}
    capacity = {stratum: len(values) - minimum for stratum, values in available.items()}
    remaining_target = count - sum(quotas.values())
    total_capacity = sum(capacity.values())
    if remaining_target and not total_capacity:
        raise AssertionError("internal stratified partition capacity invariant failed")
    if total_capacity:
        for stratum in available:
            quotas[stratum] += remaining_target * capacity[stratum] // total_capacity
    remaining = count - sum(quotas.values())
    rng = random.Random(f"{seed}:{namespace}")
    tie_breaks = {stratum: rng.random() for stratum in sorted(available)}
    remainders = sorted(
        available,
        key=lambda stratum: (
            -(
                remaining_target * capacity[stratum] % total_capacity
                if total_capacity
                else 0
            ),
            tie_breaks[stratum],
        ),
    )
    for stratum in remainders[:remaining]:
        quotas[stratum] += 1

    selected: set[NativeId] = set()
    for stratum, values in sorted(available.items()):
        ranked = sorted(values, key=_id_key)
        rng.shuffle(ranked)
        selected.update(ranked[: quotas[stratum]])
    if len(selected) != count:
        raise AssertionError("internal stratified partition quota invariant failed")
    return frozenset(selected)


def _dataset_groups(
    dataset: FalconXDataset,
) -> tuple[
    dict[NativeId, tuple[NativeId, ...]],
    dict[NativeId, tuple[int, int, int]],
]:
    if getattr(dataset, "format", None) != DATASET_FORMAT:
        raise ValueError("parent partitions require a falcon-x dataset")
    manifest = dataset.manifest
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, Mapping) or train.get("usage") != "training":
        raise ValueError("parent partitions require the declared train split")

    images = dataset.images("train")
    sources = {row["id"]: row for row in images if "source_image_id" not in row}
    grouped: dict[NativeId, list[NativeId]] = {
        source_id: [source_id] for source_id in sources
    }
    for row in images:
        if "source_image_id" not in row:
            continue
        source_id = row["source_image_id"]
        if source_id not in grouped:
            raise ValueError(f"counterfactual {row['id']!r} has no train parent")
        grouped[source_id].append(row["id"])
    memberships = {
        source_id: tuple(sorted(image_ids, key=_id_key))
        for source_id, image_ids in grouped.items()
    }
    instance_counts: dict[NativeId, tuple[int, int, int]] = {}
    for source_id in sources:
        counts = Counter(
            dataset.canonical_category_id(row["category_id"])
            for row in dataset.present_objects("train", source_id)
        )
        instance_counts[source_id] = tuple(counts.get(index, 0) for index in (1, 2, 3))
    return memberships, instance_counts


def _report(groups: Sequence[ParentGroup]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in PARTITION_ROLES:
        selected = [group for group in groups if group.role == role]
        strata = Counter(str(group.source_instance_counts) for group in selected)
        roles[role] = {
            "parent_groups": len(selected),
            "images": sum(len(group.image_ids) for group in selected),
            "source_instance_strata": dict(sorted(strata.items())),
        }
    return {
        "source_split": "train",
        "parent_groups": len(groups),
        "images": sum(len(group.image_ids) for group in groups),
        "roles": roles,
    }


def build_partitions(
    dataset: FalconXDataset,
    *,
    validation_percent: int = 5,
    calibration_percent: int = 5,
    seed: int = 0,
) -> ParentPartitions:
    """Assign complete train parent groups to deterministic disjoint roles."""

    validation_percent = _percentage(
        validation_percent,
        "validation_percent",
        allow_zero=False,
    )
    calibration_percent = _percentage(
        calibration_percent,
        "calibration_percent",
        allow_zero=True,
    )
    if validation_percent + calibration_percent >= 100:
        raise ValueError("validation_percent + calibration_percent must be below 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    memberships, instance_counts = _dataset_groups(dataset)
    total = len(memberships)
    validation_count = _target_count(total, validation_percent)
    calibration_count = _target_count(total, calibration_percent)
    if validation_count + calibration_count >= total:
        raise ValueError(
            "not enough train parent groups for disjoint train/development roles"
        )

    by_stratum: dict[tuple[int, int, int], list[NativeId]] = defaultdict(list)
    for source_id, stratum in instance_counts.items():
        by_stratum[stratum].append(source_id)
    validation_ids = _stratified_selection(
        by_stratum,
        validation_count,
        seed=seed,
        namespace="validation",
    )
    calibration_ids = _stratified_selection(
        by_stratum,
        calibration_count,
        seed=seed,
        namespace="calibration",
        excluded=validation_ids,
    )

    groups = []
    for source_id in sorted(memberships, key=_id_key):
        if source_id in validation_ids:
            role = "validation"
        elif source_id in calibration_ids:
            role = "calibration"
        else:
            role = "train"
        groups.append(
            ParentGroup(
                source_image_id=source_id,
                role=role,
                image_ids=memberships[source_id],
                source_instance_counts=instance_counts[source_id],
            )
        )
    report = _report(groups)
    return ParentPartitions(
        seed=seed,
        validation_percent=validation_percent,
        calibration_percent=calibration_percent,
        groups=tuple(groups),
        report=report,
        _dataset_root=dataset.root,
    )


def write_partitions(plan: ParentPartitions, path: PathSource) -> Path:
    """Atomically write a new external partition document; never touch dataset assets."""

    requested = Path(path).expanduser()
    dataset_root = plan._dataset_root.resolve(strict=True)
    candidate = requested.resolve(strict=False)
    try:
        candidate.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("partition output must be outside the immutable dataset root")
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"partition output already exists: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    try:
        target.resolve(strict=False).relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("partition output must be outside the immutable dataset root")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"partition output already exists: {target}")
    document = plan.to_dict()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _load_document(path: PathSource) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("partition file must not be a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
        before = resolved.stat()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        after = resolved.stat()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read partition file {unresolved}: {exc}") from exc
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("partition file changed while loading")
    if not isinstance(value, dict):
        raise ValueError("partition document must be an object")
    return resolved, value


def load_partition(
    dataset: FalconXDataset,
    path: PathSource,
    *,
    role: str = "train",
) -> PartitionSelection:
    """Load one role after rechecking complete, indivisible train parent groups."""

    role = _role(role)
    resolved, document = _load_document(path)
    if document.get("format") not in (PARTITION_FORMAT, "falcon-parent-partitions-v1"):
        raise ValueError(f"partition format must be {PARTITION_FORMAT!r}")
    if document.get("source_split") != "train":
        raise ValueError(
            "partition source_split must be 'train'; test is never development data"
        )
    raw_groups = document.get("groups")
    if isinstance(raw_groups, str | bytes) or not isinstance(raw_groups, Sequence):
        raise ValueError("partition groups must be an array")
    memberships, instance_counts = _dataset_groups(dataset)
    groups: list[ParentGroup] = []
    seen_sources: set[NativeId] = set()
    seen_images: set[NativeId] = set()
    for index, raw in enumerate(raw_groups):
        if not isinstance(raw, Mapping):
            raise ValueError(f"partition groups[{index}] must be an object")
        source_id = _identifier(
            raw.get("source_image_id"), f"groups[{index}].source_image_id"
        )
        group_role = _role(raw.get("role"))
        raw_image_ids = raw.get("image_ids")
        if isinstance(raw_image_ids, str | bytes) or not isinstance(
            raw_image_ids, Sequence
        ):
            raise ValueError(f"groups[{index}].image_ids must be an array")
        image_ids = tuple(
            _identifier(item, f"groups[{index}].image_ids") for item in raw_image_ids
        )
        raw_counts = raw.get("source_instance_counts")
        if isinstance(raw_counts, str | bytes) or not isinstance(raw_counts, Sequence):
            raise ValueError(f"groups[{index}].source_instance_counts must be an array")
        counts = tuple(raw_counts)
        if len(counts) != 3 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts
        ):
            raise ValueError(
                f"groups[{index}].source_instance_counts must have three counts"
            )
        if source_id in seen_sources:
            raise ValueError(f"duplicate partition parent {source_id!r}")
        overlap = seen_images.intersection(image_ids)
        if overlap:
            raise ValueError(
                f"partition images occur in multiple groups: {sorted(map(str, overlap))}"
            )
        expected_members = memberships.get(source_id)
        if expected_members is None:
            raise ValueError(f"partition has unknown train parent {source_id!r}")
        if image_ids != expected_members:
            raise ValueError(
                f"partition parent {source_id!r} is incomplete or reordered"
            )
        if counts != instance_counts[source_id]:
            raise ValueError(
                f"partition parent {source_id!r} stratum disagrees with source objects"
            )
        seen_sources.add(source_id)
        seen_images.update(image_ids)
        groups.append(ParentGroup(source_id, group_role, image_ids, counts))
    if seen_sources != set(memberships):
        raise ValueError("partition does not cover every train parent exactly once")
    expected_images = {item for values in memberships.values() for item in values}
    if seen_images != expected_images:
        raise ValueError("partition does not cover every train image exactly once")
    computed_report = _report(groups)
    if document.get("report") != computed_report:
        raise ValueError("partition report disagrees with recomputed memberships")

    image_ids = frozenset(
        image_id
        for group in groups
        if group.role == role
        for image_id in group.image_ids
    )
    source_ids = frozenset(
        group.source_image_id for group in groups if group.role == role
    )
    return PartitionSelection(
        role=role,
        image_ids=image_ids,
        source_image_ids=source_ids,
        report={
            **computed_report["roles"][role],
        },
        source_path=resolved,
    )


__all__ = [
    "PARTITION_FORMAT",
    "PARTITION_ROLES",
    "ParentGroup",
    "ParentPartitions",
    "PartitionSelection",
    "build_partitions",
    "load_partition",
    "write_partitions",
]
