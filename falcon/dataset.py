"""Read-only access to falcon-x image, object, and task tables by split-scoped ID."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .data import (
    CANONICAL_CATEGORY_IDS,
    DATASET_FORMAT,
    DATASET_FORMATS,
    FalconXDatasetError,
    PathSource,
    _annotation_sources,
)
from .tasks import PANOPTIC_FAMILIES, TASK_REGISTRY, component_membership_targets

_BBOX_FORMAT = "xywh_pixels"
_SEGMENTATION_FORMAT = "COCO"
_ANNOTATION_FORMAT = "falcon-x-annotations-v1"
_SAFETY_ANNOTATIONS = "safety_labels"
_PANOPTIC_ANNOTATIONS = "panoptic_metadata"
_ANNOTATION_KINDS = frozenset({_SAFETY_ANNOTATIONS, _PANOPTIC_ANNOTATIONS})
_ASSET_DIRECTORIES = ("images", "counterfactual_images", "panoptic")
_TARGET_FAMILIES = frozenset(
    name
    for name, spec in TASK_REGISTRY.items()
    if spec.training_target_kind == "object_union"
) | {"category_presence"}
_CONTEXT_FAMILIES = frozenset({"instance_description", "vqa"})
_SOURCE_ONLY_FAMILIES = frozenset(
    name for name, spec in TASK_REGISTRY.items() if spec.image_scope == "source"
)
_COUNTERFACTUAL_ONLY_FAMILIES = frozenset(
    name for name, spec in TASK_REGISTRY.items() if spec.image_scope == "counterfactual"
)
_KNOWN_FAMILIES = frozenset(TASK_REGISTRY)

NativeId = int | str


@dataclass(frozen=True)
class _DatasetSplitSpec:
    name: str
    usage: str
    images_path: Path
    objects_path: Path
    tasks_path: Path
    source_images: int
    images_count: int
    objects_count: int
    tasks_count: int
    families: Mapping[str, int]


@dataclass
class _SplitIndex:
    images: tuple[dict[str, Any], ...]
    image_by_id: dict[NativeId, dict[str, Any]]
    objects: tuple[dict[str, Any], ...]
    object_by_id: dict[NativeId, dict[str, Any]]
    objects_by_source: dict[NativeId, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class _AnnotationSpec:
    kind: str
    manifest_path: Path
    records_path: Path
    records_count: int
    provenance: Mapping[str, str]


def _fail(code: str, message: str) -> FalconXDatasetError:
    return FalconXDatasetError(message, code=code)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("invalid_type", f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise _fail("invalid_type", f"{context} must be an array")
    return value


def _identifier(value: Any, context: str) -> NativeId:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise _fail("invalid_id", f"{context} must be an integer or non-empty string")
    if isinstance(value, int) and value < 1:
        raise _fail("invalid_id", f"{context} integer IDs must be positive")
    if isinstance(value, str) and not value:
        raise _fail("invalid_id", f"{context} string IDs must not be empty")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail("invalid_integer", f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail("invalid_number", f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _fail("invalid_number", f"{context} must be a finite number")
    return result


def _probability(value: Any, context: str) -> float:
    result = _number(value, context)
    if not 0.0 <= result <= 1.0:
        raise _fail("invalid_probability", f"{context} must lie in [0, 1]")
    return result


def _relative_posix(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise _fail(
            "unsafe_path", f"{context} must be a normalized relative POSIX path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise _fail(
            "unsafe_path", f"{context} must be a normalized relative POSIX path"
        )
    return value


def _json_document(path: Path, context: str) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail("invalid_json", f"cannot read {context} {path}: {exc}") from exc
    return _mapping(value, context)


def _jsonl_rows(path: Path, table: str) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise _fail(
            "unreadable_table", f"cannot read {table} table {path}: {exc}"
        ) from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _fail(
                    "invalid_jsonl",
                    f"invalid JSON in {table} table at {path}:{line_number}: {exc}",
                ) from exc
            if not isinstance(value, dict):
                raise _fail(
                    "invalid_jsonl_row",
                    f"{table} row at {path}:{line_number} must be an object",
                )
            yield line_number, value


class _Issues:
    def __init__(self, limit: int = 50) -> None:
        self.limit = limit
        self.errors: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()

    def add(self, code: str, message: str, **details: Any) -> None:
        self.counts[code] += 1
        if len(self.errors) < self.limit:
            self.errors.append({"code": code, "message": message, **details})

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class FalconXDataset:
    """Read-only facade for one validated ``falcon-x`` package."""

    def __init__(
        self,
        root_or_manifest: PathSource,
        *,
        annotations: PathSource | Iterable[PathSource] | None = None,
    ):
        requested = Path(root_or_manifest).expanduser()
        if requested.is_dir():
            manifest_path = requested / "dataset.json"
        else:
            manifest_path = requested
        try:
            manifest_path = manifest_path.resolve(strict=True)
        except OSError as exc:
            raise _fail(
                "missing_manifest",
                f"dataset manifest does not exist: {manifest_path}",
            ) from exc
        if not manifest_path.is_file() or manifest_path.name != "dataset.json":
            raise _fail(
                "invalid_manifest_path",
                "falcon-x datasets require an explicit dataset.json",
            )

        self.root = manifest_path.parent
        self._manifest_path = manifest_path
        self._manifest = copy.deepcopy(
            dict(_json_document(manifest_path, "dataset manifest"))
        )
        self.format = DATASET_FORMAT
        self._splits = self._parse_manifest(self._manifest)
        self._indexes: dict[str, _SplitIndex] = {}
        self._metadata_identity: dict[Path, tuple[int, int, int, int, int]] = {
            path: self._identity(path)
            for path in (
                self._manifest_path,
                *(spec.images_path for spec in self._splits.values()),
                *(spec.objects_path for spec in self._splits.values()),
                *(spec.tasks_path for spec in self._splits.values()),
            )
        }
        self._annotations: dict[str, _AnnotationSpec] = {}
        self._safety_labels: dict[tuple[str, NativeId], dict[str, Any]] = {}
        self._panoptic_overrides: dict[tuple[str, NativeId], dict[str, Any]] = {}
        self._panoptic_asset_bindings: dict[
            tuple[str, NativeId], tuple[int, int, int, int, int]
        ] = {}
        for source in _annotation_sources(annotations):
            self._load_annotation(source)

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive copy of the compact manifest."""

        return copy.deepcopy(self._manifest)

    def protected_roots(self) -> tuple[Path, ...]:
        """Dataset and annotation directories that must not receive output files."""

        roots = [self.root]
        for spec in self._annotations.values():
            package_root = spec.manifest_path.parent
            if package_root not in roots:
                roots.append(package_root)
        return tuple(roots)

    @staticmethod
    def _identity(path: Path) -> tuple[int, int, int, int, int]:
        try:
            stat = path.stat()
        except OSError as exc:
            raise _fail(
                "missing_metadata", f"metadata file is unavailable: {path}"
            ) from exc
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _assert_metadata_unchanged(self) -> None:
        for path, expected in self._metadata_identity.items():
            if self._identity(path) != expected:
                raise _fail(
                    "metadata_changed", f"metadata changed after dataset open: {path}"
                )

    def _resolve_root_file(self, value: Any, context: str) -> Path:
        relpath = _relative_posix(value, context)
        candidate_unresolved = self.root.joinpath(*PurePosixPath(relpath).parts)
        if self._contains_symlink(self.root, PurePosixPath(relpath)):
            raise _fail(
                "symlink_metadata", f"{context} must not be a symlink: {relpath}"
            )
        try:
            candidate = candidate_unresolved.resolve(strict=True)
            candidate.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise _fail(
                "unsafe_path",
                f"{context} is missing or escapes dataset root: {relpath}",
            ) from exc
        if not candidate.is_file():
            raise _fail("not_a_file", f"{context} is not a file: {relpath}")
        return candidate

    def _resolve_split_asset(self, split: str, value: Any, context: str) -> Path:
        relpath = _relative_posix(value, context)
        split_root = self.root / split
        unresolved = split_root.joinpath(*PurePosixPath(relpath).parts)
        if split_root.is_symlink() or self._contains_symlink(
            split_root,
            PurePosixPath(relpath),
        ):
            raise _fail("symlink_asset", f"{context} must not be a symlink: {relpath}")
        try:
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(split_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise _fail(
                "unsafe_path",
                f"{context} is missing or escapes split root: {relpath}",
            ) from exc
        if not resolved.is_file():
            raise _fail("not_a_file", f"{context} is not a file: {relpath}")
        return resolved

    @staticmethod
    def _contains_symlink(root: Path, relpath: PurePosixPath) -> bool:
        candidate = root
        for part in relpath.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                return True
        return False

    @classmethod
    def _resolve_external_file(cls, root: Path, value: Any, context: str) -> Path:
        relpath = _relative_posix(value, context)
        unresolved = root.joinpath(*PurePosixPath(relpath).parts)
        if cls._contains_symlink(root, PurePosixPath(relpath)):
            raise _fail(
                "symlink_annotation", f"{context} must not be a symlink: {relpath}"
            )
        try:
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise _fail(
                "unsafe_annotation_path",
                f"{context} is missing or escapes the annotation root: {relpath}",
            ) from exc
        if not resolved.is_file():
            raise _fail("not_a_file", f"{context} is not a file: {relpath}")
        return resolved

    def metadata_summary(self) -> dict[str, Any]:
        """Dataset categories and split sizes for training and annotation records."""
        self._assert_metadata_unchanged()
        return {
            "format": self.format,
            "categories": self.manifest["categories"],
            "splits": {
                name: {
                    key: value
                    for key, value in spec.items()
                    if key
                    in ("images_count", "objects_count", "tasks_count", "families")
                }
                for name, spec in self.manifest["splits"].items()
            },
        }

    @staticmethod
    def _annotation_provenance(value: Any) -> dict[str, str]:
        provenance = _mapping(value, "annotation.provenance")
        required = ("authority", "source", "version")
        missing = [field for field in required if field not in provenance]
        if missing:
            raise _fail(
                "annotation_provenance",
                f"annotation.provenance lacks {', '.join(missing)}",
            )
        result: dict[str, str] = {}
        for field in required:
            raw = provenance[field]
            if not isinstance(raw, str) or not raw.strip():
                raise _fail(
                    "annotation_provenance",
                    f"annotation.provenance.{field} must be a non-empty string",
                )
            result[field] = raw
        return result

    def _load_annotation(self, source: PathSource) -> None:
        requested = Path(source).expanduser()
        manifest_unresolved = requested
        if requested.is_dir():
            candidates = [requested / "annotations.json"]
            found = [path for path in candidates if path.exists() or path.is_symlink()]
            if len(found) != 1:
                raise _fail(
                    "annotation_manifest",
                    f"annotation directory must contain exactly one manifest: {requested}",
                )
            manifest_unresolved = found[0]
        if manifest_unresolved.is_symlink():
            raise _fail(
                "symlink_annotation",
                f"annotation manifest must not be a symlink: {source}",
            )
        try:
            manifest_path = manifest_unresolved.resolve(strict=True)
        except OSError as exc:
            raise _fail(
                "missing_annotation",
                f"annotation manifest does not exist: {source}",
            ) from exc
        if not manifest_path.is_file():
            raise _fail(
                "invalid_annotation",
                f"annotation manifest is not a file: {manifest_path}",
            )

        manifest_identity = self._identity(manifest_path)
        manifest = _json_document(manifest_path, "annotation manifest")
        if self._identity(manifest_path) != manifest_identity:
            raise _fail(
                "metadata_changed", f"annotation changed while loading: {manifest_path}"
            )
        if manifest.get("format") != _ANNOTATION_FORMAT:
            raise _fail(
                "unsupported_annotation",
                f"annotation format must be {_ANNOTATION_FORMAT!r}",
            )
        kind = manifest.get("kind")
        if kind not in _ANNOTATION_KINDS:
            raise _fail(
                "unsupported_annotation", f"unsupported annotation kind {kind!r}"
            )
        if kind in self._annotations:
            raise _fail(
                "duplicate_annotation", f"multiple {kind!r} annotations are not allowed"
            )
        if manifest.get("dataset") != self.metadata_summary():
            raise _fail(
                "annotation_dataset_mismatch",
                "annotation.dataset must match dataset.metadata_summary()",
            )
        provenance = self._annotation_provenance(manifest.get("provenance"))
        records_path = self._resolve_external_file(
            manifest_path.parent,
            manifest.get("records"),
            "annotation.records",
        )
        records_count = _integer(
            manifest.get("records_count"), "annotation.records_count"
        )
        records_identity = self._identity(records_path)
        panoptic_bindings: dict[
            tuple[str, NativeId], tuple[int, int, int, int, int]
        ] = {}
        if kind == _SAFETY_ANNOTATIONS:
            values = self._load_safety_annotation(records_path, records_count)
        else:
            values, panoptic_bindings = self._load_panoptic_annotation(
                records_path,
                records_count,
            )
        if self._identity(records_path) != records_identity:
            raise _fail(
                "metadata_changed", f"annotation changed while loading: {records_path}"
            )
        if kind == _SAFETY_ANNOTATIONS:
            self._safety_labels.update(values)
        else:
            self._panoptic_overrides.update(values)
            self._panoptic_asset_bindings.update(panoptic_bindings)

        spec = _AnnotationSpec(
            kind=kind,
            manifest_path=manifest_path,
            records_path=records_path,
            records_count=records_count,
            provenance=provenance,
        )
        self._annotations[kind] = spec
        self._metadata_identity[manifest_path] = manifest_identity
        self._metadata_identity[records_path] = records_identity

    def _parse_manifest(
        self, manifest: Mapping[str, Any]
    ) -> dict[str, _DatasetSplitSpec]:
        if "schema_version" in manifest:
            raise _fail(
                "mixed_manifest",
                "dataset manifest cannot also declare the legacy schema_version",
            )
        if manifest.get("format") not in DATASET_FORMATS:
            raise _fail(
                "unsupported_format",
                f"dataset format must be {DATASET_FORMAT!r}, got {manifest.get('format')!r}",
            )
        if manifest.get("bbox_format") != _BBOX_FORMAT:
            raise _fail("unsupported_bbox", f"bbox_format must be {_BBOX_FORMAT!r}")
        if manifest.get("segmentation_format") != _SEGMENTATION_FORMAT:
            raise _fail(
                "unsupported_segmentation",
                f"segmentation_format must be {_SEGMENTATION_FORMAT!r}",
            )

        categories_raw = _sequence(manifest.get("categories"), "categories")
        categories: dict[str, int] = {}
        observed_ids: set[int] = set()
        for index, raw in enumerate(categories_raw):
            category = _mapping(raw, f"categories[{index}]")
            name = category.get("name")
            identifier = category.get("id")
            if not isinstance(name, str) or not name:
                raise _fail(
                    "invalid_category", f"categories[{index}].name must be non-empty"
                )
            identifier = _integer(identifier, f"categories[{index}].id", minimum=1)
            if name in categories or identifier in observed_ids:
                raise _fail(
                    "duplicate_category", "category names and IDs must be unique"
                )
            categories[name] = identifier
            observed_ids.add(identifier)
        if set(categories) != set(CANONICAL_CATEGORY_IDS):
            raise _fail(
                "category_mapping",
                "categories must contain each canonical component name exactly once",
            )
        self._category_name_by_id = {value: key for key, value in categories.items()}
        self._raw_category_id_by_name = dict(categories)

        splits_raw = _mapping(manifest.get("splits"), "splits")
        if not splits_raw:
            raise _fail("missing_splits", "splits must not be empty")
        result: dict[str, _DatasetSplitSpec] = {}
        seen_table_paths: set[Path] = set()
        for name, raw in splits_raw.items():
            if not isinstance(name, str) or not name:
                raise _fail("invalid_split", "split names must be non-empty strings")
            value = _mapping(raw, f"splits.{name}")
            usage = value.get("usage")
            if usage not in ("training", "evaluation"):
                raise _fail(
                    "invalid_usage",
                    f"splits.{name}.usage must be 'training' or 'evaluation'",
                )
            if name == "train" and usage != "training":
                raise _fail("split_usage", "the train split must have training usage")
            if name == "test" and usage != "evaluation":
                raise _fail("split_usage", "the test split must have evaluation usage")
            table_paths = {
                field: self._resolve_root_file(
                    value.get(field), f"splits.{name}.{field}"
                )
                for field in ("images", "objects", "tasks")
            }
            overlap = seen_table_paths.intersection(table_paths.values())
            if overlap:
                raise _fail(
                    "shared_table",
                    f"split tables must be distinct: {sorted(map(str, overlap))}",
                )
            seen_table_paths.update(table_paths.values())
            families_raw = _mapping(value.get("families"), f"splits.{name}.families")
            families: dict[str, int] = {}
            for family, count in families_raw.items():
                if not isinstance(family, str) or not family:
                    raise _fail(
                        "invalid_family", f"splits.{name}.families has an invalid name"
                    )
                if family not in _KNOWN_FAMILIES:
                    raise _fail(
                        "unknown_family",
                        f"splits.{name}.families declares unsupported family {family!r}",
                    )
                families[family] = _integer(
                    count,
                    f"splits.{name}.families.{family}",
                )
            tasks_count = _integer(
                value.get("tasks_count"), f"splits.{name}.tasks_count"
            )
            if sum(families.values()) != tasks_count:
                raise _fail(
                    "family_count",
                    f"splits.{name}.families does not sum to tasks_count",
                )
            result[name] = _DatasetSplitSpec(
                name=name,
                usage=usage,
                images_path=table_paths["images"],
                objects_path=table_paths["objects"],
                tasks_path=table_paths["tasks"],
                source_images=_integer(
                    value.get("source_images"),
                    f"splits.{name}.source_images",
                ),
                images_count=_integer(
                    value.get("images_count"), f"splits.{name}.images_count"
                ),
                objects_count=_integer(
                    value.get("objects_count"),
                    f"splits.{name}.objects_count",
                ),
                tasks_count=tasks_count,
                families=dict(families),
            )

        weights = manifest.get("train_family_weights")
        if weights is not None:
            weights = _mapping(weights, "train_family_weights")
            train_spec = result.get("train")
            train_families = (
                set(train_spec.families) if train_spec is not None else set()
            )
            if set(weights) != train_families:
                raise _fail(
                    "family_weights",
                    "train_family_weights must cover every train family exactly",
                )
            total = 0.0
            for family, raw_weight in weights.items():
                weight = _number(raw_weight, f"train_family_weights.{family}")
                if weight <= 0:
                    raise _fail(
                        "family_weights", "training family weights must be positive"
                    )
                total += weight
            if not math.isclose(total, 100.0, abs_tol=1e-6):
                raise _fail("family_weights", "train_family_weights must sum to 100")
        return result

    def _spec(self, split: str) -> _DatasetSplitSpec:
        if not isinstance(split, str) or split not in self._splits:
            raise _fail("unknown_split", f"unknown dataset split {split!r}")
        return self._splits[split]

    def _annotation_image(
        self,
        row: Mapping[str, Any],
        context: str,
    ) -> tuple[str, NativeId, dict[str, Any]]:
        self._required(row, ("split", "image_id"), context)
        split = row["split"]
        if not isinstance(split, str):
            raise _fail("invalid_split", f"{context}.split must be a string")
        image_id = _identifier(row["image_id"], f"{context}.image_id")
        index = self._load_split(split)
        try:
            image = index.image_by_id[image_id]
        except KeyError as exc:
            raise _fail(
                "unknown_annotation_image",
                f"{context} references unknown image {image_id!r} in split {split!r}",
            ) from exc
        return split, image_id, image

    def _load_safety_annotation(
        self,
        path: Path,
        expected_count: int,
    ) -> dict[tuple[str, NativeId], dict[str, Any]]:
        result: dict[tuple[str, NativeId], dict[str, Any]] = {}
        allowed = {"split", "image_id", "risk", "links"}
        for line_number, row in _jsonl_rows(path, "safety annotation"):
            context = f"{path}:{line_number}"
            self._required(row, allowed, context)
            extras = set(row).difference(allowed)
            if extras:
                raise _fail(
                    "unknown_annotation_field",
                    f"{context} has unsupported fields {sorted(extras)!r}",
                )
            split, image_id, _ = self._annotation_image(row, context)
            key = split, image_id
            if key in result:
                raise _fail(
                    "duplicate_annotation_row", f"duplicate safety label for {key!r}"
                )
            risk = row["risk"]
            if risk is not None:
                risk = _probability(risk, f"{context}.risk")
            links_raw = _sequence(row["links"], f"{context}.links")
            if len(links_raw) != 3:
                raise _fail(
                    "invalid_links", f"{context}.links must have three canonical slots"
                )
            links = [
                None
                if value is None
                else _probability(value, f"{context}.links[{index}]")
                for index, value in enumerate(links_raw)
            ]
            if risk is None and all(value is None for value in links):
                raise _fail(
                    "empty_safety_label",
                    f"{context} supplies no risk or functional-link supervision",
                )
            result[key] = {"risk": risk, "links": links}
        if len(result) != expected_count:
            raise _fail(
                "annotation_record_count",
                f"safety annotation has {len(result)} records, expected {expected_count}",
            )
        return result

    def _load_panoptic_annotation(
        self,
        path: Path,
        expected_count: int,
    ) -> tuple[
        dict[tuple[str, NativeId], dict[str, Any]],
        dict[tuple[str, NativeId], tuple[int, int, int, int, int]],
    ]:
        result: dict[tuple[str, NativeId], dict[str, Any]] = {}
        asset_bindings: dict[tuple[str, NativeId], tuple[int, int, int, int, int]] = {}
        allowed = {"split", "image_id", "segments_info"}
        for line_number, row in _jsonl_rows(path, "panoptic annotation"):
            context = f"{path}:{line_number}"
            self._required(row, allowed, context)
            extras = set(row).difference(allowed)
            if extras:
                raise _fail(
                    "unknown_annotation_field",
                    f"{context} has unsupported fields {sorted(extras)!r}",
                )
            split, image_id, image = self._annotation_image(row, context)
            key = split, image_id
            if key in result:
                raise _fail(
                    "duplicate_annotation_row",
                    f"duplicate panoptic override for {key!r}",
                )
            if "source_image_id" in image or "panoptic" not in image:
                raise _fail(
                    "panoptic_annotation_image",
                    f"{context} must reference a source image with a native panoptic asset",
                )
            native = image["panoptic"]
            asset_path = self._resolve_split_asset(
                split,
                native["file_name"],
                f"{context}.panoptic asset",
            )
            asset_identity = self._identity(asset_path)
            metadata = {
                "file_name": native["file_name"],
                "segments_info": copy.deepcopy(
                    list(_sequence(row["segments_info"], f"{context}.segments_info"))
                ),
            }
            candidate_image = {
                "width": image["width"],
                "height": image["height"],
                "panoptic": metadata,
            }
            self._validate_panoptic_metadata(split, candidate_image, context)
            issues = _Issues(limit=3)
            self._panoptic_metadata_raster_issues(
                split,
                image,
                asset_path,
                metadata,
                issues,
            )
            if issues.total:
                first = issues.errors[0]
                raise _fail(
                    "panoptic_annotation_raster",
                    f"{context} fails native raster closure: {first['code']}: {first['message']}",
                )
            if self._identity(asset_path) != asset_identity:
                raise _fail(
                    "asset_changed", f"{context} panoptic PNG changed while validating"
                )
            result[key] = metadata
            asset_bindings[key] = asset_identity
        if len(result) != expected_count:
            raise _fail(
                "annotation_record_count",
                f"panoptic annotation has {len(result)} records, expected {expected_count}",
            )
        return result, asset_bindings

    @staticmethod
    def _required(row: Mapping[str, Any], fields: Iterable[str], context: str) -> None:
        missing = [field for field in fields if field not in row]
        if missing:
            raise _fail("missing_field", f"{context} lacks {', '.join(missing)}")

    def _validate_panoptic_metadata(
        self,
        split: str,
        image: Mapping[str, Any],
        context: str,
    ) -> None:
        raw = _mapping(image.get("panoptic"), f"{context}.panoptic")
        file_name = _relative_posix(
            raw.get("file_name"), f"{context}.panoptic.file_name"
        )
        if PurePosixPath(file_name).parts[0] != "panoptic":
            raise _fail(
                "panoptic_path",
                f"{context}.panoptic.file_name must be beneath panoptic/",
            )
        self._resolve_split_asset(split, file_name, f"{context}.panoptic.file_name")
        segments = _sequence(
            raw.get("segments_info"), f"{context}.panoptic.segments_info"
        )
        seen: set[int] = set()
        for offset, item in enumerate(segments):
            segment = _mapping(item, f"{context}.panoptic.segments_info[{offset}]")
            self._required(
                segment,
                ("id", "category_id", "iscrowd", "bbox", "area"),
                f"{context}.panoptic.segments_info[{offset}]",
            )
            segment_id = _integer(
                segment["id"],
                f"{context}.panoptic.segments_info[{offset}].id",
                minimum=1,
            )
            if segment_id in seen:
                raise _fail(
                    "duplicate_panoptic_id",
                    f"{context} repeats panoptic ID {segment_id}",
                )
            seen.add(segment_id)
            category_id = _integer(
                segment["category_id"],
                f"{context}.panoptic.segments_info[{offset}].category_id",
                minimum=1,
            )
            if category_id not in self._category_name_by_id:
                raise _fail(
                    "unknown_category", f"{context} has unknown panoptic category"
                )
            if segment["iscrowd"] not in (0, 1):
                raise _fail(
                    "invalid_iscrowd", f"{context} panoptic iscrowd must be 0 or 1"
                )
            self._validate_bbox(
                segment["bbox"],
                int(image["width"]),
                int(image["height"]),
                f"{context}.panoptic.segments_info[{offset}].bbox",
            )
            if (
                _number(
                    segment["area"], f"{context}.panoptic.segments_info[{offset}].area"
                )
                <= 0
            ):
                raise _fail("invalid_area", f"{context} panoptic area must be positive")

    @staticmethod
    def _validate_bbox(value: Any, width: int, height: int, context: str) -> None:
        bbox = _sequence(value, context)
        if len(bbox) != 4:
            raise _fail("invalid_bbox", f"{context} must contain four values")
        x, y, box_width, box_height = (_number(item, context) for item in bbox)
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
            raise _fail("invalid_bbox", f"{context} has invalid origin or size")
        if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
            raise _fail("bbox_out_of_bounds", f"{context} exceeds image dimensions")

    @staticmethod
    def _validate_segmentation(
        value: Any, width: int, height: int, context: str
    ) -> None:
        if isinstance(value, Mapping):
            size = value.get("size")
            if (
                isinstance(size, str | bytes)
                or not isinstance(size, Sequence)
                or tuple(size) != (height, width)
                or not isinstance(value.get("counts"), str | list)
            ):
                raise _fail("invalid_segmentation", f"{context} is malformed COCO RLE")
            return
        polygons = _sequence(value, context)
        if not polygons:
            raise _fail("invalid_segmentation", f"{context} polygon list is empty")
        for index, raw_polygon in enumerate(polygons):
            polygon = _sequence(raw_polygon, f"{context}[{index}]")
            if len(polygon) < 6 or len(polygon) % 2:
                raise _fail(
                    "invalid_segmentation",
                    f"{context}[{index}] must contain at least three coordinate pairs",
                )
            coordinates = [_number(item, f"{context}[{index}]") for item in polygon]
            for offset in range(0, len(coordinates), 2):
                x, y = coordinates[offset : offset + 2]
                if x < 0 or y < 0 or x > width or y > height:
                    raise _fail(
                        "segmentation_out_of_bounds",
                        f"{context}[{index}] exceeds image dimensions",
                    )

    def _load_split(self, split: str) -> _SplitIndex:
        self._assert_metadata_unchanged()
        if split in self._indexes:
            return self._indexes[split]
        spec = self._spec(split)
        images: list[dict[str, Any]] = []
        image_by_id: dict[NativeId, dict[str, Any]] = {}
        image_paths: set[str] = set()
        source_count = 0
        for line_number, row in _jsonl_rows(spec.images_path, "images"):
            context = f"{spec.images_path}:{line_number}"
            self._required(row, ("id", "file_name", "width", "height"), context)
            image_id = _identifier(row["id"], f"{context}.id")
            if image_id in image_by_id:
                raise _fail(
                    "duplicate_image_id",
                    f"duplicate image ID {image_id!r} at {context}",
                )
            file_name = _relative_posix(row["file_name"], f"{context}.file_name")
            if file_name in image_paths:
                raise _fail(
                    "duplicate_image_path", f"duplicate image path {file_name!r}"
                )
            image_paths.add(file_name)
            width = _integer(row["width"], f"{context}.width", minimum=1)
            height = _integer(row["height"], f"{context}.height", minimum=1)
            is_counterfactual = "source_image_id" in row
            expected_folder = "counterfactual_images" if is_counterfactual else "images"
            if PurePosixPath(file_name).parts[0] != expected_folder:
                raise _fail(
                    "image_path_kind",
                    f"{context}.file_name must be beneath {expected_folder}/",
                )
            self._resolve_split_asset(split, file_name, f"{context}.file_name")
            if is_counterfactual:
                _identifier(row["source_image_id"], f"{context}.source_image_id")
                present = _sequence(
                    row.get("present_annotation_ids"),
                    f"{context}.present_annotation_ids",
                )
                present_ids = [
                    _identifier(item, f"{context}.present_annotation_ids")
                    for item in present
                ]
                if len(present_ids) != len(set(present_ids)):
                    raise _fail(
                        "duplicate_present_id",
                        f"{context}.present_annotation_ids contains duplicates",
                    )
                if "panoptic" in row:
                    raise _fail(
                        "counterfactual_panoptic",
                        f"{context} cannot define panoptic data",
                    )
            else:
                source_count += 1
                if "present_annotation_ids" in row:
                    raise _fail(
                        "base_membership",
                        f"{context} base image cannot define present_annotation_ids",
                    )
                if "panoptic" in row:
                    self._validate_panoptic_metadata(split, row, context)
            copied = copy.deepcopy(row)
            copied["width"], copied["height"] = width, height
            images.append(copied)
            image_by_id[image_id] = copied
        if len(images) != spec.images_count:
            raise _fail(
                "image_count",
                f"{split} images count {len(images)} disagrees with manifest {spec.images_count}",
            )
        if source_count != spec.source_images:
            raise _fail(
                "source_image_count",
                f"{split} source count {source_count} disagrees with manifest {spec.source_images}",
            )

        objects: list[dict[str, Any]] = []
        object_by_id: dict[NativeId, dict[str, Any]] = {}
        objects_by_source_lists: dict[NativeId, list[dict[str, Any]]] = defaultdict(
            list
        )
        for line_number, row in _jsonl_rows(spec.objects_path, "objects"):
            context = f"{spec.objects_path}:{line_number}"
            self._required(
                row,
                ("id", "image_id", "category_id", "bbox", "area", "segmentation"),
                context,
            )
            object_id = _identifier(row["id"], f"{context}.id")
            if object_id in object_by_id:
                raise _fail("duplicate_object_id", f"duplicate object ID {object_id!r}")
            image_id = _identifier(row["image_id"], f"{context}.image_id")
            image = image_by_id.get(image_id)
            if image is None or "source_image_id" in image:
                raise _fail(
                    "object_source",
                    f"object {object_id!r} must belong to a source image in split {split}",
                )
            category_id = _integer(
                row["category_id"], f"{context}.category_id", minimum=1
            )
            if category_id not in self._category_name_by_id:
                raise _fail(
                    "unknown_category", f"object {object_id!r} has unknown category"
                )
            self._validate_bbox(
                row["bbox"], image["width"], image["height"], f"{context}.bbox"
            )
            if _number(row["area"], f"{context}.area") <= 0:
                raise _fail("invalid_area", f"{context}.area must be positive")
            self._validate_segmentation(
                row["segmentation"],
                image["width"],
                image["height"],
                f"{context}.segmentation",
            )
            copied = copy.deepcopy(row)
            objects.append(copied)
            object_by_id[object_id] = copied
            objects_by_source_lists[image_id].append(copied)
        if len(objects) != spec.objects_count:
            raise _fail(
                "object_count",
                (
                    f"{split} objects count {len(objects)} disagrees with manifest "
                    f"{spec.objects_count}"
                ),
            )

        for image in images:
            if "source_image_id" not in image:
                continue
            image_id = image["id"]
            source_id = image["source_image_id"]
            source = image_by_id.get(source_id)
            if source is None or "source_image_id" in source:
                raise _fail(
                    "counterfactual_source",
                    (
                        f"counterfactual {image_id!r} has no source image "
                        f"{source_id!r} in split {split}"
                    ),
                )
            if (image["width"], image["height"]) != (source["width"], source["height"]):
                raise _fail(
                    "counterfactual_dimensions",
                    f"counterfactual {image_id!r} dimensions differ from its source",
                )
            source_ids = {
                item["id"] for item in objects_by_source_lists.get(source_id, [])
            }
            present_ids = set(image["present_annotation_ids"])
            unknown = present_ids.difference(source_ids)
            if unknown:
                raise _fail(
                    "counterfactual_membership",
                    (
                        f"counterfactual {image_id!r} has IDs outside its source: "
                        f"{sorted(map(str, unknown))}"
                    ),
                )
            if present_ids == source_ids:
                raise _fail(
                    "counterfactual_no_removal",
                    f"counterfactual {image_id!r} does not remove any source object",
                )

        index = _SplitIndex(
            images=tuple(images),
            image_by_id=image_by_id,
            objects=tuple(objects),
            object_by_id=object_by_id,
            objects_by_source={
                key: tuple(value) for key, value in objects_by_source_lists.items()
            },
        )
        self._indexes[split] = index
        return index

    def images(self, split: str) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of every image row in one split."""

        return tuple(copy.deepcopy(row) for row in self._load_split(split).images)

    def objects(self, split: str) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of every source-object row in one split."""

        return tuple(copy.deepcopy(row) for row in self._load_split(split).objects)

    def image(self, split: str, image_id: NativeId) -> dict[str, Any]:
        """Resolve a native image ID within an explicit split."""

        image_id = _identifier(image_id, "image_id")
        try:
            row = self._load_split(split).image_by_id[image_id]
        except KeyError as exc:
            raise _fail(
                "unknown_image", f"unknown image {image_id!r} in split {split!r}"
            ) from exc
        return copy.deepcopy(row)

    def present_objects(
        self, split: str, image_id: NativeId
    ) -> tuple[dict[str, Any], ...]:
        """Return source objects retained in a source or counterfactual state."""

        index = self._load_split(split)
        image = self.image(split, image_id)
        rows = self._present_object_rows(index, image)
        return tuple(copy.deepcopy(row) for row in rows)

    @staticmethod
    def _present_object_rows(
        index: _SplitIndex,
        image: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        if "source_image_id" not in image:
            return index.objects_by_source.get(image["id"], ())
        return tuple(
            index.object_by_id[item] for item in image["present_annotation_ids"]
        )

    def removed_objects(
        self, split: str, image_id: NativeId
    ) -> tuple[dict[str, Any], ...]:
        """Return the authoritative source-minus-present counterfactual delta."""

        index = self._load_split(split)
        image = self.image(split, image_id)
        if "source_image_id" not in image:
            return ()
        present = set(image["present_annotation_ids"])
        return tuple(
            copy.deepcopy(row)
            for row in index.objects_by_source.get(image["source_image_id"], ())
            if row["id"] not in present
        )

    def derived_presence(self, split: str, image_id: NativeId) -> tuple[int, int, int]:
        """Derive only the canonical component-presence vector from object membership."""

        present_categories = {
            self.category_name(row["category_id"])
            for row in self.present_objects(split, image_id)
        }
        return tuple(int(name in present_categories) for name in CANONICAL_CATEGORY_IDS)

    def structured_targets(self, split: str, image_id: NativeId) -> dict[str, Any]:
        """Return presence and reviewed safety labels; missing labels remain ``None``."""

        image_id = _identifier(image_id, "image_id")
        # Resolve the image before consulting the optional annotation so unknown
        # split-scoped IDs cannot look like merely unavailable supervision.
        self.image(split, image_id)
        safety = self._safety_labels.get((split, image_id), {})
        return {
            "presence": list(self.derived_presence(split, image_id)),
            "risk": safety.get("risk"),
            "links": list(safety.get("links", (None, None, None))),
        }

    def annotation_coverage(self, split: str | None = None) -> dict[str, Any]:
        """Count available external labels without claiming missing values are negative."""

        if split is not None:
            self._spec(split)
        selected_safety = [
            value
            for (row_split, _), value in self._safety_labels.items()
            if split is None or row_split == split
        ]
        selected_panoptic = [
            key for key in self._panoptic_overrides if split is None or key[0] == split
        ]
        return {
            "safety": {
                "records": len(selected_safety),
                "risk": sum(value["risk"] is not None for value in selected_safety),
                "links": [
                    sum(value["links"][index] is not None for value in selected_safety)
                    for index in range(3)
                ],
                "any_link": sum(
                    any(item is not None for item in value["links"])
                    for value in selected_safety
                ),
            },
            "panoptic": {"records": len(selected_panoptic)},
        }

    def category_name(self, category_id: int) -> str:
        """Resolve a dataset category ID to its canonical component name."""

        category_id = _integer(category_id, "category_id", minimum=1)
        try:
            return self._category_name_by_id[category_id]
        except KeyError as exc:
            raise _fail(
                "unknown_category", f"unknown dataset category ID {category_id}"
            ) from exc

    def canonical_category_id(self, category_id: int) -> int:
        """Normalize a dataset-local category ID by canonical component name."""

        return CANONICAL_CATEGORY_IDS[self.category_name(category_id)]

    def resolve_image(self, split: str, image_id: NativeId) -> Path:
        row = self.image(split, image_id)
        return self._resolve_split_asset(split, row["file_name"], f"image {image_id!r}")

    def resolve_panoptic(
        self,
        split: str,
        image_id: NativeId,
    ) -> tuple[Path, dict[str, Any]]:
        """Return the native panoptic asset and untouched metadata, without inference."""

        row = self.image(split, image_id)
        if "panoptic" not in row:
            raise _fail(
                "missing_panoptic",
                f"image {image_id!r} in split {split!r} has no panoptic data",
            )
        metadata = copy.deepcopy(
            self._panoptic_overrides.get(
                (split, image_id),
                dict(_mapping(row["panoptic"], "panoptic")),
            )
        )
        path = self._resolve_split_asset(
            split,
            metadata["file_name"],
            f"image {image_id!r} panoptic",
        )
        accepted_identity = self._panoptic_asset_bindings.get((split, image_id))
        if accepted_identity is not None and self._identity(path) != accepted_identity:
            raise _fail(
                "asset_changed",
                f"image {image_id!r} panoptic PNG changed after loading",
            )
        return path, metadata

    def resolve_task_panoptic(
        self,
        split: str,
        task: Mapping[str, Any] | str,
    ) -> tuple[Path, dict[str, Any]]:
        """Resolve panoptic targets; callers zero raster IDs absent from returned metadata."""

        row = self._task_row(split, task)
        if row["family"] not in PANOPTIC_FAMILIES:
            raise _fail("task_modality", "task does not have a panoptic target")
        path, metadata = self.resolve_panoptic(split, row["image_id"])
        if row["family"] == "referring_panoptic_segmentation":
            selected = set(row["target_segment_ids"])
            metadata["segments_info"] = [
                segment
                for segment in metadata["segments_info"]
                if segment["id"] in selected
            ]
        return path, metadata

    def decode_object_mask(
        self,
        split: str,
        object_or_id: Mapping[str, Any] | NativeId,
    ) -> Any:
        """Decode one authoritative COCO instance mask as a boolean HxW array."""

        index = self._load_split(split)
        if isinstance(object_or_id, Mapping):
            object_id = _identifier(object_or_id.get("id"), "object.id")
        else:
            object_id = _identifier(object_or_id, "object_id")
        try:
            row = index.object_by_id[object_id]
        except KeyError as exc:
            raise _fail(
                "unknown_object",
                f"unknown object {object_id!r} in split {split!r}",
            ) from exc
        image = index.image_by_id[row["image_id"]]
        return self._decode_instance_mask(
            row["segmentation"],
            image["height"],
            image["width"],
        ).copy()

    def _validate_task(
        self,
        split: str,
        row: dict[str, Any],
        context: str,
        index: _SplitIndex | None = None,
    ) -> None:
        spec = self._spec(split)
        index = self._load_split(split) if index is None else index
        self._required(row, ("id", "family", "image_id", "prompt", "answer"), context)
        task_id = row["id"]
        if not isinstance(task_id, str) or not task_id:
            raise _fail("invalid_task_id", f"{context}.id must be a non-empty string")
        family = row["family"]
        if not isinstance(family, str) or family not in spec.families:
            raise _fail(
                "unknown_family", f"{context}.family {family!r} is not declared"
            )
        image_id = _identifier(row["image_id"], f"{context}.image_id")
        if image_id not in index.image_by_id:
            raise _fail(
                "unknown_task_image", f"task {task_id!r} references unknown image"
            )
        image = index.image_by_id[image_id]
        is_counterfactual = "source_image_id" in image
        if family in _SOURCE_ONLY_FAMILIES and is_counterfactual:
            raise _fail(
                "task_image_kind",
                f"task family {family!r} requires a source image",
            )
        if family in _COUNTERFACTUAL_ONLY_FAMILIES and not is_counterfactual:
            raise _fail(
                "task_image_kind",
                f"task family {family!r} requires a counterfactual image",
            )
        for field in ("prompt", "answer"):
            if not isinstance(row[field], str) or not row[field]:
                raise _fail(
                    "invalid_task_text", f"task {task_id!r} {field} must be non-empty"
                )

        if family in _TARGET_FAMILIES and "target_annotation_ids" not in row:
            raise _fail(
                "missing_task_object_field",
                f"task {task_id!r} lacks target_annotation_ids",
            )
        if family not in _TARGET_FAMILIES and "target_annotation_ids" in row:
            raise _fail(
                "unexpected_task_object_field",
                f"task {task_id!r} unexpectedly defines target_annotation_ids",
            )
        if family in _CONTEXT_FAMILIES and "context_annotation_ids" not in row:
            raise _fail(
                "missing_task_object_field",
                f"task {task_id!r} lacks context_annotation_ids",
            )
        if family not in _CONTEXT_FAMILIES and "context_annotation_ids" in row:
            raise _fail(
                "unexpected_task_object_field",
                f"task {task_id!r} unexpectedly defines context_annotation_ids",
            )

        object_fields: dict[str, list[NativeId]] = {}
        present_rows = self._present_object_rows(index, image)
        present_ids = {item["id"] for item in present_rows}
        for field in ("target_annotation_ids", "context_annotation_ids"):
            if field not in row:
                continue
            values = _sequence(row[field], f"task {task_id!r} {field}")
            identifiers = [
                _identifier(value, f"task {task_id!r} {field}") for value in values
            ]
            object_fields[field] = identifiers
            if len(identifiers) != len(set(identifiers)):
                raise _fail(
                    "duplicate_task_object", f"task {task_id!r} {field} has duplicates"
                )
            missing = set(identifiers).difference(present_ids)
            if missing:
                raise _fail(
                    "absent_task_object",
                    f"task {task_id!r} {field} references absent IDs {sorted(map(str, missing))}",
                )

        if family in ("category_or_all_instance_grounding", "category_presence"):
            if "category_id" not in row:
                raise _fail("missing_field", f"task {task_id!r} lacks category_id")
            raw_category_id = row["category_id"]
            if family == "category_presence" and raw_category_id is None:
                raise _fail(
                    "missing_category",
                    f"category-presence task {task_id!r} needs a category_id",
                )
            if raw_category_id is not None:
                category_id = _integer(
                    raw_category_id,
                    f"task {task_id!r} category_id",
                    minimum=1,
                )
                if category_id not in self._category_name_by_id:
                    raise _fail(
                        "unknown_category", f"task {task_id!r} has unknown category"
                    )
                expected_ids = {
                    item["id"]
                    for item in present_rows
                    if item["category_id"] == category_id
                }
            else:
                expected_ids = present_ids
            target_ids = set(object_fields["target_annotation_ids"])
            if target_ids != expected_ids:
                raise _fail(
                    "category_target_mismatch",
                    f"task {task_id!r} targets do not match its category and image state",
                )
            if family == "category_presence":
                expected_answer = "yes" if expected_ids else "no"
                if row["answer"] != expected_answer:
                    raise _fail(
                        "presence_answer_mismatch",
                        f"task {task_id!r} answer disagrees with immutable membership",
                    )
        elif family == "referring_panoptic_segmentation":
            category_id = _integer(
                row.get("category_id"), "task.category_id", minimum=1
            )
            if category_id not in self._category_name_by_id:
                raise _fail(
                    "unknown_category", f"task {task_id!r} has unknown category"
                )
        elif "category_id" in row:
            raise _fail(
                "unexpected_category",
                f"task family {family!r} cannot define category_id",
            )

        if family == "vqa":
            question_type = row.get("question_type")
            if not isinstance(question_type, str) or not question_type:
                raise _fail(
                    "invalid_question_type",
                    f"task {task_id!r} question_type is invalid",
                )
        elif "question_type" in row:
            raise _fail(
                "unexpected_question_type",
                f"task family {family!r} cannot define question_type",
            )

        for field, owner in (
            ("missing_categories", "missing_component_identification"),
            ("completeness", "functional_completeness"),
            ("target_segment_ids", "referring_panoptic_segmentation"),
        ):
            if family != owner and field in row:
                raise _fail(
                    "unexpected_task_field", f"task {task_id!r} cannot define {field}"
                )
        if family in {"missing_component_identification", "functional_completeness"}:
            missing, complete = component_membership_targets(
                [
                    self._category_name_by_id[item["category_id"]]
                    for item in present_rows
                ]
            )
            try:
                answer = json.loads(row["answer"])
            except json.JSONDecodeError as exc:
                raise _fail(
                    "invalid_task_answer", f"task {task_id!r} answer must be JSON"
                ) from exc
            if family == "missing_component_identification":
                values = row.get("missing_categories")
                if (
                    values != missing
                    or not isinstance(answer, list)
                    or answer != missing
                ):
                    raise _fail(
                        "missing_component_mismatch",
                        f"task {task_id!r} missing categories disagree with image membership",
                    )
            elif (
                _number(row.get("completeness"), "task.completeness") != complete
                or _number(answer, "task.answer") != complete
            ):
                raise _fail(
                    "completeness_mismatch",
                    f"task {task_id!r} completeness disagrees with required category membership",
                )
        if family == "referring_functional_grounding":
            if set(object_fields["target_annotation_ids"]) != present_ids:
                raise _fail(
                    "functional_target_mismatch",
                    f"task {task_id!r} must target all visible types in the functional taxonomy",
                )

        if (
            family == "instance_description"
            and len(object_fields["context_annotation_ids"]) != 1
        ):
            raise _fail(
                "context_cardinality",
                f"instance-description task {task_id!r} must have one context object",
            )
        if family == "vqa" and not object_fields["context_annotation_ids"]:
            raise _fail(
                "context_cardinality", f"VQA task {task_id!r} needs context objects"
            )
        if (
            family == "referring_expression"
            and len(object_fields["target_annotation_ids"]) != 1
        ):
            raise _fail(
                "target_cardinality",
                f"referring-expression task {task_id!r} must have one target object",
            )
        if TASK_REGISTRY[family].prediction_kind == "binary_mask":
            if row["answer"] != "<SEG>":
                raise _fail(
                    "invalid_placeholder", f"task {task_id!r} must answer <SEG>"
                )
        if family in PANOPTIC_FAMILIES:
            if "source_image_id" in image or "panoptic" not in image:
                raise _fail(
                    "panoptic_task_image",
                    f"panoptic task {task_id!r} must reference a source image with panoptic data",
                )
            if row["answer"] != "<PANOPTIC>":
                raise _fail(
                    "invalid_placeholder", f"task {task_id!r} must answer <PANOPTIC>"
                )
            if family == "referring_panoptic_segmentation":
                selected = [
                    _integer(value, "task.target_segment_ids[]", minimum=1)
                    for value in _sequence(
                        row.get("target_segment_ids"), "task.target_segment_ids"
                    )
                ]
                metadata = self._panoptic_overrides.get(
                    (split, image_id), image["panoptic"]
                )
                expected = {
                    item["id"]
                    for item in metadata["segments_info"]
                    if item["category_id"] == row["category_id"]
                }
                if len(selected) != len(set(selected)) or set(selected) != expected:
                    raise _fail(
                        "panoptic_query_mismatch",
                        f"task {task_id!r} segments disagree with its queried category",
                    )

    def _iter_tasks_validated(self, split: str) -> Iterator[dict[str, Any]]:
        self._assert_metadata_unchanged()
        spec = self._spec(split)
        self._load_split(split)
        seen: set[str] = set()
        counts: Counter[str] = Counter()
        total = 0
        index = self._load_split(split)
        for line_number, row in _jsonl_rows(spec.tasks_path, "tasks"):
            context = f"{spec.tasks_path}:{line_number}"
            self._validate_task(split, row, context, index)
            task_id = row["id"]
            if task_id in seen:
                raise _fail(
                    "duplicate_task_id", f"duplicate task ID {task_id!r} at {context}"
                )
            seen.add(task_id)
            counts[row["family"]] += 1
            total += 1
            yield copy.deepcopy(row)
        if total != spec.tasks_count:
            raise _fail(
                "task_count",
                f"{split} task count {total} disagrees with manifest {spec.tasks_count}",
            )
        observed_families = {family: counts.get(family, 0) for family in spec.families}
        if observed_families != dict(spec.families):
            raise _fail(
                "task_family_count",
                f"{split} task family counts disagree with the manifest",
            )

    def iter_tasks(self, split: str) -> Iterator[dict[str, Any]]:
        """Stream validated task-row copies; exhausting the iterator checks all counts."""

        yield from self._iter_tasks_validated(split)

    def _task_row(self, split: str, task: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(task, Mapping):
            row = copy.deepcopy(dict(task))
            self._validate_task(split, row, "task", self._load_split(split))
            return row
        if not isinstance(task, str) or not task:
            raise _fail(
                "invalid_task_id", "task must be a row mapping or non-empty task ID"
            )
        found = None
        for row in self.iter_tasks(split):
            if row["id"] == task:
                found = row
        if found is None:
            raise _fail("unknown_task", f"unknown task {task!r} in split {split!r}")
        return found

    def _task_objects(
        self,
        split: str,
        task: Mapping[str, Any] | str,
        field: str,
    ) -> tuple[dict[str, Any], ...]:
        row = self._task_row(split, task)
        if field not in row:
            raise _fail(
                "missing_task_object_field", f"task {row['id']!r} lacks {field}"
            )
        index = self._load_split(split)
        return tuple(copy.deepcopy(index.object_by_id[item]) for item in row[field])

    def target_objects(
        self,
        split: str,
        task: Mapping[str, Any] | str,
    ) -> tuple[dict[str, Any], ...]:
        """Resolve an explicitly present target list; an explicit empty list returns ``()``."""

        return self._task_objects(split, task, "target_annotation_ids")

    def context_objects(
        self,
        split: str,
        task: Mapping[str, Any] | str,
    ) -> tuple[dict[str, Any], ...]:
        """Resolve an explicitly present context list without treating it as a mask target."""

        return self._task_objects(split, task, "context_annotation_ids")

    @staticmethod
    def _decode_instance_mask(
        segmentation: Any,
        height: int,
        width: int,
    ) -> Any:
        try:
            import numpy as np
            from pycocotools import mask as mask_utils

            value = copy.deepcopy(segmentation)
            if isinstance(value, list):
                encoded = mask_utils.merge(mask_utils.frPyObjects(value, height, width))
            else:
                encoded = value
                if isinstance(encoded.get("counts"), str):
                    encoded["counts"] = encoded["counts"].encode("ascii")
                elif isinstance(encoded.get("counts"), list):
                    encoded = mask_utils.frPyObjects(encoded, height, width)
            decoded = np.asarray(mask_utils.decode(encoded))
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            decoded = decoded.astype(bool, copy=False)
        except Exception as exc:
            raise _fail(
                "invalid_segmentation",
                "cannot decode immutable COCO segmentation",
            ) from exc
        if decoded.shape != (height, width) or not bool(decoded.any()):
            raise _fail(
                "invalid_segmentation",
                "immutable COCO segmentation is empty or has the wrong shape",
            )
        return decoded

    def _panoptic_raster_issues(
        self,
        split: str,
        image: Mapping[str, Any],
        issues: _Issues,
    ) -> None:
        try:
            path, metadata = self.resolve_panoptic(split, image["id"])
            self._panoptic_metadata_raster_issues(
                split,
                image,
                path,
                metadata,
                issues,
            )
        except FalconXDatasetError as exc:
            issues.add(exc.code, str(exc), split=split, image_id=image.get("id"))
        except Exception as exc:
            issues.add(
                "panoptic_decode",
                f"cannot decode panoptic PNG: {exc}",
                split=split,
                image_id=image.get("id"),
            )

    @staticmethod
    def _panoptic_metadata_raster_issues(
        split: str,
        image: Mapping[str, Any],
        path: Path,
        metadata: Mapping[str, Any],
        issues: _Issues,
    ) -> None:
        try:
            import numpy as np
            from PIL import Image

            from .artifacts import panoptic_ids_from_image

            with Image.open(path) as opened:
                if opened.size != (image["width"], image["height"]):
                    issues.add(
                        "panoptic_dimensions",
                        "panoptic PNG dimensions disagree with its image",
                        split=split,
                        image_id=image["id"],
                    )
                    return
                id_map = panoptic_ids_from_image(opened)
            raster_ids = {int(item) for item in np.unique(id_map) if int(item) != 0}
            declared = {int(item["id"]): item for item in metadata["segments_info"]}
            for segment_id, segment in declared.items():
                mask = id_map == segment_id
                if not bool(mask.any()):
                    issues.add(
                        "panoptic_missing_segment_id",
                        f"declared panoptic ID {segment_id} is absent from the PNG",
                        split=split,
                        image_id=image["id"],
                    )
                    continue
                ys, xs = np.nonzero(mask)
                bbox = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ]
                area = int(mask.sum())
                if list(segment["bbox"]) != bbox:
                    issues.add(
                        "panoptic_bbox_mismatch",
                        f"panoptic ID {segment_id} bbox {segment['bbox']!r} != raster {bbox!r}",
                        split=split,
                        image_id=image["id"],
                    )
                if not math.isclose(float(segment["area"]), float(area), abs_tol=1e-6):
                    issues.add(
                        "panoptic_area_mismatch",
                        f"panoptic ID {segment_id} area {segment['area']!r} != raster {area}",
                        split=split,
                        image_id=image["id"],
                    )
            extras = sorted(raster_ids.difference(declared))
            if extras:
                issues.add(
                    "panoptic_undeclared_segment_ids",
                    f"panoptic PNG has undeclared nonzero IDs {extras!r}",
                    split=split,
                    image_id=image["id"],
                )
        except Exception as exc:
            issues.add(
                "panoptic_decode",
                f"cannot decode panoptic PNG: {exc}",
                split=split,
                image_id=image.get("id"),
            )

    def _full_validation(
        self,
        split: str,
        selected_families: set[str],
        index: _SplitIndex,
        issues: _Issues,
        image_ids: frozenset[NativeId] | None = None,
    ) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            issues.add(
                "missing_dependency", f"Pillow is required for full validation: {exc}"
            )
            return {}

        images = (
            index.images
            if image_ids is None
            else tuple(image for image in index.images if image["id"] in image_ids)
        )
        expected_assets: set[str] = set()
        asset_by_path: dict[str, Path] = {}
        for image in images:
            relative = image["file_name"]
            expected_assets.add(relative)
            try:
                path = self.resolve_image(split, image["id"])
                asset_by_path[relative] = path
                with Image.open(path) as opened:
                    if opened.size != (image["width"], image["height"]):
                        issues.add(
                            "image_dimensions",
                            "image header dimensions disagree with metadata",
                            split=split,
                            image_id=image["id"],
                        )
            except (OSError, FalconXDatasetError) as exc:
                code = (
                    exc.code if isinstance(exc, FalconXDatasetError) else "image_decode"
                )
                issues.add(code, str(exc), split=split, image_id=image["id"])
            if "panoptic" in image:
                panoptic_relative = image["panoptic"]["file_name"]
                expected_assets.add(panoptic_relative)
                try:
                    path, _ = self.resolve_panoptic(split, image["id"])
                    asset_by_path[panoptic_relative] = path
                    with Image.open(path) as opened:
                        if opened.size != (image["width"], image["height"]):
                            issues.add(
                                "panoptic_dimensions",
                                "panoptic PNG dimensions disagree with its image",
                                split=split,
                                image_id=image["id"],
                            )
                except (OSError, FalconXDatasetError) as exc:
                    code = (
                        exc.code
                        if isinstance(exc, FalconXDatasetError)
                        else "panoptic_decode"
                    )
                    issues.add(code, str(exc), split=split, image_id=image["id"])

        actual_assets: set[str] = set()
        split_root = self.root / split
        if image_ids is None:
            for directory_name in _ASSET_DIRECTORIES:
                directory = split_root / directory_name
                if not directory.exists():
                    continue
                for path in directory.rglob("*"):
                    relative = path.relative_to(split_root).as_posix()
                    if path.is_symlink():
                        issues.add(
                            "symlink_asset", f"asset must not be a symlink: {relative}"
                        )
                    elif path.is_file():
                        actual_assets.add(relative)
        else:
            # A scoped audit deliberately makes no claim about unrelated assets.
            # Each selected path was already resolved with no-symlink containment.
            actual_assets.update(asset_by_path)
        missing = sorted(expected_assets.difference(actual_assets))
        if missing:
            issues.add("missing_assets", f"missing listed assets: {missing[:10]!r}")
        if image_ids is None:
            extras = sorted(actual_assets.difference(expected_assets))
            if extras:
                issues.add("unlisted_assets", f"unlisted assets: {extras[:10]!r}")

        parent_ids = {image.get("source_image_id", image["id"]) for image in images}
        objects = (
            index.objects
            if image_ids is None
            else tuple(obj for obj in index.objects if obj["image_id"] in parent_ids)
        )
        for obj in objects:
            image = index.image_by_id[obj["image_id"]]
            try:
                self._decode_instance_mask(
                    obj["segmentation"], image["height"], image["width"]
                )
            except FalconXDatasetError as exc:
                issues.add(exc.code, str(exc), split=split, object_id=obj["id"])

        if PANOPTIC_FAMILIES.intersection(selected_families):
            for image in images:
                if "panoptic" in image:
                    self._panoptic_raster_issues(split, image, issues)

        return {
            "asset_scope": "all" if image_ids is None else "selected_images",
            "validated_images": len(images),
            "validated_objects": len(objects),
            "validated_assets": len(expected_assets.intersection(actual_assets)),
        }

    def validate(
        self,
        split: str,
        tasks: Iterable[str] | str | None = None,
        full: bool = False,
        *,
        image_ids: Iterable[NativeId] | None = None,
    ) -> dict[str, Any]:
        """Validate one split and return a bounded, machine-readable report.

        ``tasks`` selects task families for expensive modality-specific checks.
        Structural task joins are always checked for the whole split.  In
        particular, native panoptic raster closure runs only when the selected
        families include a panoptic task.  ``image_ids`` scopes only
        expensive asset/mask/raster checks; whole-split metadata and task joins
        remain mandatory and it may therefore be used only with ``full=True``.
        """

        issues = _Issues()
        counts: dict[str, Any] = {}
        selected: set[str] = set()
        selected_image_ids: frozenset[NativeId] | None = None
        try:
            spec = self._spec(split)
            if tasks is None:
                selected = set(spec.families)
            elif isinstance(tasks, str):
                selected = {tasks}
            else:
                selected = set(tasks)
            unknown = selected.difference(spec.families)
            if unknown:
                raise _fail(
                    "unknown_family",
                    f"unknown selected task families: {sorted(unknown)!r}",
                )
            index = self._load_split(split)
            if image_ids is not None:
                if not full:
                    raise _fail(
                        "invalid_validation_scope", "image_ids requires full=True"
                    )
                if isinstance(image_ids, str | bytes):
                    raise _fail(
                        "invalid_validation_scope",
                        "image_ids must be an iterable of IDs",
                    )
                selected_image_ids = frozenset(
                    _identifier(value, "validate.image_ids[]") for value in image_ids
                )
                unknown_images = selected_image_ids.difference(index.image_by_id)
                if unknown_images:
                    raise _fail(
                        "unknown_image",
                        "validation scope contains unknown image IDs: "
                        f"{sorted(map(str, unknown_images))[:10]!r}",
                    )
            family_counts: Counter[str] = Counter()
            task_count = 0
            for task in self.iter_tasks(split):
                family_counts[task["family"]] += 1
                task_count += 1
            counts = {
                "images": len(index.images),
                "source_images": sum(
                    "source_image_id" not in row for row in index.images
                ),
                "counterfactual_images": sum(
                    "source_image_id" in row for row in index.images
                ),
                "objects": len(index.objects),
                "tasks": task_count,
                "families": dict(sorted(family_counts.items())),
                "annotation_coverage": self.annotation_coverage(split),
            }
            if full:
                counts.update(
                    self._full_validation(
                        split,
                        selected,
                        index,
                        issues,
                        selected_image_ids,
                    )
                )
        except FalconXDatasetError as exc:
            issues.add(exc.code, str(exc), split=split)
        return {
            "ok": issues.total == 0,
            "format": self.format,
            "split": split,
            "full": bool(full),
            "image_scope": "all" if selected_image_ids is None else "selected_images",
            "selected_families": sorted(selected),
            "counts": counts,
            "error_count": issues.total,
            "error_counts": dict(sorted(issues.counts.items())),
            "errors": issues.errors,
            "suppressed_error_count": max(0, issues.total - len(issues.errors)),
        }


__all__ = ["FalconXDataset", "FalconXDatasetError", "NativeId"]
