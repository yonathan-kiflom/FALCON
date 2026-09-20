"""Versioned schemas for saved task predictions.

Only identifiers and envelope fields are validated here.  A row with a valid
envelope but an unusable answer/mask is a model-invalid result and is kept in
the scoring denominator; it is not confused with a missing task row.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREDICTION_SCHEMA_VERSION = "falcon-task-prediction-v1"
RUN_SCHEMA_VERSION = "falcon-task-run-v3"
LEGACY_RUN_SCHEMA_VERSION = "falcon-task-run-v2"

_REQUIRED_RUN_FIELDS = {
    "schema_version",
    "prediction_schema_version",
    "protocol",
    "metric_recipe_version",
    "split",
    "task_families",
    "task_count",
    "provenance",
    "created_at",
}
_PARTITION_FIELDS = {"partition_role", "partition_path", "partition_image_count"}
_RUN_FIELDS = (
    _REQUIRED_RUN_FIELDS
    | _PARTITION_FIELDS
    | {
        "dataset",
        "dataset_categories",
        "task_counts",
    }
)
_LEGACY_RUN_FIELDS = _REQUIRED_RUN_FIELDS | {
    "metric_recipe_sha256",
    "query_sha256",
    "dataset_fingerprint",
    "image_fingerprint",
    "run_fingerprint",
    "partition_role",
    "partition_fingerprint",
    "partition_image_count",
}


class EvaluationError(ValueError):
    """Base error for the task-evaluation contract."""


class PredictionFormatError(EvaluationError):
    """A prediction file or row does not satisfy the versioned envelope."""


class PredictionCoverageError(EvaluationError):
    """Prediction task IDs do not exactly cover the requested task set."""


class ModelOutputError(EvaluationError):
    """One present prediction row has an unusable model payload."""


class ReferenceValidationError(EvaluationError):
    """Evaluation ground truth is internally inconsistent or incomplete."""


_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "split",
        "task_id",
        "status",
        "answer",
        "regions",
        "panoptic",
        "error",
        "diagnostics",
        "artifact_sha256",
    }
)


@dataclass(frozen=True)
class PredictionSet:
    """Validated prediction envelopes plus their artifact root and manifest."""

    rows: tuple[Mapping[str, Any], ...]
    artifact_root: Path
    manifest: Mapping[str, Any] | None = None

    def index(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        result: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in self.rows:
            key = (str(row["split"]), str(row["task_id"]))
            if key in result:
                raise PredictionCoverageError(
                    f"duplicate prediction for split/task_id {key!r}"
                )
            result[key] = row
        return result


def _reject_constant(value: str) -> None:
    raise PredictionFormatError(f"non-finite JSON constant {value!r} is not allowed")


def validate_run_manifest(
    value: Any, source: str = "run manifest"
) -> Mapping[str, Any]:
    """Validate run settings. Legacy provenance is retained, not re-certified."""
    if not isinstance(value, Mapping):
        raise PredictionFormatError(f"{source} must contain a JSON object")
    legacy = value.get("schema_version") == LEGACY_RUN_SCHEMA_VERSION
    if not legacy and value.get("schema_version") != RUN_SCHEMA_VERSION:
        raise PredictionFormatError(f"{source} has an unsupported schema version")
    required = (
        _REQUIRED_RUN_FIELDS
        if legacy
        else _REQUIRED_RUN_FIELDS
        | {
            "dataset",
            "dataset_categories",
            "task_counts",
        }
    )
    unknown = sorted(set(value) - (_LEGACY_RUN_FIELDS if legacy else _RUN_FIELDS))
    missing = sorted(required - set(value))
    if unknown or missing:
        raise PredictionFormatError(
            f"{source} fields differ; missing={missing}, unknown={unknown}"
        )
    if value["prediction_schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise PredictionFormatError(f"{source} prediction schema is unsupported")
    for key in ("protocol", "split", "created_at", "metric_recipe_version"):
        if not isinstance(value[key], str) or not value[key]:
            raise PredictionFormatError(f"{source} {key} must be a non-empty string")
    from falcon.tasks import METRIC_RECIPE_VERSION, TASK_REGISTRY

    if value["metric_recipe_version"] != METRIC_RECIPE_VERSION:
        raise PredictionFormatError(f"{source} metric recipe version is unsupported")
    families = value["task_families"]
    if (
        not isinstance(families, list)
        or not families
        or any(
            not isinstance(item, str) or item not in TASK_REGISTRY for item in families
        )
        or len(families) != len(set(families))
    ):
        raise PredictionFormatError(
            f"{source} task_families must name unique registered families"
        )
    count = value["task_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise PredictionFormatError(f"{source} task_count must be a positive integer")
    if not isinstance(value["provenance"], Mapping) or not value["provenance"]:
        raise PredictionFormatError(f"{source} provenance must be a non-empty object")
    if not legacy:
        if not isinstance(value["dataset"], str) or not value["dataset"]:
            raise PredictionFormatError(f"{source} dataset must be a path")
        if not isinstance(value["dataset_categories"], list):
            raise PredictionFormatError(f"{source} dataset_categories must be an array")
        counts = value["task_counts"]
        if (
            not isinstance(counts, Mapping)
            or set(counts) != set(families)
            or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 1
                for n in counts.values()
            )
            or sum(counts.values()) != count
        ):
            raise PredictionFormatError(
                f"{source} task_counts disagree with task coverage"
            )
    partition_fields = (
        {"partition_role", "partition_fingerprint", "partition_image_count"}
        if legacy
        else _PARTITION_FIELDS
    )
    present = partition_fields.intersection(value)
    if present:
        if present != partition_fields:
            raise PredictionFormatError(f"{source} partition settings are incomplete")
        if value["partition_role"] not in ("validation", "calibration"):
            raise PredictionFormatError(f"{source} partition_role is invalid")
        partition_count = value["partition_image_count"]
        if (
            isinstance(partition_count, bool)
            or not isinstance(partition_count, int)
            or partition_count < 1
        ):
            raise PredictionFormatError(
                f"{source} partition_image_count must be positive"
            )
        if not legacy and (
            not isinstance(value["partition_path"], str) or not value["partition_path"]
        ):
            raise PredictionFormatError(f"{source} partition_path must be a path")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PredictionFormatError(
            f"{source} must contain finite JSON values"
        ) from exc
    return dict(value)


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionFormatError(f"cannot read {path}: {exc}") from exc


def load_run_manifest(path: str | Path) -> Mapping[str, Any]:
    """Load and validate only ``run.json``, without touching prediction ledgers."""

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise PredictionFormatError(
            f"run manifest input must not be a symlink: {unresolved}"
        )
    supplied = unresolved.resolve()
    manifest_path = supplied / "run.json" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise PredictionFormatError(f"run manifest does not exist: {manifest_path}")
    if manifest_path.is_symlink():
        raise PredictionFormatError(
            f"run manifest must not be a symlink: {manifest_path}"
        )
    return validate_run_manifest(_read_json(manifest_path), str(manifest_path))


def validate_prediction_envelope(row: Any, source: str) -> Mapping[str, Any]:
    """Validate fields common to every task prediction."""

    if not isinstance(row, Mapping):
        raise PredictionFormatError(f"{source} must contain a JSON object")
    unknown = sorted(set(row).difference(_ROW_FIELDS))
    if unknown:
        raise PredictionFormatError(f"{source} has unknown fields: {unknown}")
    if row.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise PredictionFormatError(
            f"{source} schema_version must be {PREDICTION_SCHEMA_VERSION!r}"
        )
    if not isinstance(row.get("split"), str) or not row["split"]:
        raise PredictionFormatError(f"{source} requires a non-empty string split")
    if not isinstance(row.get("task_id"), str) or not row["task_id"]:
        raise PredictionFormatError(f"{source} requires a non-empty string task_id")
    status = row.get("status")
    if status not in ("ok", "error"):
        raise PredictionFormatError(f"{source} status must be 'ok' or 'error'")
    if status == "error" and not isinstance(row.get("error"), str | Mapping):
        raise PredictionFormatError(
            f"{source} error status requires an error description"
        )
    if "diagnostics" in row and not isinstance(row["diagnostics"], Mapping):
        raise PredictionFormatError(f"{source} diagnostics must be an object")
    if "artifact_sha256" in row:
        digest = row["artifact_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise PredictionFormatError(
                f"{source} artifact_sha256 must be lowercase SHA-256"
            )
    return dict(row)


def prediction_set_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    artifact_root: str | Path = ".",
    manifest: Mapping[str, Any] | None = None,
) -> PredictionSet:
    """Build a prediction set from in-memory rows (primarily for APIs/tests)."""

    checked = tuple(
        validate_prediction_envelope(row, f"prediction row {index}")
        for index, row in enumerate(rows, 1)
    )
    checked_manifest = None if manifest is None else validate_run_manifest(manifest)
    result = PredictionSet(checked, Path(artifact_root).resolve(), checked_manifest)
    result.index()
    return result


def load_prediction_set(
    path: str | Path, *, require_manifest: bool = True
) -> PredictionSet:
    """Load an official saved run directory or a JSONL plus manifest file.

    A directory contains ``run.json`` and ``predictions.jsonl``.  A standalone
    JSONL uses ``<name>.jsonl.manifest.json``.  The scorer requires the manifest
    for official results so dataset/protocol drift cannot be silent.
    """

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise PredictionFormatError(
            f"prediction input must not be a symlink: {unresolved}"
        )
    supplied = unresolved.resolve()
    if supplied.is_dir():
        rows_path = supplied / "predictions.jsonl"
        manifest_path = supplied / "run.json"
        artifact_root = supplied
    else:
        rows_path = supplied
        manifest_path = supplied.with_suffix(supplied.suffix + ".manifest.json")
        artifact_root = supplied.parent
    if not rows_path.is_file():
        raise PredictionFormatError(f"prediction JSONL does not exist: {rows_path}")
    if rows_path.is_symlink():
        raise PredictionFormatError(
            f"prediction JSONL must not be a symlink: {rows_path}"
        )
    if require_manifest and not manifest_path.is_file():
        raise PredictionFormatError(
            f"prediction manifest does not exist: {manifest_path}"
        )
    if manifest_path.is_symlink():
        raise PredictionFormatError(
            f"prediction manifest must not be a symlink: {manifest_path}"
        )
    manifest = None
    if manifest_path.is_file():
        manifest = load_run_manifest(manifest_path)

    rows: list[Mapping[str, Any]] = []
    try:
        with rows_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                except json.JSONDecodeError as exc:
                    raise PredictionFormatError(
                        f"{rows_path}:{line_number} is invalid JSON: {exc}"
                    ) from exc
                rows.append(
                    validate_prediction_envelope(row, f"{rows_path}:{line_number}")
                )
    except OSError as exc:
        raise PredictionFormatError(f"cannot read {rows_path}: {exc}") from exc

    result = PredictionSet(
        tuple(rows),
        artifact_root,
        None if manifest is None else dict(manifest),
    )
    result.index()
    return result
