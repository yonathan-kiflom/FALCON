"""Model-agnostic, resumable task prediction ledger.

Backend callbacks receive :func:`inference_task_view` and return family-specific
predictions; scoring does not require model dependencies.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from falcon.artifacts import atomic_json, external_output
from falcon.tasks import (
    DATASET_PROTOCOL,
    TASK_REGISTRY,
    inference_task_view,
    parse_task_selection,
)

from .metrics import safe_artifact_path
from .schemas import (
    PREDICTION_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    EvaluationError,
    PredictionFormatError,
    load_prediction_set,
    load_run_manifest,
    validate_prediction_envelope,
    validate_run_manifest,
)
from .scorer import (
    _partition_contract,
    checkpoint_publication_status,
    require_final_stage3_provenance,
    expected_run_manifest,
    select_dataset_tasks,
)

Predictor = Callable[[Mapping[str, Any]], Mapping[str, Any]]

_LEDGER_EVENT_SCHEMA_VERSION = "falcon-task-ledger-event-v1"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_PAYLOAD_FIELDS = {
    "text": frozenset({"answer", "diagnostics"}),
    "answer": frozenset({"answer", "diagnostics"}),
    "binary_mask": frozenset({"regions", "diagnostics"}),
    "panoptic": frozenset({"panoptic", "artifact_sha256", "diagnostics"}),
}


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PredictionFormatError(f"run metadata is not finite JSON: {exc}") from exc


def build_run_manifest(
    dataset: Any,
    *,
    split: str,
    tasks: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    protocol: str,
    provenance: Mapping[str, Any],
    partition: Any | None = None,
    partition_role: str = "validation",
) -> dict[str, Any]:
    """Build the immutable portion of an official run manifest."""

    if not isinstance(provenance, Mapping):
        raise PredictionFormatError("run provenance must be an object")
    base = expected_run_manifest(
        dataset,
        split,
        tasks,
        families,
        protocol,
        partition=partition,
        partition_role=partition_role,
    )
    base.update(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "provenance": dict(provenance),
        }
    )
    base["created_at"] = datetime.now(timezone.utc).isoformat()
    # Validate serializability now, before any prediction is attempted.
    _canonical_json(base)
    return dict(validate_run_manifest(base))


def _immutable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "created_at"}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _reject_symlink(path, "atomic JSON destination")
    atomic_json(path, value)
    _fsync_directory(path.parent)


def _rewrite_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _reject_symlink(path, "prediction ledger")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _success_row(
    split: str,
    task: Mapping[str, Any],
    payload: Any,
    artifact_root: Path,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PredictionFormatError("predictor must return an object")
    kind = TASK_REGISTRY[str(task["family"])].prediction_kind
    allowed = _PAYLOAD_FIELDS[kind]
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise PredictionFormatError(
            f"predictor returned fields not allowed by {kind} grammar: {unknown}"
        )
    required = (
        "answer"
        if kind in ("text", "answer")
        else ("regions" if kind == "binary_mask" else "panoptic")
    )
    if required not in payload:
        raise PredictionFormatError(f"predictor {kind} payload lacks {required!r}")
    if kind == "panoptic":
        panoptic = payload["panoptic"]
        if not isinstance(panoptic, Mapping):
            raise PredictionFormatError("predictor panoptic payload must be an object")
        safe_artifact_path(
            artifact_root, panoptic.get("file_name"), "predictor panoptic"
        )
    row = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "split": split,
        "task_id": task["id"],
        "status": "ok",
        **dict(payload),
    }
    checked = dict(validate_prediction_envelope(row, f"prediction {task['id']!r}"))
    _canonical_json(checked)
    return checked


def _error_row(split: str, task_id: str, exc: Exception) -> dict[str, Any]:
    message = str(exc).strip() or type(exc).__name__
    row = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "split": split,
        "task_id": task_id,
        "status": "error",
        "error": f"{type(exc).__name__}: {message}",
    }
    checked = dict(validate_prediction_envelope(row, f"prediction {task_id!r}"))
    _canonical_json(checked)
    return checked


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    _reject_symlink(path, "JSONL ledger")
    existed = os.path.lexists(path)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_symlink(path: Path, name: str) -> None:
    if path.is_symlink():
        raise EvaluationError(f"{name} must not be a symlink: {path}")


def _validate_run_layout(directory: Path) -> None:
    for name in ("run.json", "predictions.jsonl", "attempts.jsonl", ".run.lock"):
        _reject_symlink(directory / name, f"run ledger {name}")
    panoptic = directory / "panoptic"
    _reject_symlink(panoptic, "panoptic artifact directory")
    if panoptic.exists():
        if not panoptic.is_dir():
            raise EvaluationError(
                f"panoptic artifact path is not a directory: {panoptic}"
            )
        for path in panoptic.rglob("*"):
            _reject_symlink(path, "panoptic artifact path")


def validate_resume_run(
    run_dir: str | Path,
    *,
    expected_manifest: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Check an existing run's identity without creating, locking, or repairing it.

    Every supplied top-level manifest/provenance field must match exactly;
    nested field values are compared in full. Ledger contents are intentionally
    not parsed here: a recoverable truncated final line must remain untouched
    until the execution path holds the run lock. The locked runner still repeats
    its complete identity checks before repairing or appending either ledger.
    """

    if not isinstance(expected_manifest, Mapping) or not expected_manifest:
        raise PredictionFormatError(
            "resume preflight requires expected manifest fields"
        )
    if not isinstance(expected_provenance, Mapping) or not expected_provenance:
        raise PredictionFormatError(
            "resume preflight requires expected provenance fields"
        )
    _canonical_json(expected_manifest)
    _canonical_json(expected_provenance)
    requested = Path(run_dir).expanduser()
    _reject_symlink(requested, "resume run directory")
    if not requested.is_dir():
        raise PredictionFormatError(
            f"resume requires an existing run directory, not a fresh run: {requested}"
        )
    directory = requested.resolve(strict=True)
    _validate_run_layout(directory)
    for name in ("run.json", "predictions.jsonl"):
        if not (directory / name).is_file():
            raise PredictionFormatError(
                f"resume requires a regular {name}: {directory}"
            )
    for name in ("attempts.jsonl", ".run.lock"):
        path = directory / name
        if path.exists() and not path.is_file():
            raise PredictionFormatError(f"resume {name} must be a regular file: {path}")
    manifest = load_run_manifest(directory / "run.json")
    if manifest["schema_version"] != RUN_SCHEMA_VERSION:
        raise PredictionFormatError(
            "Legacy runs are read-only; use a new run directory with the current workflow"
        )
    mismatches = [
        field
        for field, value in expected_manifest.items()
        if field not in manifest or manifest[field] != value
    ]
    if mismatches:
        raise PredictionFormatError(
            "resume manifest differs from expected dataset/query/metric identity: "
            + ", ".join(sorted(mismatches))
        )
    provenance = manifest["provenance"]
    mismatches = [
        field
        for field, value in expected_provenance.items()
        if field not in provenance or provenance[field] != value
    ]
    if mismatches:
        raise PredictionFormatError(
            "resume provenance differs from expected execution: "
            + ", ".join(sorted(mismatches))
        )
    _validate_run_layout(directory)
    return {
        "schema_version": "falcon-task-resume-preflight-v1",
        "valid": True,
        "run_dir": str(directory),
        "manifest_fields_checked": sorted(expected_manifest),
        "provenance_fields_checked": sorted(expected_provenance),
        "ledger_contents_validated": False,
        "repairs_performed": False,
        "lock_checked": False,
        "scope": "read-only layout and manifest identity; ledger recovery and locking deferred to run",
    }


def _acquire_run_lock(path: Path) -> int:
    """Acquire a kernel-released lock; a stale on-disk lock file is harmless."""

    flags = os.O_CREAT | os.O_RDWR | _NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise EvaluationError(
            f"run directory is locked by an active process: {path}"
        ) from exc
    metadata = {
        "schema_version": "falcon-task-lock-v1",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = _canonical_json(metadata) + b"\n"
    try:
        os.ftruncate(descriptor, 0)
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise OSError("short write while recording run lock metadata")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except Exception:
        _release_run_lock(descriptor)
        raise
    return descriptor


def _release_run_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _recover_jsonl_tail(path: Path) -> dict[str, Any] | None:
    """Repair only a syntactically truncated final line after the run lock is held."""

    _reject_symlink(path, "JSONL recovery target")
    if not path.exists():
        return None
    descriptor = os.open(path, os.O_RDWR | _NOFOLLOW)
    with os.fdopen(descriptor, "r+b") as stream:
        line_number = 0
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                return None
            line_number += 1
            terminated = raw.endswith(b"\n")
            if not raw.strip():
                if terminated:
                    continue
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                return {
                    "schema_version": _LEDGER_EVENT_SCHEMA_VERSION,
                    "event": "truncated_tail_discarded",
                    "file": path.name,
                    "line": line_number,
                    "discarded_bytes": len(raw),
                    "recovered_at": datetime.now(timezone.utc).isoformat(),
                }
            try:
                decoded = raw.decode("utf-8")
                value = json.loads(decoded, parse_constant=_invalid_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if terminated:
                    raise PredictionFormatError(
                        f"{path}:{line_number} contains corrupt complete JSON: {exc}"
                    ) from exc
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                return {
                    "schema_version": _LEDGER_EVENT_SCHEMA_VERSION,
                    "event": "truncated_tail_discarded",
                    "file": path.name,
                    "line": line_number,
                    "discarded_bytes": len(raw),
                    "recovered_at": datetime.now(timezone.utc).isoformat(),
                }
            except ValueError as exc:
                raise PredictionFormatError(
                    f"{path}:{line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise PredictionFormatError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            if not terminated:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                return {
                    "schema_version": _LEDGER_EVENT_SCHEMA_VERSION,
                    "event": "missing_final_newline_repaired",
                    "file": path.name,
                    "line": line_number,
                    "preserved_bytes": len(raw),
                    "recovered_at": datetime.now(timezone.utc).isoformat(),
                }


def _validate_attempt_log(path: Path) -> None:
    _reject_symlink(path, "attempt audit")
    if not path.exists():
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, parse_constant=_invalid_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise PredictionFormatError(
                        f"{path}:{line_number} is invalid audit JSON: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise PredictionFormatError(
                        f"{path}:{line_number} audit row must be an object"
                    )
                if value.get("schema_version") == PREDICTION_SCHEMA_VERSION:
                    validate_prediction_envelope(value, f"{path}:{line_number}")
                elif value.get("schema_version") == _LEDGER_EVENT_SCHEMA_VERSION:
                    if value.get("event") not in {
                        "truncated_tail_discarded",
                        "missing_final_newline_repaired",
                    }:
                        raise PredictionFormatError(
                            f"{path}:{line_number} has an unknown ledger event"
                        )
                else:
                    raise PredictionFormatError(
                        f"{path}:{line_number} has an unknown audit schema"
                    )
    except OSError as exc:
        raise PredictionFormatError(f"cannot read attempt audit {path}: {exc}") from exc


def run_predictions(
    dataset: Any,
    predictor: Predictor,
    *,
    run_dir: str | Path,
    split: str = "test",
    tasks: str | Sequence[str] | None = "all",
    protocol: str = DATASET_PROTOCOL,
    provenance: Mapping[str, Any],
    resume: bool = False,
    retry_errors: bool = False,
    partition: Any | None = None,
    partition_role: str = "validation",
) -> dict[str, Any]:
    """Execute a restricted predictor callback with crash-safe incremental output.

    The function intentionally has no model imports.  Retried error rows are
    archived in ``attempts.jsonl`` and atomically removed from the official
    one-row-per-task ``predictions.jsonl`` before retrying.
    """

    require_final_stage3_provenance(provenance)
    if retry_errors and not resume:
        raise ValueError("retry_errors requires resume=True")
    if protocol != DATASET_PROTOCOL:
        raise EvaluationError("only dataset-v1 has an executable task registry")
    families = (
        parse_task_selection(tasks)
        if isinstance(tasks, str) or tasks is None
        else parse_task_selection(",".join(tasks))
    )
    partition_image_ids, _partition_fields = _partition_contract(
        partition, partition_role, split
    )
    selected = select_dataset_tasks(
        dataset, split, families, image_ids=partition_image_ids
    )
    protected_roots = getattr(dataset, "protected_roots", None)
    directory = external_output(
        run_dir,
        dataset.root,
        protected_roots=(protected_roots() if callable(protected_roots) else ()),
    )
    manifest_path = directory / "run.json"
    rows_path = directory / "predictions.jsonl"
    attempts_path = directory / "attempts.jsonl"
    lock_path = directory / ".run.lock"

    if directory.exists() and not resume:
        raise FileExistsError(f"run directory already exists; use resume: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    _validate_run_layout(directory)
    lock_fd = _acquire_run_lock(lock_path)
    try:
        proposed = build_run_manifest(
            dataset,
            split=split,
            tasks=selected,
            families=families,
            protocol=protocol,
            provenance=provenance,
            partition=partition,
            partition_role=partition_role,
        )
        existing_rows: list[Mapping[str, Any]] = []
        recoveries: list[Mapping[str, Any]] = []
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"run manifest already exists: {manifest_path}")
            # Establish identity before repairing either ledger.  Otherwise a
            # mistargeted --resume could mutate a different run's crash tail
            # before its dataset/query/provenance mismatch was discovered.
            saved_manifest = load_run_manifest(manifest_path)
            if saved_manifest["schema_version"] != RUN_SCHEMA_VERSION:
                raise PredictionFormatError(
                    "Legacy runs are read-only; choose a new run directory"
                )
            if _immutable_manifest(saved_manifest) != _immutable_manifest(proposed):
                raise PredictionFormatError(
                    "resume manifest differs from dataset/tasks/images/provenance"
                )
            attempts_recovery = _recover_jsonl_tail(attempts_path)
            if attempts_recovery is not None:
                recoveries.append(attempts_recovery)
            _validate_attempt_log(attempts_path)
            rows_recovery = _recover_jsonl_tail(rows_path)
            if rows_recovery is not None:
                recoveries.append(rows_recovery)
            for event in recoveries:
                _append_row(attempts_path, event)
            loaded = load_prediction_set(directory)
            existing_rows = list(loaded.rows)
            if loaded.manifest != saved_manifest:
                raise PredictionFormatError("run manifest changed while resuming")
            expected_keys = {(split, str(task["id"])) for task in selected}
            unexpected = sorted(set(loaded.index()).difference(expected_keys))
            if unexpected:
                raise PredictionFormatError(
                    f"resume ledger contains foreign or extra task rows: {unexpected[:20]}"
                )
        else:
            non_lock_contents = [
                path for path in directory.iterdir() if path != lock_path
            ]
            if resume and non_lock_contents:
                raise PredictionFormatError(
                    "cannot resume a run directory without run.json"
                )
            _atomic_json(manifest_path, proposed)
            descriptor = os.open(
                rows_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW,
                0o600,
            )
            os.close(descriptor)
            _fsync_directory(directory)

        if any(
            TASK_REGISTRY[family].prediction_kind == "panoptic" for family in families
        ):
            panoptic_root = directory / "panoptic"
            panoptic_root.mkdir(mode=0o700, exist_ok=True)
            _validate_run_layout(directory)
            _fsync_directory(directory)

        if retry_errors:
            retained = []
            for row in existing_rows:
                if row["status"] != "error":
                    retained.append(row)
            if len(retained) != len(existing_rows):
                _rewrite_rows(rows_path, retained)
                existing_rows = retained

        completed = {(str(row["split"]), str(row["task_id"])) for row in existing_rows}
        written_ok = written_error = skipped = 0
        started = time.monotonic()
        for task in selected:
            key = (split, str(task["id"]))
            if key in completed:
                skipped += 1
                continue
            fatal_error = None
            try:
                view = inference_task_view(dataset, split, task)
                row = _success_row(split, task, predictor(view), directory)
                written_ok += 1
            except Exception as exc:  # one model failure must not drop later task IDs
                row = _error_row(split, str(task["id"]), exc)
                written_error += 1
                if (
                    isinstance(exc, MemoryError)
                    or type(exc).__name__ == "OutOfMemoryError"
                    or any(
                        marker in str(exc).lower()
                        for marker in (
                            "cuda out of memory",
                            "device-side assert",
                            "illegal memory access",
                        )
                    )
                ):
                    fatal_error = exc
            _append_row(attempts_path, row)
            _append_row(rows_path, row)
            completed.add(key)
            attempted = written_ok + written_error
            if (
                attempted == 1
                or attempted % 100 == 0
                or len(completed) == len(selected)
            ):
                print(
                    f"[{split}] saved {len(completed)}/{len(selected)} "
                    f"(new_ok={written_ok}, new_errors={written_error}, "
                    f"elapsed={time.monotonic() - started:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
            if fatal_error is not None:
                raise EvaluationError(
                    "model execution stopped after a fatal memory/device error; outputs are saved. "
                    "Resolve the runtime issue, then use --resume --retry-errors."
                ) from fatal_error
        final_rows = load_prediction_set(directory).rows
        final_error_count = sum(row["status"] == "error" for row in final_rows)
        complete = len(completed) == len(selected)
        checkpoint_status = checkpoint_publication_status(provenance)
        publication_reasons = list(checkpoint_status["reasons"])
        if not complete:
            publication_reasons.append("prediction_coverage_incomplete")
        if final_error_count:
            publication_reasons.append("model_invalid_predictions")
        publication_reasons = list(dict.fromkeys(publication_reasons))
        official = complete and final_error_count == 0 and checkpoint_status["eligible"]
        return {
            "schema_version": "falcon-task-run-result-v1",
            "run_dir": str(directory),
            "task_count": len(selected),
            "written_ok": written_ok,
            "written_error": written_error,
            "skipped": skipped,
            "recoveries": [dict(event) for event in recoveries],
            "complete": complete,
            "model_invalid": final_error_count,
            "official": official,
            "publication_status": {
                "official_result_eligible": official,
                "checkpoint_official_result_eligible": checkpoint_status["eligible"],
                "reasons": publication_reasons,
            },
        }
    finally:
        _release_run_lock(lock_fd)
