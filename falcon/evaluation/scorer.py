"""Task-level, model-free scoring for falcon-x datasets."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from falcon.data import DATASET_FORMAT
from falcon.tasks import (
    BINARY_MASK_FAMILIES,
    CAPTION_FAMILIES,
    DATASET_PROTOCOL,
    TEXT_METRIC_FAMILIES,
    TASK_REGISTRY,
    VQA_QUESTION_TYPES,
    evaluation_metric_recipe,
    parse_task_selection,
    protocol_capabilities,
)

from .metrics import (
    binary_classification,
    checked_segments,
    iou_threshold_accuracies,
    mask_intersection_union,
    normalize_text,
    panoptic_ids,
    parse_category_set,
    parse_unit_interval,
    parse_vqa_label,
    parse_yes_no,
    safe_artifact_path,
    semantic_map,
    union_object_masks,
    union_prediction_regions,
)
from .schemas import (
    EvaluationError,
    LEGACY_RUN_SCHEMA_VERSION,
    ModelOutputError,
    PredictionCoverageError,
    PredictionSet,
    ReferenceValidationError,
    validate_run_manifest,
)
from .text import TEXT_METRIC_BACKEND, caption_backend_status, score_captions


def metric_recipe_metadata(families: Sequence[str]) -> dict[str, Any]:
    """Describe the scoring rules without hashing implementation files."""
    return evaluation_metric_recipe(families)


def require_final_stage3_provenance(provenance: Any) -> None:
    """Reject partial, Stage 2, ablated or oracle runs, including legacy runs."""
    if not isinstance(provenance, Mapping):
        raise EvaluationError("evaluation requires completed Stage 3 provenance")
    scope = provenance.get("checkpoint_scope", {})
    legacy_final = provenance.get("final_stage3_export", {})
    legacy = isinstance(legacy_final, Mapping) and legacy_final.get("verified") is True
    final = provenance.get("final_stage3") is True or legacy
    stage = legacy_final.get("stage") if legacy else provenance.get("stage")
    ablation = provenance.get("ssa_ablation", "none")
    if isinstance(ablation, Mapping):
        ablation = ablation.get("name")
    if (
        not final
        or stage != 3
        or provenance.get("training_complete") is not True
        or not isinstance(scope, Mapping)
        or scope.get("training_complete") is not True
        or scope.get("partial_checkpoint_allowed") is not False
        or ablation != "none"
        or provenance.get("checkpoint_ssa_ablation", "none") != "none"
        or provenance.get("oracle_detector", "none") != "none"
        or not checkpoint_publication_status(provenance)["eligible"]
    ):
        raise EvaluationError(
            "evaluation requires a completed, unablated, non-oracle Stage 3 run"
        )


def checkpoint_publication_status(provenance: Any) -> dict[str, Any]:
    """Require an explicit checkpoint-export eligibility decision for official results."""

    if not isinstance(provenance, Mapping):
        return {"eligible": False, "reasons": ["checkpoint_scope_not_verified"]}
    scope = provenance.get("checkpoint_scope")
    if not isinstance(scope, Mapping):
        return {"eligible": False, "reasons": ["checkpoint_scope_not_verified"]}
    raw_reasons = scope.get(
        "diagnostic_training_reasons",
        provenance.get("diagnostic_training_reasons", ()),
    )
    if isinstance(raw_reasons, str | bytes) or not isinstance(raw_reasons, Sequence):
        reasons = ["checkpoint_diagnostic_reasons_invalid"]
    else:
        reasons = [str(reason) for reason in raw_reasons]
    explicitly_eligible = scope.get("official_result_eligible") is True
    if not explicitly_eligible and not reasons:
        reasons.append("checkpoint_not_official_result_eligible")
    if explicitly_eligible and reasons:
        reasons.append("checkpoint_eligibility_metadata_contradiction")
    # Keep reports deterministic even if an exporter repeated one reason.
    reasons = list(dict.fromkeys(reasons))
    return {"eligible": explicitly_eligible and not reasons, "reasons": reasons}


def _partition_contract(
    partition: Any | None,
    partition_role: str,
    split: str,
) -> tuple[frozenset[Any] | None, dict[str, Any] | None]:
    if partition is None:
        return None, None
    if split != "train":
        raise ReferenceValidationError(
            "partitions can only select the native train split"
        )
    if partition_role not in ("validation", "calibration"):
        raise ReferenceValidationError(
            "evaluation partition role must be 'validation' or 'calibration'"
        )
    if getattr(partition, "role", None) != partition_role:
        raise ReferenceValidationError(
            "partition selection role differs from the requested evaluation role"
        )
    image_ids = getattr(partition, "image_ids", None)
    if not isinstance(image_ids, frozenset):
        raise ReferenceValidationError("partition.image_ids must be a frozenset")
    source_path = getattr(partition, "source_path", None)
    if source_path is None:
        raise ReferenceValidationError("partition selection requires its source_path")
    return image_ids, {
        "partition_role": partition_role,
        "partition_path": str(Path(source_path).expanduser().resolve()),
        "partition_image_count": len(image_ids),
    }


def expected_run_manifest(
    dataset: Any,
    split: str,
    tasks: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    protocol: str,
    *,
    partition: Any | None = None,
    partition_role: str = "validation",
) -> dict[str, Any]:
    """Record dataset locations and explicit task counts for resume checks."""
    _image_ids, partition_fields = _partition_contract(partition, partition_role, split)
    result = {
        "protocol": protocol,
        "split": split,
        "task_families": list(families),
        "task_count": len(tasks),
        "task_counts": {
            family: sum(task["family"] == family for task in tasks)
            for family in families
        },
        "dataset": str(Path(dataset.root).expanduser().resolve()),
        "dataset_categories": list(dataset.manifest["categories"]),
        "metric_recipe_version": evaluation_metric_recipe(families)["version"],
    }
    if partition_fields is not None:
        result.update(partition_fields)
    return result


def select_dataset_tasks(
    dataset: Any,
    split: str,
    families: Sequence[str],
    *,
    image_ids: frozenset[Any] | None = None,
) -> list[Mapping[str, Any]]:
    """Select and validate references, preserving native split-scoped image IDs."""

    selected = []
    seen: set[str] = set()
    for index, task in enumerate(dataset.iter_tasks(split), 1):
        if not isinstance(task, Mapping):
            raise ReferenceValidationError(f"task row {index} is not an object")
        identifier = task.get("id")
        family = task.get("family")
        if not isinstance(identifier, str) or not identifier:
            raise ReferenceValidationError(f"task row {index} has invalid id")
        if identifier in seen:
            raise ReferenceValidationError(
                f"duplicate task id in {split}: {identifier!r}"
            )
        seen.add(identifier)
        if family not in TASK_REGISTRY:
            raise ReferenceValidationError(
                f"task {identifier!r} has unknown family {family!r}"
            )
        if family in families and (
            image_ids is None or task.get("image_id") in image_ids
        ):
            if not isinstance(task.get("prompt"), str) or not isinstance(
                task.get("answer"), str
            ):
                raise ReferenceValidationError(
                    f"task {identifier!r} requires string prompt and answer"
                )
            spec = TASK_REGISTRY[str(family)]
            image = dataset.image(split, task["image_id"])
            is_counterfactual = "source_image_id" in image
            if spec.image_scope == "source" and is_counterfactual:
                raise ReferenceValidationError(
                    f"task {identifier!r} must reference a source image"
                )
            if spec.image_scope == "counterfactual" and not is_counterfactual:
                raise ReferenceValidationError(
                    f"task {identifier!r} must reference a counterfactual image"
                )
            if family == "vqa" and task.get("question_type") not in VQA_QUESTION_TYPES:
                raise ReferenceValidationError(
                    f"VQA task {identifier!r} has unsupported question_type "
                    f"{task.get('question_type')!r}"
                )
            if family in BINARY_MASK_FAMILIES | {"category_presence"}:
                targets = task.get("target_annotation_ids")
                if isinstance(targets, str | bytes) or not isinstance(
                    targets, Sequence
                ):
                    raise ReferenceValidationError(
                        f"task {identifier!r} requires explicit target_annotation_ids"
                    )
            if family in {
                "category_presence",
                "category_or_all_instance_grounding",
                "referring_panoptic_segmentation",
            }:
                if "category_id" not in task:
                    raise ReferenceValidationError(
                        f"task {identifier!r} requires category_id (null means all objects)"
                    )
                category_id = task["category_id"]
                if (
                    family != "category_or_all_instance_grounding"
                    and category_id is None
                ):
                    raise ReferenceValidationError(
                        f"task {identifier!r} requires a concrete category_id"
                    )
                if category_id is not None and category_id not in _categories(dataset):
                    raise ReferenceValidationError(
                        f"task {identifier!r} has unknown category_id {category_id!r}"
                    )
            if family == "category_presence":
                label = parse_yes_no(task["answer"])
                if label is None:
                    raise ReferenceValidationError(
                        f"presence reference {identifier!r} is not canonical yes/no"
                    )
                if bool(task["target_annotation_ids"]) != label:
                    raise ReferenceValidationError(
                        f"presence reference {identifier!r} answer disagrees with target IDs"
                    )
            if (
                family == "vqa"
                and parse_vqa_label(str(task["question_type"]), task["answer"]) is None
            ):
                raise ReferenceValidationError(
                    f"VQA reference {identifier!r} cannot be parsed by dataset-v1"
                )
            if family in BINARY_MASK_FAMILIES and task["answer"] != "<SEG>":
                raise ReferenceValidationError(
                    f"mask task {identifier!r} reference answer must be '<SEG>'"
                )
            if family == "missing_component_identification":
                expected = parse_category_set(task["answer"], _category_names(dataset))
                raw_categories = task.get("missing_categories")
                if (
                    expected is None
                    or not isinstance(raw_categories, list)
                    or parse_category_set(
                        json.dumps(raw_categories), _category_names(dataset)
                    )
                    != expected
                ):
                    raise ReferenceValidationError(
                        f"missing-component reference {identifier!r} requires matching "
                        "canonical answer and missing_categories arrays"
                    )
            if family == "functional_completeness":
                expected_completeness = parse_unit_interval(task["answer"])
                raw_completeness = task.get("completeness")
                if (
                    expected_completeness not in (0.0, 1.0)
                    or isinstance(raw_completeness, bool)
                    or not isinstance(raw_completeness, int | float)
                    or raw_completeness != expected_completeness
                ):
                    raise ReferenceValidationError(
                        f"completeness reference {identifier!r} requires matching binary "
                        "answer and completeness target"
                    )
            if spec.prediction_kind == "panoptic" and task["answer"] != "<PANOPTIC>":
                raise ReferenceValidationError(
                    f"panoptic task {identifier!r} reference answer must be '<PANOPTIC>'"
                )
            if family == "referring_panoptic_segmentation":
                segment_ids = task.get("target_segment_ids")
                if (
                    not isinstance(segment_ids, list)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in segment_ids
                    )
                    or len(segment_ids) != len(set(segment_ids))
                ):
                    raise ReferenceValidationError(
                        f"referring-panoptic task {identifier!r} requires unique positive "
                        "target_segment_ids"
                    )
            selected.append(task)
    observed = {str(task["family"]) for task in selected}
    unsupported = [family for family in families if family not in observed]
    if unsupported:
        raise ReferenceValidationError(
            f"selected task families have no reference rows in {split}: {unsupported}"
        )
    return selected


def _prediction_rows(
    predictions: PredictionSet,
    split: str,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed = predictions.index()
    expected = {str(task["id"]) for task in tasks}
    actual = {task_id for row_split, task_id in indexed if row_split == split}
    foreign = sorted(key for key in indexed if key[0] != split)
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if foreign or missing or extra:
        raise PredictionCoverageError(
            "prediction split/task coverage differs; "
            f"missing={missing[:20]}, extra={extra[:20]}, foreign={foreign[:20]}"
        )
    return {task_id: indexed[(split, task_id)] for task_id in expected}


def _manifest_errors(
    predictions: PredictionSet,
    expected: Mapping[str, Any],
) -> list[str]:
    if predictions.manifest is None:
        return ["prediction manifest is required for an official score"]
    try:
        validate_run_manifest(predictions.manifest)
    except EvaluationError as exc:
        return [str(exc)]
    errors = []
    legacy = predictions.manifest.get("schema_version") == LEGACY_RUN_SCHEMA_VERSION
    common = {
        "protocol",
        "split",
        "task_families",
        "task_count",
        "metric_recipe_version",
        "partition_role",
        "partition_image_count",
    }
    for field, value in expected.items():
        if legacy and field not in common:
            continue
        if predictions.manifest.get(field) != value:
            errors.append(
                f"prediction manifest {field} differs: "
                f"expected {value!r}, got {predictions.manifest.get(field)!r}"
            )
    return errors


def _categories(dataset: Any) -> set[int]:
    raw = dataset.manifest.get("categories")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ReferenceValidationError("dataset manifest categories must be an array")
    result: set[int] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ReferenceValidationError(f"categories[{index}] must be an object")
        identifier = row.get("id")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier <= 0
        ):
            raise ReferenceValidationError(f"categories[{index}] has invalid id")
        if identifier in result:
            raise ReferenceValidationError(f"duplicate category id {identifier}")
        result.add(identifier)
    return result


def _category_names(dataset: Any) -> set[str]:
    _categories(dataset)
    names = [row.get("name") for row in dataset.manifest["categories"]]
    if any(not isinstance(name, str) or not name for name in names):
        raise ReferenceValidationError(
            "dataset categories require non-empty string names"
        )
    if len(names) != len(set(names)):
        raise ReferenceValidationError("dataset category names must be unique")
    return set(names)


def _reference_panoptic(
    dataset: Any,
    split: str,
    task: Mapping[str, Any],
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    image = dataset.image(split, task["image_id"])
    height, width = int(image["height"]), int(image["width"])
    try:
        resolver = getattr(dataset, "resolve_task_panoptic", None)
        if callable(resolver):
            path, metadata = resolver(split, task)
        elif task["family"] == "panoptic_segmentation":
            path, metadata = dataset.resolve_panoptic(split, task["image_id"])
        else:
            raise ReferenceValidationError("dataset lacks resolve_task_panoptic")
        if not isinstance(metadata, Mapping):
            raise ReferenceValidationError(
                "resolve_panoptic metadata must be an object"
            )
        id_map = panoptic_ids(Path(path), height, width, "reference")
        if task["family"] == "referring_panoptic_segmentation":
            id_map = np.where(np.isin(id_map, task["target_segment_ids"]), id_map, 0)
        segments = checked_segments(
            id_map,
            metadata.get("segments_info"),
            _categories(dataset),
            "reference",
        )
        if task["family"] == "referring_panoptic_segmentation" and (
            set(segments) != set(task["target_segment_ids"])
            or any(
                row["category_id"] != task["category_id"] for row in segments.values()
            )
        ):
            raise ReferenceValidationError(
                "referring-panoptic segments disagree with query category/target IDs"
            )
    except ReferenceValidationError:
        raise
    except (EvaluationError, KeyError, TypeError, ValueError, OSError) as exc:
        raise ReferenceValidationError(
            f"panoptic reference for task {task['id']!r} failed native raster closure: {exc}"
        ) from exc
    return id_map, segments


def validate_evaluation(
    dataset: Any,
    *,
    split: str,
    tasks: str | Sequence[str] | None = "all",
    predictions: PredictionSet | None = None,
    protocol: str = DATASET_PROTOCOL,
    full: bool = False,
    require_manifest: bool = True,
    partition: Any | None = None,
    partition_role: str = "validation",
) -> dict[str, Any]:
    """Validate references, capabilities, and optional saved predictions."""

    families = (
        parse_task_selection(tasks)
        if isinstance(tasks, str) or tasks is None
        else parse_task_selection(",".join(tasks))
    )
    capabilities = protocol_capabilities(protocol)
    errors: list[str] = []
    warnings: list[str] = []
    if (
        predictions is not None
        and predictions.manifest is not None
        and predictions.manifest.get("schema_version") == LEGACY_RUN_SCHEMA_VERSION
    ):
        warnings.append(
            "Legacy run metadata is readable, but its recorded hashes are not reverified against current code or inputs."
        )
    if not capabilities["supported"]:
        errors.append(
            "paper-v2 is not executable from the falcon-x dataset; see capabilities.missing"
        )
    if getattr(dataset, "format", None) != DATASET_FORMAT:
        errors.append(
            f"task evaluation requires {DATASET_FORMAT!r}, got "
            f"{getattr(dataset, 'format', None)!r}"
        )
    partition_image_ids: frozenset[Any] | None = None
    partition_fields: dict[str, Any] | None = None
    try:
        partition_image_ids, partition_fields = _partition_contract(
            partition, partition_role, split
        )
        selected = select_dataset_tasks(
            dataset, split, families, image_ids=partition_image_ids
        )
    except (EvaluationError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        selected = []
        partition_fields = None

    if selected:
        counts = {family: 0 for family in families}
        for task in selected:
            counts[str(task["family"])] += 1
        if any(family in TEXT_METRIC_FAMILIES for family in families):
            status = caption_backend_status()
            if not status["available"]:
                errors.append(
                    f"caption/VQA text metrics unavailable: {status['reason']}; "
                    f"install {status['requirement']}"
                )
        if any(
            TASK_REGISTRY[family].prediction_kind == "panoptic" for family in families
        ):
            for task in selected:
                if TASK_REGISTRY[str(task["family"])].prediction_kind != "panoptic":
                    continue
                try:
                    _reference_panoptic(dataset, split, task)
                except ReferenceValidationError as exc:
                    errors.append(str(exc))
                    if not full:
                        break
        if predictions is not None:
            try:
                _prediction_rows(predictions, split, selected)
            except EvaluationError as exc:
                errors.append(str(exc))
            # A diagnostic JSONL may omit a manifest.  If one is present, it is
            # never ignored merely because absence was permitted.
            if require_manifest or predictions.manifest is not None:
                expected = expected_run_manifest(
                    dataset,
                    split,
                    selected,
                    families,
                    protocol,
                    partition=partition,
                    partition_role=partition_role,
                )
                errors.extend(_manifest_errors(predictions, expected))
    else:
        counts = {family: 0 for family in families}

    # A dataset reader may expose additional structural diagnostics.  They are
    # retained without assuming a particular report dataclass.
    reader_report: Any = None
    try:
        validation_kwargs: dict[str, Any] = {"tasks": families, "full": full}
        if full and partition_image_ids is not None:
            validation_kwargs["image_ids"] = partition_image_ids
        reader_report = dataset.validate(split, **validation_kwargs)
    except Exception as exc:  # dataset contract errors belong in this report
        errors.append(f"dataset validation failed: {exc}")
    if isinstance(reader_report, Mapping):
        reader_errors = reader_report.get("errors", ())
        if isinstance(reader_errors, Sequence) and not isinstance(
            reader_errors, str | bytes
        ):
            errors.extend(str(value) for value in reader_errors)
        reader_warnings = reader_report.get("warnings", ())
        if isinstance(reader_warnings, Sequence) and not isinstance(
            reader_warnings, str | bytes
        ):
            warnings.extend(str(value) for value in reader_warnings)

    recipe = metric_recipe_metadata(families)
    return {
        "schema_version": "falcon-task-validation-v1",
        "valid": not errors,
        "protocol": protocol,
        "split": split,
        "families": list(families),
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
        "capabilities": capabilities,
        "caption_backend": caption_backend_status(),
        "metric_recipe_version": recipe["version"],
        "metric_recipe": recipe,
        "partition": (
            None
            if partition_fields is None
            else {
                **partition_fields,
                "report": dict(getattr(partition, "report", {})),
            }
        ),
    }


def _prediction_answer(row: Mapping[str, Any]) -> str:
    if row["status"] != "ok":
        raise ModelOutputError(str(row.get("error", "model execution failed")))
    answer = row.get("answer")
    if not isinstance(answer, str):
        raise ModelOutputError("prediction requires a string answer")
    return answer


def _rows_by_image_scope(
    dataset: Any,
    split: str,
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(
        list
    )
    for task, prediction in rows:
        image = dataset.image(split, task["image_id"])
        scope = "counterfactual" if "source_image_id" in image else "source"
        grouped[scope].append((task, prediction))
    return dict(grouped)


def _score_vqa(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    # A valid raw answer need not match our narrow geometry-label vocabulary.
    # Score every raw answer with the text backend, independently of that parser.
    text_metrics = _score_caption_family(rows)
    correct = label_unparsed = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    for task, prediction in rows:
        question_type = task.get("question_type")
        if question_type not in VQA_QUESTION_TYPES:
            raise ReferenceValidationError(
                f"VQA task {task['id']!r} has unsupported question_type {question_type!r}"
            )
        expected = parse_vqa_label(str(question_type), task["answer"])
        if expected is None:
            raise ReferenceValidationError(
                f"VQA reference {task['id']!r} cannot be parsed by dataset-v1"
            )
        try:
            answer = _prediction_answer(prediction)
        except ModelOutputError:
            hit = False
        else:
            inferred = parse_vqa_label(str(question_type), answer)
            label_unparsed += inferred is None
            hit = inferred == expected
        correct += hit
        by_type[str(question_type)].append(hit)
    type_metrics = {
        name: {"count": len(values), "accuracy": sum(values) / len(values)}
        for name, values in by_type.items()
    }
    return {
        **text_metrics,
        "label_unparsed": label_unparsed,
        "label_model_invalid": text_metrics["model_invalid"] + label_unparsed,
        "label_invalid_policy": "failed or unparsed labels receive zero label credit; no rows dropped",
        "label_accuracy": correct / len(rows),
        "question_types": type_metrics,
        "question_type_macro_accuracy": sum(
            value["accuracy"] for value in type_metrics.values()
        )
        / len(type_metrics),
    }


def _score_presence(
    dataset: Any,
    split: str,
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    labels: list[bool] = []
    inferred: list[bool] = []
    invalid = 0
    category_hits: dict[str, list[bool]] = defaultdict(list)
    scope_values: dict[str, tuple[list[bool], list[bool]]] = {}
    for task, prediction in rows:
        expected = parse_yes_no(task["answer"])
        if expected is None:
            raise ReferenceValidationError(
                f"presence reference {task['id']!r} is not canonical yes/no"
            )
        try:
            answer = _prediction_answer(prediction)
            output = parse_yes_no(answer)
            if output is None:
                raise ModelOutputError("answer is not a registered yes/no form")
        except ModelOutputError:
            invalid += 1
            output = (
                not expected
            )  # deterministic wrong label; never reward an invalid negative
        labels.append(expected)
        inferred.append(output)
        category_hits[str(task.get("category_id"))].append(output == expected)
        image = dataset.image(split, task["image_id"])
        scope = "counterfactual" if "source_image_id" in image else "source"
        scope_labels, scope_predictions = scope_values.setdefault(scope, ([], []))
        scope_labels.append(expected)
        scope_predictions.append(output)
    result: dict[str, Any] = {
        "count": len(rows),
        "model_invalid": invalid,
        **binary_classification(labels, inferred),
    }
    result["categories"] = {
        key: {"count": len(values), "accuracy": sum(values) / len(values)}
        for key, values in category_hits.items()
    }
    result["category_macro_accuracy"] = sum(
        value["accuracy"] for value in result["categories"].values()
    ) / len(result["categories"])
    result["image_scope"] = {
        scope: {
            "count": len(scope_labels),
            **binary_classification(scope_labels, scope_predictions),
        }
        for scope, (scope_labels, scope_predictions) in scope_values.items()
    }
    return result


def _score_missing_components(
    dataset: Any,
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    categories = _category_names(dataset)
    counts = {category: [0, 0, 0] for category in sorted(categories)}
    correct = invalid = empty_count = empty_correct = 0
    example_f1 = 0.0
    for task, prediction in rows:
        expected = parse_category_set(task["answer"], categories)
        if expected is None:
            raise ReferenceValidationError(
                f"invalid missing-component reference {task['id']!r}"
            )
        valid = True
        try:
            inferred = parse_category_set(_prediction_answer(prediction), categories)
            if inferred is None:
                raise ModelOutputError(
                    "answer must be a JSON array of unique category names"
                )
        except ModelOutputError:
            invalid += 1
            valid = False
            inferred = frozenset(categories.difference(expected))
        hit = valid and inferred == expected
        correct += hit
        if not expected:
            empty_count += 1
            empty_correct += hit
        denominator = len(inferred) + len(expected)
        example_f1 += (
            0.0
            if not valid
            else (
                1.0 if denominator == 0 else 2 * len(inferred & expected) / denominator
            )
        )
        for category, values in counts.items():
            reference = category in expected
            prediction_positive = category in inferred
            values[0] += reference and prediction_positive
            values[1] += not reference and prediction_positive
            values[2] += reference and not prediction_positive

    def f1(values: Sequence[int]) -> float:
        tp, fp, fn = values
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    totals = [sum(values[index] for values in counts.values()) for index in range(3)]
    result: dict[str, Any] = {
        "count": len(rows),
        "model_invalid": invalid,
        "accuracy": correct / len(rows),
        "micro_f1": f1(totals),
        "macro_f1": sum(f1(values) for values in counts.values()) / len(counts),
        "example_f1": example_f1 / len(rows),
        "empty_target_count": empty_count,
        "categories": {
            category: {
                "tp": values[0],
                "fp": values[1],
                "fn": values[2],
                "f1": f1(values),
            }
            for category, values in counts.items()
        },
        "conventions": {
            "invalid": "all category decisions count as wrong; exact/example scores are zero",
            "zero_support_category_f1": 0.0,
            "empty_set_exact_and_example_f1": 1.0,
        },
    }
    if empty_count:
        result["empty_set_accuracy"] = empty_correct / empty_count
    return result


def _score_completeness(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    absolute_error = squared_error = 0.0
    correct = invalid = 0
    class_totals = {
        label: {
            "count": 0,
            "model_invalid": 0,
            "absolute_error": 0.0,
            "squared_error": 0.0,
            "correct": 0,
        }
        for label in (0, 1)
    }
    for task, prediction in rows:
        expected = parse_unit_interval(task["answer"])
        if expected not in (0.0, 1.0):
            raise ReferenceValidationError(
                f"invalid completeness reference {task['id']!r}"
            )
        prediction_valid = True
        try:
            inferred = parse_unit_interval(_prediction_answer(prediction))
            if inferred is None:
                raise ModelOutputError("answer must be a finite number in [0, 1]")
        except ModelOutputError:
            invalid += 1
            prediction_valid = False
            inferred = 1.0 - expected
        error = abs(inferred - expected)
        absolute_error += error
        squared_error += error * error
        hit = (inferred >= 0.5) == bool(expected)
        correct += hit
        totals = class_totals[int(expected)]
        totals["count"] += 1
        totals["model_invalid"] += not prediction_valid
        totals["absolute_error"] += error
        totals["squared_error"] += error * error
        totals["correct"] += hit
    by_target_class = {
        str(label): {
            "count": values["count"],
            "model_invalid": values["model_invalid"],
            "mae": values["absolute_error"] / values["count"]
            if values["count"]
            else None,
            "rmse": math.sqrt(values["squared_error"] / values["count"])
            if values["count"]
            else None,
            "accuracy": values["correct"] / values["count"]
            if values["count"]
            else None,
        }
        for label, values in class_totals.items()
    }
    both_classes = all(values["count"] for values in class_totals.values())
    balanced = {
        "status": "available"
        if both_classes
        else "unavailable: both target classes required",
        "mae": None,
        "rmse": None,
        "accuracy": None,
        "definition": "equal class weights; RMSE is sqrt(mean class MSE), not mean class RMSE",
    }
    if both_classes:
        balanced.update(
            mae=sum(row["mae"] for row in by_target_class.values()) / 2,
            rmse=math.sqrt(
                sum(
                    row["squared_error"] / row["count"] for row in class_totals.values()
                )
                / 2
            ),
            accuracy=sum(row["accuracy"] for row in by_target_class.values()) / 2,
        )
    return {
        "count": len(rows),
        "model_invalid": invalid,
        "mae": absolute_error / len(rows),
        "rmse": math.sqrt(squared_error / len(rows)),
        "accuracy": correct / len(rows),
        "threshold": 0.5,
        "invalid_policy": "retained with maximum absolute/squared error 1 and wrong class",
        "target_classes": by_target_class,
        "class_balanced": balanced,
    }


def _presence_pair_diagnostics(
    dataset: Any,
    split: str,
    tasks: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Score only source/CF presence pairs that ask about the same category."""

    relevant = [task for task in tasks if task["family"] == "category_presence"]
    if not relevant:
        return None
    indexed: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    counterfactuals = []
    for task in relevant:
        indexed[(task["image_id"], task["category_id"])].append(task)
        image = dataset.image(split, task["image_id"])
        if "source_image_id" in image:
            counterfactuals.append((task, image["source_image_id"]))

    eligible = missing_source = ambiguous = invalid_pairs = joint = 0
    unchanged = changed = correct_invariance = correct_transition = 0
    endpoint_hits: dict[str, bool] = {}
    for cf_task, source_id in counterfactuals:
        candidates = indexed.get((source_id, cf_task["category_id"]), ())
        if not candidates:
            missing_source += 1
            continue
        if len(candidates) != 1:
            ambiguous += 1
            continue
        source_task = candidates[0]
        eligible += 1
        pair_hits = []
        pair_outputs = []
        pair_expected = []
        pair_valid = True
        for task in (source_task, cf_task):
            expected = parse_yes_no(task["answer"])
            if expected is None:
                raise ReferenceValidationError(
                    f"presence reference {task['id']!r} is not canonical yes/no"
                )
            try:
                output = parse_yes_no(_prediction_answer(predictions[str(task["id"])]))
                if output is None:
                    raise ModelOutputError("answer is not a registered yes/no form")
                hit = output == expected
            except ModelOutputError:
                pair_valid = False
                output = None
                hit = False
            endpoint_hits[str(task["id"])] = hit
            pair_hits.append(hit)
            pair_outputs.append(output)
            pair_expected.append(expected)
        both_correct = all(pair_hits)
        joint += both_correct
        invalid_pairs += not pair_valid
        if pair_expected[0] == pair_expected[1]:
            unchanged += 1
            correct_invariance += (
                both_correct
                and pair_outputs[0] is not None
                and pair_outputs[0] == pair_outputs[1]
            )
        else:
            changed += 1
            correct_transition += (
                both_correct
                and pair_outputs[0] is not None
                and pair_outputs[0] != pair_outputs[1]
            )

    result: dict[str, Any] = {
        "cf_query_count": len(counterfactuals),
        "eligible_same_category_pairs": eligible,
        "missing_source_query": missing_source,
        "ambiguous_query": ambiguous,
        "invalid_pairs": invalid_pairs,
        "unique_endpoint_count": len(endpoint_hits),
        "unique_endpoint_accuracy": (
            sum(endpoint_hits.values()) / len(endpoint_hits) if endpoint_hits else None
        ),
        "joint_endpoint_accuracy": joint / eligible if eligible else None,
        "unchanged_label_pairs": unchanged,
        "changed_label_pairs": changed,
    }
    if unchanged:
        result["correct_invariance_rate"] = correct_invariance / unchanged
    if changed:
        result["correct_transition_rate"] = correct_transition / changed
    return result


class _MaskAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.invalid = 0
        self.all_iou: list[float] = []
        self.positive_iou: list[float] = []
        self.positive_intersection = 0
        self.positive_union = 0
        self.negative_count = 0
        self.negative_empty = 0
        self.all_intersection = 0
        self.all_union = 0

    def add(self, target: np.ndarray, prediction: np.ndarray | None) -> None:
        self.count += 1
        positive = bool(np.any(target))
        if prediction is None:
            self.invalid += 1
            self.all_iou.append(0.0)
            if positive:
                area = int(np.count_nonzero(target))
                self.positive_iou.append(0.0)
                self.positive_union += area
                self.all_union += area
            else:
                self.negative_count += 1
            return
        intersection, union = mask_intersection_union(prediction, target)
        iou = 1.0 if union == 0 else intersection / union
        self.all_iou.append(iou)
        self.all_intersection += intersection
        self.all_union += union
        if positive:
            self.positive_iou.append(iou)
            self.positive_intersection += intersection
            self.positive_union += union
        else:
            self.negative_count += 1
            self.negative_empty += not bool(np.any(prediction))

    def report(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "count": self.count,
            "model_invalid": self.invalid,
            "all_query_miou": sum(self.all_iou) / self.count,
            "positive_count": len(self.positive_iou),
            "negative_count": self.negative_count,
            **{
                f"all_query_{key}": value
                for key, value in iou_threshold_accuracies(self.all_iou).items()
            },
        }
        if self.positive_iou:
            result["positive_miou"] = sum(self.positive_iou) / len(self.positive_iou)
            result["positive_ciou"] = (
                self.positive_intersection / self.positive_union
                if self.positive_union
                else 0.0
            )
            result.update(
                {
                    f"positive_{key}": value
                    for key, value in iou_threshold_accuracies(
                        self.positive_iou
                    ).items()
                }
            )
        if self.negative_count:
            result["negative_empty_accuracy"] = (
                self.negative_empty / self.negative_count
            )
        if self.invalid == 0 and self.all_union:
            result["all_query_ciou"] = self.all_intersection / self.all_union
        elif self.invalid:
            result["ciou_status"] = (
                "withheld: model-invalid rows lack a canonical pixel weight"
            )
        return result


def _score_binary_masks(
    dataset: Any,
    split: str,
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    overall = _MaskAccumulator()
    strata: dict[str, _MaskAccumulator] = {}
    image_scopes: dict[str, _MaskAccumulator] = {}
    categories: dict[str, _MaskAccumulator] = {}
    family = str(rows[0][0]["family"])
    if family == "category_or_all_instance_grounding":
        strata = {"category": _MaskAccumulator(), "all_objects": _MaskAccumulator()}
    for task, prediction_row in rows:
        image = dataset.image(split, task["image_id"])
        height, width = int(image["height"]), int(image["width"])
        try:
            target_objects = dataset.target_objects(split, task)
            target = union_object_masks(target_objects, height, width)
        except Exception as exc:
            raise ReferenceValidationError(
                f"cannot resolve mask target for task {task['id']!r}: {exc}"
            ) from exc
        prediction: np.ndarray | None
        try:
            if prediction_row["status"] != "ok":
                raise ModelOutputError(
                    str(prediction_row.get("error", "model execution failed"))
                )
            if "regions" not in prediction_row:
                raise ModelOutputError("binary-mask prediction lacks regions")
            prediction = union_prediction_regions(
                prediction_row["regions"], height, width
            )
        except ModelOutputError:
            prediction = None
        overall.add(target, prediction)
        scope = "counterfactual" if "source_image_id" in image else "source"
        image_scopes.setdefault(scope, _MaskAccumulator()).add(target, prediction)
        target_categories = {str(obj["category_id"]) for obj in target_objects}
        if len(target_categories) == 1:
            category = next(iter(target_categories))
            categories.setdefault(category, _MaskAccumulator()).add(target, prediction)
        if strata:
            key = "all_objects" if task.get("category_id") is None else "category"
            strata[key].add(target, prediction)
    result = overall.report()
    if strata:
        result["query_strata"] = {
            key: value.report() for key, value in strata.items() if value.count
        }
    result["image_scope"] = {key: value.report() for key, value in image_scopes.items()}
    if categories:
        result["target_categories"] = {
            key: value.report() for key, value in sorted(categories.items())
        }
    return result


def _mask_pair_diagnostics(
    dataset: Any,
    split: str,
    tasks: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Evaluate matched source/CF grounding queries without inventing missing tasks."""

    relevant = [
        task for task in tasks if task["family"] == "category_or_all_instance_grounding"
    ]
    if not relevant:
        return None
    by_image: dict[Any, dict[Any, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cf_parent: dict[Any, Any] = {}
    for task in relevant:
        by_image[task["image_id"]][task.get("category_id")].append(task)
        image = dataset.image(split, task["image_id"])
        if "source_image_id" in image:
            cf_parent[task["image_id"]] = image["source_image_id"]

    endpoint_cache: dict[str, tuple[np.ndarray, np.ndarray | None, float, bool]] = {}

    def endpoint(
        task: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray | None, float, bool]:
        identifier = str(task["id"])
        if identifier in endpoint_cache:
            return endpoint_cache[identifier]
        image = dataset.image(split, task["image_id"])
        height, width = int(image["height"]), int(image["width"])
        try:
            target = union_object_masks(
                dataset.target_objects(split, task), height, width
            )
        except Exception as exc:
            raise ReferenceValidationError(
                f"cannot resolve paired mask target for task {identifier!r}: {exc}"
            ) from exc
        row = predictions[identifier]
        try:
            if row["status"] != "ok":
                raise ModelOutputError(str(row.get("error", "model execution failed")))
            prediction = union_prediction_regions(row.get("regions"), height, width)
            intersection, union = mask_intersection_union(prediction, target)
            iou = 1.0 if union == 0 else intersection / union
            exact = bool(np.array_equal(prediction, target))
        except ModelOutputError:
            prediction = None
            iou = 0.0
            exact = False
        endpoint_cache[identifier] = target, prediction, iou, exact
        return endpoint_cache[identifier]

    eligible_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    missing_cf = missing_source = ambiguous = 0
    for cf_image_id, source_image_id in cf_parent.items():
        source_queries = by_image.get(source_image_id, {})
        cf_queries = by_image[cf_image_id]
        for category_id, source_rows in source_queries.items():
            cf_rows = cf_queries.get(category_id, ())
            if not cf_rows:
                missing_cf += 1
            elif len(source_rows) != 1 or len(cf_rows) != 1:
                ambiguous += 1
            else:
                eligible_pairs.append((source_rows[0], cf_rows[0]))
        for category_id in cf_queries:
            if category_id not in source_queries:
                missing_source += 1

    endpoint_iou: dict[str, float] = {}
    endpoint_exact: dict[str, bool] = {}
    endpoint_scope: dict[str, str] = {}
    invalid_pairs = joint_exact = joint_at_05 = non_subset = 0
    removed_ious: list[float] = []
    retained_ious: list[float] = []
    removed_intersection = removed_union = 0
    retained_intersection = retained_union = 0
    removed_support = retained_support = 0
    correct_composition = invariant_pairs = correct_invariance = 0
    pair_strata: dict[str, list[bool]] = defaultdict(list)
    for source_task, cf_task in eligible_pairs:
        source_target, source_prediction, source_iou, source_exact = endpoint(
            source_task
        )
        cf_target, cf_prediction, cf_iou, cf_exact = endpoint(cf_task)
        if source_target.shape != cf_target.shape:
            raise ReferenceValidationError(
                f"paired mask tasks {source_task['id']!r}/{cf_task['id']!r} differ in size"
            )
        for task, iou, exact, scope in (
            (source_task, source_iou, source_exact, "source"),
            (cf_task, cf_iou, cf_exact, "counterfactual"),
        ):
            identifier = str(task["id"])
            endpoint_iou[identifier] = iou
            endpoint_exact[identifier] = exact
            endpoint_scope[identifier] = scope
        pair_valid = source_prediction is not None and cf_prediction is not None
        invalid_pairs += not pair_valid
        both_exact = source_exact and cf_exact
        joint_exact += both_exact
        joint_at_05 += pair_valid and source_iou >= 0.5 and cf_iou >= 0.5
        stratum = (
            "all_objects" if source_task.get("category_id") is None else "category"
        )
        pair_strata[stratum].append(both_exact)

        added_reference = cf_target & ~source_target
        if np.any(added_reference):
            non_subset += 1
            continue
        reference_removed = source_target & ~cf_target
        reference_retained = source_target & cf_target
        removed_support += int(np.count_nonzero(reference_removed))
        retained_support += int(np.count_nonzero(reference_retained))
        if not np.any(reference_removed):
            invariant_pairs += 1
        if not pair_valid:
            removed_ious.append(0.0)
            retained_ious.append(0.0)
            continue
        assert source_prediction is not None and cf_prediction is not None
        predicted_removed = source_prediction & ~cf_prediction
        predicted_retained = source_prediction & cf_prediction
        ri, ru = mask_intersection_union(predicted_removed, reference_removed)
        ti, tu = mask_intersection_union(predicted_retained, reference_retained)
        removed_intersection += ri
        removed_union += ru
        retained_intersection += ti
        retained_union += tu
        removed_ious.append(1.0 if ru == 0 else ri / ru)
        retained_ious.append(1.0 if tu == 0 else ti / tu)
        composition_exact = bool(
            np.array_equal(predicted_removed, reference_removed)
            and np.array_equal(predicted_retained, reference_retained)
        )
        correct_composition += both_exact and composition_exact
        if not np.any(reference_removed):
            correct_invariance += both_exact and bool(
                np.array_equal(source_prediction, cf_prediction)
            )

    def endpoint_summary(scope: str) -> dict[str, Any]:
        identifiers = [key for key, value in endpoint_scope.items() if value == scope]
        return {
            "count": len(identifiers),
            "miou": (
                sum(endpoint_iou[key] for key in identifiers) / len(identifiers)
                if identifiers
                else None
            ),
            "exact_match": (
                sum(endpoint_exact[key] for key in identifiers) / len(identifiers)
                if identifiers
                else None
            ),
        }

    pair_count = len(eligible_pairs)
    composition_count = pair_count - non_subset
    result: dict[str, Any] = {
        "counterfactual_image_count": len(cf_parent),
        "eligible_same_query_pairs": pair_count,
        "missing_counterfactual_query": missing_cf,
        "missing_source_query": missing_source,
        "ambiguous_query": ambiguous,
        "invalid_pairs": invalid_pairs,
        "unique_endpoints": {
            "source": endpoint_summary("source"),
            "counterfactual": endpoint_summary("counterfactual"),
        },
        "joint_endpoint_exact_match": joint_exact / pair_count if pair_count else None,
        "joint_endpoint_iou_at_0_5": joint_at_05 / pair_count if pair_count else None,
        "pair_strata": {
            key: {
                "count": len(values),
                "joint_endpoint_exact_match": sum(values) / len(values),
            }
            for key, values in pair_strata.items()
        },
        "composition_eligible_pairs": composition_count,
        "non_subset_ground_truth_pairs": non_subset,
        "removed_gt_union_difference_pixels": removed_support,
        "retained_gt_intersection_pixels": retained_support,
        "removed_delta_miou": (
            sum(removed_ious) / len(removed_ious) if removed_ious else None
        ),
        "retained_delta_miou": (
            sum(retained_ious) / len(retained_ious) if retained_ious else None
        ),
        "correct_composition_rate": (
            correct_composition / composition_count if composition_count else None
        ),
        "invariant_pairs": invariant_pairs,
        "correct_invariance_rate": (
            correct_invariance / invariant_pairs if invariant_pairs else None
        ),
    }
    if invalid_pairs == 0:
        result["removed_delta_ciou"] = (
            1.0 if removed_union == 0 else removed_intersection / removed_union
        )
        result["retained_delta_ciou"] = (
            1.0 if retained_union == 0 else retained_intersection / retained_union
        )
    else:
        result["ciou_status"] = "withheld: model-invalid pair endpoints"
    return result


def _pq_counts(
    reference_ids: np.ndarray,
    reference_segments: Mapping[int, Mapping[str, int]],
    predicted_ids: np.ndarray,
    predicted_segments: Mapping[int, Mapping[str, int]],
) -> tuple[int, int, int, float]:
    candidates = []
    for reference_id, reference in reference_segments.items():
        ref_mask = reference_ids == reference_id
        for predicted_id, predicted in predicted_segments.items():
            if predicted["category_id"] != reference["category_id"]:
                continue
            intersection = int(
                np.count_nonzero(ref_mask & (predicted_ids == predicted_id))
            )
            if not intersection:
                continue
            union = int(np.count_nonzero(ref_mask | (predicted_ids == predicted_id)))
            iou = intersection / union
            if iou > 0.5:
                candidates.append((-iou, reference_id, predicted_id))
    matched_reference: set[int] = set()
    matched_prediction: set[int] = set()
    iou_sum = 0.0
    for negative_iou, reference_id, predicted_id in sorted(candidates):
        if reference_id in matched_reference or predicted_id in matched_prediction:
            continue
        matched_reference.add(reference_id)
        matched_prediction.add(predicted_id)
        iou_sum -= negative_iou
    tp = len(matched_reference)
    return tp, len(predicted_segments) - tp, len(reference_segments) - tp, iou_sum


def _score_panoptic(
    dataset: Any,
    split: str,
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    artifact_root: Path,
) -> dict[str, Any]:
    category_ids = _categories(dataset)
    intersections = unions = 0
    per_image: list[float] = []
    background: list[float] = []
    pq_tp = pq_fp = pq_fn = 0
    pq_iou = 0.0
    invalid = 0
    for task, prediction_row in rows:
        reference_ids, reference_segments = _reference_panoptic(dataset, split, task)
        reference_semantic = semantic_map(reference_ids, reference_segments)
        image = dataset.image(split, task["image_id"])
        height, width = int(image["height"]), int(image["width"])
        prediction_valid = True
        try:
            if prediction_row["status"] != "ok":
                raise ModelOutputError(
                    str(prediction_row.get("error", "model execution failed"))
                )
            raw = prediction_row.get("panoptic")
            if not isinstance(raw, Mapping):
                raise ModelOutputError("panoptic prediction must be an object")
            path = safe_artifact_path(artifact_root, raw.get("file_name"), "prediction")
            predicted_ids = panoptic_ids(path, height, width, "prediction")
            predicted_segments = checked_segments(
                predicted_ids, raw.get("segments_info"), category_ids, "prediction"
            )
            predicted_semantic = semantic_map(predicted_ids, predicted_segments)
        except (ModelOutputError, OSError, TypeError, ValueError):
            invalid += 1
            prediction_valid = False
            predicted_ids = np.zeros((height, width), dtype=np.int64)
            predicted_segments = {}
            predicted_semantic = np.zeros((height, width), dtype=np.int64)

        image_ious = []
        for category_id in sorted(category_ids):
            intersection, union = mask_intersection_union(
                predicted_semantic == category_id, reference_semantic == category_id
            )
            if union:
                intersections += intersection
                unions += union
                image_ious.append(intersection / union)
        per_image.append(
            0.0
            if not prediction_valid
            else (
                sum(image_ious) / len(image_ious)
                if image_ious
                else float(task["family"] == "referring_panoptic_segmentation")
            )
        )
        bg_intersection, bg_union = mask_intersection_union(
            predicted_semantic == 0, reference_semantic == 0
        )
        background.append(
            0.0
            if not prediction_valid
            else (1.0 if bg_union == 0 else bg_intersection / bg_union)
        )
        tp, fp, fn, iou_sum = _pq_counts(
            reference_ids, reference_segments, predicted_ids, predicted_segments
        )
        pq_tp += tp
        pq_fp += fp
        pq_fn += fn
        pq_iou += iou_sum
    denominator = pq_tp + 0.5 * pq_fp + 0.5 * pq_fn
    result: dict[str, Any] = {
        "count": len(rows),
        "model_invalid": invalid,
        "foreground_miou": sum(per_image) / len(per_image),
        "foreground_image_class_miou": sum(per_image) / len(per_image),
        "background_miou": sum(background) / len(background),
        "matched_segments": pq_tp,
        "false_positive_segments": pq_fp,
        "false_negative_segments": pq_fn,
    }
    if invalid:
        result["pixel_micro_and_pq_status"] = (
            "withheld: model-invalid rows make cumulative/instance metrics non-comparable"
        )
    else:
        if unions:
            result["foreground_ciou"] = intersections / unions
            result["foreground_class_pixel_ciou"] = result["foreground_ciou"]
        result.update(
            {
                "pq": 1.0 if denominator == 0 else pq_iou / denominator,
                "sq": (
                    1.0
                    if pq_tp == 0 and denominator == 0
                    else (0.0 if pq_tp == 0 else pq_iou / pq_tp)
                ),
                "rq": 1.0 if denominator == 0 else pq_tp / denominator,
            }
        )
        result.update({f"pooled_{name}": result[name] for name in ("pq", "sq", "rq")})
    return result


def _score_caption_family(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    references: dict[str, str] = {}
    outputs: dict[str, str] = {}
    exact = invalid = 0
    for task, prediction in rows:
        identifier = str(task["id"])
        references[identifier] = task["answer"]
        prediction_valid = True
        try:
            answer = _prediction_answer(prediction)
        except ModelOutputError:
            invalid += 1
            prediction_valid = False
            answer = ""
        outputs[identifier] = answer
        exact += prediction_valid and normalize_text(answer) == normalize_text(
            task["answer"]
        )
    return {
        "count": len(rows),
        "model_invalid": invalid,
        "metric_backend": TEXT_METRIC_BACKEND,
        "normalized_exact_match": exact / len(rows),
        **score_captions(references, outputs),
    }


def score_dataset(
    dataset: Any,
    predictions: PredictionSet,
    *,
    split: str,
    tasks: str | Sequence[str] | None = "all",
    protocol: str = DATASET_PROTOCOL,
    require_manifest: bool = True,
    partition: Any | None = None,
    partition_role: str = "validation",
) -> dict[str, Any]:
    """Score one exact saved-prediction set without loading a model."""

    families = (
        parse_task_selection(tasks)
        if isinstance(tasks, str) or tasks is None
        else parse_task_selection(",".join(tasks))
    )
    # Preserve the public distinction between a malformed task set and a
    # model-invalid row.  Coverage errors are raised with their specific type
    # before the aggregate validation report is assembled.
    partition_image_ids, partition_fields = _partition_contract(
        partition, partition_role, split
    )
    require_final_stage3_provenance(
        predictions.manifest.get("provenance") if predictions.manifest else None
    )
    selected = select_dataset_tasks(
        dataset, split, families, image_ids=partition_image_ids
    )
    _prediction_rows(predictions, split, selected)
    validation = validate_evaluation(
        dataset,
        split=split,
        tasks=families,
        predictions=predictions,
        protocol=protocol,
        full=False,
        require_manifest=require_manifest,
        partition=partition,
        partition_role=partition_role,
    )
    if not validation["valid"]:
        raise EvaluationError(
            "evaluation validation failed: " + "; ".join(validation["errors"])
        )
    predicted = _prediction_rows(predictions, split, selected)
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(
        list
    )
    for task in selected:
        grouped[str(task["family"])].append((task, predicted[str(task["id"])]))

    family_metrics: dict[str, Any] = {}
    for family in families:
        rows = grouped[family]
        if family in CAPTION_FAMILIES:
            family_metrics[family] = _score_caption_family(rows)
        elif family == "vqa":
            family_metrics[family] = _score_vqa(rows)
        elif family == "category_presence":
            family_metrics[family] = _score_presence(dataset, split, rows)
        elif family == "missing_component_identification":
            family_metrics[family] = _score_missing_components(dataset, rows)
            family_metrics[family]["image_scope"] = {
                scope: _score_missing_components(dataset, scope_rows)
                for scope, scope_rows in _rows_by_image_scope(
                    dataset, split, rows
                ).items()
            }
        elif family == "functional_completeness":
            family_metrics[family] = _score_completeness(rows)
            family_metrics[family]["image_scope"] = {
                scope: _score_completeness(scope_rows)
                for scope, scope_rows in _rows_by_image_scope(
                    dataset, split, rows
                ).items()
            }
        elif family in BINARY_MASK_FAMILIES:
            family_metrics[family] = _score_binary_masks(dataset, split, rows)
        elif TASK_REGISTRY[family].prediction_kind == "panoptic":
            family_metrics[family] = _score_panoptic(
                dataset,
                split,
                rows,
                predictions.artifact_root,
            )
        else:  # guarded by registry; defensive against future drift
            raise EvaluationError(f"no scorer is registered for {family!r}")

    model_invalid = sum(value["model_invalid"] for value in family_metrics.values())
    pair_diagnostics = {}
    presence_pairs = _presence_pair_diagnostics(dataset, split, selected, predicted)
    if presence_pairs is not None:
        pair_diagnostics["category_presence"] = presence_pairs
    mask_pairs = _mask_pair_diagnostics(dataset, split, selected, predicted)
    if mask_pairs is not None:
        pair_diagnostics["category_or_all_instance_grounding"] = mask_pairs
    checkpoint_status = checkpoint_publication_status(
        predictions.manifest.get("provenance")
        if predictions.manifest is not None
        else None
    )
    publication_reasons = list(checkpoint_status["reasons"])
    legacy_run = (
        predictions.manifest is not None
        and predictions.manifest.get("schema_version") == LEGACY_RUN_SCHEMA_VERSION
    )
    if legacy_run:
        publication_reasons.append("legacy_run_provenance_not_reverified")
    if not require_manifest:
        publication_reasons.append("run_manifest_not_required_diagnostic")
    if model_invalid:
        publication_reasons.append("model_invalid_predictions")
    publication_reasons = list(dict.fromkeys(publication_reasons))
    official = (
        require_manifest
        and model_invalid == 0
        and checkpoint_status["eligible"]
        and not legacy_run
    )
    recipe = metric_recipe_metadata(families)
    return {
        "schema_version": "falcon-task-metrics-v2",
        "protocol": protocol,
        "metric_recipe_version": recipe["version"],
        "metric_recipe": recipe,
        "primary_metrics": recipe["primary_metrics"],
        "split": split,
        "dataset": str(Path(dataset.root).expanduser().resolve()),
        "run_metadata_status": "legacy; recorded provenance not reverified"
        if legacy_run
        else "path/configuration checked; no content hashes",
        "partition": partition_fields,
        "task_count": len(selected),
        "model_invalid": model_invalid,
        "official": official,
        "publication_status": {
            "official_result_eligible": official,
            "checkpoint_official_result_eligible": checkpoint_status["eligible"],
            "reasons": publication_reasons,
        },
        "families": family_metrics,
        "counterfactual_pair_diagnostics": pair_diagnostics,
        "notes": [
            "No aggregate scalar is formed across task families.",
            "Counterfactual pair diagnostics are secondary and do not add task rows to "
            "the primary family denominators.",
            "Risk/link scores are absent without reviewed labels. Functional completeness "
            "measures required-category membership, not physical connectivity or scene risk.",
            "category_presence scores generated language answers, not the structured "
            "presence head.",
            "This evaluates only the selected existing dataset tasks; no queries are added "
            "and exact paper-protocol parity is not claimed.",
            "IoU threshold accuracies are diagnostics, not mIoU. Panoptic pooled PQ/SQ/RQ "
            "are not category-macro metrics; legacy names remain aliases.",
        ],
    }
