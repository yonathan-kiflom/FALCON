"""Task registry and evaluation contracts for the falcon-x dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from .data import CANONICAL_CATEGORY_IDS

DATASET_PROTOCOL = "dataset-v1"
PAPER_PROTOCOL = "paper-v2"
METRIC_RECIPE_VERSION = "falcon-existing-task-metrics-v2"
REQUIRED_COMPONENTS = tuple(CANONICAL_CATEGORY_IDS)

PredictionKind = Literal["text", "answer", "binary_mask", "panoptic"]

TEXT_METRICS = ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider_d")
MASK_METRICS = (
    "positive_ciou", "positive_miou", "all_query_ciou", "all_query_miou",
    "negative_empty_accuracy", "all_query_iou_accuracy_at_0_5",
    "all_query_iou_accuracy_at_0_75", "positive_iou_accuracy_at_0_5",
    "positive_iou_accuracy_at_0_75",
)
PANOPTIC_METRICS = (
    "foreground_class_pixel_ciou", "foreground_image_class_miou", "background_miou",
    "pooled_pq", "pooled_sq", "pooled_rq",
)


@dataclass(frozen=True)
class TaskSpec:
    """Evaluation contract for one falcon-x task family."""

    family: str
    prediction_kind: PredictionKind
    metrics: tuple[str, ...]
    image_scope: Literal["source", "counterfactual", "both"]
    training_target_kind: Literal["text", "answer", "object_union", "native_panoptic"]
    output_grammar: str
    description: str


_SPECS = (
    TaskSpec(
        "source_summary",
        "text",
        TEXT_METRICS,
        "source",
        "text",
        "utf8-text-v1",
        "Short source-image summary.",
    ),
    TaskSpec(
        "detailed_caption",
        "text",
        TEXT_METRICS,
        "source",
        "text",
        "utf8-text-v1",
        "Detailed source-image caption.",
    ),
    TaskSpec(
        "instance_description",
        "text",
        TEXT_METRICS,
        "source",
        "text",
        "utf8-text-v1",
        "Description of the instance identified by the prompt's box.",
    ),
    TaskSpec(
        "counterfactual_caption",
        "text",
        TEXT_METRICS,
        "counterfactual",
        "text",
        "utf8-text-v1",
        "Caption of a counterfactual image.",
    ),
    TaskSpec(
        "vqa",
        "answer",
        (*TEXT_METRICS, "label_accuracy", "normalized_exact_match"),
        "source",
        "answer",
        "vqa-label-v1",
        "Text similarity plus registered-label diagnostics for five geometry question types.",
    ),
    TaskSpec(
        "category_presence",
        "answer",
        ("accuracy", "positive_f1", "macro_f1"),
        "both",
        "answer",
        "yes-no-v1",
        "Binary presence of the one category named in the query.",
    ),
    TaskSpec(
        "referring_expression",
        "binary_mask",
        MASK_METRICS,
        "source",
        "object_union",
        "coco-rle-regions-v1",
        "Union mask for the referred object IDs.",
    ),
    TaskSpec(
        "category_or_all_instance_grounding",
        "binary_mask",
        MASK_METRICS,
        "both",
        "object_union",
        "coco-rle-regions-v1",
        "Union mask for a category or for all annotated instances.",
    ),
    TaskSpec(
        "panoptic_segmentation",
        "panoptic",
        PANOPTIC_METRICS,
        "source",
        "native_panoptic",
        "panoptic-id-png-v1",
        "Native non-overlapping semantic and instance panoptic map.",
    ),
    TaskSpec(
        "missing_component_identification",
        "answer",
        ("accuracy", "example_f1", "micro_f1", "macro_f1"),
        "both",
        "answer",
        "component-name-json-array-v1",
        "Absent categories from the required component taxonomy; [] means none missing.",
    ),
    TaskSpec(
        "functional_completeness",
        "answer",
        ("mae", "rmse", "accuracy"),
        "both",
        "answer",
        "completeness-score-v1",
        "All required component categories are present (paper Eq. 1), not scene risk.",
    ),
    TaskSpec(
        "referring_functional_grounding",
        "binary_mask",
        MASK_METRICS,
        "both",
        "object_union",
        "coco-rle-regions-v1",
        "Visible components relevant to the fixed functional taxonomy; no connectivity claim.",
    ),
    TaskSpec(
        "referring_panoptic_segmentation",
        "panoptic",
        PANOPTIC_METRICS,
        "source",
        "native_panoptic",
        "panoptic-id-png-v1",
        "Category-conditioned panoptic map, preserving instances and background elsewhere.",
    ),
)

TASK_REGISTRY = MappingProxyType({spec.family: spec for spec in _SPECS})
TASK_FAMILIES = tuple(TASK_REGISTRY)

CAPTION_FAMILIES = frozenset(
    {
        "source_summary",
        "detailed_caption",
        "instance_description",
        "counterfactual_caption",
    }
)
TEXT_METRIC_FAMILIES = CAPTION_FAMILIES | {"vqa"}
BINARY_MASK_FAMILIES = frozenset(
    {"referring_expression", "category_or_all_instance_grounding", "referring_functional_grounding"}
)
PANOPTIC_FAMILIES = frozenset({"panoptic_segmentation", "referring_panoptic_segmentation"})

PRIMARY_METRICS = MappingProxyType(
    {
        **{family: ("bleu_1", "meteor", "rouge_l", "cider_d") for family in TEXT_METRIC_FAMILIES},
        "category_presence": ("accuracy", "macro_f1"),
        "missing_component_identification": ("accuracy", "example_f1"),
        "functional_completeness": ("mae", "rmse"),
        **{
            family: (
                "positive_ciou", "positive_miou", "all_query_miou", "negative_empty_accuracy"
            )
            for family in BINARY_MASK_FAMILIES
        },
        **{
            family: ("foreground_class_pixel_ciou", "foreground_image_class_miou")
            for family in PANOPTIC_FAMILIES
        },
    }
)


def evaluation_metric_recipe(families: Sequence[str]) -> dict[str, Any]:
    """Describe existing-task metrics without asserting historical/paper equivalence."""

    selected = parse_task_selection(",".join(families))
    return {
        "version": METRIC_RECIPE_VERSION,
        "task_scope": "existing_selected_dataset_tasks_only; no added queries",
        "exact_paper_parity": False,
        "primary_metrics": {family: list(PRIMARY_METRICS[family]) for family in selected},
        "report_scale": "raw; no external scaling; CIDEr-D and FC errors are not percentages",
        "text": {
            "backend": "pycocoevalcap==1.2/PTBTokenizer/Cider(sigma=6.0)",
            "pooling": "one reference per task; separate corpus per family; no cross-family average",
            "primary_bleu": "BLEU-1; BLEU-2..4 remain available",
            "invalid": "failed/non-string outputs become empty hypotheses, never dropped",
            "vqa_label_parser": "separate diagnostic; unparsed text still receives text scores",
        },
        "presence_f1": "binary macro-F1: mean of yes-class and no-class F1",
        "missing_component_f1": "mean per-query set F1; empty/empty=1; invalid=0",
        "completeness": {
            "target": "all required categories present (0/1), not risk or connectivity",
            "primary": "query-weighted MAE/RMSE; invalid outputs have error 1",
            "class_balanced": "equal weight to target classes 0/1; unavailable if either is absent",
        },
        "binary_masks": {
            "prediction": "union of predicted regions; not instance-set matching",
            "ciou": "sum intersections / sum unions; positive_ciou uses nonempty targets",
            "miou": "mean per-query IoU; all_query_miou gives valid empty/empty=1",
            "threshold_accuracy": "fraction of query IoUs >= 0.5 or >= 0.75; NOT mIoU",
        },
        "panoptic": {
            "foreground_class_pixel_ciou": "sum class-pixel intersections / sum class-pixel unions",
            "foreground_image_class_miou": "mean over queries of mean nonzero-union class IoU",
            "pooled_pq_sq_rq": "same-category matches at IoU>0.5; pooled counts, not class-macro",
            "legacy_aliases": {
                "foreground_ciou": "foreground_class_pixel_ciou",
                "foreground_miou": "foreground_image_class_miou",
                "pq": "pooled_pq", "sq": "pooled_sq", "rq": "pooled_rq",
            },
        },
        "paper_comparison": (
            "BLEU-1/CPC macro-F1/MCI example-F1 are declared reporting choices informed by "
            "legacy artifacts, not proven paper parity; query coverage, text normalization, "
            "and segmentation reductions differ or remain ambiguous."
        ),
    }


def component_membership_targets(present_categories: Sequence[str]) -> tuple[list[str], int]:
    """Return missing types and the all-required-types-present target from paper Eq. 1."""

    present = set(present_categories)
    if present.difference(REQUIRED_COMPONENTS):
        raise ValueError("component membership contains an unknown category")
    missing = [name for name in REQUIRED_COMPONENTS if name not in present]
    return missing, int(not missing)

VQA_QUESTION_TYPES = (
    "width_height",
    "absolute_position",
    "horizontal_relation",
    "vertical_relation",
    "relative_area",
)

# The dataset has no supervision for these paper tasks. Keeping this
# list explicit prevents a current task from being silently relabelled as a
# paper result.
PAPER_ONLY_CAPABILITIES = MappingProxyType(
    {
        "potential_component_sets": "no set labels",
        "scene_risk_level": "no expert-reviewed risk labels",
        "component_link_risk": "no functional-link labels",
    }
)

PAPER_PROTOCOL_AMBIGUITIES = (
    "Paper section 5.1 specifies VQA exact match, counting MAE, and binary "
    "accuracy, while Table 2 reports BLEU, METEOR, ROUGE-L, and CIDEr.",
    "The paper does not define empty-mask aggregation for cIoU/mIoU.",
    "The paper does not fully specify how cIoU/mIoU reduce a semantic-instance "
    "panoptic map.",
)


def parse_task_selection(value: str | None) -> tuple[str, ...]:
    """Return a stable, de-duplicated task selection.

    ``all`` is the only wildcard.  Unknown or empty names fail closed so a
    benchmark run cannot quietly skip a misspelled family.
    """

    if value is None or value.strip() == "all":
        return TASK_FAMILIES
    requested = [item.strip() for item in value.split(",")]
    if not requested or any(not item for item in requested):
        raise ValueError("--tasks must be 'all' or a comma-separated family list")
    unknown = sorted(set(requested).difference(TASK_REGISTRY))
    if unknown:
        raise ValueError(f"unknown task families: {unknown}")
    return tuple(dict.fromkeys(requested))


def protocol_capabilities(protocol: str) -> dict[str, object]:
    """Describe protocol support without claiming unavailable paper metrics."""

    if protocol == DATASET_PROTOCOL:
        return {
            "protocol": protocol,
            "supported": True,
            "families": list(TASK_FAMILIES),
        }
    if protocol == PAPER_PROTOCOL:
        return {
            "protocol": protocol,
            "supported": False,
            "missing": dict(PAPER_ONLY_CAPABILITIES),
            "ambiguities": list(PAPER_PROTOCOL_AMBIGUITIES),
        }
    raise ValueError(f"unknown evaluation protocol {protocol!r}")


def inference_task_view(dataset: Any, split: str, task: Mapping[str, Any]) -> dict[str, Any]:
    """Create the only task mapping permitted to cross into a model runner.

    Ground-truth answers, object IDs, masks, counterfactual membership, source
    links, and panoptic metadata are intentionally not copied.  The category
    and question type are query metadata already expressed by the prompt; they
    are retained for deterministic output routing, not supplied as answers.
    """

    if not isinstance(task, Mapping):
        raise ValueError("task must be an object")
    required = ("id", "family", "image_id", "prompt")
    missing = [field for field in required if field not in task]
    if missing:
        raise ValueError(f"task is missing inference fields: {missing}")
    family = task["family"]
    if family not in TASK_REGISTRY:
        raise ValueError(f"unknown task family {family!r}")
    if not isinstance(task["id"], str) or not task["id"]:
        raise ValueError("task id must be a non-empty string")
    if not isinstance(task["prompt"], str) or not task["prompt"]:
        raise ValueError("task prompt must be a non-empty string")
    view: dict[str, Any] = {
        "split": split,
        "task_id": task["id"],
        "family": family,
        "image_id": task["image_id"],
        "image_path": str(dataset.resolve_image(split, task["image_id"])),
        "prompt": task["prompt"],
    }
    if family == "vqa":
        question_type = task.get("question_type")
        if question_type not in VQA_QUESTION_TYPES:
            raise ValueError(f"unsupported VQA question_type {question_type!r}")
        view["question_type"] = question_type
    if family in {
        "category_presence",
        "category_or_all_instance_grounding",
        "referring_panoptic_segmentation",
    }:
        category_id = task.get("category_id")
        valid_ids = {
            row.get("id")
            for row in dataset.manifest.get("categories", ())
            if isinstance(row, Mapping)
        }
        if category_id is not None and category_id not in valid_ids:
            raise ValueError(f"query category_id {category_id!r} is not in the dataset taxonomy")
        if family == "referring_panoptic_segmentation" and category_id is None:
            raise ValueError("referring-panoptic queries require a category_id")
        view["query_category_id"] = category_id
    return view
