from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import time
from collections import Counter, OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    broadcast_object_list,
    gather_object,
    set_seed,
)
from pycocotools import mask as mask_utils
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.ops import box_iou
from transformers import get_cosine_schedule_with_warmup

from .artifacts import (
    atomic_json,
    external_output,
    panoptic_ids_from_image,
)
from .capabilities import (
    ALL_SAFETY_HEADS,
    SafetyCapabilities,
    apply_ablation,
    capabilities_from_coverage,
    supervision_coverage,
)
from .checkpoint import (
    inspect_training_state,
    restore_training_state,
    save_training_state,
)
from .config import (
    apply_stage_overrides,
    load_config,
    paper_reproduction_issues,
    resolve_stage_config,
)
from .data import DATASET_FORMAT, FalconDataset, open_dataset
from .detector import RFDETRDetector
from .feature_cache import FrozenFeatureCache
from .grounding import (
    MATCHING_POLICY_VERSION,
    GroundingMatchStatus,
    GroundTruthInstance,
    encode_panoptic_match,
    encode_segmentation_match,
    match_proposals,
)
from .model import FalconModel, VisionFeatures
from .sampling import plan_epoch
from .tasks import TASK_REGISTRY, parse_task_selection


STAGE3_MEMORY_RECIPE = {
    "language_attention": "sdpa",
    "language_use_cache": False,
    "gradient_checkpointing_kwargs": {
        "use_reentrant": False,
        "preserve_rng_state": True,
    },
}


class TaskDataset(Dataset):
    def __init__(
        self,
        dataset: FalconDataset,
        split: str = "train",
        *,
        tasks: str | None = None,
        partition: str | None = None,
    ):
        self.dataset, self.split = dataset, split
        self.native = getattr(dataset, "format", None) == DATASET_FORMAT
        self.root = dataset.root if self.native else dataset.manifest.root
        self.partition = None
        if self.native:
            if dataset.manifest["splits"][split]["usage"] != "training":
                raise ValueError(
                    "Training may only read the dataset's declared training split"
                )
            selected = set(parse_task_selection(tasks))
            if partition is not None:
                from .partitions import load_partition

                self.partition = load_partition(dataset, partition, role="train")
            report = dataset.validate(
                split,
                tasks=selected,
                full=True,
                image_ids=None if self.partition is None else self.partition.image_ids,
            )
            if not report["ok"]:
                raise ValueError(
                    f"Native training data validation failed: {report['errors'][:3]}"
                )
            self.examples = [
                row for row in dataset.iter_tasks(split) if row["family"] in selected
            ]
            if self.partition is not None:
                self.examples = [
                    row
                    for row in self.examples
                    if row["image_id"] in self.partition.image_ids
                ]
            if not self.examples:
                raise ValueError("The selected training tasks are empty")
            image_ids = {row["image_id"] for row in self.examples}
            self.states = {
                row["id"]: row
                for row in dataset.images(split)
                if row["id"] in image_ids
            }
            self.family_weights = dataset.manifest.get("train_family_weights")
            return
        if tasks not in (None, "all"):
            raise ValueError("--tasks selection currently requires the falcon-x format")
        if partition is not None:
            raise ValueError("Parent-group partitions require a falcon-x dataset")
        self.family_weights = None
        spec = dataset.split(split)
        if not spec.tasks:
            raise ValueError(f"Split {split!r} has no task manifest")
        split_root = dataset.split_root(split)
        task_path = (split_root / spec.tasks).resolve(strict=True)
        try:
            task_path.relative_to(split_root)
        except ValueError as exc:
            raise ValueError(
                f"Split {split!r} task manifest escapes its split root"
            ) from exc
        if not task_path.is_file():
            raise ValueError(
                f"Split {split!r} task manifest is not a file: {task_path}"
            )
        self.states = {record.image_id: record for record in dataset.records(split)}
        self.examples = self._load_examples(task_path)
        selected_image_ids = {row["image_id"] for row in self.examples}
        self.states = {
            image_id: state
            for image_id, state in self.states.items()
            if image_id in selected_image_ids
        }

    def _load_examples(self, task_path: Path) -> list[dict[str, Any]]:
        examples = []
        identifiers = set()
        validated_exact_masks: set[tuple[int | str, int | str]] = set()
        for line_number, line in enumerate(
            task_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid task JSON at {task_path}:{line_number}"
                ) from exc
            if not isinstance(example, dict):
                raise ValueError(
                    f"Task row at {task_path}:{line_number} must be an object"
                )
            required = ("id", "image_id", "image_relpath", "task", "prompt", "answer")
            missing = [key for key in required if key not in example]
            if missing:
                raise ValueError(
                    f"Task row at {task_path}:{line_number} lacks {', '.join(missing)}"
                )
            if example["id"] in identifiers:
                raise ValueError(f"Duplicate task ID {example['id']!r}")
            identifiers.add(example["id"])
            try:
                state = self.states[example["image_id"]]
            except KeyError as exc:
                raise ValueError(
                    f"Task {example['id']!r} references unknown image {example['image_id']!r}"
                ) from exc
            if example["image_relpath"] != state.image_relpath:
                raise ValueError(
                    f"Task {example['id']!r} image_relpath disagrees with its state record"
                )
            for key in ("task", "prompt", "answer"):
                if not isinstance(example[key], str):
                    raise ValueError(f"Task {example['id']!r} {key} must be a string")
            grounding = example.get("grounding", [])
            if not isinstance(grounding, list):
                raise ValueError(f"Task {example['id']!r} grounding must be an array")
            for item in grounding:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Task {example['id']!r} grounding entries must be objects"
                    )
                category = item.get("category")
                if category not in state.present_components:
                    raise ValueError(
                        f"Task {example['id']!r} grounds a missing or unknown component"
                    )
                bbox = item.get("bbox_xywh")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(
                        f"Task {example['id']!r} grounding bbox must contain four values"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    for value in bbox
                ):
                    raise ValueError(
                        f"Task {example['id']!r} grounding bbox must be numeric"
                    )
                x, y, width, height = (float(value) for value in bbox)
                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    raise ValueError(
                        f"Task {example['id']!r} grounding bbox is invalid"
                    )
                if x + width > state.width + 1e-6 or y + height > state.height + 1e-6:
                    raise ValueError(
                        f"Task {example['id']!r} grounding bbox exceeds its image"
                    )
                if item.get("match_policy") == "exact_mask_iou":
                    source_annotation_id = item.get("source_annotation_id")
                    if source_annotation_id is None or "segmentation" not in item:
                        raise ValueError(
                            f"Task {example['id']!r} exact grounding lacks source ID/mask"
                        )
                    state_matches = [
                        candidate
                        for candidate in state.grounding
                        if candidate.get("source_annotation_id") == source_annotation_id
                    ]
                    if len(state_matches) != 1:
                        raise ValueError(
                            f"Task {example['id']!r} source annotation is not in its state"
                        )
                    authoritative = state_matches[0]
                    for key in (
                        "instance_id",
                        "source_annotation_id",
                        "category",
                        "bbox_xywh",
                        "segmentation",
                        "match_policy",
                    ):
                        if item.get(key) != authoritative.get(key):
                            raise ValueError(
                                f"Task {example['id']!r} exact grounding {key} was altered"
                            )
                    mask_key = (state.image_id, source_annotation_id)
                    if mask_key not in validated_exact_masks:
                        _decode_grounding_mask(
                            authoritative["segmentation"],
                            state.height,
                            state.width,
                        )
                        validated_exact_masks.add(mask_key)
            examples.append(example)
        if not examples:
            raise ValueError(f"Split {self.split!r} task manifest is empty")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        state = self.states[example["image_id"]]
        if self.native:
            kind = TASK_REGISTRY[example["family"]].prediction_kind
            instances = []
            if kind == "binary_mask":
                instances = [
                    GroundTruthInstance(
                        obj["id"],
                        obj["category_id"],
                        self.dataset.decode_object_mask(self.split, obj),
                    )
                    for obj in self.dataset.target_objects(self.split, example)
                ]
            elif kind == "panoptic":
                path, metadata = self.dataset.resolve_task_panoptic(self.split, example)
                with Image.open(path) as opened:
                    ids = panoptic_ids_from_image(opened)
                instances = [
                    GroundTruthInstance(
                        segment["id"], segment["category_id"], ids == segment["id"]
                    )
                    for segment in metadata["segments_info"]
                ]
            return {
                "id": example["id"],
                "family": example["family"],
                "image_id": state["id"],
                "image_path": self.dataset.resolve_image(self.split, state["id"]),
                "prompt": example["prompt"],
                "answer": example["answer"],
                "grounding": [],
                "ground_truth_instances": instances,
                "prediction_kind": kind,
                "targets": self.dataset.structured_targets(self.split, state["id"]),
            }
        return {
            "id": example["id"],
            "family": example["task"],
            "image_id": state.image_id,
            "image_path": self.dataset.resolve_image(self.split, state),
            "prompt": example["prompt"],
            "answer": example["answer"],
            "grounding": example.get("grounding", []),
            "targets": state.targets,
        }

    def supervision_rows(self, image_ids=None):
        selected = self.states if image_ids is None else image_ids
        if self.native:
            yield from (
                self.dataset.structured_targets(self.split, image_id)
                for image_id in selected
            )
        else:
            yield from (self.states[image_id].targets for image_id in selected)


class GroupedTaskSampler(Sampler[int]):
    """Shuffle images while keeping their tasks adjacent for one live detector pass."""

    def __init__(self, dataset: TaskDataset, seed: int):
        groups: dict[int | str, list[int]] = {}
        for index, example in enumerate(dataset.examples):
            groups.setdefault(example["image_id"], []).append(index)
        self.groups = tuple(tuple(indices) for indices in groups.values())
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        for group_index in torch.randperm(
            len(self.groups),
            generator=self.generator,
        ).tolist():
            group = self.groups[group_index]
            order = torch.randperm(len(group), generator=self.generator).tolist()
            yield from (group[index] for index in order)

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups)


class PlannedTaskSampler(Sampler[int]):
    """Stable epoch plans with family quotas and explicit coverage reports."""

    def __init__(self, dataset: TaskDataset, training: dict[str, Any]):
        self.dataset, self.training = dataset, training
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.plan = plan_epoch(
            self.dataset.examples,
            seed=self.training["seed"],
            epoch=epoch,
            mode=self.training.get("sampling", "all"),
            epoch_size=self.training.get("epoch_size"),
            weights=self.dataset.family_weights,
        )

    def __iter__(self):
        return iter(self.plan.indices)

    def __len__(self):
        return len(self.plan.indices)


def _collate(rows: list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    # Do not default-collate variable geometry, nullable labels, or PIL images.
    if not rows:
        raise ValueError("cannot collate an empty batch")
    return rows[0] if len(rows) == 1 else rows


def _decode_grounding_mask(value: Any, height: int, width: int) -> torch.Tensor:
    try:
        segmentation = copy.deepcopy(value)
        if isinstance(segmentation, list):
            if not segmentation:
                raise ValueError("empty polygons")
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation, dict):
            if list(segmentation.get("size", ())) != [height, width]:
                raise ValueError("RLE size mismatch")
            if isinstance(segmentation.get("counts"), str):
                segmentation["counts"] = segmentation["counts"].encode("ascii")
                rle = segmentation
            elif isinstance(segmentation.get("counts"), list):
                rle = mask_utils.frPyObjects(segmentation, height, width)
            else:
                raise ValueError("invalid RLE counts")
        else:
            raise ValueError("unsupported segmentation")
        decoded = np.asarray(mask_utils.decode(rle))
    except Exception as exc:
        raise ValueError("Malformed immutable grounding segmentation") from exc
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width) or not bool(np.any(decoded)):
        raise ValueError(
            "Immutable grounding segmentation is empty or has wrong dimensions"
        )
    return torch.as_tensor(decoded, dtype=torch.bool)


def _one_to_one_matches(overlaps: torch.Tensor, threshold: float) -> list[int]:
    if overlaps.ndim != 2:
        raise ValueError("overlap matrix must have shape [ground_truth, proposals]")
    if overlaps.numel() == 0:
        return []
    rows, columns = linear_sum_assignment(1.0 - overlaps.detach().cpu().numpy())
    return sorted(
        int(column)
        for row, column in zip(rows, columns, strict=True)
        if float(overlaps[int(row), int(column)]) >= threshold
    )


def _grounded_answer(
    answer: str,
    grounding: list[dict[str, Any]],
    boxes: torch.Tensor,
    masks: torch.Tensor | None = None,
    image_hw: tuple[int, int] | None = None,
    *,
    match_iou_threshold: float = 0.5,
) -> str:
    if not grounding:
        return answer
    if not 0.0 <= match_iou_threshold <= 1.0:
        raise ValueError("match_iou_threshold must lie in [0, 1]")
    exact = [item.get("match_policy") == "exact_mask_iou" for item in grounding]
    if any(exact) and not all(exact):
        raise ValueError("Grounding cannot mix exact-mask and legacy box contracts")
    if all(exact):
        if image_hw is None:
            raise ValueError("Exact-mask grounding requires image dimensions")
        height, width = image_hw
        targets = torch.stack(
            [
                _decode_grounding_mask(item.get("segmentation"), height, width)
                for item in grounding
            ]
        )
        if boxes.numel() == 0:
            return answer
        if masks is None:
            raise ValueError("Exact-mask grounding requires detector masks")
        proposal_masks = masks.detach().cpu().to(dtype=torch.bool)
        if proposal_masks.ndim != 3 or tuple(proposal_masks.shape[-2:]) != (
            height,
            width,
        ):
            raise ValueError(
                "Detector masks do not match the grounded image dimensions"
            )
        intersections = torch.logical_and(targets[:, None], proposal_masks[None]).sum(
            dim=(-2, -1)
        )
        unions = torch.logical_or(targets[:, None], proposal_masks[None]).sum(
            dim=(-2, -1)
        )
        overlaps = intersections.float() / unions.clamp_min(1).float()
    else:
        if boxes.numel() == 0:
            return answer
        targets = torch.tensor([x["bbox_xywh"] for x in grounding], dtype=torch.float32)
        targets[:, 2:] += targets[:, :2]
        overlaps = box_iou(targets, boxes.detach().cpu())
    matched = _one_to_one_matches(overlaps, match_iou_threshold)
    if not matched:
        return answer
    references = ", ".join(f"region_{index:03d}" for index in matched)
    return f"{answer} Regions: {references}."


def _targets(values: dict[str, Any] | list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if isinstance(values, list):
        if not values:
            raise ValueError("target batch cannot be empty")
        singles = [_targets(row) for row in values]
        return {name: torch.cat([row[name] for row in singles]) for name in singles[0]}
    if len(values["presence"]) != 3 or len(values["links"]) != 3:
        raise ValueError(
            "Structured presence and link targets must follow canonical length 3"
        )
    links = [
        float("nan") if value is None else float(value) for value in values["links"]
    ]
    return {
        "risk": torch.tensor(
            [float("nan") if values["risk"] is None else float(values["risk"])],
            dtype=torch.float32,
        ),
        "presence": torch.tensor([values["presence"]], dtype=torch.float32),
        "links": torch.tensor([links], dtype=torch.float32),
    }


def _validate_initialization(
    payload: dict[str, Any],
    settings: dict[str, Any],
    *,
    allow_legacy: bool,
    allow_partial: bool = False,
) -> bool:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        if not allow_legacy:
            raise ValueError(
                "Stage-2 initialization lacks configuration; "
                "--allow-legacy-init is required for diagnostic use"
            )
        return False
    parent_complete = (
        metadata.get("training_complete") is True
        and metadata.get("official_result_eligible", True) is True
    )
    if not parent_complete and not allow_partial:
        raise ValueError(
            "Stage 2 is incomplete; --allow-partial-init is required for diagnostic use"
        )
    for key in (
        "safety_capabilities",
        "dataset_categories",
        "grounding_policy",
        "structured_loss_recipe",
    ):
        if metadata.get(key) != settings.get(key):
            raise ValueError(f"Stage-2 initialization differs at {key}")
    parent_config = metadata.get("config")
    if not isinstance(parent_config, dict):
        raise ValueError("Stage-2 initialization lacks its effective configuration")
    for key in ("model", "detector"):
        parent_settings = dict(parent_config[key])
        current_settings = dict(settings["config"][key])
        if key == "model":
            for name in ("vision_model", "language_model"):
                parent_settings.pop(name, None)
                current_settings.pop(name, None)
        if parent_settings != current_settings:
            raise ValueError(f"Stage-2 initialization {key} configuration differs")
    for key in ("protocol", "backbone_layout", "ssa_ablation"):
        if parent_config.get("experiment", {}).get(key) != settings["config"].get(
            "experiment", {}
        ).get(key):
            raise ValueError(f"Stage-2 initialization experiment.{key} differs")
    expected_recipe = resolve_stage_config(settings["config"], 2)
    if metadata.get("resolved_stage") != expected_recipe:
        raise ValueError(
            "Stage-2 initialization recipe differs from configured stages.stage2"
        )
    return parent_complete


def _load_trainable(
    model: FalconModel,
    path: str | None,
    target_stage: int,
    *,
    settings: dict[str, Any] | None = None,
    allow_legacy: bool = False,
) -> None:
    if path is None:
        if target_stage == 3:
            raise ValueError("Stage 3 requires --init-checkpoint from Stage 2")
        return
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("stage") != 2:
        raise ValueError(
            "The initialization checkpoint must be a Falcon Stage-2 checkpoint"
        )
    if settings is not None:
        complete = _validate_initialization(
            payload,
            settings,
            allow_legacy=allow_legacy,
            allow_partial=settings.get("partial_initialization_allowed", False),
        )
        if settings.get("initialization_complete") != complete:
            raise ValueError("Stage-2 initialization metadata changed after preflight")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("The initialization checkpoint has no trainable state_dict")
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state_dict) != expected:
        raise ValueError("Stage-2 trainable parameter inventory differs from the model")
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state_dict.values()
    ):
        raise ValueError("Stage-2 checkpoint contains invalid parameter values")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Stage-2 checkpoint has unexpected keys: {incompatible.unexpected_keys[:5]}"
        )


def _task_key(identifier: Any) -> str:
    """Preserve typed legacy IDs while keeping checkpoint cursors JSON-safe."""
    return json.dumps(identifier, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _observed_report(
    dataset: TaskDataset,
    rank_reports: list[dict[str, Any]],
    training: dict[str, Any],
    capabilities: SafetyCapabilities = ALL_SAFETY_HEADS,
) -> dict[str, Any]:
    """Merge actual per-rank draws, never presenting a sampler plan as exposure."""
    seen = set()
    updated = set()
    draws: Counter[str] = Counter()
    language: Counter[str] = Counter()
    grounding: Counter[str] = Counter()
    cache: Counter[str] = Counter()
    for report in rank_reports:
        seen.update(report["seen_task_keys"])
        updated.update(report["updated_task_keys"])
        draws.update(report["family_draws"])
        language.update(report["language_supervision_draws"])
        grounding.update(report["grounding_counts"])
        cache.update(report["cache_counts"])
    selected_keys = {_task_key(row["id"]) for row in dataset.examples}
    if not seen.issubset(selected_keys):
        raise ValueError("Observed training cursor contains unknown selected task IDs")
    if not updated.issubset(seen):
        raise ValueError("Updated training task IDs were not observed")
    unique: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    image_ids = set()
    updated_image_ids = set()
    for row in dataset.examples:
        family = row.get("family", row.get("task"))
        selected[family] += 1
        if _task_key(row["id"]) in seen:
            unique[family] += 1
            image_ids.add(row["image_id"])
        if _task_key(row["id"]) in updated:
            updated_image_ids.add(row["image_id"])
    families = {
        name: {
            "selected_tasks": count,
            "draws": draws[name],
            "unique_tasks": unique[name],
            "repeated_draws": draws[name] - unique[name],
            "coverage": unique[name] / count,
            "language_supervision_draws": language[name],
        }
        for name, count in sorted(selected.items())
    }
    observed_coverage = supervision_coverage(
        row
        for row in dataset.supervision_rows(updated_image_ids)
        if _has_active_supervision(row, False, capabilities, training)
    )
    for name, weight_name in (
        ("risk", "risk_loss_weight"),
        ("presence", "presence_loss_weight"),
        ("links", "link_loss_weight"),
    ):
        if training[weight_name] == 0:
            observed_coverage[name] = 0 if name == "risk" else [0, 0, 0]
    observed_coverage["risk"] *= int(capabilities.risk)
    for name in ("presence", "links"):
        observed_coverage[name] = [
            count * int(enabled)
            for count, enabled in zip(
                observed_coverage[name], getattr(capabilities, name), strict=True
            )
        ]
    return {
        "observed_training": {
            "scope": "global_across_ranks_including_sampler_padding",
            "draws": sum(draws.values()),
            "unique_tasks": len(seen),
            "unique_images": len(image_ids),
            "update_contributing_unique_tasks": len(updated),
            "update_contributing_unique_images": len(updated_image_ids),
            "families": families,
            "batches_seen_by_rank": [item["batches_seen"] for item in rank_reports],
            "optimizer_updates_by_rank": [item["updates"] for item in rank_reports],
        },
        "exposed_label_coverage": supervision_coverage(
            dataset.supervision_rows(image_ids)
        ),
        "observed_supervision_coverage": observed_coverage,
        "observed_supervision_coverage_semantics": (
            "unique images contributing positive-weight structured loss to a successful "
            "optimizer update; availability is separate from enabled SSA capabilities"
        ),
        "training_complete": all(item["training_complete"] for item in rank_reports),
        "grounding_coverage": dict(grounding),
        "grounding_coverage_semantics": (
            "global task-draw/target-occurrence weighted; not unique-object detector recall"
        ),
        "cache_counts": dict(cache),
    }


def _has_active_supervision(targets, language, capabilities, recipe) -> bool:
    if language:
        return True
    if (
        capabilities.risk
        and recipe["risk_loss_weight"] > 0
        and targets.get("risk") is not None
    ):
        return True
    for name, weight in (
        ("presence", "presence_loss_weight"),
        ("links", "link_loss_weight"),
    ):
        if recipe[weight] > 0 and any(
            enabled and label is not None
            for enabled, label in zip(
                getattr(capabilities, name), targets[name], strict=True
            )
        ):
            return True
    return False


def _prepare_training_output(output, *, resume, accelerator, resources):
    """Protect the stage directory against concurrent writers and accidental reuse."""
    error = [None]
    if accelerator.is_main_process:
        try:
            output.mkdir(parents=True, exist_ok=True)
            lock_path = output / ".training.lock"
            if lock_path.is_symlink() or (output / "last.pt").is_symlink():
                raise ValueError("Training output files must not be symlinks")
            lock = resources.enter_context(lock_path.open("a+b"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "Training output is locked by another process"
                ) from exc
            if output.exists():
                contents = [
                    path for path in output.iterdir() if path.name != ".training.lock"
                ]
                if contents and not resume:
                    raise ValueError(
                        "Fresh training requires an empty output directory"
                    )
                if resume and contents and Path(resume).resolve() != output / "last.pt":
                    raise ValueError(
                        "Resume into this directory requires its own last.pt"
                    )
                extra_weights = [
                    path
                    for path in output.rglob("*")
                    if path.is_file()
                    and path.suffix in (".pt", ".pth", ".ckpt", ".safetensors")
                    and path != output / "last.pt"
                ]
                if extra_weights:
                    raise ValueError(
                        "Keep only last.pt in the training stage directory"
                    )
        except (OSError, ValueError) as exc:
            error[0] = str(exc)
    broadcast_object_list(error)
    if error[0] is not None:
        raise ValueError(error[0])


def train(args: argparse.Namespace) -> None:
    with ExitStack() as resources:
        _train(args, resources)


def _preflight_text_budget(
    dataset, model, *, budget: int, policy: str
) -> dict[str, Any]:
    """Scan every selected query/static answer before any optimizer updates.

    Grounding answers depend on live proposals, so only their full queries can
    be checked here. Their generated transports still use strict runtime checks.
    This reads metadata, not decoded image/mask tensors or detector features.
    """
    families: dict[str, Counter] = {}
    oversized = []
    oversized_count = maximum_prompt = maximum_static = dynamic_count = 0
    for row in dataset.examples:
        family = row.get("family", row.get("task"))
        dynamic = (
            TASK_REGISTRY[family].prediction_kind in ("binary_mask", "panoptic")
            if dataset.native
            else bool(row.get("grounding"))
        )
        try:
            counts = model.text_token_counts(
                f"USER: {row['prompt']}\nASSISTANT:", None if dynamic else row["answer"]
            )
        except ValueError as exc:
            raise ValueError(
                f"Text preflight failed for task {row['id']!r}: {exc}"
            ) from exc
        family_counts = families.setdefault(family, Counter())
        family_counts["queries"] += 1
        family_counts["dynamic_targets" if dynamic else "static_targets"] += 1
        dynamic_count += int(dynamic)
        maximum_prompt = max(maximum_prompt, counts["prompt_tokens"])
        if not dynamic:
            maximum_static = max(maximum_static, counts["total_tokens"])
        if counts["total_tokens"] > budget:
            oversized_count += 1
            family_counts["over_budget"] += 1
            if len(oversized) < 10:
                oversized.append(
                    {"task_id": row["id"], "query_only": dynamic, **counts}
                )
    report = {
        "format": "falcon-text-preflight-v1",
        "max_text_tokens": budget,
        "text_overflow_policy": policy,
        "queries_checked": len(dataset.examples),
        "static_targets_checked": len(dataset.examples) - dynamic_count,
        "dynamic_targets_runtime_checked": dynamic_count,
        "maximum_prompt_tokens": maximum_prompt,
        "maximum_static_training_tokens": maximum_static,
        "over_budget_count": oversized_count,
        "over_budget_examples": oversized,
        "families": {
            family: dict(counts) for family, counts in sorted(families.items())
        },
    }
    if oversized_count and policy == "error":
        raise ValueError(
            f"Text preflight found {oversized_count} tasks exceeding max_text_tokens={budget}; "
            "increase the stage's budget before training. No optimizer updates were made. "
            f"Examples: {oversized}"
        )
    return report


def _train(args: argparse.Namespace, resources: ExitStack) -> None:
    config = apply_stage_overrides(
        load_config(args.config),
        args.stage,
        {} if args.precision is None else {"precision": args.precision},
    )
    training = resolve_stage_config(config, args.stage)
    model_cfg, detector_cfg = config["model"], config["detector"]
    if config.get("experiment", {}).get("protocol") == "paper-v2":
        issues = paper_reproduction_issues(config)
        if issues:
            raise ValueError("Paper reproduction is not eligible: " + "; ".join(issues))
    precision = training["precision"]
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if getattr(args, "save_every", 0) < 0:
        raise ValueError("--save-every cannot be negative")
    threshold = getattr(args, "match_iou_threshold", 0.5)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("--match-iou-threshold must be finite and in [0,1]")
    dataset = TaskDataset(
        open_dataset(
            args.data_dir,
            annotations=getattr(args, "annotations", None),
        ),
        split="train",
        tasks=getattr(args, "tasks", None),
        partition=getattr(args, "partition", None),
    )
    protected_roots = getattr(dataset.dataset, "protected_roots", lambda: ())()
    output_root = external_output(
        args.output_dir, dataset.root, protected_roots=protected_roots
    )
    coverage = supervision_coverage(dataset.supervision_rows())
    available_capabilities = capabilities_from_coverage(coverage)
    ablation = config.get("experiment", {}).get("ssa_ablation", "none")
    capabilities = apply_ablation(available_capabilities, ablation)
    for name, enabled, weight in (
        ("risk", capabilities.risk, "risk_loss_weight"),
        ("presence", any(capabilities.presence), "presence_loss_weight"),
        ("links", any(capabilities.links), "link_loss_weight"),
    ):
        if enabled and training[weight] == 0:
            raise ValueError(
                f"Enabled {name} head requires positive {weight}; use an explicit SSA ablation "
                "to disable it instead of exporting an unsupervised head"
            )
    settings = {
        "stage": args.stage,
        "config": config,
        "resolved_stage": training,
        "training_complete": False,
        "dataset": {
            "root": str(dataset.root),
            "images": len(dataset.states),
            "tasks": len(dataset),
            "families": dict(
                Counter(row.get("family", row.get("task")) for row in dataset.examples)
            ),
        },
        "safety_capabilities": capabilities.as_dict(),
        "supervision_capabilities": available_capabilities.as_dict(),
        "ssa_ablation": ablation,
        "partition": None if dataset.partition is None else dataset.partition.to_dict(),
        "supervision_coverage": coverage,
        "require_observed_supervision": True,
        "structured_loss_recipe": {
            "risk": "sigmoid_l1",
            "presence": "bce_with_logits_fp32",
            "links": "sigmoid_l1",
            "reduction": "mean_over_finite_enabled_targets",
        },
        "model_locations": {
            "vision_model": args.vision_model or model_cfg["vision_model"],
            "language_model": args.language_model or model_cfg["language_model"],
            "detector_weights": args.detector_weights,
        },
        "grounding_policy": {
            "version": MATCHING_POLICY_VERSION,
            "iou_threshold": getattr(args, "match_iou_threshold", 0.5),
            "unmatched": getattr(args, "unmatched_grounding", "mask_language_loss"),
        },
        "legacy_initialization_allowed": getattr(args, "allow_legacy_init", False),
        "partial_initialization_allowed": getattr(args, "allow_partial_init", False),
        "precision": precision,
    }
    if args.stage == 3:
        settings["training_memory_recipe"] = copy.deepcopy(STAGE3_MEMORY_RECIPE)
    if dataset.native:
        settings["dataset_categories"] = dataset.dataset.manifest["categories"]
    settings["initialization_complete"] = None
    if args.init_checkpoint is not None:
        parent_payload = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=True
        )
        if not isinstance(parent_payload, dict) or parent_payload.get("stage") != 2:
            raise ValueError(
                "The initialization checkpoint must be a Falcon Stage-2 checkpoint"
            )
        settings["initialization_complete"] = _validate_initialization(
            parent_payload,
            settings,
            allow_legacy=getattr(args, "allow_legacy_init", False),
            allow_partial=getattr(args, "allow_partial_init", False),
        )
        del parent_payload
    elif args.stage == 3:
        raise ValueError("Stage 3 requires --init-checkpoint from Stage 2")
    feature_cache = None
    cache_settings = None
    if getattr(args, "feature_cache", None):
        cache_root = external_output(
            args.feature_cache, dataset.root, protected_roots=protected_roots
        )
        cache_settings = {
            "vision_model": settings["model_locations"]["vision_model"],
            "detector_weights": args.detector_weights,
            "detector_config": detector_cfg,
            "image_size": model_cfg["image_size"],
            "patch_size": model_cfg["patch_size"],
        }
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation"],
        mixed_precision=precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(training["seed"])
    requested_world = config.get("experiment", {}).get("world_size")
    if requested_world is not None and accelerator.num_processes != requested_world:
        raise ValueError(
            f"Configured world_size={requested_world}, got {accelerator.num_processes}"
        )
    resume = None
    if getattr(args, "resume", None):
        resume = inspect_training_state(
            args.resume,
            metadata=settings,
            world_size=accelerator.num_processes,
            rank=getattr(accelerator, "process_index", 0),
        )
    _prepare_training_output(
        output_root,
        resume=args.resume,
        accelerator=accelerator,
        resources=resources,
    )
    if cache_settings is not None:
        feature_cache = FrozenFeatureCache(cache_root, cache_settings)
    detector_device = (
        str(accelerator.device)
        if args.detector_device == "auto"
        else args.detector_device
    )
    detector = RFDETRDetector(
        checkpoint=args.detector_weights,
        variant=detector_cfg["variant"],
        device=detector_device,
        score_threshold=detector_cfg["score_threshold"],
        nms_threshold=detector_cfg["nms_threshold"],
        max_regions=detector_cfg["max_regions"],
        mask_threshold=detector_cfg.get("mask_threshold", 0.5),
        resolution=detector_cfg.get("resolution"),
    )
    if "tf32" in training:
        torch.backends.cuda.matmul.allow_tf32 = training["tf32"]
        torch.backends.cudnn.allow_tf32 = training["tf32"]
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model = FalconModel.from_pretrained(
        vision_model_name_or_path=args.vision_model or model_cfg["vision_model"],
        language_model_name_or_path=args.language_model or model_cfg["language_model"],
        detector=detector,
        region_dim=model_cfg["region_dim"],
        image_size=model_cfg["image_size"],
        patch_size=model_cfg["patch_size"],
        roi_size=model_cfg["roi_size"],
        max_regions=model_cfg["max_regions"],
        max_text_tokens=training["max_text_tokens"],
        text_overflow_policy=training["text_overflow_policy"],
        risk_loss_weight=training["risk_loss_weight"],
        presence_loss_weight=training["presence_loss_weight"],
        link_loss_weight=training["link_loss_weight"],
        safety_capabilities=capabilities,
        language_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
    )
    if accelerator.is_main_process:
        print(
            f"[text] checking {len(dataset)} selected queries and static targets",
            flush=True,
        )
    text_preflight = _preflight_text_budget(
        dataset,
        model,
        budget=training["max_text_tokens"],
        policy=training["text_overflow_policy"],
    )
    if accelerator.is_main_process:
        print(
            f"[text] checked {text_preflight['static_targets_checked']} static targets; "
            f"{text_preflight['dynamic_targets_runtime_checked']} proposal-dependent targets "
            "will also be checked at runtime",
            flush=True,
        )
    if args.stage == 3:
        model.enable_lora(
            rank=model_cfg["lora_rank"],
            alpha=model_cfg["lora_alpha"],
            dropout=model_cfg["lora_dropout"],
        )
    if resume is None:
        _load_trainable(
            model,
            args.init_checkpoint,
            args.stage,
            settings=settings,
            allow_legacy=getattr(args, "allow_legacy_init", False),
        )
    model.set_training_stage(args.stage)
    if args.stage == 3:
        memory_recipe = settings["training_memory_recipe"]
        model.language_model.config.use_cache = memory_recipe["language_use_cache"]
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=memory_recipe["gradient_checkpointing_kwargs"]
        )

    sampler = PlannedTaskSampler(dataset, training)
    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        sampler=sampler,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(training["seed"]),
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError(f"Training Stage {args.stage} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    batches_per_process = math.ceil(len(loader) / accelerator.num_processes)
    updates = (
        math.ceil(batches_per_process / training["gradient_accumulation"])
        * training["epochs"]
    )
    warmup = int(updates * training["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, max(updates, 1))
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    if resume is not None:
        restore_training_state(
            resume,
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    model.train()
    step = 0 if resume is None else resume["cursor"]["batches_seen"]
    optimizer_updates = 0 if resume is None else resume["cursor"]["updates"]
    start_epoch = 0 if resume is None else resume["cursor"]["epoch"]
    start_batch = 0 if resume is None else resume["cursor"]["next_batch"]
    if start_epoch > training["epochs"] or start_batch > len(loader):
        raise ValueError("Resume cursor exceeds the configured training schedule")
    last_saved_update = optimizer_updates if resume is not None else -1
    # Only frozen outputs are cached. All trainable projections/SSA run again
    # for every batch. A bounded LRU avoids retaining a dataset in GPU memory.
    frozen_cache: OrderedDict[int | str, tuple[Any, Any, Any]] = OrderedDict()
    cache_capacity = max(training["batch_size"], 2)
    prior_cursor = {} if resume is None else resume["cursor"]
    cache_counts: Counter[str] = Counter(prior_cursor.get("cache_counts", {}))
    grounding_counts: Counter[str] = Counter(
        {} if resume is None else resume["cursor"].get("grounding_counts", {})
    )
    epoch_reports = (
        [] if resume is None else resume["cursor"].get("sampling_epochs", [])
    )
    seen_task_key_list = list(prior_cursor.get("seen_task_keys", []))
    seen_task_keys = set(seen_task_key_list)
    updated_task_key_list = list(prior_cursor.get("updated_task_keys", []))
    updated_task_keys = set(updated_task_key_list)
    pending_update_tasks = set()
    family_draws: Counter[str] = Counter(prior_cursor.get("family_draws", {}))
    language_draws: Counter[str] = Counter(
        prior_cursor.get("language_supervision_draws", {})
    )
    cursor = {
        "epoch": start_epoch,
        "next_batch": start_batch,
        "batches_seen": step,
        "updates": optimizer_updates,
        "grounding_counts": dict(grounding_counts),
        "sampling_epochs": epoch_reports,
        "seen_task_keys": seen_task_key_list,
        "updated_task_keys": updated_task_key_list,
        "family_draws": dict(family_draws),
        "language_supervision_draws": dict(language_draws),
        "cache_counts": dict(cache_counts),
    }
    started_at = time.monotonic()
    initial_step = step
    total_batches = len(loader) * training["epochs"]
    for epoch in range(start_epoch, training["epochs"]):
        sampler.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        if epoch >= len(epoch_reports):
            epoch_reports.append(sampler.plan.report)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            if args.max_steps and step >= args.max_steps and accelerator.sync_gradients:
                break
            rows = [batch] if isinstance(batch, dict) else batch
            images, detected_batch, features_batch, answers, language_supervision = (
                [],
                [],
                [],
                [],
                [],
            )
            for row in rows:
                key = row["image_id"]
                if key not in frozen_cache:
                    image_key = str(row["image_path"])
                    with Image.open(row["image_path"]) as opened:
                        image = opened.convert("RGB")
                    unwrapped = accelerator.unwrap_model(model)
                    # no_grad tensors can safely feed trainable layers later;
                    # inference tensors must not escape into backward graphs.
                    cached = (
                        None
                        if feature_cache is None
                        else feature_cache.get(image_key, (image.height, image.width))
                    )
                    if cached is None:
                        with torch.no_grad():
                            detected = unwrapped.detector(image)
                            features = unwrapped.extract_vision_features(image)
                        if feature_cache is not None:
                            feature_cache.put(image_key, detected, features)
                        cache_counts["perception_passes"] += 1
                    else:
                        detected, features = cached
                        cache_counts["disk_hits"] += 1
                    frozen_cache[key] = (image, detected, features)
                else:
                    cache_counts["memory_hits"] += 1
                frozen_cache.move_to_end(key)
                image, detected, features = frozen_cache[key]
                while len(frozen_cache) > cache_capacity:
                    frozen_cache.popitem(last=False)
                images.append(image)
                detected_batch.append(detected)
                features_batch.append(features)
                supervise_language = True
                if row.get("prediction_kind") in ("binary_mask", "panoptic"):
                    matched = match_proposals(
                        row["ground_truth_instances"],
                        detected.masks,
                        iou_threshold=getattr(args, "match_iou_threshold", 0.5),
                    )
                    grounding_counts[matched.status.value] += 1
                    grounding_counts["ground_truth_instances"] += (
                        matched.ground_truth_count
                    )
                    grounding_counts["matched_instances"] += len(matched.matches)
                    # A missed positive must not become an empty-mask training answer.
                    # Mask only its unrepresentable LM target; retain the example
                    # and independently available structured supervision.
                    if matched.status in (
                        GroundingMatchStatus.PARTIAL_MATCH,
                        GroundingMatchStatus.UNMATCHED_POSITIVE,
                    ):
                        if (
                            getattr(args, "unmatched_grounding", "mask_language_loss")
                            == "error"
                        ):
                            raise ValueError(
                                f"Task {row['id']} has {matched.status.value}; "
                                "positive target cannot be represented by detector proposals. "
                                f"Coverage so far: {dict(grounding_counts)}"
                            )
                        supervise_language = False
                        answer = ""  # never supervised as an empty answer or empty-mask transport
                        grounding_counts["language_supervision_masked"] += 1
                    elif row["prediction_kind"] == "panoptic":
                        answer = encode_panoptic_match(
                            matched,
                            category_ids=tuple(
                                item["id"]
                                for item in dataset.dataset.manifest["categories"]
                            ),
                            max_regions=model_cfg["max_regions"],
                        )
                    else:
                        answer = encode_segmentation_match(
                            matched, max_regions=model_cfg["max_regions"]
                        )
                else:
                    answer = _grounded_answer(
                        row["answer"],
                        row["grounding"],
                        detected.boxes_xyxy,
                        detected.masks,
                        detected.original_hw,
                    )
                answers.append(answer)
                language_supervision.append(supervise_language)
            if len(rows) == 1:
                image_input, detector_input, feature_input = (
                    images[0],
                    detected_batch[0],
                    features_batch[0],
                )
                prompt_input = f"USER: {rows[0]['prompt']}\nASSISTANT:"
                answer_input = answers[0]
            else:
                if len({feature.grid_size for feature in features_batch}) != 1:
                    raise ValueError(
                        "all preprocessed images must use the same DINO patch grid"
                    )
                image_input, detector_input = images, detected_batch
                feature_input = VisionFeatures(
                    torch.cat([feature.patch_tokens for feature in features_batch]),
                    features_batch[0].grid_size,
                )
                prompt_input = [f"USER: {row['prompt']}\nASSISTANT:" for row in rows]
                answer_input = answers
            with accelerator.accumulate(model):
                try:
                    output = model(
                        images=image_input,
                        prompts=prompt_input,
                        answers=answer_input,
                        targets=_targets([row["targets"] for row in rows]),
                        detector_outputs=detector_input,
                        vision_features=feature_input,
                        language_supervision=language_supervision,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Training batch failed for task IDs {[row['id'] for row in rows]}: {exc}"
                    ) from exc
                if accelerator.scaler is None:
                    finite_loss = (
                        torch.isfinite(output.loss.detach()).all().to(torch.int32)
                    )
                    if (
                        accelerator.reduce(finite_loss, reduction="sum")
                        != accelerator.num_processes
                    ):
                        raise FloatingPointError(
                            f"Nonfinite training loss for task IDs {[row['id'] for row in rows]}"
                        )
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(parameters, 1.0)
                    if accelerator.scaler is None:
                        finite_grad = torch.isfinite(grad_norm).all().to(torch.int32)
                        if (
                            accelerator.reduce(finite_grad, reduction="sum")
                            != accelerator.num_processes
                        ):
                            raise FloatingPointError(
                                "Nonfinite gradients before optimizer update for task IDs "
                                f"{[row['id'] for row in rows]}"
                            )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            step += 1
            for row, supervised in zip(rows, language_supervision, strict=True):
                task_key = _task_key(row["id"])
                if _has_active_supervision(
                    row["targets"], supervised, capabilities, training
                ):
                    pending_update_tasks.add(task_key)
                if task_key not in seen_task_keys:
                    seen_task_keys.add(task_key)
                    seen_task_key_list.append(task_key)
                family_draws[row["family"]] += 1
                language_draws[row["family"]] += int(supervised)
            if accelerator.sync_gradients and not getattr(
                accelerator, "optimizer_step_was_skipped", False
            ):
                optimizer_updates += 1
                new_keys = sorted(pending_update_tasks - updated_task_keys)
                updated_task_key_list.extend(new_keys)
                updated_task_keys.update(new_keys)
            if accelerator.sync_gradients:
                pending_update_tasks.clear()
            cursor = {
                "epoch": epoch,
                "next_batch": batch_index + 1,
                "batches_seen": step,
                "updates": optimizer_updates,
                "grounding_counts": dict(grounding_counts),
                "sampling_epochs": epoch_reports,
                "seen_task_keys": seen_task_key_list,
                "updated_task_keys": updated_task_key_list,
                "family_draws": dict(family_draws),
                "language_supervision_draws": dict(language_draws),
                "cache_counts": dict(cache_counts),
            }
            if accelerator.is_main_process and (
                step == initial_step + 1 or step % 100 == 0 or step == total_batches
            ):
                elapsed = time.monotonic() - started_at
                rate = (step - initial_step) / max(elapsed, 1e-9)
                presence = (
                    output.safety.presence_probabilities.detach().float().mean(dim=0)
                )
                presence_text = ",".join(f"{value:.3f}" for value in presence.tolist())
                print(
                    f"[stage{args.stage}] epoch {epoch + 1}/{training['epochs']} "
                    f"batch {step}/{total_batches} updates={optimizer_updates} "
                    f"loss={float(output.loss.detach()):.5f} "
                    f"presence_bce={float(output.structured_losses['presence'].detach()):.5f} "
                    f"presence=[{presence_text}] "
                    f"elapsed={elapsed:.1f}s batches/s={rate:.3f}",
                    flush=True,
                )
            del output
            interval = getattr(args, "save_every", 0)
            if (
                interval
                and accelerator.sync_gradients
                and optimizer_updates % interval == 0
                and optimizer_updates != last_saved_update
            ):
                save_training_state(
                    output_root / "last.pt",
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata=settings,
                    cursor=cursor,
                )
                last_saved_update = optimizer_updates
                if accelerator.is_main_process:
                    print(
                        f"[checkpoint] saved {output_root / 'last.pt'}",
                        flush=True,
                    )
            if args.max_steps and step >= args.max_steps and accelerator.sync_gradients:
                break
        if args.max_steps and step >= args.max_steps and accelerator.sync_gradients:
            break
    rank_reports = gather_object(
        [{**cursor, "training_complete": step == len(loader) * training["epochs"]}]
    )
    observed = _observed_report(dataset, rank_reports, training, capabilities)
    global_cache_counts = observed.pop("cache_counts")
    summary = {
        **settings,
        **observed,
        "batches_seen": step,
        "sampling_epochs": epoch_reports,
        "frozen_cache": {"settings": cache_settings, "counts": global_cache_counts},
        "unmatched_grounding_policy": getattr(
            args, "unmatched_grounding", "mask_language_loss"
        ),
        "physical_batch_size": training["batch_size"],
        "world_size": accelerator.num_processes,
        "effective_batch_size": training["batch_size"]
        * training["gradient_accumulation"]
        * accelerator.num_processes,
        "text_budget_preflight": text_preflight,
    }
    diagnostic_reasons = []
    if not summary["training_complete"]:
        diagnostic_reasons.append("incomplete_training_schedule")
    observed_capabilities = capabilities_from_coverage(
        summary["observed_supervision_coverage"]
    )
    if capabilities.restrict(observed_capabilities) != capabilities:
        diagnostic_reasons.append("enabled_heads_lack_successful_supervision")
    if not summary["observed_training"]["update_contributing_unique_tasks"]:
        diagnostic_reasons.append("no_supervised_optimizer_updates")
    if training["text_overflow_policy"] == "truncate":
        diagnostic_reasons.append("query_or_target_truncation_enabled")
    if settings["initialization_complete"] is False:
        diagnostic_reasons.append("partial_stage2_initialization")
    summary["diagnostic_training_reasons"] = diagnostic_reasons
    summary["official_result_eligible"] = not diagnostic_reasons
    save_training_state(
        output_root / "last.pt",
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=summary,
        cursor=cursor,
    )
    if accelerator.is_main_process:
        atomic_json(output_root / "training_report.json", summary)
        print(f"[checkpoint] saved {output_root / 'last.pt'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Falcon Stage 2 or Stage 3")
    parser.add_argument("--stage", type=int, choices=(2, 3), required=True)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--annotations",
        action="append",
        help="Reviewed safety/panoptic annotation package",
    )
    parser.add_argument(
        "--partition", help="Parent-grouped partition manifest; uses its train role"
    )
    parser.add_argument(
        "--tasks", default="all", help="Native task-family selection (comma-separated)"
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.5,
        help="Minimum target/proposal mask IoU for a grounding match",
    )
    parser.add_argument(
        "--unmatched-grounding",
        choices=("mask_language_loss", "error"),
        default="mask_language_loss",
        help="Unrepresentable positives retain SSA supervision; never become empty masks",
    )
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--allow-legacy-init",
        action="store_true",
        help="Allow Stage-2 weights without metadata for diagnostic use",
    )
    parser.add_argument(
        "--allow-partial-init",
        action="store_true",
        help="Diagnostic-only opt-in to incomplete Stage-2 initialization",
    )
    parser.add_argument(
        "--resume", help="Resume an interrupted stage from its last.pt file"
    )
    parser.add_argument(
        "--feature-cache", help="Optional frozen detector/DINO feature cache directory"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Replace last.pt every N optimizer updates; 0 saves only at the end",
    )
    parser.add_argument("--vision-model")
    parser.add_argument("--language-model")
    parser.add_argument(
        "--detector-device",
        default="auto",
        help="Defaults to each Accelerate process's local device",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16"),
        help="Override the stage's configured precision",
    )
    parser.add_argument("--max-steps", type=int, help="Optional training-batch limit")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
