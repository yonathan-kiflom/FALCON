"""External falcon-x samples and label-free demo prompts."""

from __future__ import annotations

from pathlib import Path

from ..data import CANONICAL_COMPONENT_ORDER
from ..dataset import FalconXDataset, NativeId
from ..tasks import inference_task_view

COMPONENTS = CANONICAL_COMPONENT_ORDER
SUMMARY_PROMPT = "Summarize what is visible in this X-ray image."
DETAIL_PROMPT = (
    "Describe this X-ray image in visual detail, including the objects and their arrangement."
)
CAPTION_PROMPT = "Describe the objects and their arrangement in this X-ray image."
MISSING_PROMPT = (
    "Which required component categories are absent from this image? "
    "The required categories are detonator, explosive, and battery. "
    "Return a JSON array of category names, or [] when none are missing."
)
COMPLETE_PROMPT = (
    "Does this image contain at least one detonator, one explosive, and one battery? "
    "Return 1 if all three categories are present, otherwise 0."
)
SEGMENT_PROMPT = "Segment all annotated objects."
PANOPTIC_PROMPT = "Segment and identify the annotated objects in this image."


def category_prompt(category: str) -> str:
    if category not in COMPONENTS:
        raise ValueError(f"Unknown component category: {category!r}")
    return f"Segment all annotated {category} instances."


def referring_prompt(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Enter a referring expression.")
    return f"Segment the object described by: {expression.strip()}"


class Catalog:
    """A deterministic sample selection; annotations never become predictions."""

    def __init__(self, dataset: Path, split: str = "test", examples: int = 12):
        if isinstance(examples, bool) or not isinstance(examples, int) or examples < 1:
            raise ValueError("examples must be a positive integer")
        self._dataset = FalconXDataset(dataset)
        self._split = split
        images = self._dataset.images(split)
        originals = sorted(
            (row for row in images if "source_image_id" not in row),
            key=lambda row: row["file_name"],
        )
        if not originals:
            raise ValueError(f"Dataset split {split!r} contains no original images")
        count = min(examples, len(originals))
        offsets = (
            [0] if count == 1
            else [round(i * (len(originals) - 1) / (count - 1)) for i in range(count)]
        )
        selected = [originals[index] for index in offsets]
        selected_ids = {row["id"] for row in selected}
        variants: dict[NativeId, list[dict]] = {image_id: [] for image_id in selected_ids}
        for row in images:
            if row.get("source_image_id") in selected_ids:
                variants[row["source_image_id"]].append(row)

        self.sources: list[str] = []
        self.choices: list[tuple[str, str]] = []
        self._ids: dict[str, NativeId] = {}
        self._counterfactuals: set[str] = set()
        self._variants: dict[str, list[tuple[str, str]]] = {}
        self._questions: dict[str, list[str]] = {}
        keys_by_id: dict[NativeId, str] = {}
        for row in selected:
            source_key = str(len(self._ids))
            self._ids[source_key] = row["id"]
            keys_by_id[row["id"]] = source_key
            self.sources.append(source_key)
            self.choices.append((Path(row["file_name"]).name, source_key))
            choices = [("Original", source_key)]
            for variant in sorted(variants[row["id"]], key=lambda item: item["file_name"]):
                key = str(len(self._ids))
                self._ids[key] = variant["id"]
                keys_by_id[variant["id"]] = key
                self._counterfactuals.add(key)
                choices.append((Path(variant["file_name"]).stem, key))
            self._variants[source_key] = choices

        for key in self._ids:
            self.path(key)
            self._questions[key] = []
        # Exhaust the stream: the adapter checks split and family counts at EOF.
        for task in self._dataset.iter_tasks(split):
            key = keys_by_id.get(task["image_id"])
            if key is not None and task["family"] == "vqa":
                prompt = inference_task_view(self._dataset, split, task)["prompt"]
                if prompt not in self._questions[key]:
                    self._questions[key].append(prompt)

    def variants(self, source_key: str) -> list[tuple[str, str]]:
        try:
            return list(self._variants[source_key])
        except KeyError as exc:
            raise ValueError("Unknown original sample") from exc

    def path(self, key: str) -> Path:
        try:
            image_id = self._ids[key]
        except KeyError as exc:
            raise ValueError("Unknown sample") from exc
        return self._dataset.resolve_image(self._split, image_id)

    def is_counterfactual(self, key: str) -> bool:
        if key not in self._ids:
            raise ValueError("Unknown sample")
        return key in self._counterfactuals

    def questions(self, key: str) -> list[str]:
        try:
            return list(self._questions[key])
        except KeyError as exc:
            raise ValueError("Unknown sample") from exc
