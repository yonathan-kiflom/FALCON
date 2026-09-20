"""Export falcon-x annotations and copy their existing image assets."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_row(stream, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _copy_asset(source: str, split_root: Path, folder: str, name: str) -> str:
    name_path = Path(name)
    if name_path.is_absolute() or ".." in name_path.parts or not name_path.name:
        raise ValueError(f"Invalid asset name: {name!r}")
    relative = Path(folder) / name_path
    destination = split_root / relative
    if destination.exists():
        raise FileExistsError(f"Duplicate asset destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return relative.as_posix()


def _question_metadata(tasks: Path, categories: dict[int, str]) -> dict[str, dict]:
    """Keep evaluation groups without retaining generator requests or geometry copies."""
    metadata: dict[str, dict] = {}
    vqa_path = tasks / "vqa.json"
    if vqa_path.is_file():
        for row in _json(vqa_path).get("vqa", []):
            metadata[row["id"]] = {"question_type": row["family"]}
    semantic_path = tasks / "grounding_semantic.json"
    if semantic_path.is_file():
        for row in _json(semantic_path).get("grounding_semantic", []):
            if row.get("type") == "category_presence":
                metadata[row["id"]] = {"category_id": row["category_id"]}
    grounding_path = tasks / "category_grounding.json"
    if grounding_path.is_file():
        category_ids = {name: identifier for identifier, name in categories.items()}
        for row in _json(grounding_path).get("category_grounding", []):
            name = row["category"]
            if name != "all annotated objects" and name not in category_ids:
                raise ValueError(f"Unknown grounding category: {name!r}")
            metadata[row["id"]] = {"category_id": category_ids.get(name)}
    return metadata


def _compact_split(source: Path, output: Path, split: str, manifest: dict) -> dict:
    split_source = source / split
    split_output = output / split
    split_output.mkdir()
    categories: dict[int, str] = {}
    objects: dict[int, dict] = {}
    states: dict[int | str, set[int]] = {}
    counterfactuals: list[dict] = []
    source_count = 0

    with (split_output / "images.jsonl").open("w", encoding="utf-8") as images_file, (
        split_output / "objects.jsonl"
    ).open("w", encoding="utf-8") as objects_file:
        for record in _rows(split_source / "canonical_bundle.jsonl"):
            job = record["job"]
            image_id = record["image_id"]
            if record["split"] != split or image_id in states:
                raise ValueError(f"Duplicate image or incorrect split: {record['key']}")
            current_categories = {row["id"]: row["name"] for row in job["categories"]}
            if categories and categories != current_categories:
                raise ValueError("Category mapping changed within a split")
            categories = current_categories
            if job.get("kind") == "counterfactual":
                # Resolve after originals, even if a resumable snapshot delivered
                # counterfactual completions before their source completions.
                counterfactuals.append({"image_id": image_id, "job": job})
                continue
            image = {
                "id": image_id,
                "file_name": _copy_asset(
                    job["image_path"], split_output, "images", job["file_name"]
                ),
                "width": job["width"],
                "height": job["height"],
            }
            panoptic = job.get("panoptic")
            if panoptic and panoptic.get("image_path"):
                image["panoptic"] = {
                    "file_name": _copy_asset(
                        panoptic["image_path"], split_output, "panoptic",
                        Path(panoptic["image_path"]).name,
                    ),
                    # Panoptic areas/boxes can differ from overlapping COCO masks.
                    "segments_info": panoptic["segments_info"],
                }
            present = set()
            for obj in job["objects"]:
                identifier = obj["annotation_id"]
                if identifier in objects:
                    raise ValueError(f"Duplicate object ID within {split}: {identifier}")
                compact = {
                    "id": identifier, "image_id": image_id,
                    "category_id": obj["category_id"], "bbox": obj["bbox"],
                    "area": obj["area"], "segmentation": obj["segmentation"],
                }
                objects[identifier] = compact
                present.add(identifier)
                _write_row(objects_file, compact)
            states[image_id] = present
            _write_row(images_file, image)
            source_count += 1

            variant = job.get("variant")
            if variant:
                variant_id = f"{record['key']}:variant:{variant['variant_id']}"
                remaining = set(variant["present_annotation_ids"])
                removed = set(variant["removed_annotation_ids"])
                if remaining & removed or remaining | removed != present:
                    raise ValueError(f"Invalid counterfactual membership for {variant_id}")
                if variant_id in states:
                    raise ValueError(f"Duplicate variant ID: {variant_id}")
                states[variant_id] = remaining
                _write_row(images_file, {
                    "id": variant_id,
                    "file_name": _copy_asset(
                        variant["image_path"], split_output, "counterfactual_images",
                        variant["file_name"],
                    ),
                    "width": variant["width"], "height": variant["height"],
                    "source_image_id": image_id,
                    "present_annotation_ids": variant["present_annotation_ids"],
                })

        for record in counterfactuals:
            job = record["job"]
            image_id = record["image_id"]
            source_id = job["source_image_id"]
            variant = job["variant"]
            if image_id in states:
                raise ValueError(f"Duplicate counterfactual ID: {image_id}")
            if source_id not in states:
                raise ValueError(
                    f"Counterfactual source {source_id} is not exported yet; "
                    "export the completed snapshot"
                )
            remaining = set(variant["present_annotation_ids"])
            removed = set(variant["removed_annotation_ids"])
            if remaining & removed or remaining | removed != states[source_id]:
                raise ValueError(f"Invalid counterfactual membership for {image_id}")
            if {obj["annotation_id"] for obj in job["objects"]} != remaining:
                raise ValueError(f"Counterfactual objects do not match membership: {image_id}")
            for obj in job["objects"]:
                original = objects[obj["annotation_id"]]
                if original["image_id"] != source_id or any(
                    obj[field] != original[field]
                    for field in ("category_id", "bbox", "area", "segmentation")
                ):
                    raise ValueError(f"Conflicting counterfactual geometry: {image_id}")
            states[image_id] = remaining
            _write_row(images_file, {
                "id": image_id,
                "file_name": _copy_asset(
                    job["image_path"], split_output, "counterfactual_images", job["file_name"],
                ),
                "width": job["width"], "height": job["height"],
                "source_image_id": source_id,
                "present_annotation_ids": variant["present_annotation_ids"],
            })

    category_list = [{"id": key, "name": value} for key, value in sorted(categories.items())]
    if "categories" in manifest and manifest["categories"] != category_list:
        raise ValueError("Category mapping differs between train and test")
    manifest["categories"] = category_list
    metadata = _question_metadata(split_source / "tasks", categories)
    counts: Counter[str] = Counter()
    task_ids: set[str] = set()
    with (split_output / "tasks.jsonl").open("w", encoding="utf-8") as tasks_file:
        for row in _json(split_source / "tasks" / "llava.json"):
            if row["id"] in task_ids:
                raise ValueError(f"Duplicate task ID: {row['id']}")
            task_ids.add(row["id"])
            if row["image_id"] not in states:
                raise ValueError(f"Task references an unknown image: {row['id']}")
            turns = row["conversations"]
            if len(turns) != 2 or [turn["from"] for turn in turns] != ["human", "gpt"]:
                raise ValueError(f"Expected one prompt/answer pair: {row['id']}")
            compact = {
                "id": row["id"], "family": row["task"], "image_id": row["image_id"],
                "prompt": turns[0]["value"].removeprefix("<image>\n"),
                "answer": turns[1]["value"],
            }
            for original, destination in (
                ("grounding", "target_annotation_ids"),
                ("context_grounding", "context_annotation_ids"),
            ):
                if original not in row:
                    continue
                ids = []
                for target in row[original]:
                    identifier = target["annotation_id"]
                    if identifier not in states[row["image_id"]]:
                        raise ValueError(f"Task targets an absent/unknown object: {row['id']}")
                    obj = objects[identifier]
                    if any(target[key] != obj[key] for key in (
                        "category_id", "bbox", "area", "segmentation"
                    )):
                        raise ValueError(f"Conflicting task geometry: {row['id']}")
                    ids.append(identifier)
                compact[destination] = ids
            compact.update(metadata.get(row["id"], {}))
            counts[row["task"]] += 1
            _write_row(tasks_file, compact)

    mix = split_source / "tasks" / "derived" / "falcon_training_mix.json"
    if split == "train" and mix.is_file():
        manifest["train_family_weights"] = _json(mix)["target_percentages"]
    return {
        "usage": "training" if split == "train" else "evaluation",
        "images": f"{split}/images.jsonl", "objects": f"{split}/objects.jsonl",
        "tasks": f"{split}/tasks.jsonl", "source_images": source_count,
        "images_count": len(states), "objects_count": len(objects),
        "tasks_count": sum(counts.values()), "families": dict(sorted(counts.items())),
    }


def compact_dataset(source: Path, output: Path) -> dict:
    """Create an independent, portable dataset. Never overwrite an existing destination."""
    source, output = Path(source).resolve(), Path(output).absolute()
    splits = [split for split in ("train", "test") if (source / split).is_dir()]
    if not splits:
        raise ValueError("Source must be a direct-full-v1/v2 output folder containing train/test")
    for split in splits:
        for relative in ("canonical_bundle.jsonl", "tasks/llava.json"):
            if not (source / split / relative).is_file():
                raise FileNotFoundError(source / split / relative)
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "format": "falcon-x", "bbox_format": "xywh_pixels",
        "segmentation_format": "COCO", "splits": {},
    }
    for split in splits:
        manifest["splits"][split] = _compact_split(source, output, split, manifest)
    (output / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(DATASET_README, encoding="utf-8")
    return manifest


DATASET_README = """# falcon-x

Original and counterfactual images with Gemma/LLaVA task annotations.

## Files

- `dataset.json`: categories, coordinate conventions, split counts, task families,
  and training-family sampling weights.
- `{split}/images.jsonl`: image IDs, dimensions and paths relative to that split.
  Counterfactuals additionally identify their source image and remaining objects.
- `{split}/objects.jsonl`: each source object's box, area and COCO mask, stored once.
- `{split}/tasks.jsonl`: final prompts/answers and object IDs, with no repeated masks.
- `{split}/images/`, `counterfactual_images/`, `panoptic/`: copied original assets.

Each JSONL line is an ordinary JSON object. IDs are scoped to a split; never join
train and test on an object ID alone. Test tasks are evaluation-only.

## Resolving targets

Find an image by `id`. Its `file_name` is relative to its split directory. Original
objects are the rows with matching `image_id` in `objects.jsonl`. Counterfactuals
reuse their source objects' geometry and list only `present_annotation_ids`;
removed IDs are the source object's ID set minus this list.

Task `target_annotation_ids` specify supervised mask targets; combine those masks
by pixelwise union. An empty list means no target. `context_annotation_ids` instead
identify objects referred to by a question or description, not segmentation labels.
Panoptic tasks use their image's `panoptic.file_name` and original `segments_info`.
Panoptic geometry can differ from overlapping instance masks and is kept intact.
VQA `question_type` preserves the original question group. For category grounding,
`category_id: null` means all annotated objects. Prompt text omits only the LLaVA
`<image>` transport marker. Answers, including `<SEG>` and `<PANOPTIC>`, are unchanged.

## Export

From the FALCON repository (Python 3.10+):

```bash
python -m falcon.compact --source /path/to/annotation_output \\
  --output /path/to/falcon-x
```

The destination must not exist. Images and annotations are copied unchanged from
a completed export; original inputs are never modified.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed direct export folder")
    parser.add_argument("--output", type=Path, required=True, help="New falcon-x dataset folder")
    args = parser.parse_args()
    manifest = compact_dataset(args.source, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
