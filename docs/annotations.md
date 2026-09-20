# Optional annotations

Add expert risk/link labels or reviewed panoptic metadata without replacing
images. Each package contains `annotations.json` and `records.jsonl`.

## Manifest

Build the manifest using the target dataset, then save it as `annotations.json`:

```python
from falcon.data import open_dataset

dataset = open_dataset("/path/to/falcon-x")
manifest = {
    "format": "falcon-x-annotations-v1",
    "kind": "safety_labels",
    "dataset": dataset.metadata_summary(),
    "records": "records.jsonl",
    "records_count": 1,
    "provenance": {
        "authority": "expert-reviewed",
        "source": "review batch identifier",
        "version": "1",
    },
}
```

Use `kind: safety_labels` or `panoptic_metadata`, at most one package per kind.
`records_count` equals nonblank JSONL rows. Files must be regular, not symlinks;
record paths stay inside the package.

## Safety labels

Each row contains exactly these fields:

```json
{"split":"train","image_id":1,"risk":0.75,"links":[null,0.4,null]}
```

Values are `null` or finite numbers in `[0,1]`; at least one must be non-null.
Link order: battery–detonator, battery–explosive, detonator–explosive. Missing
labels stay unavailable. Do not add presence: it comes from object membership.

## Panoptic metadata

Each row contains exactly these fields:

```json
{
  "split": "train",
  "image_id": 1,
  "segments_info": [
    {"id": 7, "category_id": 1, "iscrowd": 0, "bbox": [10, 20, 30, 40], "area": 900}
  ]
}
```

Source images only, with an existing panoptic PNG. Only `segments_info` changes:
IDs, areas, and boxes must match the PNG. Segment IDs are positive; categories
are dataset-local; boxes use pixel `xywh`; background is 0. Evaluation requires
`iscrowd: 0`. Do not infer categories from object order.

## Attach annotations

Add repeated `--annotations PATH` to training/evaluation, or pass
`annotations=["/labels/safety", "/labels/panoptic"]` to `open_dataset()`.
Paths accept directories or `annotations.json` files. Within each package,
`(split, image_id)` must be unique; integer and string IDs are distinct.
