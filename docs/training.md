# Training

Run from the repository root.

## Setup

```bash
conda env create --file environment.yml
conda activate falcon
python -m pip check
```

Place RF-DETR segmentation weights in `checkpoints/pretrained/`, DINOv2-L/14 in
`checkpoints/dinov2-large/`, and Vicuna-7B-v1.5 in `checkpoints/vicuna-7b-v1.5/`.
Model and dataset licenses apply separately.

## Dataset

Use the prepared falcon-x dataset (`dataset.json` and split JSONL tables/assets).
For a native falcon-x dataset missing the component tasks, prepare a new directory:

```bash
python -m falcon prepare tasks --data-dir /path/to/existing_dataset \
  --output-dir /path/to/falcon-x
```

Validate the prepared dataset:

```bash
python -m falcon evaluate validate --dataset /path/to/falcon-x \
  --split train --tasks all --full
```

Create one train-only partition for all stages:

```bash
python -m falcon prepare partitions --data-dir /path/to/falcon-x \
  --output runs/partitions.json --validation-percent 5 \
  --calibration-percent 5 --seed 42

python -m falcon prepare detector --data-dir /path/to/falcon-x \
  --output-dir runs/detector_data --partition runs/partitions.json \
  --image-scope all --link-mode copy
```

Originals and their counterfactuals stay together; test images are excluded.
Keep outputs/caches outside the dataset. Add repeated `--annotations PATH` to
Stages 2 and 3 for [optional labels](annotations.md); absent labels are unsupervised.

## Stage 1: detector

```bash
python -m falcon detector --config configs/falcon.yaml \
  --data-dir runs/detector_data --output-dir checkpoints/stage1 \
  --weights checkpoints/pretrained/rf-detr-seg-xxlarge.pt
```

Default: RF-DETR 1.5.2 `seg-2xlarge`, resolution 768. Resume with
`--resume checkpoints/stage1/last.pth`.

## Stage 2: multimodal adapters

```bash
python -m falcon train --stage 2 --config configs/falcon.yaml \
  --data-dir /path/to/falcon-x --partition runs/partitions.json --tasks all \
  --detector-weights checkpoints/stage1/last.pth \
  --vision-model checkpoints/dinov2-large \
  --language-model checkpoints/vicuna-7b-v1.5 \
  --output-dir checkpoints/stage2
```

Trains adapters and supervised safety heads; detector and backbones stay frozen.

## Stage 3: LoRA fine-tuning

```bash
python -m falcon train --stage 3 --config configs/falcon.yaml \
  --data-dir /path/to/falcon-x --partition runs/partitions.json --tasks all \
  --detector-weights checkpoints/stage1/last.pth \
  --vision-model checkpoints/dinov2-large \
  --language-model checkpoints/vicuna-7b-v1.5 \
  --init-checkpoint checkpoints/stage2/last.pt \
  --output-dir checkpoints/stage3
```

Requires completed Stage-2 weights and unchanged data, annotations, partition,
and model settings. `--allow-partial-init` is diagnostic-only. Defaults: BF16,
activation checkpointing, no language-model KV cache.

## Checkpoints and training options

- Use empty output directories for fresh runs; concurrent writers are rejected.
- Stage 1 keeps `last.pth`; Stages 2/3 replace `last.pt` every `--save-every`
  optimizer updates (default 1000) and on completion. Frozen backbones stay separate.
- Resume Stages 2/3 with `--resume checkpoints/stageN/last.pt`, keeping training
  settings, data, model files, and distributed world size unchanged.
- `--feature-cache runs/frozen_features` caches frozen detector/DINO features.
  Change cache directories after changing model files or preprocessing.
- Grounding uses detector proposals; unmatched positives skip language loss but
  retain available structured supervision.
- `sampling: all` covers every selected task. Increase `max_text_tokens` on overflow.
- Independent RF-DETR/DINO backbones and float32 presence BCE-with-logits differ
  from the paper's shared backbone/L1 recipe; risk/link losses use L1.

## Export

```bash
python -m falcon export
```

Packages completed Stage-3 weights, detector, backbones, tokenizer, and preprocessing
into `checkpoints/FALCON`; no retraining or upload. The destination must not exist
(`--output PATH` to change it). Continue with [inference](../README.md#weights) or
[evaluation](evaluation.md).
