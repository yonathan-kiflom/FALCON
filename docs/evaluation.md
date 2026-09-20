# Evaluation

Requires a completed, non-ablated Stage-3 model, the caption dependencies, and Java
(included in `environment.yml`). Run from the repository root:

```bash
python -m pip install -e '.[train,caption]'
```

## Run and score

For local training, [export](training.md#export) once; skip if already packaged:

```bash
python -m falcon export
```

```bash
python -m falcon evaluate preflight \
  --dataset /path/to/falcon-x --split test --tasks all \
  --model checkpoints/FALCON \
  --run-dir runs/stage3-evaluation \
  --max-new-tokens 256 --temperature 0 --seed 42
```

Preflight checks data and model artifacts without loading weights. Replace
`preflight` with `run` to generate predictions, then score without loading the model:

```bash
python -m falcon evaluate score \
  --dataset /path/to/falcon-x --split test --tasks all \
  --predictions runs/stage3-evaluation \
  --output reports/stage3-metrics.json
```

- `--tasks FAMILY,FAMILY` selects families; repeat `--annotations PATH` on every
  command for [optional labels](annotations.md).
- For held-out train data, use `--split train --partition runs/partitions.json
  --partition-role validation` (or `calibration`, never the training role).
- Hub IDs work with `--model`; pin with `--revision COMMIT`, or use
  `--local-files-only` offline. Preflight downloads uncached Hub files. Model
  loading executes custom code: trust the repository/revision. Separate
  detector/backbone/config overrides are unsupported for packaged models.
- Resume with the same command plus `--resume`; keep data, model, tasks, and
  decoding unchanged. `--retry-errors` is for resolved runtime failures, not
  incorrect answers.

## Tasks and metrics

| Task family | Metrics |
| --- | --- |
| `source_summary`, `detailed_caption`, `instance_description`, `counterfactual_caption` | BLEU-1–4, METEOR, ROUGE-L, CIDEr-D, each family separately |
| `vqa` | Text metrics and label accuracy by question type |
| `category_presence` | Accuracy, binary macro-F1; positive-F1 diagnostic |
| `missing_component_identification` | Exact-set accuracy, example-F1; micro/macro-F1 diagnostics |
| `functional_completeness` | MAE, RMSE; threshold and class-balanced diagnostics |
| `referring_expression` | Mask cIoU/mIoU and empty-target accuracy |
| `category_or_all_instance_grounding` | Mask cIoU/mIoU and category/source/counterfactual breakdowns |
| `referring_functional_grounding` | Mask cIoU/mIoU and empty-target accuracy |
| `panoptic_segmentation` | Foreground cIoU/mIoU, background mIoU, PQ/SQ/RQ |
| `referring_panoptic_segmentation` | The same panoptic metrics for the queried category |

- Text: PTB tokenizer, `pycocoevalcap==1.2`, raw units; BLEU-1 is primary.
  CIDEr-D uses `sigma=6.0`. VQA labels cover size, position, horizontal/vertical
  relation, and relative area. Unparseable/contradictory answers get no label
  credit but remain in text scoring.
- Presence scores generated answers, not the structured head; macro-F1 averages
  yes/no class F1. Missing-component answers are JSON arrays of unique category
  names; example-F1 averages per-query set F1 (empty/empty = 1).
- Completeness is 1 iff all three categories are present, not a risk/connectivity
  label. MAE/RMSE weight queries equally; balanced diagnostics weight target
  classes equally and require both classes. Classification uses threshold 0.5.
- Binary masks union instances. Positive cIoU pools intersections/unions over
  nonempty targets; positive mIoU averages their query IoUs. All-query mIoU includes
  empty targets (empty/empty = 1). Accuracy at IoU 0.5/0.75 is separate from mIoU.
- Panoptic foreground mIoU averages class IoUs per image, then images; ID 0 is
  background. PQ matches same-category segments at IoU > 0.5. PQ/SQ/RQ pool counts,
  not class scores. Paired source/counterfactual grounding requires both endpoints
  to pass; pairs do not enlarge primary denominators.

Invalid rows stay in the denominator: empty text hypotheses, zero set/classification
credit, error 1 for completeness, and zero per-query IoU. Invalid masks withhold
all-query cIoU; invalid panoptic rows withhold foreground cIoU and PQ/SQ/RQ, not
mIoU. Reports mark unavailable scores.

The test set has 1,383 complete originals and 8,298 incomplete counterfactuals:
always-incomplete accuracy is 85.7%; inspect MAE/RMSE and class-balanced results.
The dataset lacks expert risk/link and potential-component-set labels.
Empty-target/counting metrics need corresponding queries; missing support is not
zero. Task definitions and mask reductions differ from the archived paper; this
is not an exact numerical reproduction. No combined score is formed across families.

## Output files

Keep `run.json`, `predictions.jsonl`, `attempts.jsonl`, and `panoptic/` together,
outside the dataset. Predictions save after each task. Scoring requires exactly
one row per selected task, including errors; missing/extra/duplicate/foreign-split
rows are rejected. Device/memory failures stop after saving the failed task;
invalid outputs can cause nonzero exit status. Resume preserves completed rows
and recovers interrupted final writes. Existing score reports are not overwritten.
