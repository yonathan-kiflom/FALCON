---
library_name: transformers
license: llama2
base_model:
  - lmsys/vicuna-7b-v1.5
  - facebook/dinov2-large
tags:
  - falcon-x
  - vision-language
  - image-segmentation
  - custom_code
language:
  - en
---

# FALCON

Functional Assembly and Language for Compositional Reasoning in X-ray.

[Paper](https://arxiv.org/abs/2606.25701) ·
[Project page](https://yonathan-kiflom.github.io/FALCON/page/) ·
[Code](https://github.com/yonathan-kiflom/FALCON)

This repository contains the completed Stage-3 model, including Vicuna-7B-v1.5,
DINOv2-L/14, the trained RF-DETR segmentation detector, multimodal adapters,
and unmerged LoRA weights.

## Usage

With Python 3.10 or 3.11, install the pinned inference and evaluation dependencies:

```bash
python -m pip install 'falcon-x[train,caption] @ git+https://github.com/yonathan-kiflom/FALCON.git'
```

Use `JonathanJMK/FALCON` or a local downloaded model directory. Review the custom
code before trusting it; pin a Hub commit with `revision=` for reproducible use.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "JonathanJMK/FALCON", trust_remote_code=True,
).to("cuda").eval()

result, masks = model.predict(
    image="image.png", prompt="Describe the image.", max_new_tokens=256,
)
print(result["answer"])
```

Use `predict_segmentation` for grounding prompts and `predict_panoptic` for
panoptic prompts. Tokenization, image preprocessing, and prompt formatting are
included. The model supports one device; automatic multi-device dispatch and
quantized loading are not supported. The saved per-component precision is
preserved; do not cast the entire model to half precision.

To evaluate all tasks, also install Java and ensure `java -version` works:

```bash
python -m falcon evaluate run --model JonathanJMK/FALCON \
  --dataset /path/to/falcon-x --split test --tasks all \
  --run-dir runs/falcon-evaluation --device cuda
```

## Scope and limitations

Trained on falcon-x for X-ray descriptions, questions, component
presence/completeness, instance grounding, and segmentation. Generated answers
and masks can be wrong; this is a research model, not a certified screening
system. Available structured heads are declared in `config.json`; untrained
risk and physical-link heads are not presented as predictions. Counterfactual
completeness does not establish real-world danger or physical connectivity.

This is a model export, not an optimizer-state checkpoint. See the
[evaluation guide](https://github.com/yonathan-kiflom/FALCON/blob/main/docs/evaluation.md)
for task metrics and evaluation limitations.

## Licenses

Vicuna is derived from Llama 2 and retains the
[Llama 2 Community License](https://huggingface.co/meta-llama/Llama-2-7b/blob/main/LICENSE.txt)
and [Acceptable Use Policy](https://huggingface.co/meta-llama/Llama-2-7b/blob/main/USE_POLICY.md).
Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta
Platforms, Inc. All Rights Reserved.

[DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) and
[RF-DETR segmentation 1.5.2](https://github.com/roboflow/rf-detr/blob/1.5.2/LICENSE)
use Apache-2.0. FALCON code is Apache-2.0 (`LICENSE-code`); this does not
relicense upstream model weights. Retain `LICENSE-Llama-2`, `LICENSE-code` and
`Notice` when redistributing the full package.
