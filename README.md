<h1>
  <img src="page/assets/falcon-logo.png" alt="FALCON logo" width="52" align="absmiddle">
  FALCON: Functional Assembly and Language for Compositional Reasoning in X-ray [ECCV 2026]
</h1>

---

[Yonathan Michael](https://scholar.google.com/citations?user=1NgtYpwAAAAJ&hl=en) *, [Mohamad Alansari](https://scholar.google.com/citations?user=dLQ1jLkAAAAJ&hl=en) *, [Natnael Takele](https://scholar.google.com/citations?user=dVtaIkAAAAAJ&hl=en), [Andreas Henschel](https://scholar.google.com/citations?user=jenl24IAAAAJ&hl=en&oi=ao), [Naoufel Werghi](https://scholar.google.com/citations?user=G_2Xpm0AAAAJ&hl=en)

**Khalifa University, Abu Dhabi, UAE**, * Equal Contribution

<p align="center">
  <a href="https://arxiv.org/abs/2606.25701">📄 Paper</a>
  &nbsp;|&nbsp;
  <a href="https://yonathan-kiflom.github.io/FALCON/page/">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/JonathanJMK/FALCON"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="20" height="20" align="absmiddle"> Weights</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/JonathanJMK/falcon-x">&#128202; Benchmark</a>
</p>

---

## 📰 News

- [x] FALCON [local demo](docs/demo.md) is now available!
- [x] FALCON code is now available!
- [x] FALCON project page is now live!
- [x] :tada: FALCON is accepted at ECCV 2026!

## Code

Requires Python 3.10–3.11. Caption metrics also require Java.

```bash
python -m pip install -e '.[train,caption]'
python -m falcon --help
```

See [training](docs/training.md), [evaluation](docs/evaluation.md), and
[optional annotations](docs/annotations.md).

## Weights

Download the complete model from [JonathanJMK/FALCON](https://huggingface.co/JonathanJMK/FALCON)
on Hugging Face:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "JonathanJMK/FALCON", trust_remote_code=True,
).to("cuda").eval()
result, masks = model.predict(image="image.png", prompt="Describe the image.")
print(result["answer"])
```


## Local demo

Explore images, ask questions, visualize segmentation, and compare existing
counterfactuals in a local browser. See [demo setup](docs/demo.md).

```bash
.venv-demo/bin/python -m falcon demo \
  --runtime-python /path/to/falcon-environment/bin/python \
  --model checkpoints/FALCON --dataset /path/to/falcon-x
```

## Benchmark

Download [falcon-x on Hugging Face](https://huggingface.co/datasets/JonathanJMK/falcon-x).

## Citation

```bibtex
@inproceedings{michael2026falcon,
    title={FALCON: Functional Assembly and Language for Compositional Reasoning in X-ray},
    author={Michael, Yonathan and Alansari, Mohamad and Takele, Natnael and Henschel, Andreas and Werghi, Naoufel},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2026}
}
```
