# Third-party sources

Dependencies are installed separately; upstream code is not vendored.

| Project | Revision used for integration | Runtime role | License |
|---|---|---|---|
| [Groma](https://github.com/FoundationVision/Groma) | `d3a95b6` | architecture and localized-tokenization reference | Apache-2.0 |
| [RF-DETR](https://github.com/roboflow/rf-detr) | tag `1.5.2` | trainable Stage-1 detector and live inference backend | Apache-2.0 |
| [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) | PyPI `1.2` | optional pinned caption metric implementations and Java tools | Upstream/component licenses; not vendored |

Model weights, datasets, and the PTBTokenizer/METEOR resources retain their
upstream licenses; Falcon's Apache-2.0 license does not relicense those assets.
