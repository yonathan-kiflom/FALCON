# Local demo

The demo uses a local Falcon model and one CUDA GPU.

## Setup

From the repository root, with your existing FALCON inference environment
activated:

```bash
FALCON_PYTHON="$(command -v python)"
"$FALCON_PYTHON" -m venv .venv-demo
.venv-demo/bin/python -m pip install -e '.[demo]'

.venv-demo/bin/python -m falcon demo \
  --runtime-python "$FALCON_PYTHON" \
  --model checkpoints/FALCON \
  --dataset /path/to/falcon-x
```

Open **http://127.0.0.1:7860**. The interface becomes ready after the model loads.
Use Ctrl+C in the terminal to stop the demo and release the GPU.

Gradio runs in `.venv-demo`; model inference runs in your existing environment.
Do not install `.[train,demo]` together or upgrade the inference dependencies for
the demo. Model files must already be downloaded; startup does not fetch weights.
