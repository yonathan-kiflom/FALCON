"""Private, single-worker bridge to the existing inference environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import uuid
from typing import Any


class ModelClient:
    def __init__(self, runtime_python: Path, model: Path, device: str = "cuda:0",
                 *, startup_timeout: float = 600, prediction_timeout: float = 300):
        # Resolving a venv's Python symlink would bypass its site-packages.
        runtime_python = runtime_python.expanduser().absolute()
        model = model.expanduser().resolve()
        if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
            raise ValueError(f"Inference Python is not executable: {runtime_python}")
        if not model.is_dir() or not (model / "config.json").is_file():
            raise ValueError(f"Expected a local FALCON export with config.json: {model}")
        self._temporary = tempfile.TemporaryDirectory(prefix="falcon-demo-")
        self.workspace = Path(self._temporary.name)
        self._state_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._ready = threading.Event()
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._state = "loading"
        self._busy = False
        self._message = "Loading FALCON…"
        self._metadata: dict[str, Any] = {}
        self._prediction_timeout = prediction_timeout
        self._process: subprocess.Popen[str] | None = None
        environment = os.environ.copy()
        environment.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
        )
        # Never put the UI environment's site-packages on the worker's path.
        environment.pop("PYTHONPATH", None)
        try:
            self._process = subprocess.Popen(
                [str(runtime_python), "-u", str(Path(__file__).with_name("worker.py")),
                 "--model", str(model), "--workspace", str(self.workspace),
                 "--device", device],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                encoding="utf-8", bufsize=1, env=environment, start_new_session=True,
            )
        except OSError:
            self._temporary.cleanup()
            raise
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        threading.Thread(target=self._watch_startup, args=(startup_timeout,), daemon=True).start()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            if self._state == "ready" and self._busy:
                return {"state": "running", "message": "Running FALCON…",
                        "metadata": dict(self._metadata)}
            return {"state": self._state, "message": self._message,
                    "metadata": dict(self._metadata)}

    def _fail(self, message: str) -> None:
        with self._state_lock:
            if self._state in ("closed", "failed"):
                return
            self._state, self._message = "failed", message
        self._ready.set()
        self._responses.put({"status": "error", "error": message, "fatal": True})

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("worker response is not an object")
                except (ValueError, TypeError):
                    self._fail("Invalid response from the inference worker. Restart the demo.")
                    self._stop_process()
                    return
                if message.get("event") == "ready":
                    with self._state_lock:
                        if self._state != "loading":
                            continue
                        self._metadata = message.get("metadata", {})
                        self._state, self._message = "ready", "FALCON is ready."
                    self._ready.set()
                elif message.get("event") == "fatal":
                    self._fail(message.get("error", "Model loading failed."))
                    self._stop_process()
                    return
                else:
                    if message.get("fatal"):
                        self._fail(message.get("error", "Inference worker failed."))
                        self._stop_process()
                        return
                    else:
                        self._responses.put(message)
        except (OSError, ValueError):
            self._fail("Lost connection to the inference worker. Restart the demo.")
        finally:
            self._fail("Inference worker stopped. Check the terminal and restart the demo.")

    def _watch_startup(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            self._fail("Model loading timed out. Check the terminal and restart the demo.")
            self._stop_process()

    def predict(self, image: Any, prompt: str, mode: str = "text", *,
                max_new_tokens: int = 128, query_category_id: int | None = None) -> dict[str, Any]:
        if mode not in ("text", "segmentation", "panoptic"):
            raise ValueError("Unsupported prediction mode")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a prompt first.")
        if len(prompt) > 32_768:
            raise ValueError("The prompt is too long. Shorten it before retrying.")
        if max_new_tokens not in (64, 128, 256):
            raise ValueError("Choose 64, 128, or 256 generated tokens.")
        with self._request_lock:
            state = self.status()
            if state["state"] != "ready":
                raise RuntimeError(state["message"])
            width, height = image.size
            if width < 1 or height < 1 or width * height > 16_000_000:
                raise ValueError("Images must contain at most 16 megapixels.")
            request_id = uuid.uuid4().hex
            request_dir = self.workspace / request_id
            request_dir.mkdir(mode=0o700)
            image_path = request_dir / "input.png"
            image.convert("RGB").save(image_path)
            request = {"id": request_id, "image": f"{request_id}/input.png",
                       "mode": mode, "prompt": prompt.strip(),
                       "max_new_tokens": max_new_tokens}
            if query_category_id is not None:
                request["query_category_id"] = query_category_id
            message = json.dumps(request, allow_nan=False, ensure_ascii=False) + "\n"
            if len(message) > 65_536:
                raise ValueError("The prompt is too long. Shorten it before retrying.")
            assert self._process is not None and self._process.stdin is not None
            with self._state_lock:
                self._busy = True
            try:
                self._process.stdin.write(message)
                self._process.stdin.flush()
                response = self._responses.get(timeout=self._prediction_timeout)
            except queue.Empty as exc:
                message = "Prediction timed out. Restart the demo before trying again."
                self._fail(message)
                self._stop_process()
                raise RuntimeError(message) from exc
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._fail("Inference worker is unavailable. Restart the demo.")
                raise RuntimeError(self.status()["message"]) from exc
            finally:
                with self._state_lock:
                    self._busy = False
            if response.get("fatal"):
                raise RuntimeError(response.get("error", "Inference worker failed."))
            if response.get("id") != request_id:
                self._fail("Inference response did not match the request. Restart the demo.")
                self._stop_process()
                raise RuntimeError(self.status()["message"])
            if response.get("status") != "ok":
                raise RuntimeError(response.get("error", "Prediction failed."))
            return response

    def _stop_process(self) -> None:
        with self._process_lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        with self._state_lock:
            if self._state == "closed":
                return
            self._state, self._message = "closed", "Demo stopped."
        self._ready.set()
        self._responses.put({"status": "error", "error": "Demo stopped.", "fatal": True})
        self._stop_process()
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=5)
        if self._process is not None:
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    stream.close()
        with self._request_lock:
            self._temporary.cleanup()
