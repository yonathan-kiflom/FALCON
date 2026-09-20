"""Pinned optional reference-text metrics for captions and VQA.

Caption scores are never approximated by home-grown substitutes.  The exact
COCO backend is optional; selecting a caption or VQA family without it fails during
preflight with an actionable message.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
from contextlib import redirect_stderr, redirect_stdout
from importlib import metadata
from io import StringIO
from typing import Any

from .schemas import EvaluationError

CAPTION_BACKEND_DISTRIBUTION = "pycocoevalcap"
CAPTION_BACKEND_VERSION = "1.2"
TEXT_METRIC_BACKEND = (
    "pycocoevalcap==1.2/PTBTokenizer/Cider(sigma=6.0; labelled CIDEr-D)"
)
METEOR_TIMEOUT_SECONDS = 120.0


class CaptionMetricsUnavailable(EvaluationError):
    """The pinned caption metric implementation is unavailable."""


def caption_backend_status() -> dict[str, Any]:
    """Return a deterministic dependency preflight result."""

    try:
        installed = metadata.version(CAPTION_BACKEND_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return {
            "available": False,
            "requirement": f"{CAPTION_BACKEND_DISTRIBUTION}=={CAPTION_BACKEND_VERSION}",
            "reason": "distribution is not installed",
        }
    if installed != CAPTION_BACKEND_VERSION:
        return {
            "available": False,
            "requirement": f"{CAPTION_BACKEND_DISTRIBUTION}=={CAPTION_BACKEND_VERSION}",
            "installed": installed,
            "reason": "installed version does not match the evaluation protocol",
        }
    if shutil.which("java") is None:
        return {
            "available": False,
            "requirement": f"{CAPTION_BACKEND_DISTRIBUTION}=={CAPTION_BACKEND_VERSION} and Java",
            "installed": installed,
            "reason": "Java is required by the pinned METEOR implementation",
        }
    return {
        "available": True,
        "requirement": f"{CAPTION_BACKEND_DISTRIBUTION}=={CAPTION_BACKEND_VERSION}",
        "installed": installed,
    }


def require_caption_backend() -> None:
    status = caption_backend_status()
    if not status["available"]:
        raise CaptionMetricsUnavailable(
            "caption metrics unavailable: "
            f"{status['reason']}; install {status['requirement']}"
        )


def score_captions(
    references: dict[str, str], predictions: dict[str, str]
) -> dict[str, float]:
    """Compute the pinned text bundle for one caption or VQA family.

    Values are returned in backend-native units, without percentage scaling.
    All supplied task IDs, including empty failure hypotheses, stay in the corpus.
    """

    require_caption_backend()
    if references.keys() != predictions.keys():
        raise ValueError("caption reference and prediction IDs differ")
    if not references:
        raise ValueError("caption inputs are empty")

    # Imports are deliberately deferred: non-text scoring has no dependency
    # on the optional caption package or its Java-backed METEOR process.
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    class ManagedMeteor(Meteor):
        """Add deterministic cleanup missing from pycocoevalcap 1.2."""

        def compute_score(
            self, ground_truth: dict[str, list[str]], result: dict[str, list[str]]
        ) -> tuple[float, list[float]]:
            if ground_truth.keys() != result.keys():
                raise CaptionMetricsUnavailable(
                    "METEOR reference and prediction IDs differ"
                )
            process = self.meteor_p
            timed_out = threading.Event()

            def kill_on_timeout() -> None:
                timed_out.set()
                if process.poll() is None:
                    process.kill()

            timer = threading.Timer(METEOR_TIMEOUT_SECONDS, kill_on_timeout)
            timer.daemon = True
            timer.start()
            try:
                with self.lock:
                    if process.stdin is None or process.stdout is None:
                        raise RuntimeError("METEOR process pipes are unavailable")
                    statistics = []
                    for key in ground_truth:
                        hypothesis = (
                            result[key][0].replace("|||", "").replace("  ", " ")
                        )
                        references_for_key = " ||| ".join(ground_truth[key])
                        request = f"SCORE ||| {references_for_key} ||| {hypothesis}\n"
                        process.stdin.write(request.encode())
                        process.stdin.flush()
                        response = process.stdout.readline()
                        if not response:
                            raise RuntimeError(
                                "METEOR process closed while computing statistics"
                            )
                        statistics.append(response.decode().strip())
                    process.stdin.write(
                        (
                            "EVAL"
                            + "".join(f" ||| {value}" for value in statistics)
                            + "\n"
                        ).encode()
                    )
                    process.stdin.flush()
                    scores = []
                    for _key in ground_truth:
                        response = process.stdout.readline()
                        if not response:
                            raise RuntimeError(
                                "METEOR process closed while returning item scores"
                            )
                        scores.append(float(response.strip()))
                    response = process.stdout.readline()
                    if not response:
                        raise RuntimeError(
                            "METEOR process closed while returning corpus score"
                        )
                    return float(response.strip()), scores
            except Exception as exc:
                if timed_out.is_set():
                    raise CaptionMetricsUnavailable(
                        f"METEOR exceeded its {METEOR_TIMEOUT_SECONDS:g}-second timeout"
                    ) from exc
                if isinstance(exc, CaptionMetricsUnavailable):
                    raise
                raise CaptionMetricsUnavailable(
                    f"METEOR backend failed: {exc}"
                ) from exc
            finally:
                timer.cancel()
                timer.join()

        def close(self) -> None:
            process = getattr(self, "meteor_p", None)
            if process is None:
                return
            self.lock.acquire()
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                self.meteor_p = None
            finally:
                self.lock.release()

        def __del__(self) -> None:
            try:
                self.close()
            except Exception:
                # Destructors must never hide the scorer's real exception.
                pass

    ground_truth = {
        key: [{"image_id": key, "caption": value}] for key, value in references.items()
    }
    result = {
        key: [{"image_id": key, "caption": predictions[key]}] for key in references
    }
    # The reference package prints tokenizer and BLEU diagnostics.  Capturing
    # them keeps the command's stdout a single valid JSON document.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        tokenizer = PTBTokenizer()
        tokenized_gt = tokenizer.tokenize(ground_truth)
        tokenized_result = tokenizer.tokenize(result)

        scores: dict[str, float] = {}
        bleu, _ = Bleu(4).compute_score(tokenized_gt, tokenized_result)
        for index, value in enumerate(bleu, 1):
            scores[f"bleu_{index}"] = float(value)
        meteor = ManagedMeteor()
        try:
            value, _ = meteor.compute_score(tokenized_gt, tokenized_result)
        finally:
            meteor.close()
        scores["meteor"] = float(value)
        value, _ = Rouge().compute_score(tokenized_gt, tokenized_result)
        scores["rouge_l"] = float(value)
        value, _ = Cider().compute_score(tokenized_gt, tokenized_result)
        # pycocoevalcap's class is named ``Cider`` but version 1.2 applies the
        # Gaussian length penalty (sigma=6.0), i.e. the CIDEr-D variant.  Keep
        # that distinction explicit rather than presenting it as paper CIDEr.
        scores["cider_d"] = float(value)
    non_finite = [name for name, value in scores.items() if not math.isfinite(value)]
    if non_finite:
        raise CaptionMetricsUnavailable(
            f"caption backend returned non-finite metrics: {non_finite}"
        )
    return scores
