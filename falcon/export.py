"""Export the completed model as a self-contained Transformers repository."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from importlib.metadata import distribution
from pathlib import Path


def export_model(args: argparse.Namespace) -> dict:
    import torch

    from .config import resolve_stage_config
    from .hf_support import validate_model_package
    from .infer import _load_model, verify_final_stage3_evaluation
    from .modeling_falcon import FalconModel

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    readiness = verify_final_stage3_evaluation(
        checkpoint=args.checkpoint,
        detector_weights=args.detector_weights,
        vision_model=args.vision_model,
        language_model=args.language_model,
    )
    for source in (args.vision_model, args.language_model):
        root = Path(source).expanduser().resolve(strict=True)
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Export must be outside the source backbone directories")
    config = readiness["config"]
    precision = resolve_stage_config(config, 3)["precision"]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    loading = argparse.Namespace(
        checkpoint=args.checkpoint,
        detector_weights=args.detector_weights,
        vision_model=args.vision_model,
        language_model=args.language_model,
        device=args.device,
        language_dtype=dtype,
        ssa_ablation="none",
        oracle_detector="none",
        allow_partial_checkpoint=False,
        final_stage3_readiness=readiness,
    )
    print("Loading completed Stage-3 model and detector", flush=True)
    core = _load_model(loading, None)
    model = FalconModel.from_falcon(core)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-export-", dir=output.parent))
    try:
        print(f"Saving self-contained model to {output}", flush=True)
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size=args.max_shard_size)
        source_root = Path(__file__).resolve().parent
        shutil.copyfile(source_root / "model_card.md", temporary / "README.md")
        code_license = source_root.parent / "LICENSE"
        if not code_license.is_file():
            installed = distribution("falcon-x")
            code_license = next(
                (Path(installed.locate_file(item)) for item in installed.files or []
                 if item.name == "LICENSE" and ".dist-info" in str(item)),
                code_license,
            )
        shutil.copyfile(code_license, temporary / "LICENSE-code")
        for notice in (source_root / "licenses").iterdir():
            if notice.is_file():
                shutil.copyfile(notice, temporary / notice.name)
        inspection = validate_model_package(temporary)
        # The final path appears only after every shard and required asset exists.
        if output.exists():
            raise FileExistsError(f"Export destination appeared while saving: {output}")
        os.rename(temporary, output)
    except Exception as error:
        raise RuntimeError(f"Export did not finish; partial files retained at {temporary}: {error}") from error
    return {
        "model": str(output),
        "model_type": "falcon_x",
        "stage": 3,
        "training_complete": True,
        "tensor_count": inspection["tensor_count"],
        "weight_shards": len(inspection["weight_files"]),
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/stage3/last.pt")
    parser.add_argument("--detector-weights", default="checkpoints/stage1/last.pth")
    parser.add_argument("--vision-model", default="checkpoints/dinov2-large")
    parser.add_argument("--language-model", default="checkpoints/vicuna-7b-v1.5")
    parser.add_argument("--output", default="checkpoints/FALCON")
    parser.add_argument("--device", default="cpu", help="Device used while assembling weights (default: cpu)")
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()
    print(json.dumps(export_model(args), indent=2))


if __name__ == "__main__":
    main()
