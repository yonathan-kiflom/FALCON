"""Local browser demo with an isolated inference worker."""

from __future__ import annotations


def main() -> None:
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True,
                        help="Python executable in the FALCON inference environment")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/FALCON"),
                        help="Local completed Stage-3 export")
    parser.add_argument("--dataset", type=Path, help="Local falcon-x dataset (optional)")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.device != "cuda" and not (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        parser.error("--device must be cuda or cuda:N")

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    from .catalog import Catalog
    from .client import ModelClient

    try:
        catalog = Catalog(args.dataset, args.split, args.examples) if args.dataset else None
        client = ModelClient(args.runtime_python, args.model, args.device)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    os.environ["GRADIO_TEMP_DIR"] = str(client.workspace / "gradio")
    try:
        from .app import build_app

        app = build_app(client, catalog)
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False,
                   allowed_paths=[], max_file_size=20 * 1024 * 1024,
                   blocked_paths=[str(args.model.expanduser().resolve())],
                   show_error=False, inbrowser=False, footer_links=[], run_history=False)
    except ImportError as exc:
        raise SystemExit("Install '.[demo]' in a separate UI environment; see docs/demo.md") from exc
    finally:
        client.close()
