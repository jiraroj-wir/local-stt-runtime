"""Internal Python transcription runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from app.audio import validate_supported_audio_path
from app.backends import FAKE_BACKEND_NAME, transcribe_with_backend
from app.config import get_device_config
from app.output import write_all_outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    stem = args.stem or input_path.stem

    try:
        validate_supported_audio_path(input_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    config = get_device_config(args.device)
    segments = transcribe_with_backend(args.backend, input_path, config)
    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "device": config.device,
        "model": config.model,
        "backend": args.backend,
    }
    log_lines = [
        f"input_path={input_path}",
        f"output_dir={output_dir}",
        f"device={config.device}",
        f"model={config.model}",
        f"backend={args.backend}",
        f"segments={len(segments)}",
    ]

    paths = write_all_outputs(
        segments=segments,
        output_dir=output_dir,
        stem=stem,
        metadata=metadata,
        log_lines=log_lines,
    )

    print("Transcription complete")
    print(f"  Input      : {input_path}")
    print(f"  Output dir : {output_dir}")
    print(f"  Device     : {config.device}")
    print(f"  Backend    : {args.backend}")
    print(f"  Model      : {config.model}")
    print(f"  Text       : {paths['txt']}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local transcription inside the runtime.")
    parser.add_argument("input_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--stem")
    parser.add_argument("--backend", choices=[FAKE_BACKEND_NAME], default=FAKE_BACKEND_NAME)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
