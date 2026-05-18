"""Internal Python transcription runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from app.audio import (
    INACCURATE_DURATION_WARNING_TEXT,
    ChunkPlanItem,
    create_chunk,
    create_source_audio,
    detect_volume,
    inspect_audio,
    plan_chunks,
    should_auto_normalize,
    validate_supported_audio_path,
)
from app.backends import FAKE_BACKEND_NAME, SUPPORTED_BACKENDS, create_backend_session
from app.config import get_device_config
from app.errors import FailureReport
from app.output import write_all_outputs
from app.segments import TranscriptionSegment, offset_segments
from app.ui import (
    format_audio_info,
    format_clock,
    format_failure_summary,
    format_finish_summary,
    format_start_summary,
)

DEFAULT_CHUNK_MINUTES = 20.0
DEFAULT_PREPROCESS_MODE = "auto"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    stem = args.stem or input_path.stem

    try:
        validate_supported_audio_path(input_path)
    except ValueError as exc:
        print(
            format_failure_summary(FailureReport(stage="input validation", error=str(exc))),
            file=sys.stderr,
        )
        return 2

    config = get_device_config(args.device)
    mode = args.mode or args.device
    output_dir.mkdir(parents=True, exist_ok=True)

    original_audio_inspection = inspect_audio(input_path)
    original_audio_metadata = original_audio_inspection.metadata
    start_time = time.monotonic()
    try:
        source_preparation = _prepare_source_audio(
            input_path=input_path,
            output_dir=output_dir,
            stem=stem,
            preprocess_mode=args.preprocess,
        )
    except subprocess.CalledProcessError as exc:
        print(
            format_failure_summary(
                FailureReport(stage="audio preparation", error=_called_process_error_message(exc))
            ),
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(
            format_failure_summary(
                FailureReport(stage="audio preparation", error=f"ffmpeg failed: {exc}")
            ),
            file=sys.stderr,
        )
        return 1

    source_path = source_preparation["source_path"]
    preprocessing_applied = bool(source_preparation["preprocessing_applied"])
    source_audio_inspection = inspect_audio(source_path)
    source_audio_metadata = source_audio_inspection.metadata
    chunks = plan_chunks(
        duration_seconds=source_audio_metadata.duration_seconds,
        chunk_minutes=args.chunk_minutes,
        output_dir=output_dir,
        stem=stem,
        enabled=not args.no_chunk,
    )

    print(
        format_start_summary(
            input_path=input_path,
            mode=mode,
            selected_device=config.device,
            model=config.model,
            duration_seconds=original_audio_metadata.duration_seconds,
            audio=format_audio_info(original_audio_metadata),
            chunking=_format_chunking_summary(chunks, args.chunk_minutes, not args.no_chunk),
            preprocess=_format_preprocess_summary(args.preprocess, preprocessing_applied),
        )
    )

    try:
        if chunks:
            _create_chunks(source_path, chunks)
            segments = _transcribe_chunks(args.backend, config, chunks)
        else:
            backend_session = create_backend_session(args.backend, config)
            segments = backend_session.transcribe(source_path)
    except RuntimeError as exc:
        print(
            format_failure_summary(FailureReport(stage="transcription", error=str(exc))),
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            format_failure_summary(
                FailureReport(stage="chunking", error=_called_process_error_message(exc))
            ),
            file=sys.stderr,
        )
        return 1
    elapsed_seconds = time.monotonic() - start_time
    transcribed_duration_seconds = _transcribed_duration_seconds(segments)
    finish_audio_duration_seconds = source_audio_metadata.duration_seconds
    realtime_duration_seconds = finish_audio_duration_seconds
    if realtime_duration_seconds is None:
        realtime_duration_seconds = transcribed_duration_seconds
    realtime_factor = (
        realtime_duration_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    warnings = _warnings(
        original_audio_inspection.warning,
        source_audio_inspection.warning,
        source_preparation["warning"],
    )
    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "device": config.device,
        "model": config.model,
        "backend": args.backend,
        "input_duration_seconds": _reliable_input_duration_seconds(
            original_audio_metadata,
            original_audio_inspection.warning,
        ),
        "source_duration_seconds": source_audio_metadata.duration_seconds,
        "transcribed_duration_seconds": transcribed_duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "realtime_factor": realtime_factor,
        "audio": asdict(original_audio_metadata),
        "original_audio": asdict(original_audio_metadata),
        "source_audio": asdict(source_audio_metadata),
        "chunking_enabled": not args.no_chunk,
        "chunk_minutes": args.chunk_minutes,
        "chunks_count": len(chunks),
        "chunks": [_chunk_metadata(chunk) for chunk in chunks],
        "preprocess_mode": args.preprocess,
        "preprocessing_applied": source_preparation["preprocessing_applied"],
        "audio_preparation_applied": source_preparation["audio_preparation_applied"],
        "mean_volume_db": source_preparation["mean_volume_db"],
        "max_volume_db": source_preparation["max_volume_db"],
        "canonical_source_path": source_path.name,
        "source_path": source_path.name,
        "preprocessed_path": source_path.name if preprocessing_applied else None,
        "warnings": warnings,
    }
    log_lines = [
        f"input_path={input_path}",
        f"output_dir={output_dir}",
        f"device={config.device}",
        f"model={config.model}",
        f"backend={args.backend}",
        f"input_duration_seconds={metadata['input_duration_seconds']}",
        f"source_duration_seconds={source_audio_metadata.duration_seconds}",
        f"canonical_source_path={source_path.name}",
        f"source_path={source_path.name}",
        f"transcribed_duration_seconds={transcribed_duration_seconds}",
        f"elapsed_seconds={elapsed_seconds}",
        f"realtime_factor={realtime_factor}",
        f"chunking_enabled={not args.no_chunk}",
        f"chunk_minutes={args.chunk_minutes}",
        f"chunks_count={len(chunks)}",
        f"preprocess_mode={args.preprocess}",
        f"preprocessing_applied={source_preparation['preprocessing_applied']}",
        f"audio_preparation_applied={source_preparation['audio_preparation_applied']}",
        f"mean_volume_db={source_preparation['mean_volume_db']}",
        f"max_volume_db={source_preparation['max_volume_db']}",
        f"segments={len(segments)}",
        *[f"warning={warning}" for warning in metadata["warnings"]],
    ]

    paths = write_all_outputs(
        segments=segments,
        output_dir=output_dir,
        stem=stem,
        metadata=metadata,
        log_lines=log_lines,
    )

    print(
        format_finish_summary(
            audio_duration_seconds=finish_audio_duration_seconds,
            transcribed_duration_seconds=transcribed_duration_seconds,
            elapsed_seconds=elapsed_seconds,
            device=config.device,
            model=config.model,
            files=paths,
        )
    )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local transcription inside the runtime.")
    parser.add_argument("input_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--mode", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--stem")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=FAKE_BACKEND_NAME)
    parser.add_argument("--chunk-minutes", type=_positive_float, default=DEFAULT_CHUNK_MINUTES)
    parser.add_argument("--no-chunk", action="store_true")
    parser.add_argument(
        "--preprocess",
        choices=["auto", "off", "normalize"],
        default=DEFAULT_PREPROCESS_MODE,
    )
    return parser


def _transcribed_duration_seconds(segments) -> float:
    if not segments:
        return 0.0
    return max(segment.end for segment in segments)


def _reliable_input_duration_seconds(
    original_audio_metadata,
    original_warning: str | None,
) -> float | None:
    if _has_inaccurate_duration_warning(original_warning):
        return None
    return original_audio_metadata.duration_seconds


def _prepare_source_audio(
    *,
    input_path: Path,
    output_dir: Path,
    stem: str,
    preprocess_mode: str,
) -> dict[str, object]:
    stats = None
    warning = None
    normalize = False

    if preprocess_mode == "auto":
        stats = detect_volume(input_path)
        warning = stats.warning
        normalize = warning is None and should_auto_normalize(stats)
    elif preprocess_mode == "normalize":
        normalize = True

    source_path = output_dir / ".work" / "source" / f"{stem}.source.wav"
    create_source_audio(input_path, source_path, normalize=normalize)

    return {
        "source_path": source_path,
        "preprocessing_applied": normalize,
        "audio_preparation_applied": True,
        "mean_volume_db": stats.mean_volume_db if stats is not None else None,
        "max_volume_db": stats.max_volume_db if stats is not None else None,
        "warning": warning,
    }


def _create_chunks(input_path: Path, chunks: list[ChunkPlanItem]) -> None:
    for chunk in chunks:
        print(
            f"Chunk {chunk.index}/{len(chunks)} "
            f"offset {format_clock(chunk.offset_seconds)} "
            f"duration {format_clock(chunk.duration_seconds)}",
            flush=True,
        )
        create_chunk(input_path, chunk)


def _transcribe_chunks(
    backend: str,
    config,
    chunks: list[ChunkPlanItem],
) -> list[TranscriptionSegment]:
    backend_session = create_backend_session(backend, config)
    merged_segments: list[TranscriptionSegment] = []
    for chunk in chunks:
        chunk_segments = backend_session.transcribe(chunk.path)
        merged_segments.extend(offset_segments(chunk_segments, chunk.offset_seconds))
    return merged_segments


def _format_chunking_summary(
    chunks: list[ChunkPlanItem],
    chunk_minutes: float,
    enabled: bool,
) -> str:
    if not enabled:
        return "off"
    if chunks:
        return f"{_format_minutes(chunk_minutes)} min x {len(chunks)} chunks"
    return f"{_format_minutes(chunk_minutes)} min threshold"


def _format_minutes(minutes: float) -> str:
    if minutes.is_integer():
        return str(int(minutes))
    return f"{minutes:g}"


def _format_preprocess_summary(mode: str, applied: bool) -> str:
    if mode == "off":
        return "off"
    return f"{mode}/{'applied' if applied else 'skipped'}"


def _chunk_metadata(chunk: ChunkPlanItem) -> dict[str, object]:
    return {
        "name": chunk.path.name,
        "offset_seconds": chunk.offset_seconds,
        "duration_seconds": chunk.duration_seconds,
    }


def _warnings(*warnings: object) -> list[str]:
    unique_warnings = []
    for warning in warnings:
        if not warning:
            continue
        warning_text = str(warning)
        if warning_text not in unique_warnings:
            unique_warnings.append(warning_text)
    return unique_warnings


def _has_inaccurate_duration_warning(warning: str | None) -> bool:
    return bool(warning and INACCURATE_DURATION_WARNING_TEXT in warning)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _called_process_error_message(exc: subprocess.CalledProcessError) -> str:
    command = exc.cmd
    if isinstance(command, list):
        command_text = " ".join(str(part) for part in command)
    else:
        command_text = str(command)
    stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
    if stderr:
        return f"{command_text}: {stderr}"
    return f"{command_text} exited with status {exc.returncode}"


if __name__ == "__main__":
    raise SystemExit(main())
