"""Internal Python transcription runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
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
FALLBACK_CHUNK_SECONDS = 5 * 60.0
MODEL_ALIASES = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium.en": "Systran/faster-whisper-medium.en",
}


@dataclass
class IsolatedTranscriptionResult:
    segments: list[TranscriptionSegment]
    child_output_dirs: list[Path]
    child_processes_count: int
    fallback_events: list[dict[str, object]]


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
    if args.model:
        config = replace(config, model=_resolve_model(args.model))
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

    isolated_chunks = bool(chunks and args.isolate_chunks)
    chunk_output_dirs = (
        _isolated_chunk_output_dirs(output_dir, chunks) if isolated_chunks else []
    )
    child_processes_count = len(chunk_output_dirs)
    fallback_events: list[dict[str, object]] = []

    try:
        if chunks:
            if args.adaptive_fallback and not args.isolate_chunks:
                raise RuntimeError(
                    "--adaptive-fallback requires --isolate-chunks when chunking is active"
                )
            _create_chunks(source_path, chunks, isolated=isolated_chunks)
            if isolated_chunks:
                isolated_result = _transcribe_chunks_isolated(
                    backend=args.backend,
                    config=config,
                    chunks=chunks,
                    output_dir=output_dir,
                    mode=mode,
                    adaptive_fallback=args.adaptive_fallback,
                )
                segments = isolated_result.segments
                chunk_output_dirs = isolated_result.child_output_dirs
                child_processes_count = isolated_result.child_processes_count
                fallback_events = isolated_result.fallback_events
            else:
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
        "isolated_chunks": isolated_chunks,
        "chunk_output_dirs": [
            _relative_output_path(path, output_dir) for path in chunk_output_dirs
        ],
        "child_processes_count": child_processes_count,
        "adaptive_fallback": args.adaptive_fallback,
        "fallback_events": fallback_events,
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
        f"isolated_chunks={str(isolated_chunks).lower()}",
        f"child_processes_count={child_processes_count}",
        f"adaptive_fallback={str(args.adaptive_fallback).lower()}",
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
    parser.add_argument("--isolate-chunks", action="store_true")
    parser.add_argument("--adaptive-fallback", action="store_true")
    parser.add_argument("--model")
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


def _create_chunks(
    input_path: Path,
    chunks: list[ChunkPlanItem],
    *,
    isolated: bool = False,
) -> None:
    for chunk in chunks:
        suffix = " isolated" if isolated else ""
        print(
            f"Chunk {chunk.index}/{len(chunks)} "
            f"offset {format_clock(chunk.offset_seconds)} "
            f"duration {format_clock(chunk.duration_seconds)}"
            f"{suffix}",
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


def _transcribe_chunks_isolated(
    *,
    backend: str,
    config,
    chunks: list[ChunkPlanItem],
    output_dir: Path,
    mode: str,
    adaptive_fallback: bool,
) -> IsolatedTranscriptionResult:
    merged_segments: list[TranscriptionSegment] = []
    child_output_dirs: list[Path] = []
    child_processes_count = 0
    fallback_events: list[dict[str, object]] = []
    for chunk in chunks:
        child_output_dir = _isolated_chunk_output_dir(output_dir, chunk)
        command = _build_isolated_chunk_command(
            chunk=chunk,
            child_output_dir=child_output_dir,
            backend=backend,
            device=config.device,
            mode=_child_process_mode(mode, config.device),
            model=config.model,
        )
        child_output_dir.mkdir(parents=True, exist_ok=True)
        result = _run_isolated_chunk_command(command=command, chunk=chunk)
        child_processes_count += 1
        child_output_dirs.append(child_output_dir)

        if result.returncode != 0:
            if adaptive_fallback and config.device == "cuda" and _is_cuda_oom(result):
                fallback_result = _transcribe_chunk_with_adaptive_fallback(
                    backend=backend,
                    chunk=chunk,
                    output_dir=output_dir,
                    mode=mode,
                    fallback_events=fallback_events,
                )
                merged_segments.extend(fallback_result.segments)
                child_output_dirs.extend(fallback_result.child_output_dirs)
                child_processes_count += fallback_result.child_processes_count
                continue
            raise RuntimeError(
                _isolated_chunk_failure_message(
                    chunk=chunk,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )

        chunk_segments = _read_chunk_segments(child_output_dir, chunk.path.stem)
        merged_segments.extend(offset_segments(chunk_segments, chunk.offset_seconds))
    return IsolatedTranscriptionResult(
        segments=merged_segments,
        child_output_dirs=child_output_dirs,
        child_processes_count=child_processes_count,
        fallback_events=fallback_events,
    )


def _build_isolated_chunk_command(
    *,
    chunk: ChunkPlanItem,
    child_output_dir: Path,
    backend: str,
    device: str,
    mode: str,
    model: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "app.transcribe",
        str(chunk.path),
        "--output-dir",
        str(child_output_dir),
        "--device",
        device,
        "--backend",
        backend,
        "--no-chunk",
        "--preprocess",
        "off",
        "--stem",
        chunk.path.stem,
        "--mode",
        mode,
    ]
    if model:
        command.extend(["--model", model])
    return command


def _run_isolated_chunk_command(
    *,
    command: list[str],
    chunk: ChunkPlanItem,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            _isolated_chunk_failure_message(
                chunk=chunk,
                returncode=None,
                stdout="",
                stderr=f"failed to start child process: {exc}",
            )
        ) from exc


def _transcribe_chunk_with_adaptive_fallback(
    *,
    backend: str,
    chunk: ChunkPlanItem,
    output_dir: Path,
    mode: str,
    fallback_events: list[dict[str, object]],
) -> IsolatedTranscriptionResult:
    print(
        f"Chunk {chunk.index} CUDA OOM; retrying as 5 min subchunks",
        flush=True,
    )
    fallback_events.append(
        _fallback_event(
            chunk=chunk,
            action="split",
            model="Systran/faster-whisper-large-v3",
            device="cuda",
            reason="cuda_oom",
        )
    )

    subchunks = _fallback_subchunks(output_dir=output_dir, chunk=chunk)
    for subchunk in subchunks:
        create_chunk(chunk.path, subchunk)

    merged_segments: list[TranscriptionSegment] = []
    child_output_dirs: list[Path] = []
    child_processes_count = 0
    for subchunk in subchunks:
        sub_result = _transcribe_fallback_subchunk(
            backend=backend,
            parent_chunk=chunk,
            subchunk=subchunk,
            output_dir=output_dir,
            mode=mode,
            fallback_events=fallback_events,
        )
        merged_segments.extend(
            offset_segments(
                sub_result.segments,
                chunk.offset_seconds + subchunk.offset_seconds,
            )
        )
        child_output_dirs.extend(sub_result.child_output_dirs)
        child_processes_count += sub_result.child_processes_count

    return IsolatedTranscriptionResult(
        segments=merged_segments,
        child_output_dirs=child_output_dirs,
        child_processes_count=child_processes_count,
        fallback_events=[],
    )


def _transcribe_fallback_subchunk(
    *,
    backend: str,
    parent_chunk: ChunkPlanItem,
    subchunk: ChunkPlanItem,
    output_dir: Path,
    mode: str,
    fallback_events: list[dict[str, object]],
) -> IsolatedTranscriptionResult:
    attempts = [
        ("large-v3", "Systran/faster-whisper-large-v3", "cuda"),
        ("medium.en", "Systran/faster-whisper-medium.en", "cuda"),
        ("medium.en", "Systran/faster-whisper-medium.en", "cpu"),
    ]
    child_output_dirs: list[Path] = []
    child_processes_count = 0
    previous_oom = False

    for attempt_index, (label, model, device) in enumerate(attempts, start=1):
        absolute_offset = parent_chunk.offset_seconds + subchunk.offset_seconds
        if attempt_index == 1:
            print(
                f"Subchunk {parent_chunk.index}.{subchunk.index} "
                f"offset {format_clock(absolute_offset)} "
                f"duration {format_clock(subchunk.duration_seconds)} "
                f"{label} CUDA",
                flush=True,
            )
        elif device == "cuda":
            print(
                f"Subchunk {parent_chunk.index}.{subchunk.index} "
                f"CUDA OOM; retrying medium.en CUDA",
                flush=True,
            )
            fallback_events.append(
                _fallback_event(
                    chunk=parent_chunk,
                    subchunk=subchunk,
                    action="retry",
                    model=model,
                    device=device,
                    reason="cuda_oom",
                )
            )
        else:
            print(
                f"Subchunk {parent_chunk.index}.{subchunk.index} "
                f"medium.en CUDA OOM; retrying medium.en CPU",
                flush=True,
            )
            fallback_events.append(
                _fallback_event(
                    chunk=parent_chunk,
                    subchunk=subchunk,
                    action="retry",
                    model=model,
                    device=device,
                    reason="cuda_oom",
                )
            )

        child_output_dir = _fallback_subchunk_output_dir(
            output_dir=output_dir,
            parent_chunk=parent_chunk,
            subchunk=subchunk,
            attempt_index=attempt_index,
        )
        command = _build_isolated_chunk_command(
            chunk=subchunk,
            child_output_dir=child_output_dir,
            backend=backend,
            device=device,
            mode="gpu" if device == "cuda" else "cpu",
            model=model,
        )
        child_output_dir.mkdir(parents=True, exist_ok=True)
        result = _run_isolated_chunk_command(command=command, chunk=subchunk)
        child_processes_count += 1
        child_output_dirs.append(child_output_dir)

        if result.returncode == 0:
            segments = _read_chunk_segments(child_output_dir, subchunk.path.stem)
            return IsolatedTranscriptionResult(
                segments=segments,
                child_output_dirs=child_output_dirs,
                child_processes_count=child_processes_count,
                fallback_events=[],
            )

        if not (device == "cuda" and _is_cuda_oom(result)):
            raise RuntimeError(
                _isolated_subchunk_failure_message(
                    parent_chunk=parent_chunk,
                    subchunk=subchunk,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
        previous_oom = True

    if previous_oom:
        raise RuntimeError(
            f"adaptive fallback failed for chunk {parent_chunk.index} "
            f"subchunk {subchunk.index} at offset "
            f"{format_clock(parent_chunk.offset_seconds + subchunk.offset_seconds)}: "
            "medium.en CPU failed after CUDA OOM retries"
        )
    raise RuntimeError("adaptive fallback failed unexpectedly")


def _fallback_subchunks(
    *,
    output_dir: Path,
    chunk: ChunkPlanItem,
) -> list[ChunkPlanItem]:
    chunks_dir = output_dir / ".work" / "fallback_chunks" / f"chunk-{chunk.index:04d}"
    subchunks = []
    offset = 0.0
    index = 1
    while offset < chunk.duration_seconds:
        duration = min(FALLBACK_CHUNK_SECONDS, chunk.duration_seconds - offset)
        subchunks.append(
            ChunkPlanItem(
                index=index,
                offset_seconds=offset,
                duration_seconds=duration,
                path=chunks_dir
                / f"{chunk.path.stem}.fallback-{index:04d}.wav",
            )
        )
        offset += FALLBACK_CHUNK_SECONDS
        index += 1
    return subchunks


def _fallback_subchunk_output_dir(
    *,
    output_dir: Path,
    parent_chunk: ChunkPlanItem,
    subchunk: ChunkPlanItem,
    attempt_index: int,
) -> Path:
    return (
        output_dir
        / ".work"
        / "chunk_outputs"
        / f"chunk-{parent_chunk.index:04d}"
        / f"fallback-{subchunk.index:04d}-attempt-{attempt_index}"
    )


def _fallback_event(
    *,
    chunk: ChunkPlanItem,
    action: str,
    model: str,
    device: str,
    reason: str,
    subchunk: ChunkPlanItem | None = None,
) -> dict[str, object]:
    offset_seconds = chunk.offset_seconds
    duration_seconds = chunk.duration_seconds
    if subchunk is not None:
        offset_seconds += subchunk.offset_seconds
        duration_seconds = subchunk.duration_seconds
    return {
        "chunk_index": chunk.index,
        "offset_seconds": offset_seconds,
        "duration_seconds": duration_seconds,
        "action": action,
        "model": model,
        "device": device,
        "reason": reason,
    }


def _is_cuda_oom(result: subprocess.CompletedProcess[str]) -> bool:
    text = f"{result.stdout}\n{result.stderr}".casefold()
    return (
        "out of memory" in text
        or "cuda failed with error out of memory" in text
        or "cuda out of memory" in text
    )


def _resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _child_process_mode(mode: str, device: str) -> str:
    if mode in {"auto", "gpu", "cpu"}:
        return mode
    if device == "cuda":
        return "gpu"
    return "cpu"


def _read_chunk_segments(
    child_output_dir: Path,
    stem: str,
) -> list[TranscriptionSegment]:
    output_path = child_output_dir / f"{stem}.json"
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        raw_segments = data["segments"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"failed to read isolated chunk output {output_path}: {exc}") from exc

    segments = []
    for index, segment in enumerate(raw_segments, start=1):
        try:
            segments.append(
                TranscriptionSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    text=str(segment["text"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid segment {index} in isolated chunk output {output_path}: {exc}"
            ) from exc
    return segments


def _isolated_chunk_output_dirs(
    output_dir: Path,
    chunks: list[ChunkPlanItem],
) -> list[Path]:
    return [_isolated_chunk_output_dir(output_dir, chunk) for chunk in chunks]


def _isolated_chunk_output_dir(output_dir: Path, chunk: ChunkPlanItem) -> Path:
    return output_dir / ".work" / "chunk_outputs" / f"chunk-{chunk.index:04d}"


def _relative_output_path(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _isolated_chunk_failure_message(
    *,
    chunk: ChunkPlanItem,
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> str:
    if returncode is None:
        first_line = (
            f"isolated chunk {chunk.index} failed at offset "
            f"{format_clock(chunk.offset_seconds)}"
        )
    else:
        first_line = (
            f"isolated chunk {chunk.index} failed at offset "
            f"{format_clock(chunk.offset_seconds)} with return code {returncode}"
        )
    summary = _process_output_summary(stdout=stdout, stderr=stderr)
    if summary:
        return f"{first_line}\n{summary}"
    return first_line


def _isolated_subchunk_failure_message(
    *,
    parent_chunk: ChunkPlanItem,
    subchunk: ChunkPlanItem,
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> str:
    absolute_offset = parent_chunk.offset_seconds + subchunk.offset_seconds
    if returncode is None:
        first_line = (
            f"adaptive fallback subchunk {parent_chunk.index}.{subchunk.index} "
            f"failed at offset {format_clock(absolute_offset)}"
        )
    else:
        first_line = (
            f"adaptive fallback subchunk {parent_chunk.index}.{subchunk.index} "
            f"failed at offset {format_clock(absolute_offset)} "
            f"with return code {returncode}"
        )
    summary = _process_output_summary(stdout=stdout, stderr=stderr)
    if summary:
        return f"{first_line}\n{summary}"
    return first_line


def _process_output_summary(*, stdout: str, stderr: str) -> str:
    sections = []
    stderr_tail = _tail_text(stderr)
    stdout_tail = _tail_text(stdout)
    if stderr_tail:
        sections.append(f"stderr tail:\n{stderr_tail}")
    if stdout_tail:
        sections.append(f"stdout tail:\n{stdout_tail}")
    return "\n".join(sections)


def _tail_text(text: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()[-max_lines:]
    return "\n".join(lines)[-max_chars:]


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
