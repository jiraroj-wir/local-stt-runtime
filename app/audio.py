"""Audio metadata, preprocessing, and chunking helper functions."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg"})
INACCURATE_DURATION_WARNING_TEXT = "Estimating duration from bitrate, this may be inaccurate"


@dataclass(frozen=True)
class AudioMetadata:
    duration_seconds: float | None
    codec_name: str | None
    sample_rate: int | None
    channels: int | None
    bit_rate: int | None
    format_name: str | None


@dataclass(frozen=True)
class AudioInspection:
    metadata: AudioMetadata
    warning: str | None = None


@dataclass(frozen=True)
class VolumeStats:
    mean_volume_db: float | None
    max_volume_db: float | None
    warning: str | None = None


@dataclass(frozen=True)
class ChunkPlanItem:
    index: int
    offset_seconds: float
    duration_seconds: float
    path: Path


def is_supported_audio_path(path: str | PathLike[str]) -> bool:
    """Return whether a path has a supported audio extension."""
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def validate_supported_audio_path(path: str | PathLike[str]) -> None:
    """Raise ValueError if a path does not have a supported audio extension."""
    if not is_supported_audio_path(path):
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio file extension for {path!s}; supported: {supported}")


def build_ffprobe_command(input_path: Path) -> list[str]:
    """Build an ffprobe command for JSON audio metadata."""
    return [
        "ffprobe",
        "-v",
        "warning",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]


def inspect_audio(input_path: Path) -> AudioInspection:
    """Inspect audio metadata with ffprobe, returning nullable fields on failure."""
    try:
        result = subprocess.run(
            build_ffprobe_command(input_path),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return AudioInspection(empty_audio_metadata(), warning=f"ffprobe failed: {exc}")

    if result.returncode != 0:
        error = _compact_process_error(
            result.stderr,
            f"ffprobe exited with status {result.returncode}",
        )
        return AudioInspection(empty_audio_metadata(), warning=f"ffprobe failed: {error}")

    try:
        data = json.loads(result.stdout)
        return AudioInspection(
            parse_ffprobe_metadata(data),
            warning=_ffprobe_duration_warning(result.stderr),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return AudioInspection(empty_audio_metadata(), warning=f"ffprobe parse failed: {exc}")


def parse_ffprobe_metadata(ffprobe_json: dict[str, Any]) -> AudioMetadata:
    """Parse ffprobe JSON output into AudioMetadata."""
    format_section = ffprobe_json.get("format")
    if not isinstance(format_section, dict):
        format_section = {}

    audio_stream = _first_audio_stream(ffprobe_json.get("streams"))

    return AudioMetadata(
        duration_seconds=_optional_duration(format_section, audio_stream),
        codec_name=_optional_str(audio_stream.get("codec_name") if audio_stream else None),
        sample_rate=_optional_int(audio_stream.get("sample_rate") if audio_stream else None),
        channels=_optional_int(audio_stream.get("channels") if audio_stream else None),
        bit_rate=_optional_int(
            _first_present(
                audio_stream.get("bit_rate") if audio_stream else None,
                format_section.get("bit_rate"),
            )
        ),
        format_name=_optional_str(format_section.get("format_name")),
    )


def empty_audio_metadata() -> AudioMetadata:
    """Return an AudioMetadata object with all nullable fields empty."""
    return AudioMetadata(
        duration_seconds=None,
        codec_name=None,
        sample_rate=None,
        channels=None,
        bit_rate=None,
        format_name=None,
    )


def build_preprocess_command(input_path: Path, output_wav_path: Path) -> list[str]:
    """Build a conservative loudness normalization command."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav_path),
    ]


def build_volumedetect_command(input_path: Path) -> list[str]:
    """Build an ffmpeg command that measures audio volume without writing output."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]


def detect_volume(input_path: Path) -> VolumeStats:
    """Run ffmpeg volumedetect and parse mean/max volume when available."""
    try:
        result = subprocess.run(
            build_volumedetect_command(input_path),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return VolumeStats(None, None, warning=f"volumedetect failed: {exc}")

    if result.returncode != 0:
        error = _compact_process_error(
            result.stderr,
            f"ffmpeg exited with status {result.returncode}",
        )
        return VolumeStats(None, None, warning=f"volumedetect failed: {error}")

    stats = parse_volumedetect_output(result.stderr)
    if stats.mean_volume_db is None and stats.max_volume_db is None:
        return VolumeStats(None, None, warning="volumedetect output did not include volume stats")
    return stats


def parse_volumedetect_output(stderr: str) -> VolumeStats:
    """Parse ffmpeg volumedetect stderr for mean and max volume."""
    return VolumeStats(
        mean_volume_db=_parse_volume_field(stderr, "mean_volume"),
        max_volume_db=_parse_volume_field(stderr, "max_volume"),
    )


def should_auto_normalize(stats: VolumeStats) -> bool:
    """Return whether conservative auto preprocessing should normalize this input."""
    return (
        stats.mean_volume_db is not None
        and stats.mean_volume_db <= -35
        or stats.max_volume_db is not None
        and stats.max_volume_db <= -12
    )


def normalize_audio(input_path: Path, output_wav_path: Path) -> None:
    """Create a normalized 16 kHz mono WAV."""
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_preprocess_command(input_path, output_wav_path),
        text=True,
        capture_output=True,
        check=True,
    )


def plan_chunks(
    *,
    duration_seconds: float | None,
    chunk_minutes: float,
    output_dir: Path,
    stem: str,
    enabled: bool,
) -> list[ChunkPlanItem]:
    """Plan chunk output paths for long audio; exact-threshold files are not chunked."""
    if not enabled or duration_seconds is None:
        return []

    chunk_seconds = chunk_minutes * 60
    if chunk_seconds <= 0 or duration_seconds <= chunk_seconds:
        return []

    chunks_dir = output_dir / ".work" / "chunks"
    chunks = []
    offset = 0.0
    index = 1
    while offset < duration_seconds:
        chunk_duration = min(chunk_seconds, duration_seconds - offset)
        chunks.append(
            ChunkPlanItem(
                index=index,
                offset_seconds=offset,
                duration_seconds=chunk_duration,
                path=chunks_dir / f"{stem}.chunk-{index:04d}.wav",
            )
        )
        offset += chunk_seconds
        index += 1
    return chunks


def build_chunk_command(
    input_path: Path,
    output_wav_path: Path,
    *,
    offset_seconds: float,
    duration_seconds: float,
) -> list[str]:
    """Build an ffmpeg command that creates one 16 kHz mono WAV chunk."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        _format_seconds_arg(offset_seconds),
        "-t",
        _format_seconds_arg(duration_seconds),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav_path),
    ]


def create_chunk(input_path: Path, chunk: ChunkPlanItem) -> None:
    """Create one chunk file from a planned chunk item."""
    chunk.path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_chunk_command(
            input_path,
            chunk.path,
            offset_seconds=chunk.offset_seconds,
            duration_seconds=chunk.duration_seconds,
        ),
        text=True,
        capture_output=True,
        check=True,
    )


def _first_audio_stream(streams: Any) -> dict[str, Any] | None:
    if not isinstance(streams, list):
        return None

    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    return None


def _optional_duration(
    format_section: dict[str, Any],
    audio_stream: dict[str, Any] | None,
) -> float | None:
    if _has_value(format_section.get("duration")):
        duration = _optional_float(format_section["duration"])
        if duration is not None:
            return duration

    if audio_stream and _has_value(audio_stream.get("duration")):
        return _optional_float(audio_stream["duration"])

    return None


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _optional_int(value: Any) -> int | None:
    if not _has_value(value):
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    return parsed


def _optional_str(value: Any) -> str | None:
    if not _has_value(value):
        return None
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if _has_value(value):
            return value
    return None


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _parse_volume_field(stderr: str, field_name: str) -> float | None:
    match = re.search(rf"{field_name}:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB", stderr)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _format_seconds_arg(seconds: float) -> str:
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def _compact_process_error(stderr: str, fallback: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return fallback
    return lines[-1]


def _ffprobe_duration_warning(stderr: str) -> str | None:
    if INACCURATE_DURATION_WARNING_TEXT in stderr:
        return f"ffprobe warning: {INACCURATE_DURATION_WARNING_TEXT}"
    return None
