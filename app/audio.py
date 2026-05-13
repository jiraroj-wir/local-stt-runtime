"""Audio validation and command helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg"})


@dataclass(frozen=True)
class AudioMetadata:
    duration_seconds: float
    format_name: str | None
    sample_rate_hz: int | None
    channels: int | None
    codec_name: str | None


def is_supported_audio_path(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def validate_supported_audio_path(path: Path) -> None:
    if not is_supported_audio_path(path):
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio extension: {path.suffix or '<none>'}. Supported: {supported}"
        )


def build_ffprobe_command(input_path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]


def parse_ffprobe_metadata(ffprobe_json: dict) -> AudioMetadata:
    format_data = ffprobe_json.get("format", {})
    streams = ffprobe_json.get("streams", [])

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _parse_float(format_data.get("duration"))
    if duration is None and audio_stream is not None:
        duration = _parse_float(audio_stream.get("duration"))
    if duration is None:
        raise ValueError("Unable to determine audio duration from ffprobe metadata.")

    sample_rate = _parse_int(audio_stream.get("sample_rate")) if audio_stream else None
    channels = _parse_int(audio_stream.get("channels")) if audio_stream else None
    codec_name = audio_stream.get("codec_name") if audio_stream else None
    format_name = format_data.get("format_name")

    return AudioMetadata(
        duration_seconds=duration,
        format_name=format_name,
        sample_rate_hz=sample_rate,
        channels=channels,
        codec_name=codec_name,
    )


def build_preprocess_command(input_path: Path, output_wav_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
