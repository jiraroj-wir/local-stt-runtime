"""Transcription backend contracts."""

from __future__ import annotations

from pathlib import Path

from app.config import TranscriptionConfig
from app.segments import TranscriptionSegment
from app.whisper_backend import transcribe_faster_whisper

FAKE_BACKEND_NAME = "fake"
FASTER_WHISPER_BACKEND_NAME = "faster-whisper"
SUPPORTED_BACKENDS = (FAKE_BACKEND_NAME, FASTER_WHISPER_BACKEND_NAME)


def transcribe_fake(
    input_path: Path,
    config: TranscriptionConfig,
) -> list[TranscriptionSegment]:
    _ = input_path
    _ = config
    return [
        TranscriptionSegment(start=0.0, end=2.5, text="Hello everyone."),
        TranscriptionSegment(start=2.5, end=5.0, text="Today we discuss local transcription."),
    ]


def transcribe_with_backend(
    backend: str,
    input_path: Path,
    config: TranscriptionConfig,
) -> list[TranscriptionSegment]:
    if backend == FAKE_BACKEND_NAME:
        return transcribe_fake(input_path, config)
    if backend == FASTER_WHISPER_BACKEND_NAME:
        return transcribe_faster_whisper(input_path, config)
    raise ValueError(f"Unsupported transcription backend: {backend}")
