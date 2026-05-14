"""Transcription runtime configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str
    device: str
    compute_type: str
    beam_size: int
    language: str
    vad_filter: bool
    condition_on_previous_text: bool
    word_timestamps: bool | None = None
    batch_size: int | None = None
    cpu_threads: str | None = None


def get_device_config(device: str) -> TranscriptionConfig:
    if device == "cpu":
        return cpu_config()
    if device == "cuda":
        return cuda_config()
    raise ValueError(f"Unsupported transcription device: {device}")


def cpu_config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model="Systran/faster-whisper-medium.en",
        device="cpu",
        compute_type="int8",
        beam_size=5,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False,
        cpu_threads="auto",
    )


def cuda_config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model="Systran/faster-whisper-large-v3",
        device="cuda",
        compute_type="int8",
        beam_size=5,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=False,
        batch_size=1,
    )
