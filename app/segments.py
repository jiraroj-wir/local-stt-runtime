"""Transcription segment data model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str


def offset_segments(
    segments: list[TranscriptionSegment],
    offset_seconds: float,
) -> list[TranscriptionSegment]:
    """Return segments shifted by a chunk offset."""
    return [
        TranscriptionSegment(
            start=segment.start + offset_seconds,
            end=segment.end + offset_seconds,
            text=segment.text,
        )
        for segment in segments
    ]
