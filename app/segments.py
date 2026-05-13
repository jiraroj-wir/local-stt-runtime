"""Transcription segment data model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
