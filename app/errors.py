"""Error types for transcription reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FailureReport:
    stage: str
    error: str
    log_path: Path | None = None
