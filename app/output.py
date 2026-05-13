"""Output writers for transcription results."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from app.segments import TranscriptionSegment
from app.timefmt import format_srt_timestamp


def write_txt(segments: Iterable[TranscriptionSegment], output_path: Path) -> None:
    lines = [text for text in _non_empty_text(segments)]
    content = "\n".join(lines)
    if content:
        content += "\n"

    Path(output_path).write_text(content, encoding="utf-8")


def write_srt(segments: Iterable[TranscriptionSegment], output_path: Path) -> None:
    cues = []
    for cue_number, segment in enumerate(_non_empty_segments(segments), start=1):
        cues.append(
            "\n".join(
                [
                    str(cue_number),
                    (
                        f"{format_srt_timestamp(segment.start)} --> "
                        f"{format_srt_timestamp(segment.end)}"
                    ),
                    segment.text.strip(),
                ]
            )
        )

    content = "\n\n".join(cues)
    if content:
        content += "\n"

    Path(output_path).write_text(content, encoding="utf-8")


def write_json(
    segments: Iterable[TranscriptionSegment],
    output_path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    data = {
        "segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            for segment in segments
        ],
        "metadata": dict(metadata or {}),
    }

    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_metadata(metadata: Mapping[str, Any], output_path: Path) -> None:
    Path(output_path).write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_run_log(lines: Iterable[str], output_path: Path) -> None:
    content = "\n".join(str(line) for line in lines)
    if content:
        content += "\n"

    Path(output_path).write_text(content, encoding="utf-8")


def write_all_outputs(
    segments: Iterable[TranscriptionSegment],
    output_dir: Path,
    stem: str,
    metadata: Mapping[str, Any],
    log_lines: Iterable[str],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_list = list(segments)
    paths = {
        "txt": output_dir / f"{stem}.txt",
        "srt": output_dir / f"{stem}.srt",
        "json": output_dir / f"{stem}.json",
        "metadata": output_dir / "metadata.json",
        "log": output_dir / "run.log",
    }

    write_txt(segment_list, paths["txt"])
    write_srt(segment_list, paths["srt"])
    write_json(segment_list, paths["json"], metadata=metadata)
    write_metadata(metadata, paths["metadata"])
    write_run_log(log_lines, paths["log"])

    return paths


def _non_empty_segments(
    segments: Iterable[TranscriptionSegment],
) -> Iterable[TranscriptionSegment]:
    for segment in segments:
        if segment.text.strip():
            yield segment


def _non_empty_text(segments: Iterable[TranscriptionSegment]) -> Iterable[str]:
    for segment in _non_empty_segments(segments):
        yield segment.text.strip()
