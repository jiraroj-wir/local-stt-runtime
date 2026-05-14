#!/usr/bin/env python3
"""Score transcript text with simple WER and CER metrics."""

from __future__ import annotations

import argparse
import json
import string
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

SEPARATOR = "────────────────────────────────────────"
DASH_CHARACTERS = "-‐‑‒–—―"
PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class TranscriptScore:
    reference_word_count: int
    hypothesis_word_count: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float
    reference_char_count: int
    hypothesis_char_count: int
    character_edit_distance: int
    cer: float


def normalize_text(text: str) -> str:
    text = text.lower()
    for dash in DASH_CHARACTERS:
        text = text.replace(dash, " ")
    text = text.translate(PUNCTUATION_TRANSLATION)
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return " ".join(text.split())


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def levenshtein_counts(
    reference_words: Sequence[str],
    hypothesis_words: Sequence[str],
) -> EditCounts:
    rows = len(reference_words) + 1
    columns = len(hypothesis_words) + 1
    table = [[EditCounts(0, 0, 0) for _column in range(columns)] for _row in range(rows)]

    for row in range(1, rows):
        table[row][0] = EditCounts(0, row, 0)
    for column in range(1, columns):
        table[0][column] = EditCounts(0, 0, column)

    for row in range(1, rows):
        for column in range(1, columns):
            if reference_words[row - 1] == hypothesis_words[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue

            substitution = _add_substitution(table[row - 1][column - 1])
            deletion = _add_deletion(table[row - 1][column])
            insertion = _add_insertion(table[row][column - 1])
            table[row][column] = min(
                (substitution, deletion, insertion),
                key=lambda counts: (
                    counts.distance,
                    counts.substitutions,
                    counts.deletions,
                    counts.insertions,
                ),
            )

    return table[-1][-1]


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_char in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_char in enumerate(hypothesis, start=1):
            cost = 0 if reference_char == hypothesis_char else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def score_transcript(reference_text: str, hypothesis_text: str) -> TranscriptScore:
    reference_words = tokenize_words(reference_text)
    hypothesis_words = tokenize_words(hypothesis_text)
    if not reference_words:
        raise ValueError("reference transcript is empty after normalization")

    counts = levenshtein_counts(reference_words, hypothesis_words)
    normalized_reference = normalize_text(reference_text)
    normalized_hypothesis = normalize_text(hypothesis_text)
    # CER ignores whitespace for v1 so line wrapping and spacing changes do not dominate scoring.
    reference_chars = normalized_reference.replace(" ", "")
    hypothesis_chars = normalized_hypothesis.replace(" ", "")
    character_distance = edit_distance(reference_chars, hypothesis_chars)

    return TranscriptScore(
        reference_word_count=len(reference_words),
        hypothesis_word_count=len(hypothesis_words),
        substitutions=counts.substitutions,
        deletions=counts.deletions,
        insertions=counts.insertions,
        wer=counts.distance / len(reference_words),
        reference_char_count=len(reference_chars),
        hypothesis_char_count=len(hypothesis_chars),
        character_edit_distance=character_distance,
        cer=character_distance / len(reference_chars) if reference_chars else 0.0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a hypothesis transcript against reference text."
    )
    parser.add_argument("reference_path")
    parser.add_argument("hypothesis_path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    reference_path = Path(args.reference_path)
    hypothesis_path = Path(args.hypothesis_path)

    try:
        reference_text = reference_path.read_text(encoding="utf-8")
        hypothesis_text = hypothesis_path.read_text(encoding="utf-8")
        score = score_transcript(reference_text, hypothesis_text)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    metrics = {
        "reference_path": str(reference_path),
        "hypothesis_path": str(hypothesis_path),
        **asdict(score),
    }
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print(format_summary(metrics))
    return 0


def format_summary(metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            SEPARATOR,
            "TRANSCRIPT SCORE",
            "",
            "Input",
            f"  Reference  : {metrics['reference_path']}",
            f"  Hypothesis : {metrics['hypothesis_path']}",
            "",
            "Words",
            f"  Reference  : {metrics['reference_word_count']}",
            f"  Hypothesis : {metrics['hypothesis_word_count']}",
            f"  Substitutions : {metrics['substitutions']}",
            f"  Deletions     : {metrics['deletions']}",
            f"  Insertions    : {metrics['insertions']}",
            f"  WER        : {_format_rate(metrics['wer'])}",
            "",
            "Characters",
            f"  Reference  : {metrics['reference_char_count']}",
            f"  Hypothesis : {metrics['hypothesis_char_count']}",
            f"  Edit distance : {metrics['character_edit_distance']}",
            f"  CER        : {_format_rate(metrics['cer'])}",
            SEPARATOR,
        ]
    )


def _format_rate(value: object) -> str:
    return f"{float(value):.4f}"


def _add_substitution(counts: EditCounts) -> EditCounts:
    return EditCounts(counts.substitutions + 1, counts.deletions, counts.insertions)


def _add_deletion(counts: EditCounts) -> EditCounts:
    return EditCounts(counts.substitutions, counts.deletions + 1, counts.insertions)


def _add_insertion(counts: EditCounts) -> EditCounts:
    return EditCounts(counts.substitutions, counts.deletions, counts.insertions + 1)


if __name__ == "__main__":
    raise SystemExit(main())
