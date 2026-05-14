import json
import subprocess
import sys

import pytest

from scripts.score_transcript import normalize_text, score_transcript


def test_exact_match_gives_zero_wer_and_cer() -> None:
    score = score_transcript("hello world", "hello world")

    assert score.wer == 0
    assert score.cer == 0


def test_one_substitution_counted_correctly() -> None:
    score = score_transcript("hello world", "hello there")

    assert score.substitutions == 1
    assert score.deletions == 0
    assert score.insertions == 0
    assert score.wer == 0.5


def test_one_deletion_counted_correctly() -> None:
    score = score_transcript("hello brave world", "hello world")

    assert score.substitutions == 0
    assert score.deletions == 1
    assert score.insertions == 0


def test_one_insertion_counted_correctly() -> None:
    score = score_transcript("hello world", "hello brave world")

    assert score.substitutions == 0
    assert score.deletions == 0
    assert score.insertions == 1


def test_case_and_punctuation_normalize_away() -> None:
    score = score_transcript("Hello, WORLD!", "hello world")

    assert normalize_text("Hello, WORLD!") == "hello world"
    assert score.wer == 0
    assert score.cer == 0


def test_hyphenated_text_normalizes_sensibly() -> None:
    score = score_transcript("speech-to-text", "speech to text")

    assert normalize_text("speech-to-text") == "speech to text"
    assert score.wer == 0
    assert score.cer == 0


def test_empty_reference_raises_value_error() -> None:
    with pytest.raises(ValueError, match="reference transcript is empty"):
        score_transcript("   !!!   ", "hello")


def test_cli_text_output_contains_wer_and_cer(tmp_path) -> None:
    reference = tmp_path / "reference.txt"
    hypothesis = tmp_path / "hypothesis.txt"
    reference.write_text("hello world", encoding="utf-8")
    hypothesis.write_text("hello there", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/score_transcript.py", str(reference), str(hypothesis)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "WER" in result.stdout
    assert "CER" in result.stdout


def test_cli_json_outputs_parseable_json(tmp_path) -> None:
    reference = tmp_path / "reference.txt"
    hypothesis = tmp_path / "hypothesis.txt"
    reference.write_text("hello world", encoding="utf-8")
    hypothesis.write_text("hello world", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_transcript.py",
            str(reference),
            str(hypothesis),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    data = json.loads(result.stdout)
    assert result.returncode == 0
    assert data["reference_word_count"] == 2
    assert data["hypothesis_word_count"] == 2
    assert data["wer"] == 0
    assert data["cer"] == 0


def test_cli_empty_reference_exits_nonzero(tmp_path) -> None:
    reference = tmp_path / "reference.txt"
    hypothesis = tmp_path / "hypothesis.txt"
    reference.write_text("", encoding="utf-8")
    hypothesis.write_text("hello", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/score_transcript.py", str(reference), str(hypothesis)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "reference transcript is empty" in result.stderr
