import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE = REPO_ROOT / "transcribe"


def run_transcribe(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(TRANSCRIBE), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help() -> None:
    result = run_transcribe("--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_list_presets() -> None:
    result = run_transcribe("--list-presets")
    assert result.returncode == 0
    assert "linux-gpu" in result.stdout
    assert "linux-cpu" in result.stdout
    assert "macos-cpu" in result.stdout


def test_dry_run_auto_mode() -> None:
    result = run_transcribe("input.m4a", "--dry-run")
    assert result.returncode == 0
    assert "mode=auto" in result.stdout


def test_dry_run_gpu_mode() -> None:
    result = run_transcribe("input.m4a", "--gpu", "--dry-run")
    assert result.returncode == 0
    assert "mode=gpu" in result.stdout


def test_dry_run_cpu_mode() -> None:
    result = run_transcribe("input.m4a", "--cpu", "--dry-run")
    assert result.returncode == 0
    assert "mode=cpu" in result.stdout


def test_gpu_and_cpu_conflict() -> None:
    result = run_transcribe("input.m4a", "--gpu", "--cpu")
    assert result.returncode != 0


def test_out_dir_override() -> None:
    result = run_transcribe("input.m4a", "--out", "transcripts", "--dry-run")
    assert result.returncode == 0
    assert "output_dir=transcripts" in result.stdout


def test_unknown_option() -> None:
    result = run_transcribe("input.m4a", "--nope")
    assert result.returncode != 0


def test_missing_out_value() -> None:
    result = run_transcribe("input.m4a", "--out")
    assert result.returncode != 0


def test_missing_input() -> None:
    result = run_transcribe("--dry-run")
    assert result.returncode != 0
