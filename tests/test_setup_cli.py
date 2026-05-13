import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "setup"


def run_setup(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SETUP), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_exits_zero() -> None:
    result = run_setup("--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_dry_run_exits_zero_and_reports_placeholder() -> None:
    result = run_setup("--dry-run")
    assert result.returncode == 0
    assert "not implemented yet" in result.stdout


def test_unknown_option_returns_nonzero() -> None:
    result = run_setup("--unknown")
    assert result.returncode != 0
