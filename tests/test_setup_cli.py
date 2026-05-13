import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "setup"
BASH = shutil.which("bash") or "/bin/bash"


def run_setup(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(SETUP), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def write_fake_podman(bin_dir: Path, log_file: Path) -> Path:
    podman = bin_dir / "podman"
    podman.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'podman %s\\n' \"$*\" >> \"$FAKE_PODMAN_LOG\"\n"
    )
    podman.chmod(podman.stat().st_mode | stat.S_IXUSR)
    return podman


def test_help_exits_zero() -> None:
    result = run_setup("--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_dry_run_exits_zero_and_prints_plan() -> None:
    result = run_setup("--dry-run")
    assert result.returncode == 0
    assert "image_name=local-stt-runtime" in result.stdout
    assert "containerfile=Containerfile" in result.stdout
    assert "build_command=podman build -t local-stt-runtime -f Containerfile ." in result.stdout


def test_setup_missing_podman_returns_nonzero(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)

    result = run_setup(env=env)
    assert result.returncode != 0
    assert "podman was not found" in result.stderr
    assert "install Podman" in result.stderr


def test_setup_runs_podman_build_with_fake_podman(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_file = tmp_path / "podman.log"
    write_fake_podman(bin_dir, log_file)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_PODMAN_LOG"] = str(log_file)
    env["HOME"] = str(tmp_path)

    result = run_setup(env=env)
    assert result.returncode == 0
    command_log = log_file.read_text()
    assert "podman build -t local-stt-runtime -f Containerfile ." in command_log


def test_setup_rebuild_uses_no_cache_with_fake_podman(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_file = tmp_path / "podman.log"
    write_fake_podman(bin_dir, log_file)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_PODMAN_LOG"] = str(log_file)
    env["HOME"] = str(tmp_path)

    result = run_setup("--rebuild", env=env)
    assert result.returncode == 0
    command_log = log_file.read_text()
    assert "podman build --no-cache -t local-stt-runtime -f Containerfile ." in command_log


def test_unknown_option_returns_nonzero() -> None:
    result = run_setup("--unknown")
    assert result.returncode != 0
