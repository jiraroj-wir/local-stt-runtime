import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE = REPO_ROOT / "transcribe"


def run_transcribe(
    *args: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    transcribe_path: Path = TRANSCRIBE,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(transcribe_path), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_exits_zero_and_prints_usage() -> None:
    result = run_transcribe("--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_list_presets_exits_zero_and_lists_expected_presets() -> None:
    result = run_transcribe("--list-presets")
    assert result.returncode == 0
    assert "linux-gpu" in result.stdout
    assert "linux-cpu" in result.stdout
    assert "macos-cpu" in result.stdout


def test_dry_run_default_mode_auto() -> None:
    result = run_transcribe("input.m4a", "--dry-run")
    assert result.returncode == 0
    assert "mode=auto" in result.stdout
    assert "selected_device=auto" in result.stdout
    assert "podman_image=local-stt-runtime" in result.stdout


def test_dry_run_gpu_mode() -> None:
    result = run_transcribe("input.m4a", "--gpu", "--dry-run")
    assert result.returncode == 0
    assert "mode=gpu" in result.stdout
    assert "selected_device=cuda" in result.stdout


def test_dry_run_cpu_mode() -> None:
    result = run_transcribe("input.m4a", "--cpu", "--dry-run")
    assert result.returncode == 0
    assert "mode=cpu" in result.stdout
    assert "selected_device=cpu" in result.stdout


def test_dry_run_does_not_execute_podman(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log, cuda_check_success=False)

    result = run_transcribe("input.m4a", "--dry-run", env=env)

    assert result.returncode == 0
    assert not podman_log.exists()


def test_gpu_and_cpu_conflict_returns_nonzero() -> None:
    result = run_transcribe("input.m4a", "--gpu", "--cpu")
    assert result.returncode != 0


def test_out_dir_override() -> None:
    result = run_transcribe("input.m4a", "--out", "transcripts", "--dry-run")
    assert result.returncode == 0
    assert "output_dir=transcripts" in result.stdout


def test_unknown_option_returns_nonzero() -> None:
    result = run_transcribe("input.m4a", "--unknown")
    assert result.returncode != 0


def test_missing_out_value_returns_nonzero() -> None:
    result = run_transcribe("input.m4a", "--out")
    assert result.returncode != 0


def test_missing_input_returns_nonzero() -> None:
    result = run_transcribe("--dry-run")
    assert result.returncode != 0


def test_non_dry_run_with_fake_podman_executes_podman_run(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log)

    result = run_transcribe("input.m4a", "--cpu", env=env)

    assert result.returncode == 0
    log = podman_log.read_text(encoding="utf-8")
    assert "run --rm" in log
    assert "local-stt-runtime" in log
    assert "-m app.transcribe" in log


def test_cpu_mode_passes_cpu_device_to_runner(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log)

    result = run_transcribe("input.m4a", "--cpu", env=env)

    assert result.returncode == 0
    log = podman_log.read_text(encoding="utf-8")
    assert "nvidia-smi" not in log
    assert "nvidia.com/gpu=all" not in transcribe_run_line(log)
    assert "--device cpu" in transcribe_run_line(log)


def test_gpu_mode_passes_cuda_device_to_runner(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log)

    result = run_transcribe("input.m4a", "--gpu", env=env)

    assert result.returncode == 0
    log = podman_log.read_text(encoding="utf-8")
    run_line = transcribe_run_line(log)
    assert "nvidia-smi" in log
    assert "nvidia.com/gpu=all" in run_line
    assert "--security-opt=label=disable" in run_line
    assert "--device cuda" in run_line


def test_out_dir_passes_output_dir_to_runner(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log)

    result = run_transcribe("input.m4a", "--out", "transcripts", env=env)

    assert result.returncode == 0
    assert "--output-dir transcripts" in podman_log.read_text(encoding="utf-8")


def test_gpu_mode_cuda_check_failure_returns_nonzero_without_cpu_fallback(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log, cuda_check_success=False)

    result = run_transcribe("input.m4a", "--gpu", env=env)

    log = podman_log.read_text(encoding="utf-8")
    assert result.returncode != 0
    assert "CUDA/CDI GPU check failed" in result.stderr
    assert "nvidia-smi" in log
    assert "local-stt-runtime -m app.transcribe" not in log


def test_auto_mode_cuda_check_success_uses_cuda_and_gpu_flags(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log, cuda_check_success=True)

    result = run_transcribe("input.m4a", env=env)

    log = podman_log.read_text(encoding="utf-8")
    run_line = transcribe_run_line(log)
    assert result.returncode == 0
    assert "nvidia-smi" in log
    assert "nvidia.com/gpu=all" in run_line
    assert "--security-opt=label=disable" in run_line
    assert "--device cuda" in run_line


def test_auto_mode_cuda_check_failure_falls_back_to_cpu_without_gpu_flags(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log, cuda_check_success=False)

    result = run_transcribe("input.m4a", env=env)

    log = podman_log.read_text(encoding="utf-8")
    run_line = transcribe_run_line(log)
    assert result.returncode == 0
    assert "CUDA/CDI GPU check failed; falling back to CPU." in result.stderr
    assert "Runtime fallback" in result.stderr
    assert "Continuing : CPU" in result.stderr
    assert "nvidia-smi" in log
    assert "nvidia.com/gpu=all" not in run_line
    assert "--security-opt=label=disable" not in run_line
    assert "--device cpu" in run_line


def test_podman_run_uses_expected_mounts_and_image(tmp_path) -> None:
    podman_log = tmp_path / "podman.log"
    env = fake_podman_env(tmp_path, podman_log, home=tmp_path / "home")

    result = run_transcribe("input.m4a", env=env)

    log = podman_log.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert f"-v {REPO_ROOT}:/work" in log
    assert f"-v {tmp_path / 'home' / '.cache' / 'local-stt-runtime'}:/cache" in log
    assert "local-stt-runtime -m app.transcribe" in log
    assert "--backend faster-whisper" in log


def test_image_missing_runs_setup_from_transcribe_script_directory(tmp_path) -> None:
    script_dir = tmp_path / "script-dir"
    script_dir.mkdir()
    transcribe_copy = script_dir / "transcribe"
    shutil.copy2(TRANSCRIBE, transcribe_copy)
    transcribe_copy.chmod(0o755)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    podman_log = tmp_path / "podman.log"
    setup_log = tmp_path / "setup.log"
    env = fake_podman_env(
        tmp_path,
        podman_log,
        image_exists=False,
    )
    adjacent_setup = script_dir / "setup"
    adjacent_setup.write_text(
        f"#!/usr/bin/env bash\npwd >> {setup_log}\n",
        encoding="utf-8",
    )
    adjacent_setup.chmod(0o755)
    cwd_setup = work_dir / "setup"
    cwd_setup.write_text(
        "#!/usr/bin/env bash\nexit 97\n",
        encoding="utf-8",
    )
    cwd_setup.chmod(0o755)

    result = run_transcribe(
        "input.m4a",
        "--out",
        "out",
        cwd=work_dir,
        env=env,
        transcribe_path=transcribe_copy,
    )

    assert result.returncode == 0
    assert setup_log.read_text(encoding="utf-8") == f"{script_dir}\n"
    assert "run --rm" in podman_log.read_text(encoding="utf-8")


def test_podman_missing_returns_nonzero_with_clear_error(tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "bash").symlink_to("/bin/bash")
    env = {"PATH": str(bin_dir)}

    result = run_transcribe("input.m4a", "--out", "out", env=env)

    assert result.returncode != 0
    assert "podman is required" in result.stderr


def fake_podman_env(
    tmp_path: Path,
    podman_log: Path,
    *,
    image_exists: bool = True,
    cuda_check_success: bool = True,
    home: Path | None = None,
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    podman = bin_dir / "podman"
    podman.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$PODMAN_LOG"',
                'if [[ "$1" == "image" && "$2" == "exists" ]]; then',
                '  if [[ "$PODMAN_IMAGE_EXISTS" == "1" ]]; then',
                "    exit 0",
                "  fi",
                "  exit 1",
                "fi",
                'if [[ "$*" == *"nvidia-smi"* ]]; then',
                '  if [[ "$PODMAN_CUDA_CHECK_SUCCESS" == "1" ]]; then',
                "    exit 0",
                "  fi",
                "  exit 1",
                "fi",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    podman.chmod(0o755)

    return {
        "PATH": f"{bin_dir}:{get_default_path()}",
        "HOME": str(home or tmp_path / "home"),
        "PODMAN_LOG": str(podman_log),
        "PODMAN_IMAGE_EXISTS": "1" if image_exists else "0",
        "PODMAN_CUDA_CHECK_SUCCESS": "1" if cuda_check_success else "0",
    }


def transcribe_run_line(log: str) -> str:
    for line in log.splitlines():
        if "local-stt-runtime -m app.transcribe" in line:
            return line
    raise AssertionError("missing transcription podman run")


def get_default_path() -> str:
    return "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
