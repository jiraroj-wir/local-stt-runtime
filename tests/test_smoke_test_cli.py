import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_TEST = REPO_ROOT / "smoke_test"


def run_smoke_test(
    *args: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SMOKE_TEST), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_success_path_uses_macos_say_and_prints_summary(tmp_path) -> None:
    transcribe = make_fake_transcribe(tmp_path)
    env = smoke_env(tmp_path, transcribe=transcribe, platform="darwin")
    make_fake_command(tmp_path, "say", "touch \"$2\"")
    make_fake_ffmpeg(tmp_path)

    result = run_smoke_test("--out", str(tmp_path / "work"), env=env)

    assert result.returncode == 0
    assert "LOCAL-STT-RUNTIME SMOKE TEST" in result.stdout
    assert "Result" in result.stdout
    assert "PASS" in result.stdout
    assert (tmp_path / "work" / "output" / "validation.txt").read_text(encoding="utf-8")


def test_linux_path_uses_espeak_ng_when_available(tmp_path) -> None:
    transcribe = make_fake_transcribe(tmp_path)
    espeak_log = tmp_path / "espeak.log"
    env = smoke_env(tmp_path, transcribe=transcribe, platform="linux")
    make_fake_command(tmp_path, "espeak-ng", f"printf 'espeak-ng\\n' >> {espeak_log}; touch \"$4\"")
    make_fake_ffmpeg(tmp_path)

    result = run_smoke_test("--out", str(tmp_path / "work"), env=env)

    assert result.returncode == 0
    assert espeak_log.read_text(encoding="utf-8") == "espeak-ng\n"


def test_missing_ffmpeg_returns_nonzero(tmp_path) -> None:
    transcribe = make_fake_transcribe(tmp_path)
    env = smoke_env(tmp_path, transcribe=transcribe, platform="darwin")
    make_fake_command(tmp_path, "say", "touch \"$2\"")

    result = run_smoke_test("--out", str(tmp_path / "work"), env=env)

    assert result.returncode != 0
    assert "ffmpeg is required" in result.stderr


def test_missing_tts_returns_nonzero(tmp_path) -> None:
    transcribe = make_fake_transcribe(tmp_path)
    env = smoke_env(tmp_path, transcribe=transcribe, platform="linux")
    make_fake_ffmpeg(tmp_path)

    result = run_smoke_test("--out", str(tmp_path / "work"), env=env)

    assert result.returncode != 0
    assert "no supported TTS tool found" in result.stderr


def test_keep_temp_prints_work_dir(tmp_path) -> None:
    transcribe = make_fake_transcribe(tmp_path)
    env = smoke_env(tmp_path, transcribe=transcribe, platform="darwin")
    make_fake_command(tmp_path, "say", "touch \"$2\"")
    make_fake_ffmpeg(tmp_path)

    result = run_smoke_test("--keep-temp", env=env)

    assert result.returncode == 0
    assert "Work dir" in result.stdout


def test_out_dir_is_not_deleted(tmp_path) -> None:
    transcribe = make_fake_transcribe(tmp_path)
    work_dir = tmp_path / "kept"
    env = smoke_env(tmp_path, transcribe=transcribe, platform="darwin")
    make_fake_command(tmp_path, "say", "touch \"$2\"")
    make_fake_ffmpeg(tmp_path)

    result = run_smoke_test("--out", str(work_dir), env=env)

    assert result.returncode == 0
    assert (work_dir / "reference.txt").exists()
    assert (work_dir / "output" / "validation.txt").exists()


def make_fake_transcribe(tmp_path: Path) -> Path:
    transcribe = tmp_path / "fake-transcribe"
    transcribe.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'out_dir=""',
                "while (($# > 0)); do",
                '  if [[ "$1" == "--out" ]]; then',
                "    shift",
                '    out_dir="$1"',
                "  fi",
                "  shift || true",
                "done",
                'mkdir -p "$out_dir"',
                'printf "Local speech to text validation.\\n" > "$out_dir/validation.txt"',
                'printf "1\\n00:00:00,000 --> 00:00:01,000\\nText\\n" > "$out_dir/validation.srt"',
                'printf "{}\\n" > "$out_dir/validation.json"',
                'printf "{}\\n" > "$out_dir/metadata.json"',
                'printf "ok\\n" > "$out_dir/run.log"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    transcribe.chmod(0o755)
    return transcribe


def make_fake_ffmpeg(tmp_path: Path) -> None:
    make_fake_command(tmp_path, "ffmpeg", 'touch "${@: -1}"')


def make_fake_command(tmp_path: Path, name: str, body: str) -> None:
    command = tmp_path / "bin" / name
    command.parent.mkdir(exist_ok=True)
    command.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )
    command.chmod(0o755)


def smoke_env(tmp_path: Path, *, transcribe: Path, platform: str) -> dict[str, str]:
    env = {
        "PATH": f"{tmp_path / 'bin'}:/bin:/usr/bin",
        "LOCAL_STT_SMOKE_PLATFORM": platform,
        "LOCAL_STT_SMOKE_TRANSCRIBE": str(transcribe),
        "LOCAL_STT_SMOKE_PYTHON": "",
    }
    return env
