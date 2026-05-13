from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAINERFILE = REPO_ROOT / "Containerfile"
REQS = REPO_ROOT / "requirements-container.txt"


def test_containerfile_exists() -> None:
    assert CONTAINERFILE.exists()


def test_containerfile_has_cuda_base_image() -> None:
    content = CONTAINERFILE.read_text()
    assert "FROM nvidia/cuda:" in content


def test_containerfile_installs_ffmpeg() -> None:
    content = CONTAINERFILE.read_text()
    assert "ffmpeg" in content


def test_containerfile_copies_requirements() -> None:
    content = CONTAINERFILE.read_text()
    assert "COPY requirements-container.txt" in content


def test_requirements_container_has_faster_whisper_only_runtime_deps() -> None:
    lines = [line.strip() for line in REQS.read_text().splitlines() if line.strip()]
    assert "faster-whisper" in lines
    assert all("pytest" not in line for line in lines)
    assert all("ruff" not in line for line in lines)
