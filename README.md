# local-stt-runtime

Personal local speech-to-text tooling for long English lecture recordings.

The main command is:

```bash
./transcribe input.m4a
```

It runs `faster-whisper` inside a Podman container, writes transcript files next to the input by default, and keeps all speech-to-text work local. It does not call external STT APIs.

## Index

- [What It Does](#what-it-does)
- [Current Status](#current-status)
- [Requirements](#requirements)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Output Files](#output-files)
- [Runtime Behavior](#runtime-behavior)
- [Options](#options)
- [Models](#models)
- [Chunking](#chunking)
- [Performance Expectations](#performance-expectations)
- [Accuracy Expectations](#accuracy-expectations)
- [Troubleshooting](#troubleshooting)

## What It Does

`local-stt-runtime` transcribes local audio files to:

- plain text
- SRT subtitles
- JSON segments
- run metadata
- run log

It is designed for lecture and meeting recordings, mostly in English. The implementation favors portability and reliability over maximum speed.

Supported input extensions:

```text
.aac .m4a .mp3 .wav .flac .ogg
```

## Current Status

This is a personal v1 tool. It was built to make lecture recordings searchable and easier to study from.

Expected best path:

```text
Linux / Fedora + NVIDIA GPU + CUDA driver stack + Podman
```

Expected fallback paths:

```text
Linux CPU through Podman
macOS Apple Silicon CPU through Podman
```

macOS currently runs through CPU mode. `faster-whisper` in this container does not provide Apple Metal acceleration.

## Requirements

Required before running `./setup`:

- Git, to clone the repository.
- Bash.
- Podman installed and working.
- Internet access during setup and first transcription.
- Enough disk space for the container image, Hugging Face model cache, and temporary WAV/chunk files.

Required for normal CPU use:

- Podman only. Host `ffmpeg` is not required for normal transcription because `ffmpeg` is installed inside the container image.

Required for Linux GPU use:

- NVIDIA GPU.
- NVIDIA Linux driver installed on the host.
- `nvidia-smi` working on the host.
- NVIDIA Container Toolkit configured for Podman.
- NVIDIA CDI devices available to Podman.
- A working Podman CUDA probe with `--device nvidia.com/gpu=all`.

The runtime checks GPU access with this shape of command:

```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi
```

If that fails in auto mode, `./transcribe` falls back to CPU. If it fails with `--gpu`, the run stops.

## Setup

Clone the repository:

```bash
git clone https://github.com/jiraroj-wir/local-stt-runtime.git
cd local-stt-runtime
```

Build the local runtime image:

```bash
./setup
```

The setup step builds a Podman image named:

```text
local-stt-runtime
```

It also uses this cache directory:

```text
~/.cache/local-stt-runtime
```

The first real transcription may still take extra time because the Whisper model is downloaded into the cache.

## Quick Start

Run from a directory where the input path is available under the current working directory. Prefer relative paths.

```bash
./transcribe audio/lecture.m4a
```

Force CPU mode:

```bash
./transcribe audio/lecture.m4a --cpu
```

Force GPU mode:

```bash
./transcribe audio/lecture.m4a --gpu
```

Use an exact output directory:

```bash
./transcribe audio/lecture.m4a --out transcripts
```

Disable chunking on a stronger machine:

```bash
./transcribe audio/lecture.m4a --no-chunk
```

Use isolated chunks and adaptive CUDA OOM fallback on small VRAM GPUs:

```bash
./transcribe audio/lecture.m4a --isolate-chunks --adaptive-fallback
```

Preview resolved settings without running transcription:

```bash
./transcribe audio/lecture.m4a --dry-run
```

## Output Files

Default output:

```bash
./transcribe audio/lecture.m4a
```

Creates:

```text
audio/lecture/
  lecture.txt
  lecture.srt
  lecture.json
  metadata.json
  run.log
```

With `--out DIR`:

```bash
./transcribe audio/lecture.m4a --out transcripts
```

Creates:

```text
transcripts/
  lecture.txt
  lecture.srt
  lecture.json
  metadata.json
  run.log
```

`--out` means use that exact output directory.

The output directory may also contain a hidden `.work` directory with internal source WAVs, chunk WAVs, isolated chunk outputs, and fallback chunk outputs.

## Runtime Behavior

Default `./transcribe input.m4a` behavior:

1. Resolves the input and output paths.
2. Checks whether Podman is available.
3. Selects runtime mode from `--cpu`, `--gpu`, or default auto mode.
4. In auto mode, checks whether CUDA/CDI GPU access works.
5. Falls back to CPU if the CUDA/CDI check fails.
6. Builds the local runtime image automatically if it is missing.
7. Mounts the current working directory into the container at `/work`.
8. Mounts the repository into the container at `/repo`.
9. Mounts `~/.cache/local-stt-runtime` into the container at `/cache`.
10. Validates the input file extension inside the container.
11. Decodes the input to an internal 16 kHz mono WAV.
12. Applies loudness normalization when requested or when auto preprocessing decides the input is quiet.
13. Plans chunks from the canonical WAV duration.
14. Runs `faster-whisper`.
15. Writes transcript outputs, metadata, and log files.

Path expectation:

- The current working directory is mounted into the container.
- Use input and output paths that are reachable from the current working directory.
- Relative paths are the safest option.

## Options

`./transcribe` options:

| Option | Meaning |
| --- | --- |
| `--help` | Show CLI help. |
| `--list-presets` | Show runtime preset names. |
| `--gpu` | Force CUDA mode. GPU check failure stops the run. |
| `--cpu` | Force CPU mode. Skips CUDA probing. |
| `--out DIR` | Use `DIR` as the exact output directory. |
| `--chunk-minutes N` | Use `N` minute chunks for long audio. Default is `20`. |
| `--no-chunk` | Disable automatic chunking. |
| `--isolate-chunks` | Run each planned chunk in a fresh Python process. Slower, but releases memory between chunks. |
| `--adaptive-fallback` | For isolated chunk runs, retry CUDA OOM chunks as smaller fallback chunks. |
| `--preprocess auto` | Decode to canonical WAV and normalize only if volume detection says the input is quiet. Default. |
| `--preprocess off` | Decode to canonical WAV without loudness normalization. |
| `--preprocess normalize` | Decode to canonical WAV and always apply loudness normalization. |
| `--dry-run` | Print resolved settings without running Podman transcription. |

Preset names:

```text
linux-gpu  Linux with NVIDIA CUDA runtime.
linux-cpu  Linux CPU runtime through Podman.
macos-cpu  macOS CPU runtime through Podman.
```

## Models

GPU default:

| Setting | Value |
| --- | --- |
| model | `Systran/faster-whisper-large-v3` |
| device | `cuda` |
| compute type | `int8` |
| beam size | `5` |
| language | `en` |
| VAD filter | `true` |
| condition on previous text | `false` |
| word timestamps | `false` |
| batch size | `1` |

CPU default:

| Setting | Value |
| --- | --- |
| model | `Systran/faster-whisper-medium.en` |
| device | `cpu` |
| compute type | `int8` |
| beam size | `5` |
| language | `en` |
| VAD filter | `true` |
| condition on previous text | `false` |
| CPU threads | `auto` |

There is no general automatic model fallback in the normal path. Adaptive fallback is a specific CUDA OOM recovery mode for isolated chunk runs.

## Chunking

Chunking exists to make long recordings work on more machines, including small VRAM GPUs.

Default behavior:

- Chunking is enabled.
- The threshold is 20 minutes.
- Audio longer than the threshold is split into sequential chunks.
- Chunks are transcribed sequentially.
- Chunk timing is offset back into one final transcript.

Useful modes:

```bash
./transcribe lecture.m4a --chunk-minutes 10
```

Uses smaller planned chunks.

```bash
./transcribe lecture.m4a --no-chunk
```

Disables chunking. Use this on machines that can handle the full file in one run.

```bash
./transcribe lecture.m4a --isolate-chunks
```

Runs each chunk in a new Python process. This is slower because the model reloads per chunk, but it can reduce CUDA memory problems.

```bash
./transcribe lecture.m4a --isolate-chunks --adaptive-fallback
```

If a planned CUDA chunk runs out of memory, it is split into 5 minute fallback subchunks and retried through:

1. `large-v3` on CUDA
2. `medium.en` on CUDA
3. `medium.en` on CPU

Current limitation: chunk overlap and deduplication are not implemented.

## Performance Expectations

This is not intended to be the fastest STT setup. It is containerized and has chunking behavior to make it work on more machines.

Known reference machines:

- Fedora workstation, x86, NVIDIA GTX 1050 Ti, sufficient system RAM.
- MacBook Air M1, CPU mode through Podman, sufficient system RAM.

Expected behavior:

- Linux + NVIDIA GPU should be the best current path.
- GTX 1050 Ti class GPUs may need chunking, isolated chunks, or adaptive fallback for long files.
- CPU mode should work, but it can be much slower than GPU mode.
- Apple Silicon macOS currently runs in CPU mode. On the known MacBook Air M1 path, leaving long lectures overnight is reasonable. It should usually finish faster than 1:1 playback time, but that is not guaranteed.
- First runs are slower because model downloads happen on demand.

## Accuracy Expectations

The current setup has been lightly tested on lecture and meeting style audio.

Observed quality so far:

- Good enough for personal study notes and searchable transcripts.
- Quite accurate on clear English speech.
- Can miss small binding words.
- Can miss or alter occasional words.
- Not yet benchmarked against a controlled audiobook/reference-text dataset.

Formal accuracy benchmarking is planned.

## Troubleshooting

Podman missing:

```text
Error: podman is required but was not found in PATH.
```

Install Podman and rerun `./setup`.

GPU check fails in auto mode:

```text
CUDA/CDI GPU check failed; falling back to CPU.
```

The run continues on CPU.

GPU check fails with `--gpu`:

```text
Error: CUDA/CDI GPU check failed; cannot continue with --gpu.
```

Fix the host NVIDIA driver, NVIDIA Container Toolkit, or CDI setup, or run without `--gpu`.

CUDA out of memory:

```bash
./transcribe lecture.m4a --isolate-chunks --adaptive-fallback
```

If needed, also reduce planned chunk size:

```bash
./transcribe lecture.m4a --chunk-minutes 10 --isolate-chunks --adaptive-fallback
```

Input path does not resolve inside the container:

- Run from a directory that contains the input file.
- Use relative input paths.
- Keep output paths under the current working directory.

First run is slow:

- The container image may need to build.
- The CUDA probe image may need to pull.
- The Whisper model may need to download.
- Later runs reuse the local cache.
