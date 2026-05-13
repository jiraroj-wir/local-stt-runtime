# AGENTS.md

## Project

`local-stt-runtime` is a minimal local speech-to-text CLI for long English lecture recordings.

Version 1 uses:

- Bash scripts for the user-facing CLI.
- Podman as the runtime wrapper.
- Python inside the container.
- `ffmpeg` / `ffprobe` for audio inspection and preprocessing.
- `faster-whisper` for transcription.
- Local execution only. Do not call external STT APIs.

The main user command is:

```bash
./transcribe input.m4a
```

## Instruction-following rules

When working on this repository:

- Follow the requested task scope exactly.
- Do not add unrelated features.
- Keep each change small and reviewable.
- Prefer simple, boring, maintainable code.
- If the task is ambiguous, make the smallest reasonable assumption and state it in the PR.
- If a task conflicts with this file, mention the conflict before changing the design.
- If something cannot be tested in Codex Cloud, explain why and suggest a local validation step.

## Version 1 constraints

Do not implement these unless explicitly requested:

- Directory batch processing.
- Speaker diarization.
- Speaker labels.
- Summarization.
- LLM cleanup.
- GUI.
- `whisper.cpp`.
- macOS Metal support.
- Manual audio chunking.
- GPU usage limiting.
- ETA preflight.
- Interactive keypress progress UI.
- Host Python virtual environment.
- Large model downloads in automated tests.

## Runtime targets

Primary target:

```text
Linux + NVIDIA CUDA + Podman + faster-whisper
```

Fallback targets:

```text
Linux CPU through Podman
macOS CPU through Podman
```

The project may check for Podman, NVIDIA driver, NVIDIA Container Toolkit, and CDI GPU availability, but it must not install these host-level dependencies automatically.

## Default behavior

`./transcribe input.m4a` should:

1. Validate the input file.
2. Check whether the local runtime image is ready.
3. If the runtime is not ready, run setup automatically.
4. Try CUDA mode first when available.
5. Fall back to CPU automatically if CUDA/CDI is unavailable.
6. Write output files.
7. Print clear start, progress, and finish summaries.

If the user passes `--gpu`, CUDA failure should stop instead of falling back.

## Output behavior

Default command:

```bash
./transcribe audio/lecture.m4a
```

should create:

```text
audio/lecture/
├── lecture.txt
├── lecture.srt
├── lecture.json
├── run.log
└── metadata.json
```

With `--out DIR`:

```bash
./transcribe audio/lecture.m4a --out transcripts
```

should create:

```text
transcripts/
├── lecture.txt
├── lecture.srt
├── lecture.json
├── run.log
└── metadata.json
```

`--out` means use this exact output directory.

## Model defaults

GPU default:

```text
model: Systran/faster-whisper-large-v3
device: cuda
compute_type: int8
beam_size: 5
language: en
vad_filter: true
condition_on_previous_text: false
word_timestamps: false
batch_size: 1
```

CPU default:

```text
model: Systran/faster-whisper-medium.en
device: cpu
compute_type: int8
beam_size: 5
language: en
vad_filter: true
condition_on_previous_text: false
cpu_threads: auto
```

Do not implement automatic model fallback in v1 unless explicitly requested.

## Terminal output style

Use plain terminal output with readable sections.

Good style:

```text
────────────────────────────────────────
LOCAL-STT-RUNTIME

Input
  File       : audio/lecture.m4a
  Duration   : 02:13:42
  Audio      : AAC, 44.1 kHz, stereo

Runtime
  Mode       : auto
  Selected   : CUDA
  Model      : Systran/faster-whisper-large-v3
────────────────────────────────────────
```

Avoid noisy, overly decorative, or unclear output.

## Testing rules

After completing a task, run relevant tests before opening or finalizing the PR.

Preferred commands:

```bash
ruff check .
pytest
shellcheck transcribe setup
```

If a command cannot run because the related files do not exist yet, state that clearly in the PR.

Automated tests should not require:

- Real CUDA GPU access.
- Real CDI setup.
- Real Podman GPU passthrough.
- Large Whisper model downloads.
- Long audio files.

Use mocks, fake command outputs, temporary files, and small fixtures where possible.

## PR summary requirements

Every PR should include:

- What changed.
- Why it changed.
- Files added or modified.
- Exact tests run.
- Test results.
- Known limitations.
- Assumptions made.

If tests fail, report the failure honestly and explain the likely cause.
