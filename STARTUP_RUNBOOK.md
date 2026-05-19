# Target Runtime Runbook

Date: 2026-05-19

This runbook is for the target repository:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0
```

## Environment

Use the existing project environment. For local MuJoCo rendering on headless
machines, set the appropriate backend before running tasks:

```bash
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-roco
mkdir -p "$MPLCONFIGDIR"
```

Configure the LLM using the repository's existing OpenAI-compatible client
environment. This migration does not add new LLM client variables.

## Smoke Tests

Baseline non-regression:

```bash
python run_dialog.py \
  --task sort \
  --num_runs 1 \
  --skip_display \
  --skip_video \
  --tsteps 1 \
  --comm_mode plan \
  --num_replans 1
```

IoA-Lite:

```bash
python run_dialog.py \
  --task sort \
  --num_runs 1 \
  --skip_display \
  --skip_video \
  --tsteps 1 \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 1 \
  --monitor_interval 30
```

Six-task smoke:

```bash
for task in sort cabinet rope sweep sandwich pack; do
  python run_dialog.py \
    --task "$task" \
    --num_runs 1 \
    --skip_display \
    --skip_video \
    --tsteps 1 \
    --comm_mode ioa_lite \
    --ioa_preset full \
    --num_replans 1 \
    --monitor_interval 30
done
```

## Evaluator

Official-style pilot:

```bash
python evaluator.py \
  --tasks sort cabinet rope sweep sandwich pack \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 5 \
  --monitor_interval 30 \
  --run_timeout 600
```

Use `--save_video` only when artifacts are needed. By default evaluator skips
video export for speed.
