# Experiment Notes

Date: 2026-05-19

This repository is the target scoring-oriented merge of `code_v1.0` with the
IoA-Lite coordination-state architecture from `code_original`.

## Method Label

Use this description in run notes and reports:

```text
IoA-inspired local communication-state coordinator
+ task-aware fallback and execution robustness
```

Do not claim this is a faithful full IoA runtime.

## Configuration Classes

Baseline-equivalent:

- `comm_mode=plan` or the original requested baseline.
- No mandatory `--fallback_first` unless the target task already enables it internally.
- Keep default reward, done, parser, feedback, RRT, and MuJoCo execution.

Scoring:

- `comm_mode=ioa_lite`
- `ioa_preset=full`
- `num_replans=5`
- `run_timeout=600`
- `skip_video=True`
- `monitor_interval=30`
- Preserve target runtime settings: rope uses 6 tsteps; pack uses at least 12 tsteps and a 30-iteration RRT timeout.

Ablation:

- Compare `plan` versus `ioa_lite`.
- Compare no `--fallback_first` versus `--fallback_first`.
- Compare `ioa_preset=schema` versus `ioa_preset=full`.

## Invariants

- Official success remains environment reward/done logic.
- Do not modify task reward functions or success thresholds.
- Parser, feedback, RRT, policy, and MuJoCo execution remain on the scoring path.
- Video export is an artifact only and may be skipped for speed.
- This migration does not change `prompting/llm_client.py`.

## Recommended Smoke Commands

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode plan --num_replans 1
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset schema --num_replans 1
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset full --num_replans 1
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

Evaluator pilot:

```bash
python evaluator.py \
  --tasks sort cabinet rope sweep sandwich pack \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 5 \
  --monitor_interval 30 \
  --run_timeout 600
```
