# IoA-Lite Full Migration Action Plan

Date: 2026-05-19

This document is the action blueprint for merging the useful architecture from:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_original
```

into the target scoring-oriented repository:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0
```

The target system is:

```text
IoA-inspired local communication-state coordinator
+ task-aware fallback and execution robustness
```

Do not describe the merged system as a faithful full IoA runtime.

## Global Principles

Use `code_v1.0` as the mainline. Do not overwrite its core scoring stack.

Preserve from `code_v1.0`:

- `prompting/plan_prompter.py` fallback planner and candidate validation.
- `prompting/parser.py`.
- `rocobench/policy.py`.
- `rocobench/rrt.py`.
- `rocobench/rrt_multi_arm.py`.
- `rocobench/envs/task_*.py`.
- `run_dialog.py` task-specific settings for rope and pack.

Migrate from `code_original`:

- `prompting/ioa_lite_prompter.py`.
- `run_dialog.py` support for `comm_mode=ioa_lite`, `ioa_preset`, monitor, and skip-video behavior.
- `evaluator.py`.
- Experiment notes and runbook discipline.

## Phase 0: Safety Snapshot

Before code edits:

```bash
git status --short --untracked-files=all
git log --oneline -5
```

Do not stash, remove, or overwrite existing untracked analysis documents.

## Phase 1: Add IoA-Lite Prompter

Add:

```text
prompting/ioa_lite_prompter.py
```

Adapt `code_original/prompting/ioa_lite_prompter.py` so it inherits target
`SingleThreadPrompter`.

Requirements:

- `prompt_one_round()` must call `super().prompt_one_round()` so target fallback, parser, feedback, RRT, and MuJoCo remain on the scoring path.
- Constructor must pass target-specific kwargs, including `fallback_first`.
- Keep `self.use_history = False` to avoid raw prompt-history bloat.
- Preserve IoA-Lite sections: communication header, observation context, robot registry, task table, dynamic plan, validator feedback, local turn trace, completion check, and output contract.
- Adjust `OUTPUT_CONTRACT` based on `self.use_waypoints`; rope and pack require `PATH`.
- This fix is target-only. Do not modify `/Users/charly/VScodeProjects/ChuangZhi/code_original`.

Verification:

```bash
python -m py_compile prompting/ioa_lite_prompter.py
```

## Phase 2: Integrate `ioa_lite` In `run_dialog.py`

Make minimal additive edits:

- Add `ioa_lite` to `--comm_mode` choices.
- Add `--ioa_preset`.
- Add lazy import branch for `IoaLitePrompter`.
- Pass `fallback_first=self.fallback_first` into `IoaLitePrompter`.
- Preserve `plan`, `chat`, and `dialog` branches.
- Preserve target `--fallback_first`, `--rrt_timeout`, and `--skip_smooth_path`.
- Preserve rope and pack automatic runtime settings.

Desired branch structure:

```text
plan/chat -> SingleThreadPrompter
ioa_lite -> IoaLitePrompter(..., comm_mode="plan", fallback_first=...)
dialog -> DialogPrompter
```

## Phase 3: Complete Skip-Video Support

Implement no-video evaluation without changing reward or done logic.

Needed layers:

- `run_dialog.py`: do not call `env.export_render_to_video()` when `--skip_video` is set.
- `prompting/display_utils.py`: add `include_video=False` support to `save_episode_html()`.
- `rocobench/envs/base_env.py`: add `record_video=False` support to avoid unnecessary camera render buffers.

Verification:

```bash
python -m py_compile run_dialog.py prompting/display_utils.py rocobench/envs/base_env.py
```

## Phase 4: Add Monitor / Heartbeat

Migrate the source `RunMonitor` concept into target `run_dialog.py`.

Monitor phases should include:

- bootstrap start.
- environment creation.
- run start.
- step start.
- LLM prompt start, done, and error.
- RRT planning start and done.
- execution progress.
- video skipped or exported.
- post-execute update.
- episode finish.

Default `--monitor_interval` should be `0` or disabled unless explicitly set.
Evaluation runs can use `--monitor_interval 30`.

This phase must not change execution semantics.

## Phase 5: Add Evaluator

Add:

```text
evaluator.py
```

Base it on `code_original/evaluator.py`, but match target `run_dialog.py`
arguments.

Defaults:

```text
comm_mode=ioa_lite
ioa_preset=full
skip_video=True
num_replans=5
run_timeout=600
monitor_interval=30
```

Use robust result selection:

```text
steps*_success_*.json
```

Keep metric formulas unchanged:

```text
success_rate = success_cnt / total_cnt
timeout_count unchanged
avg_steps unchanged
```

Do not modify reward functions or success thresholds.

Do not change `prompting/llm_client.py` as part of this migration.

## Phase 6: Add Experiment Discipline

Add or adapt:

```text
EXPERIMENT_NOTES.md
experiment_notes/runs/current-status.md
STARTUP_RUNBOOK.md
```

Content must use target commands and target paths.

Always separate:

- baseline-equivalent config.
- scoring config.
- ablation config.

Explicitly record that rope 6 tsteps and pack 12 tsteps are target scoring
runtime settings, not official reward changes.

## Phase 7: Verification Ladder

Static checks:

```bash
python -m py_compile run_dialog.py evaluator.py prompting/*.py rocobench/rrt.py rocobench/rrt_multi_arm.py rocobench/policy.py
git diff --check
```

Baseline non-regression:

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode plan --num_replans 1
```

IoA-Lite smoke:

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset schema --num_replans 1
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset full --num_replans 1
```

Six-task smoke:

```bash
for task in sort cabinet rope sweep sandwich pack; do
  python run_dialog.py --task "$task" --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset full --num_replans 1 --monitor_interval 30
done
```

Official-style pilot:

```bash
python evaluator.py --tasks sort cabinet rope sweep sandwich pack --comm_mode ioa_lite --ioa_preset full --num_replans 5 --monitor_interval 30
```

## Phase 8: Ablation And Attribution

Run at least these configurations:

```text
plan, no fallback_first
plan, fallback_first
ioa_lite schema, no fallback_first
ioa_lite full, no fallback_first
ioa_lite full, fallback_first
```

Keep method attribution precise:

- IoA-Lite coordination-state prompt layer.
- Deterministic fallback safety layer.
- Task-specific execution and motion-planning robustness.

Do not attribute all gains to IoA-Lite alone.

## Recommended Implementation Order

1. Phase 1-2: get `ioa_lite` callable in target.
2. Phase 3-4: add no-video and heartbeat for long runs.
3. Phase 5: add evaluator.
4. Phase 6: add target-specific experiment records.
5. Phase 7-8: run smoke, pilots, and ablations.

Every phase should have a clean verification gate before proceeding.
