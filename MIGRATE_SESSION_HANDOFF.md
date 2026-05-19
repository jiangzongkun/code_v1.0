# Migration Session Handoff: Merge `code_original` Architecture Into `code_v1.0`

Date: 2026-05-19

This handoff is for a new session whose main task is to migrate the useful
architecture from:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_original
```

into:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0
```

Do not modify `SESSION_HANDOFF.md` for this task. That file is mostly a server
run handoff. This document is the migration handoff for combining two code
paths.

## Current Objective

Use `code_v1.0` as the target scoring-oriented codebase, while migrating the
architecture and tooling that `code_original` already implemented:

```text
code_original strengths:
  IoA-Lite prompter mode
  + evaluator entry-point robustness
  + monitor/heartbeat for long runs
  + API/Ollama timeout hardening
  + experiment note discipline

code_v1.0 strengths:
  deterministic task fallback planners
  + candidate validation
  + task-specific prompt and state hints
  + parser/path/RRT/policy physical execution fixes
```

The intended final merged system is not "original IoA". It should be described
as:

```text
IoA-inspired local communication-state coordinator
+ task-aware fallback and execution robustness
```

Do not claim that this faithfully reproduces the full IoA runtime.

## Important Repositories And Documents

Source repo:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_original
```

Target repo:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0
```

Official task statement:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_original/Task/题目：多机器人协同的具身操作.md
```

Target-side analysis already written:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0/TECHNICAL_SUMMARY.md
/Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0/COMPARISON_WITH_CODE_ORIGINAL.md
```

Source experiment notes:

```text
/Users/charly/VScodeProjects/ChuangZhi/code_original/EXPERIMENT_NOTES.md
/Users/charly/VScodeProjects/ChuangZhi/code_original/experiment_notes/runs/current-status.md
/Users/charly/VScodeProjects/ChuangZhi/code_original/STARTUP_RUNBOOK.md
```

## Hard Constraints From The Task

Keep these constraints intact in the merged target:

- Model must be under 80B parameters.
- Each run has a 10 minute timeout. Current code uses 600 seconds.
- Official success should remain the environment reward/done logic.
- Do not change task reward functions or success thresholds to inflate scores.
- Do not remove parser, feedback, RRT, or MuJoCo execution from the scoring path.
- It is acceptable to modify interaction/cooperation strategy and implement
  custom planning methods.
- It is acceptable to skip video export for speed, as video is an artifact and
  not the success metric.

`num_replans` is not explicitly fixed by the task, but the safest default is
still 5 because it matches the original `run_dialog.py` default. Test larger
values only as an ablation and only keep them if they improve success without
causing timeout.

## Source Architecture To Migrate

### 1. IoA-Lite Prompter

Source file:

```text
code_original/prompting/ioa_lite_prompter.py
```

Core class:

```text
IoaLitePrompter(SingleThreadPrompter)
```

It inherits the existing centralized execution loop and only augments prompt
state. It currently adds:

```text
[COMM_HEADER]
[OBSERVATION_CONTEXT]
[ROBOT_REGISTRY]
[CONVERSATION_CONTEXT]
[GROUP_STATE]
[TASK_TABLE]
[AWAIT_STATUS]
[DYNAMIC_PLAN]
[VALIDATOR_FEEDBACK]
[LOCAL_TURN_TRACE]
[COMPLETION_CHECK]
[OUTPUT_CONTRACT]
```

Important implementation traits:

- `self.use_history = False` to prevent raw response-history prompt bloat.
- Presets: `schema`, `task_table`, `feedback`, `fsm`, `dynamic`, `full`.
- Task profiles for all six tasks.
- Dynamic robot registry includes holding, visible objects, reachable regions,
  reachable objects, preferred role, and collaboration information where
  available.
- Completion checks currently add task-specific terminal guidance for sweep and
  cabinet.

Migration principle:

```text
Port the class as a separate file first.
Do not rewrite code_v1.0 SingleThreadPrompter around it.
```

### 2. `run_dialog.py` Integration

Source behavior to migrate:

- Add `comm_mode="ioa_lite"` as a new mode.
- Lazy import `IoaLitePrompter` only inside the `ioa_lite` branch.
- Add `--ioa_preset`.
- Add `--skip_video`.
- Add `--monitor_interval`.
- Preserve baseline `plan`, `chat`, and `dialog` paths.

Critical source design:

```text
if llm_comm_mode in ["plan", "chat"]:
    use SingleThreadPrompter
elif llm_comm_mode == "ioa_lite":
    from prompting.ioa_lite_prompter import IoaLitePrompter
    use IoaLitePrompter(..., comm_mode="plan", ioa_preset=...)
elif llm_comm_mode == "dialog":
    use DialogPrompter
```

Target warning:

`code_v1.0/run_dialog.py` already has task-specific auto settings for rope and
pack, plus `--fallback_first`, `--rrt_timeout`, and `--skip_smooth_path`.
Do not overwrite those. Add IoA-Lite as an additional mode around the existing
runner structure.

### 3. Evaluator Robustness

Source file:

```text
code_original/evaluator.py
```

Target repo may not have a tracked evaluator. If the official evaluator must be
run from target, port the source evaluator carefully.

Useful source features:

- Default `comm_mode=ioa_lite`.
- Default `ioa_preset=full`.
- Default `skip_video=True`.
- `--save_video` opt-in.
- `--monitor_interval`.
- `num_replans=5`.
- Robust result JSON selection using files named:

```text
steps*_success_*.json
```

Important: this is allowed only as an entry-point/runtime robustness change. It
must not change metric formulas:

```text
success_rate = success_cnt / total_cnt
timeout_count unchanged
avg_steps unchanged
```

### 4. API / Local Model Robustness

Source code has DashScope/OpenAI-compatible/Ollama hardening inside
`prompting/plan_prompter.py`.

Target code already has a cleaner `prompting/llm_client.py`. Prefer target's
client abstraction, but migrate missing source behaviors if needed:

- no-thinking support for Ollama/Qwen;
- timeout or subprocess protection for LLM calls;
- retry/backoff;
- avoid empty `choices[0].message.content`;
- avoid printing or committing API keys.

Do not paste keys from:

```text
code_original/scripts/setup_dashscope_env.sh
code_original/scripts/dashscope_profile.sh
```

Use those only locally when running API tests.

### 5. Monitor / Heartbeat

Source `run_dialog.py` has bootstrap and per-run monitoring so long RoCoBench
runs do not appear stuck for 10 minutes.

Migrate the concept, not necessarily exact code:

- print task, run, step, replan, elapsed time;
- report remaining timeout budget;
- keep logs useful during RRT and LLM stalls;
- do not change execution semantics.

This is important because a real task run can legitimately take several
minutes before returning.

## Target Architecture To Preserve

Do not discard these `code_v1.0` capabilities. They are likely useful for final
score.

### 1. Deterministic Fallback Planner

Target file:

```text
code_v1.0/prompting/plan_prompter.py
```

Important target methods:

```text
build_fallback_response()
build_fallback_candidates()
validate_fallback_candidates()
build_sort_fallback_response()
build_sweep_fallback_response()
build_sandwich_fallback_response()
build_rope_fallback_candidates()
build_pack_fallback_candidates()
build_cabinet_fallback_response()
```

This is the strongest scoring mechanism in `code_v1.0`. Preserve it.

### 2. Candidate Validation

Target fallback candidates are parsed and checked by
`FeedbackManager.give_feedback()` before execution. This is exactly the kind of
validator-guided planning we wanted from the literature review.

Recommended merge direction:

```text
IoaLitePrompter should inherit target SingleThreadPrompter,
so it can reuse target fallback/candidate validation automatically.
```

Do not copy `code_original`'s parent class implementation over the target
prompter, because that would erase fallback logic.

### 3. Task-Specific Runtime Settings

Target `run_dialog.py` adjusts rope and pack:

- rope: `action_and_path`, split parsed plans, shorter horizon/config changes.
- pack: `action_and_path`, split parsed plans, at least 12 tsteps, shorter RRT
  timeout, serial fallback behavior.

Preserve these if they are still compatible with official constraints. If
changing tsteps or RRT timeout, record it in experiment notes and separate
"scoring configuration" from "baseline-equivalent configuration".

### 4. Parser / Path / RRT Fixes

Target has execution-layer changes in:

```text
prompting/parser.py
rocobench/policy.py
rocobench/rrt.py
rocobench/rrt_multi_arm.py
rocobench/envs/task_*.py
```

These should not be overwritten by source files. Review them before modifying.
They likely improve Rope and Pack.

## Recommended Merge Strategy

### Phase 0: Safety Snapshot

In both repos:

```bash
git status --short --untracked-files=all
git log --oneline -5
```

If target has uncommitted work, do not overwrite it. Commit or stash only after
the user approves.

### Phase 1: Add IoA-Lite As A New Target File

Copy/adapt:

```text
code_original/prompting/ioa_lite_prompter.py
```

into:

```text
code_v1.0/prompting/ioa_lite_prompter.py
```

Required adaptation:

- Ensure it imports target `SingleThreadPrompter`.
- Ensure constructor signature matches target `SingleThreadPrompter`.
- Ensure it passes through target-specific args such as `fallback_first` if
  needed.
- Do not import it in `prompting/__init__.py` unless the target already uses
  that pattern.

First verification:

```bash
python -m py_compile prompting/ioa_lite_prompter.py
```

### Phase 2: Add `ioa_lite` Mode To Target `run_dialog.py`

Make minimal edits:

- add `ioa_lite` to `--comm_mode` choices;
- add `--ioa_preset`;
- add `--skip_video` if absent;
- add `--monitor_interval` if absent;
- add lazy import branch for `IoaLitePrompter`;
- keep target fallback, parser, and runtime task settings untouched.

The target currently does not have the exact same runner structure as source,
so do not paste the source branch blindly. Match target constructor arguments.

### Phase 3: Integrate IoA-Lite With Target Fallback

Preferred behavior:

```text
--comm_mode ioa_lite --fallback_first
```

should mean:

1. Target deterministic fallback candidates are tried/validated.
2. If fallback is not enough, IoA-Lite prompt sections guide the LLM.
3. Final output still goes through target parser/feedback/RRT/MuJoCo.

If `IoaLitePrompter` overrides too much and bypasses fallback, adjust it so
target `SingleThreadPrompter.prompt_one_round()` remains the parent execution
loop.

### Phase 4: Port Evaluator/Monitor Carefully

If target needs an official runner:

- add or adapt `evaluator.py` from source;
- default to `comm_mode=ioa_lite`;
- keep `num_runs=5`, `tsteps` and `run_timeout=600` unless deliberately
  configured;
- keep result metric formulas unchanged;
- skip video by default for speed;
- add `--save_video` for artifacts;
- robustly select `steps*_success_*.json`.

### Phase 5: Run Smoke Tests Before Full Runs

Use real one-step runs, not `num_runs=0`.

Minimum target smoke:

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode plan --num_replans 1
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset schema --num_replans 1
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset full --num_replans 1
```

Then run six-task smoke:

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

### Phase 6: Full Pilot And Targeted Iteration

Do not diagnose from smoke failures unless smoke cannot execute at all. For
strategy failures, run full pilots.

Initial full-pilot order:

```text
sort -> sandwich -> rope -> sweep -> cabinet -> pack
```

Use one seed first, then targeted reruns:

```bash
python run_dialog.py \
  --task <task> \
  --num_runs 1 \
  --skip_display \
  --skip_video \
  --tsteps 10 \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 5 \
  --monitor_interval 30 \
  --run_timeout 600 \
  --seed 0
```

For target `pack`, respect target's existing automatic tstep/path settings if
they are still in `run_dialog.py`.

## Known Current Results From Source

Do not treat these as target results. They are only source-repo guidance.

Source DashScope/API:

- sort: success
- sweep: success
- sandwich: success
- cabinet: failed from near-terminal repair / timeout
- rope: failed from API attempts exhaustion
- pack: failed from API attempts exhaustion / partial progress

Source local Ollama round 1:

- sort: success
- sandwich: success
- rope: success
- sweep: failed after many successful executions but no final success
- cabinet: failed from near-terminal all-WAIT
- pack: failed from timeout during near-bin placement

Most useful source lessons:

- smoke passing is only a pipeline gate;
- diagnose failures from full pilots;
- sweep needs DUMP / remaining-cube terminal awareness;
- cabinet needs anti-all-WAIT near terminal state;
- pack needs sequential/collision-aware bin placement and occupancy tracking.

## What Not To Claim

Do not say the merged method is original IoA or a faithful IoA runtime.

The current source IoA-Lite does not implement:

- IoA server/client/WebSocket routing;
- real Alice/Bob/Chad clients;
- networked `AgentMessage` passing;
- true next-speaker selection;
- formal proposal/vote/pause/trigger flow;
- `_rephrase_task_to_tool_agent()` LLM extraction;
- per-agent independent prompt -> discussion -> conclusion.

The honest claim:

```text
We localize IoA's protocol-level coordination abstractions into RoCoBench:
robot registry, communication header, task table, dynamic plan, await status,
typed feedback, local turn trace, and parser-compatible execution contract.
```

## Merge Risk Checklist

Before editing target, inspect:

```bash
git -C /Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0 status --short --untracked-files=all
rg -n "class SingleThreadPrompter|def prompt_one_round|fallback_first|build_fallback|validate_fallback" /Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0/prompting/plan_prompter.py
rg -n "comm_mode|fallback_first|rrt_timeout|skip_smooth_path|output_mode|split_parsed_plans" /Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0/run_dialog.py
```

High-risk conflicts:

- Target `SingleThreadPrompter.__init__` has different arguments from source.
- Target `prompt_one_round()` includes fallback logic; source IoA-Lite must not
  bypass it.
- Target parser has `action_and_path` assumptions for rope/pack.
- Target path/RRT changes should not be overwritten by source.
- Target may not track `evaluator.py`; add only if needed.

## Suggested File-Level Merge Plan

Port or adapt from source:

```text
prompting/ioa_lite_prompter.py
run_dialog.py: ioa_lite branch, ioa_preset, skip_video, monitor_interval
evaluator.py: only if target needs official evaluation entry
STARTUP_RUNBOOK.md: only commands relevant to target runtime
experiment_notes/: optional, if target will track experiment discipline
```

Preserve from target:

```text
prompting/llm_client.py
prompting/plan_prompter.py fallback methods
prompting/parser.py action/path fixes
rocobench/policy.py
rocobench/rrt.py
rocobench/rrt_multi_arm.py
rocobench/envs/task_*.py task hints
*_SOLUTION.md docs
```

Likely best implementation shape:

```text
code_v1.0/prompting/ioa_lite_prompter.py
  class IoaLitePrompter(target SingleThreadPrompter)
    override compose_system_prompt()
    override prompt_one_round() only to set current_obs and call super()
    override post_execute_update() to update IoA-Lite memory and call super()
    override post_episode_update() to clear IoA-Lite memory and call super()

code_v1.0/run_dialog.py
  add ioa_lite branch
  pass target fallback args through unchanged
```

## Verification Commands

Static checks in target:

```bash
cd /Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0
python -m py_compile run_dialog.py prompting/plan_prompter.py prompting/ioa_lite_prompter.py prompting/parser.py
git diff --check -- run_dialog.py prompting/ioa_lite_prompter.py
```

Baseline non-regression:

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode plan --num_replans 1
```

IoA-Lite smoke:

```bash
python run_dialog.py --task sort --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset full --num_replans 1 --monitor_interval 30
```

Six-task smoke:

```bash
for task in sort cabinet rope sweep sandwich pack; do
  python run_dialog.py --task "$task" --num_runs 1 --skip_display --skip_video --tsteps 1 --comm_mode ioa_lite --ioa_preset full --num_replans 1 --monitor_interval 30
done
```

Official-style evaluator, after smoke passes:

```bash
python evaluator.py --tasks sort cabinet rope sweep sandwich pack --comm_mode ioa_lite --ioa_preset full --num_replans 5 --monitor_interval 30
```

Only run this after target evaluator exists and one-step smoke is clean.

## Final Next Step For The New Session

Start with target inspection, not direct copying:

```bash
cd /Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0
git status --short --untracked-files=all
rg -n "class SingleThreadPrompter|def __init__|def compose_system_prompt|def prompt_one_round|fallback_first|validate_fallback_candidates" prompting/plan_prompter.py
rg -n "comm_mode|fallback_first|output_mode|rrt_timeout|skip_smooth_path|split_parsed_plans" run_dialog.py
```

Then implement Phase 1 and Phase 2 only. Do not attempt a full six-task
optimization until `ioa_lite` can run a one-step `sort` smoke in the target.
