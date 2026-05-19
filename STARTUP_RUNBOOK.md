# RoCoBench IoA-Lite Startup Runbook

Date: 2026-05-19

This runbook matches the current Linux workspace:

```text
/inspire/qb-ilm2/project/26summer-camp-09/26220358/workspace/code_v1.0
```

Use the shared Ollama runtime from:

```text
/inspire/qb-ilm2/project/26summer-camp-09/26220358/workspace/CZ-SummerCamp-G11/runtime
```

## 1. Start Ollama

Run this in a separate terminal or tmux pane:

```bash
export PROJECT_ROOT=/inspire/qb-ilm2/project/26summer-camp-09/26220358
export OLLAMA_RUNTIME=$PROJECT_ROOT/workspace/CZ-SummerCamp-G11/runtime

mkdir -p "$OLLAMA_RUNTIME/ollama_home" "$OLLAMA_RUNTIME/ollama_models"

export HOME=$OLLAMA_RUNTIME/ollama_home
export OLLAMA_MODELS=$OLLAMA_RUNTIME/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

ollama serve
```

In another terminal, the expected model name is:

```bash
export HOME=/inspire/qb-ilm2/project/26summer-camp-09/26220358/workspace/CZ-SummerCamp-G11/runtime/ollama_home
export OLLAMA_MODELS=/inspire/qb-ilm2/project/26summer-camp-09/26220358/workspace/CZ-SummerCamp-G11/runtime/ollama_models
export OLLAMA_HOST=127.0.0.1:11434

ollama list
```

The current validation model is:

```bash
qwen3.5:27b
```

## 2. Prepare RoCoBench Shell

Run this in the `code_v1.0` experiment terminal:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate roco

cd /inspire/qb-ilm2/project/26summer-camp-09/26220358/workspace/code_v1.0
export PY=/root/miniconda3/envs/roco/bin/python

export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=qwen3.5:27b
export OPENAI_TIMEOUT=600
export OLLAMA_REASONING_EFFORT=none

export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-roco
mkdir -p "$MPLCONFIGDIR"
```

Important: use `$PY`, not the system `python`. The system Python in this
workspace may miss RoCoBench dependencies such as `numpy`.

`OLLAMA_REASONING_EFFORT=none` is required for the local Qwen/Ollama path.
Without it, the model can return HTTP 200 while spending all output tokens on
hidden reasoning and leaving the visible planner response empty.

## 3. Static Checks

Run these before runtime validation:

```bash
$PY -m py_compile \
  run_dialog.py \
  evaluator.py \
  prompting/llm_client.py \
  prompting/plan_prompter.py \
  prompting/ioa_lite_prompter.py \
  prompting/parser.py \
  rocobench/rrt.py \
  rocobench/rrt_multi_arm.py \
  rocobench/policy.py

git diff --check
$PY run_dialog.py --help
$PY evaluator.py --help
```

## 4. One-Task Smoke

Start with Sort IoA-Lite. This validates:

```text
Ollama -> IoA-Lite prompt -> parser -> feedback -> RRT -> MuJoCo -> no-video save
```

```bash
$PY run_dialog.py \
  --task sort \
  --num_runs 1 \
  --skip_display \
  --skip_video \
  --tsteps 1 \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 1 \
  --monitor_interval 30 \
  --run_name ollama_sort_ioa_smoke_reasoning_none
```

For `tsteps=1`, `success=False` is normal. The gate is whether the run reaches
LLM response, parsing, RRT planning, execution, and HTML/JSON output without
crashing.

To confirm the LLM output was visible and not fallback-only:

```bash
sed -n '1,220p' data/ollama_sort_ioa_smoke_reasoning_none/run_*/step_0/prompts/replan0_*.json
```

The `Planner` field should contain an `EXECUTE` plan.

## 5. Six-Task Smoke

Run the six pipeline gates:

```bash
for task in sort cabinet rope sweep sandwich pack; do
  $PY run_dialog.py \
    --task "$task" \
    --num_runs 1 \
    --skip_display \
    --skip_video \
    --tsteps 1 \
    --comm_mode ioa_lite \
    --ioa_preset full \
    --num_replans 1 \
    --monitor_interval 30 \
    --run_name "ollama_${task}_ioa_smoke_reasoning_none"
done
```

Known interpretation from current validation:

- Sort, rope, sweep reached executable plans.
- Rope can finish successfully even in the smoke because `run_dialog.py`
  automatically uses rope-specific runtime settings.
- Cabinet and sandwich may fail at parser/format level with only one replan.
- Pack may expose bad LLM path geometry. Current `run_dialog.py` should record
  policy initialization or IK failure as a failed plan instead of crashing the
  whole loop.

For parser-format tasks, rerun with more feedback attempts:

```bash
for task in cabinet sandwich; do
  $PY run_dialog.py \
    --task "$task" \
    --num_runs 1 \
    --skip_display \
    --skip_video \
    --tsteps 1 \
    --comm_mode ioa_lite \
    --ioa_preset full \
    --num_replans 5 \
    --monitor_interval 30 \
    --run_name "ollama_${task}_ioa_smoke_replans5"
done
```

For pack crash regression:

```bash
$PY run_dialog.py \
  --task pack \
  --num_runs 1 \
  --skip_display \
  --skip_video \
  --tsteps 1 \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 1 \
  --monitor_interval 30 \
  --run_name ollama_pack_ioa_smoke_policy_init_fix
```

## 6. Full Pilots

After smoke tests execute cleanly, run full pilots one task at a time. Use this
order first:

```text
sort -> sandwich -> rope -> sweep -> cabinet -> pack
```

Template:

```bash
$PY run_dialog.py \
  --task sort \
  --num_runs 1 \
  --skip_display \
  --skip_video \
  --tsteps 10 \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 5 \
  --monitor_interval 30 \
  --run_timeout 600 \
  --seed 0 \
  --run_name ollama_ioa_full_sort_s0
```

Repeat by changing `--task` and `--run_name`.

Rope and pack are automatically adjusted inside `run_dialog.py`:

- Rope uses `action_and_path`, split parsed plans, and 6 tsteps when feedback is
  enabled.
- Pack uses `action_and_path`, split parsed plans, at least 12 tsteps, and a
  reduced RRT timeout.

## 7. Evaluator Pilot

Only run evaluator after the one-task and six-task smoke gates are clean:

```bash
$PY evaluator.py \
  --tasks sort cabinet rope sweep sandwich pack \
  --comm_mode ioa_lite \
  --ioa_preset full \
  --num_replans 5 \
  --monitor_interval 30 \
  --run_timeout 600
```

By default evaluator skips video export. Use `--save_video` only when an
artifact is needed.

## 8. Ablations

Recommended comparisons:

```text
plan, no fallback_first
plan, fallback_first
ioa_lite full, no fallback_first
ioa_lite full, fallback_first
ioa_lite schema, no fallback_first
```

Keep method attribution precise:

```text
IoA-inspired local communication-state coordinator
+ task-aware fallback and execution robustness
```

Do not describe this system as a faithful full IoA runtime.
