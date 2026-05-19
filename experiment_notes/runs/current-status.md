# Run Registry

Date: 2026-05-19

## Current Branch

```text
migrate-ioa-lite-full
```

## Migration Status

| Area | Status | Notes |
| --- | --- | --- |
| IoA-Lite prompter | static compile passed | Inherits target `SingleThreadPrompter` and preserves fallback validation. |
| `run_dialog.py` integration | static compile passed | Adds `ioa_lite`, `ioa_preset`, skip-video, and monitor. |
| evaluator | help/CLI verified | Defaults to IoA-Lite full, no-video evaluation. |
| Ollama/Qwen client | smoke verified | Local Ollama uses `OPENAI_MODEL=qwen3.5:27b` and `OLLAMA_REASONING_EFFORT=none`; visible planner responses now contain `EXECUTE` output. |
| RRT policy init robustness | smoke verified | Invalid LLM paths can fail IK during `PlannedPathPolicy` construction; runner now records the failed plan, rewinds, and continues instead of aborting the process. |
| experiment docs | initialized | Keep concise run outcomes here. |

## Smoke Gates

| Gate | Command | Result | Notes |
| --- | --- | --- | --- |
| static compile | `python -m py_compile ...` | passed | System Python compile completed. |
| plan baseline | `sort`, `comm_mode=plan`, `tsteps=1` | pending | Pipeline non-regression. |
| IoA schema | `sort`, `comm_mode=ioa_lite`, `ioa_preset=schema`, `tsteps=1` | pending | Minimal IoA prompt. |
| IoA full | `sort`, `comm_mode=ioa_lite`, `ioa_preset=full`, `tsteps=1` | passed | After Ollama reasoning suppression, visible LLM response contained parser-compatible `EXECUTE` output. |
| six-task smoke | all tasks, `tsteps=1` / task auto settings | partial, no crash | Sort/rope/sweep reached execution. Sandwich and pack IK failures were handled by `rrt_policy_init_error` + rewind. Pack continued through 12 auto-adjusted steps and made partial progress but did not succeed. |

## Local Verification Notes

- `python -m py_compile evaluator.py run_dialog.py prompting/*.py rocobench/rrt.py rocobench/rrt_multi_arm.py rocobench/policy.py` passed.
- `git diff --check` passed for tracked modifications.
- `python evaluator.py --help` passed.
- Local Ollama validation should use `OPENAI_MODEL=qwen3.5:27b` and `OLLAMA_REASONING_EFFORT=none`.
- On 2026-05-19, `ioa_lite/full` sort smoke produced a non-empty LLM response and executed through RRT/MuJoCo.
- On 2026-05-19, the initial six-task smoke reached a runner crash in pack when an LLM path failed IK during `PlannedPathPolicy` construction.
- On 2026-05-19, rerun after the policy-init patch confirmed the runner no longer crashes on IK initialization failures. Sandwich logged `failed to compute IK for Dave`; pack logged repeated `failed to compute IK for Alice`, rewound, and continued until step 11.
- Current strategy quality gaps: rope can regress into repeated invalid pick plans after partial progress; pack tends to repeat unreachable Alice bread paths after a successful banana pick/place; sandwich can choose geometrically infeasible first picks.

## Result Recording Rules

Update this file whenever:

- a smoke gate is run or rerun;
- a full pilot starts producing interpretable outcomes;
- a task changes status;
- a run name, output directory, seed policy, or task order becomes the new recommended baseline.

Keep verbose logs in `logs/` or evaluator output directories, not in this file.
