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
| experiment docs | initialized | Keep concise run outcomes here. |

## Smoke Gates

| Gate | Command | Result | Notes |
| --- | --- | --- | --- |
| static compile | `python -m py_compile ...` | passed | System Python compile completed. |
| plan baseline | `sort`, `comm_mode=plan`, `tsteps=1` | pending | Pipeline non-regression. |
| IoA schema | `sort`, `comm_mode=ioa_lite`, `ioa_preset=schema`, `tsteps=1` | pending | Minimal IoA prompt. |
| IoA full | `sort`, `comm_mode=ioa_lite`, `ioa_preset=full`, `tsteps=1` | pending | Full IoA prompt. |
| six-task smoke | all tasks, `tsteps=1` | pending | Pipeline gate only. |

## Local Verification Notes

- `python -m py_compile evaluator.py run_dialog.py prompting/*.py rocobench/rrt.py rocobench/rrt_multi_arm.py rocobench/policy.py` passed.
- `git diff --check` passed for tracked modifications.
- `python evaluator.py --help` passed.
- `run_dialog.py --help` and runtime smoke are blocked in the current local environment because `open3d` is not installed in the available Python/conda environment. This is an environment dependency issue, not a known migration syntax failure.

## Result Recording Rules

Update this file whenever:

- a smoke gate is run or rerun;
- a full pilot starts producing interpretable outcomes;
- a task changes status;
- a run name, output directory, seed policy, or task order becomes the new recommended baseline.

Keep verbose logs in `logs/` or evaluator output directories, not in this file.
