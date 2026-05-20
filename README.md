# RocoBench 多机器人协作任务优化仓库

本仓库用于改进 RoCoBench 多机器人协作任务的执行稳定性、评测效率和成功率。当前版本围绕六个基础常规任务进行优化：

- Sort
- Cabinet
- Rope
- Sweep
- Sandwich
- Pack Grocery

各任务的策略、失败案例和优化记录分布在对应的方案文档中。最新一次六任务基础评测结果见 [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md)。

## 环境准备

推荐使用 Python 3.8，并通过 `uv` 管理环境：

```bash
uv sync
```

也可以使用已有环境直接运行：

```bash
python run_dialog.py --task sort --comm_mode plan --num_runs 1 --tsteps 3 --skip_display --run_name sort_smoke
```

## LLM 配置

默认通过 OpenAI-compatible 接口调用大模型，例如 Ollama：

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=qwen3.5:27b
export OLLAMA_MODEL=qwen3.5:27b
```

运行前建议先确认模型接口可用：

```bash
python -c "from prompting.llm_client import chat_completion,response_content; r=chat_completion(messages=[{'role':'user','content':'Say hello'}],max_tokens=30); print(repr(response_content(r)))"
```

## 基础评测命令

全部六个常规任务：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks sort cabinet rope sweep sandwich pack --runs 5 --full
```

单独测试 Rope：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks rope --runs 5 --tsteps 6 --timeout 600
```

如果需要强制先调用大模型，而不是优先使用确定性 fallback：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks rope --runs 5 --tsteps 6 --timeout 600 --no_fallback_first
```

## 最新基础评测结果

评测目录：`output/run_20260520_121100`

| 任务 | 成功率 | 成功次数 | 超时 | 平均成功步数 | 总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sort | 100.0% | 5/5 | 0 | 5.40 | 372.18s |
| Cabinet | 100.0% | 5/5 | 0 | 3.40 | 892.04s |
| Rope | 80.0% | 4/5 | 1 | 1.50 | 1098.72s |
| Sweep | 80.0% | 4/5 | 0 | 8.00 | 893.32s |
| Sandwich | 100.0% | 5/5 | 0 | 6.20 | 522.77s |
| Pack Grocery | 80.0% | 4/5 | 0 | 11.00 | 835.47s |

整体成功率：`27/30 = 90.0%`。

详细分析见 [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md)。

## 任务方案文档

- [Sort](SORT_SOLUTION.md)
- [Cabinet](CABINET_SOLUTION.md)
- [Rope](ROPE_SOLUTION.md)
- [Sweep](SWEEP_SOLUTION.md)
- [Sandwich](SANDWICH_SOLUTION.md)
- [Pack Grocery](PACK_GROCERY_SOLUTION.md)

## 输出目录

`run_dialog.py` 和 `evaluator.py` 会生成本地运行产物，例如：

```text
output/run_YYYYMMDD_HHMMSS/
|-- evaluator.log
`-- tasks/
    |-- 01_sort/
    |-- 02_cabinet/
    `-- ...
```

`output/`、`data/`、视频、日志、pickle 中间状态等均为本地评测产物，默认不提交到远程仓库。

## 常用检查命令

语法检查：

```bash
python -m compileall run_dialog.py prompting rocobench
```

查看某次评测摘要：

```bash
find output/run_20260520_121100/tasks -name summary.json -print -exec cat {} \;
```

查看失败反馈：

```bash
grep -R "IK failed\|Collision detected\|Parsing failed\|timed_out" -n output/run_20260520_121100/tasks
```

## 提交注意事项

远程仓库不应包含：

- 个人 API key 或代理配置
- 本地 LLM 客户端覆盖文件
- `data/`、`output/`、视频、日志、pickle 等运行产物
- 与当前任务无关的临时文件
