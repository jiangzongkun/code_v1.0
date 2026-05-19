# RocoBench 协作任务仓库

本仓库用于改进 RoCoBench 多机器人协作任务的执行稳定性和成功率。各任务的具体优化思路、运行命令和验证记录统一写在对应的 `*_SOLUTION.md` 文件中。

## 环境准备

项目使用 Python 3.8。推荐通过 `uv` 管理环境：

```bash
uv sync
```

也可以使用已有 Python 环境直接运行：

```bash
python run_dialog.py --task sort --comm_mode plan --num_runs 1 --tsteps 3 --skip_display --run_name sort_smoke
```

## LLM 配置

默认通过 OpenAI-compatible 环境变量配置模型服务：

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=Qwen/Qwen3.5-27B
```

如需本地个人客户端，可创建被 `.gitignore` 忽略的 `prompting/openai_client.py`。公共代码默认使用 `prompting/llm_client.py`，没有个人覆盖文件时也能运行。

## 任务方案文档

- [Sort](SORT_SOLUTION.md)
- [Cabinet](CABINET_SOLUTION.md)
- [Rope](ROPE_SOLUTION.md)
- [Sweep](SWEEP_SOLUTION.md)
- [Sandwich](SANDWICH_SOLUTION.md)
- [Pack Grocery](PACK_GROCERY_SOLUTION.md)

## Pack Grocery 最新更新

Pack 任务的主要改动记录在 [PACK_GROCERY_SOLUTION.md](PACK_GROCERY_SOLUTION.md)。

最新优化点：

- 调整 Pack 任务的机器人-物品偏好：Alice 优先 `bread`、`banana`；Bob 优先 `milk`、`soda_can`、`cereal`、`apple`。
- 保持每轮一个机器人执行 `PICK` 或 `PLACE`，另一个机器人 `WAIT`，降低箱口附近碰撞概率。
- 新增 Pack fallback 候选方案队列。fallback 不再只返回一个贪心动作，而是生成多个候选动作并逐个通过 parser 和环境反馈验证。
- 当某个候选在目标点发生碰撞，例如 `milk-Alice`，系统会自动尝试下一个候选，避免重复卡在同一个失败动作。
- PLACE 阶段会尝试多个空槽位候选，减少固定槽位反复失败。

Pack 快速运行命令：

```bash
python run_dialog.py --task pack --comm_mode plan --num_runs 1 --tsteps 12 --num_replans 1 --skip_display --run_name pack_eval_12 --rrt_timeout 15 --skip_smooth_path --pack_fallback_first
```

使用 `uv`：

```bash
uv run python run_dialog.py --task pack --comm_mode plan --num_runs 1 --tsteps 12 --num_replans 1 --skip_display --run_name pack_eval_12 --rrt_timeout 15 --skip_smooth_path --pack_fallback_first
```

## 输出目录

`run_dialog.py` 默认将运行产物写入 `data/<run_name>/`。部分批量评测脚本也可能写入 `output/<run_name>/`。

```text
data/<run_name>/
|-- args_YYYYMM_HHMM.json
`-- run_0/
    |-- step_0/
    |-- steps*_success_*.json
    `-- *.html / *.mp4
```

`data/`、`output/`、视频、日志和中间 pickle 均为本地产物，应保持在 `.gitignore` 中，不提交到远程仓库。

## 常用检查命令

语法检查：

```bash
python -m py_compile prompting/plan_prompter.py
```

批量编译主要模块：

```bash
python -m compileall run_dialog.py prompting rocobench
```

查看 Pack fallback 日志：

```bash
find output/run_20260519_153732/tasks/01_pack/runs -path "*prompts*fallback*.json" -print -exec cat {} \;
```

查看 Pack 碰撞反馈：

```bash
grep -R "Collision detected\|FallbackCandidates\|SelectedFallback" -n output/run_20260519_153732/tasks/01_pack/runs
```

## 单任务运行示例

Sort：

```bash
uv run python run_dialog.py --task sort --comm_mode plan --num_runs 1 --tsteps 8 --num_replans 2 --skip_display --skip_smooth_path --fallback_first --run_name sort_debug
```

Pack：

```bash
uv run python run_dialog.py --task pack --comm_mode plan --num_runs 1 --tsteps 12 --num_replans 1 --skip_display --skip_smooth_path --pack_fallback_first --run_name pack_debug
```

## 协作规则

提交前请阅读 [AGENTS.md](AGENTS.md)。

远程仓库不应包含：

- 个人密钥或代理配置
- 个人 LLM 客户端覆盖文件
- 本地 runner / evaluator 临时脚本
- `data/`、`output/`、视频、日志、pickle 等运行产物
- 与当前任务无关的临时文件

## pack_code.sh

`pack_code.sh` 用于按 `.gitignore` 规则打包当前工作区：

```bash
./pack_code.sh
./pack_code.sh myproject.zip
./pack_code.sh myproject
```

如果没有执行权限：

```bash
chmod +x pack_code.sh
```
