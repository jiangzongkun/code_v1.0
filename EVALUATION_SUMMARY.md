# 基础常规任务评测总结

## 评测信息

- 评测时间：2026-05-20
- 输出目录：`output/run_20260520_121100`
- 运行模式：`comm_mode=plan`
- 运行次数：每个任务 5 次
- 运行配置：`--full`，即 10 步、3 次 replan、600s 单 run 超时、200s RRT 超时
- 规划策略：默认启用 `fallback_first`
- 说明：`output/` 为本地评测产物，按 `.gitignore` 不提交到远程仓库。

## 总体结果

| 任务 | Run IDs | 成功率 | 成功次数 | 超时 | 平均成功步数 | 总耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sort | 0-4 | 100.0% | 5/5 | 0 | 5.40 | 372.18s |
| Cabinet | 5-9 | 100.0% | 5/5 | 0 | 3.40 | 892.04s |
| Rope | 10-14 | 80.0% | 4/5 | 1 | 1.50 | 1098.72s |
| Sweep | 15-19 | 80.0% | 4/5 | 0 | 8.00 | 893.32s |
| Sandwich | 20-24 | 100.0% | 5/5 | 0 | 6.20 | 522.77s |
| Pack Grocery | 25-29 | 80.0% | 4/5 | 0 | 11.00 | 835.47s |

总体成功率：`27/30 = 90.0%`。

总体耗时：约 `4614.49s`，约 `76.91min`。

## 任务分析

### Sort

Sort 达到 `5/5` 成功，平均 `5.40` 步。当前 panel relay 策略表现稳定，能够处理跨机器人可达区域的传递。

观察点：
- 无超时。
- 无失败 run。
- 当前 fallback-first 对 Sort 任务有明显稳定作用。

### Cabinet

Cabinet 达到 `5/5` 成功，平均 `3.40` 步。开门、保持门打开、取出杯子/马克杯的阶段划分稳定。

观察点：
- stderr 中出现若干 IK 收敛 warning，但未影响最终成功。
- 任务耗时相对较长，主要来自 RRT/IK 求解成本。

### Rope

Rope 达到 `4/5` 成功，失败 run 为 `run_14`，并出现 1 次超时。

主要失败原因：
- `run_14` 中 Bob 抓取 `rope_back_end` 的目标点约为 `(-0.53, 0.24, 0.32)`，反馈显示：

```text
IK failed: on Bob (-0.53, 0.24, 0.32)
```

- fallback 候选重复尝试 Bob 抓 `rope_back_end`，后续 LLM replan 又出现空 `EXECUTE`，导致解析失败并最终超时。

后续优化方向：
- Rope fallback 应增加交换抓取端候选，例如 `Alice PICK rope_back_end`、`Bob PICK rope_front_end`，避免 Bob 被固定分配到不可达端点。
- 对 fallback-first 模式限制单段 RRT 超时，减少不可达候选拖慢整轮评测。
- 对已接近 groove 的状态增加“不要重复 pick/put”的保护，减少放置后未触发 done 时的二次移动偏差。

### Sweep

Sweep 达到 `4/5` 成功，失败 run 为 `run_15`，无超时。

主要失败原因：
- `run_15` 第 9 步视频和场景描述看起来已完成，三个 cube 均显示 `DONE - already swept into trash_bin`。
- 但 `SweepTask.get_reward_done()` 的成功判定是每个 cube 中心点到 `trash_bin_bottom` 的三维距离小于 `0.2`。因此 cube 接触 trash bin 并显示 DONE，不一定满足最终 reward。

后续优化方向：
- 统一任务描述中的 DONE 逻辑和 reward 判定逻辑。
- 或优化 DUMP 动作，使 cube 更靠近 `trash_bin_bottom` 中心，避免“视觉完成但 reward 失败”。

### Sandwich

Sandwich 达到 `5/5` 成功，平均 `6.20` 步，是本轮中较稳定的任务之一。

观察点：
- 无超时。
- 任务流程符合逐步取放食材的预期。
- 当前 recipe fallback 对该任务足够稳定。

### Pack Grocery

Pack Grocery 达到 `4/5` 成功，失败 run 为 `run_29`，无超时。

主要失败特征：
- fallback 候选和 parser 均通过，但最终 step 11 未达到 reward 成功条件。
- Pack 的平均成功步数为 `11.00`，接近本次 `--full` 的 10-12 步上限，说明任务对步数预算敏感。

后续优化方向：
- 对 Pack 增加更保守的 bin slot 选择，避免最后一两个物体放置后未被判定进箱。
- 对接近成功但未 done 的状态增加末端校正动作。
- 若用于正式评测，可考虑 `--tsteps 12` 或更高步数预算。

## 建议复测命令

完整基础任务：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks sort cabinet rope sweep sandwich pack --runs 5 --full
```

只复测 Rope：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks rope --runs 5 --tsteps 6 --timeout 600
```

只复测 Sweep：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks sweep --runs 5 --tsteps 12 --timeout 600
```

只复测 Pack：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks pack --runs 5 --tsteps 12 --timeout 600
```

## 当前结论

本次六个基础常规任务的整体成功率为 `90.0%`。Sort、Cabinet、Sandwich 已达到 `100%`；Rope、Sweep、Pack 均为 `80%`，失败原因集中在：

- Rope：Bob 端点 IK 不可达和 LLM replan 格式失效。
- Sweep：视觉/接触完成与 reward 距离判定不一致。
- Pack：最后阶段放置接近成功但未满足 reward，且步数预算偏紧。

下一轮优化应优先聚焦 Rope 的抓取端交换策略、Sweep 的 reward 判定一致性、Pack 的末端放置校正。
