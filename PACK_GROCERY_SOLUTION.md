# Pack Grocery 任务优化方案

本文件记录 Pack Grocery 多机器人协同装箱任务的实现思路、关键改动、运行命令和调试方法。任务基于 RoCoBench：Alice 为 UR5E + Robotiq，Bob 为 Franka Panda，两台机器人需要把桌面上的 6 个物品放入箱子。

物品包括：

- `apple`
- `banana`
- `milk`
- `soda_can`
- `bread`
- `cereal`

核心难点不是单个物体的抓取，而是两个机械臂在同一张桌子和同一个箱子附近协作时，需要避免机械臂互撞、物体碰撞、不可达目标、RRT 超时以及 LLM 输出格式不稳定。

## 关键改动

### 1. Pack 专用提示词

在 `prompting/plan_prompter.py` 中为 `PackGroceryTask` 增强了任务提示：

- 明确每轮每个机器人只能输出一个动作。
- 空手只能 `PICK` 或 `WAIT`，持物必须优先 `PLACE`。
- 已经在箱内的物品不能再次抓取。
- 已占用的箱子槽位不能再次放置。
- 推荐每轮只让一个机器人执行 `PICK` 或 `PLACE`，另一个机器人 `WAIT`。
- 固定批次顺序：`bread + milk`，`cereal + soda_can`，`banana + apple`。
- 更新机器人偏好：Alice 优先处理 `bread`、`banana`；Bob 优先处理 `milk`、`soda_can`、`cereal`、`apple`。
- `cereal` 更推荐 Bob 处理，因为 Alice 放置大物体时更容易超时或碰撞。
- `PLACE` 路径保持较高安全高度，尤其是 `cereal` 和 `milk`。
- 如果反馈提示物体不可达、目标姿态异常或发生碰撞，则跳过该候选动作，继续尝试其他物体或槽位。

### 2. 保守 fallback planner

Pack 任务新增了确定性 fallback planner，用于在 LLM 输出不可用或反复失败时继续推进任务。

主要策略：

- 一轮只激活一个机器人，另一个机器人保持 `WAIT`。
- 如果有机器人持物，则优先执行 `PLACE`。
- 如果两个机器人都空手，则按照批次和代价选择一个可抓取物体。
- Bob 不抓取 x 坐标过小的物体，避免 Panda 远距离抓取失败。
- 跳过已经打包、姿态异常、掉出工作区或目标点不合法的物体。
- 使用固定安全高度生成 4 个均匀路径点。

### 3. fallback 候选方案队列

这次更新的重点是：Pack fallback 不再只返回一个贪心方案，而是返回一组按代价排序的候选方案。

之前的问题：

- fallback 可能反复选择同一个看似最优动作，例如 `Alice PICK bread`。
- 该动作能通过解析，但在目标抓取姿态处可能和邻近物体碰撞，例如 `milk-Alice`。
- 下一轮 fallback 仍然生成相同动作，导致任务卡住。

现在的流程：

1. `build_pack_fallback_candidates()` 生成多个候选方案。
2. `validate_fallback_candidates()` 逐个解析并调用环境反馈检查。
3. 如果第一个候选发生 IK、可达性或碰撞失败，自动尝试下一个候选。
4. 找到第一个通过检查的候选后返回执行。

涉及的核心函数：

- `_build_pack_pick_candidate`
- `_pack_pick_candidate_responses`
- `build_pack_fallback_candidates`
- `validate_fallback_candidates`
- `build_fallback_candidates`

### 4. PLACE 槽位候选

对于持物状态下的 fallback，系统现在会为同一个物体生成多个空槽位候选：

- 优先使用 `PACK_ITEM_SLOT_PREFERENCE` 中的推荐槽位。
- 如果推荐槽位失败，会尝试其他空槽位。
- 如果历史失败计划中已经出现某个物体放入某个槽位失败，会优先避开该槽位。

这能减少大物体靠近箱子边缘时反复碰撞的问题。

## 关键参数

```python
SAFE_PICK_HEIGHT = 0.62
SAFE_PLACE_HEIGHT = 0.68
MIN_PATH_HEIGHT = 0.45
MAX_PATH_HEIGHT = 0.78
MIN_PACK_ITEM_Z = 0.05
MAX_PACK_ITEM_Z = 0.90
PACK_ROBOT_HANDOFF_PENALTY = 2.0
```

机器人与物品偏好：

```python
PACK_ROBOT_ITEM_PREFERENCE = {
    "ur5e_robotiq": ["bread", "banana"],
    "panda": ["milk", "soda_can", "cereal", "apple"],
}

PACK_ROBOT_ALLOWED_ITEMS = {
    "ur5e_robotiq": {"bread", "banana"},
    "panda": {"milk", "soda_can", "cereal", "apple"},
}
```

## 推荐运行命令

快速检查前 3 步：

```bash
python run_dialog.py --task pack --comm_mode plan --num_runs 1 --tsteps 3 --num_replans 1 --skip_display --run_name pack_debug_3 --rrt_timeout 15 --skip_smooth_path --pack_fallback_first
```

完整 12 步测试：

```bash
python run_dialog.py --task pack --comm_mode plan --num_runs 1 --tsteps 12 --num_replans 1 --skip_display --run_name pack_eval_12 --rrt_timeout 15 --skip_smooth_path --pack_fallback_first
```

如果使用 `uv`：

```bash
uv run python run_dialog.py --task pack --comm_mode plan --num_runs 1 --tsteps 12 --num_replans 1 --skip_display --run_name pack_eval_12 --rrt_timeout 15 --skip_smooth_path --pack_fallback_first
```

## 日志检查命令

查看 fallback 记录：

```bash
find output/run_20260519_153732/tasks/01_pack/runs -path "*prompts*fallback*.json" -print -exec cat {} \;
```

带分隔符查看：

```bash
find output/run_20260519_153732/tasks/01_pack/runs -path "*prompts*fallback*.json" -print -exec sh -c 'echo "===== $1 ====="; cat "$1"; echo' _ {} \;
```

只看碰撞相关反馈：

```bash
grep -R "Collision detected\|milk-Alice\|FallbackCandidates\|SelectedFallback" -n output/run_20260519_153732/tasks/01_pack/runs
```

语法检查：

```bash
python -m py_compile prompting/plan_prompter.py
```

## 验证记录

已执行基础语法检查：

```bash
python -m py_compile prompting/plan_prompter.py
```

检查通过。完整仿真成功率需要按上述 Pack 运行命令重新评估。

## 代码文件

| 文件 | 作用 |
| --- | --- |
| `prompting/plan_prompter.py` | Pack 提示词、fallback planner、候选方案生成和验证入口 |
| `prompting/feedback.py` | 环境反馈、IK、碰撞、路径平滑检查 |
| `prompting/parser.py` | LLM 输出解析 |
| `rocobench/envs/task_pack.py` | Pack 任务环境、物体状态、箱子槽位、任务反馈 |
| `rocobench/policy.py` | 策略执行和路径规划 |
| `rocobench/rrt_multi_arm.py` | 多机械臂 RRT / IK 规划 |

## 总结

本次 Pack 任务优化的重点是提升 fallback 的抗卡死能力。原先的单一贪心 fallback 一旦选中会碰撞的动作，容易在同一状态下重复失败；现在 fallback 会生成候选队列，并让环境反馈逐个筛选，从而能自动绕过局部碰撞方案，继续尝试其他机器人、物体或槽位。

## 引用

```bibtex
@misc{mandi2023roco,
      title={RoCo: Dialectic Multi-Robot Collaboration with Large Language Models},
      author={Zhao Mandi and Shreeya Jain and Shuran Song},
      year={2023},
      eprint={2307.04738},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
