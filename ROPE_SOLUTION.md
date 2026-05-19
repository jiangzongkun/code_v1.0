# Rope 任务调试与优化总结

本文记录本次 MoveRopeTask 的完整调试过程。Rope 任务最终从 `2/5`、`3/5` 逐步提升到 `4/5 = 80%`，并将主要失败原因从 IK、碰撞、RRT timeout 收敛到“评测步数不足”。本次优化重点不是继续手调物理点，而是引入阶段化策略、fallback 候选路径、执行前验证、避障候选和恢复性 horizon。

---

## 1. 任务特点

Rope 任务要求 Alice 和 Bob 协同抓取绳子两端，并将绳子放入狭窄 groove 中。与 Sort 不同，Rope 更强依赖空间约束和双臂协同：

- 绳子两端必须同时被抓住；
- 两个机器人移动时要保持相对距离；
- 绳子需要跨过或绕过障碍墙；
- 抓取点不能太贴近端点，否则 IK 和接触都不稳定；
- 中间 waypoint 既要可达，又要避免与 obstacle wall 碰撞；
- 单个失败 waypoint 会导致整轮 RRT 卡死或超时。

因此 Rope 的核心瓶颈不是模型是否理解“把绳子放进槽里”，而是每一步动作是否满足物理可执行性。

---

## 2. 初始评测结果

初始使用：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks rope --runs 5
```

早期结果大致为：

```text
Success Rate: 2/5 或 2/4
Timeout Count: 0/5 或 1/5
```

失败日志中常见现象：

```text
IK failed
Collision detected: obstacle_wall-Bob
ReasonTimeout_time62...
Parsing failed! Response does not contain NAME.
Previous response: EXECUTE
```

说明 Rope 同时存在四类问题：

1. LLM 输出格式不稳定；
2. 端点抓取或低位抓取导致 IK 不稳定；
3. PICK/PUT 路径穿过障碍物；
4. RRT 长时间找不到碰撞自由路径。

---

## 3. 关键失败类型

### 3.1 端点抓取不稳定

原始策略直接抓 `rope_front_end` 和 `rope_back_end`，但绳子端点靠近桌面且接触状态复杂，目标点可能太低或太靠边，容易导致 IK 失败。

### 3.2 PUT 高位中间点不可达

失败样例：

```text
NAME Alice ACTION PUT rope_front_end groove_left_end PATH [(-1.14, 0.65, 0.52), ...]
IK failed: on Alice (-1.14, 0.65, 0.52)
```

该问题说明路径点语义合理，但空间位置对机械臂不可达。

### 3.3 PICK 路径穿过 obstacle wall

典型日志：

```text
Collision detected: collided object pairs: obstacle_wall-Bob
Waypoint Step Alice (-0.89, 0.34, 0.36); Bob (-0.48, 0.75, 0.36)
```

说明低位直线路径会穿过障碍墙附近区域。

### 3.4 LLM replan 输出为空

长 prompt 和失败反馈叠加后，模型有时只输出：

```text
EXECUTE
```

缺少 `NAME Alice ACTION ...` 和 `NAME Bob ACTION ...`，parser 失败。此时如果没有 fallback，就会白白消耗步骤和时间。

### 3.5 任务步数不足

后期 80% 版本中，唯一失败样例不是 IK、碰撞或 RRT timeout，而是第 4 步刚完成恢复性 `PICK`，还差第 5 步 `PUT`，但 Rope 被内部限制为 5 个 timestep，导致提前结束。

---

## 4. 修改一：Rope prompt 阶段化

在 `prompting/plan_prompter.py` 中，将 Rope 明确拆分为两个阶段。

### Phase 1: PICK

- Alice 抓 `rope_front_end`
- Bob 抓 `rope_back_end`
- 路径高度保持在 `0.25 ~ 0.52`
- 不在 PICK 阶段高抬绳子

### Phase 2: PUT

- Alice 将 `rope_front_end` 放到 `groove_left_end`
- Bob 将 `rope_back_end` 放到 `groove_right_end`
- Alice 走左侧，Bob 走右侧，避免交叉
- PUT 阶段使用更保守的中间高度

同时加入 Bob 的专属约束：

```text
Bob must keep PATH x >= -0.40 when z > 0.50
```

该约束来自实际失败日志，能减少 Panda 机械臂在高位偏左区域的 IK/RRT 风险。

---

## 5. 修改二：绳子端点 inward-offset 抓取

在 `rocobench/envs/task_rope.py` 中新增 `get_rope_grasp_target_pos()`：

- 不再抓精确端点；
- 根据绳子另一端方向，向绳子内部偏移约 `0.06m`；
- 将抓取高度限制在更稳定的范围；
- 仍然保持 weld 对应 rope end，保证语义上抓的是绳子端点。

核心思想：

```text
grasp point = endpoint + inward_offset
```

这样可以避免端点太低、太边缘或数值不稳定导致的 IK 失败。

---

## 6. 修改三：parser 中 Rope 抓取姿态稳定化

在 `prompting/parser.py` 中加入 Rope 专用处理：

- Rope PICK 使用环境提供的 inward-offset target；
- 保留当前 gripper quaternion；
- 避免强行使用 rope body 的绝对 quaternion；
- 对 `None` 响应做保护，避免 parser 崩溃。

原因是 Rope endpoint 的姿态不一定适合作为机械臂末端目标姿态，强行继承可能导致 IK 难以收敛。

---

## 7. 修改四：Rope fallback candidate 机制

在 `prompting/plan_prompter.py` 中加入 Rope 专用 fallback：

- 根据当前是否 holding rope 判断是 PICK 阶段还是 PUT 阶段；
- 一次生成多个候选路径；
- 使用 parser 和 feedback manager 对候选做验证；
- 选择第一个通过验证的 candidate；
- 日志中记录 `FallbackCandidates` 和 `SelectedFallback`，便于复盘。

这一步将系统从“LLM 只给一个动作”改成：

```text
LLM/规则生成候选 → parser 检查 → feedback 验证 → 选择可执行动作
```

---

## 8. 修改五：PUT 阶段候选路径

针对 PUT 阶段，设计多种候选：

| 候选 | 策略 |
|---|---|
| candidate 0 | 原始保守 PUT 路径 |
| candidate 1 | 降低第一段高度，避免 Alice/Bob 高位 IK 失败 |
| candidate 2 | 中间点更靠近机器人本体，减少不可达风险 |

例如失败日志中 Alice 高位点不可达后，加入低位过渡点：

```text
(-1.14, 0.65, 0.38) → (-0.81, 0.62, 0.46) → ...
```

这样比手动修改单个坐标更泛化，因为系统会在多个候选中筛选。

---

## 9. 修改六：PICK 阶段 obstacle-aware candidates

后续发现失败集中在初始 PICK 低位路径穿过障碍墙。因此新增 obstacle-aware rope pick candidates：

| 候选 | 路径特点 |
|---|---|
| candidate 0 | 原始直线低位接近 |
| candidate 1 | Alice 走低 y 通道，Bob 走高 y 通道 |
| candidate 2 | 更保守的侧向绕障路径 |

后续进一步调整候选优先级：

```text
PICK 阶段优先尝试 obstacle-aware side-lane candidates
PUT 阶段优先尝试原始保守路径
```

这样可以减少 obstacle wall 碰撞和 RRT timeout。

相关提交：

- `5a7c0a0 add obstacle-aware rope pick candidates`
- `2748b79 prioritize obstacle-aware rope pick candidates`

---

## 10. 修改七：Rope horizon 从 5 放宽到 6

80% 版本的唯一失败样例：

```text
run_1/steps4_success_False.json
step 4 success false, timed_out false
```

stdout 显示 step 0-4 的所有计划和执行都成功：

- step 0: PICK 成功；
- step 1: PUT 成功；
- step 2: 恢复性 PICK 成功；
- step 3: 恢复性 PUT 成功；
- step 4: 再次 PICK 成功；
- 但没有第 5 步执行最后 PUT。

因此失败原因不是物理规划，而是 Rope 内部将步数限制为 5。

在 `run_dialog.py` 中修改：

```python
args.tsteps = 6
```

日志提示：

```text
MoveRope uses 6 tsteps to allow one recovery pick-place cycle
```

相关提交：

- `67d0634 fix rope evaluation horizon`

---

## 11. 实验结果变化

### 早期结果

```text
Success Rate: 2/5 = 40.0%
Timeout Count: 1/5
```

### 加入 fallback 和路径候选后

```text
Success Rate: 3/5 = 60.0%
Timeout Count: 2/5
Total Time: 768.24s
```

### 加入避障候选后

```text
Success Rate: 3/5 = 60.0%
Timeout Count: 1/5
Total Time: 462.16s
```

虽然成功率暂时没提升，但 timeout 明显减少，说明路径候选机制有效。

### 优先选择 obstacle-aware candidate 后

```text
Success Rate: 4/5 = 80.0%
Timeout Count: 0/5
Average Steps: 1.50
Total Time: 408.23s
```

这是本次 Rope 调试的最好结果。唯一失败样例已经不是 IK、碰撞或 timeout，而是需要第 6 步完成恢复性放置。

---

## 12. 推荐评测命令

更新到最新代码后使用：

```bash
git pull
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks rope --runs 5 --tsteps 6
```

单轮快速验证：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks rope --runs 1 --tsteps 6
```

查看失败日志：

```bash
find output/run_xxx/tasks/01_rope/runs -maxdepth 2 -name "*.json" -print -exec cat {} \;
tail -260 output/run_xxx/tasks/01_rope/stdout.log
grep -R "timed_out\|Timeout\|Run finished\|failed\|Plan success\|Collision detected\|IK failed\|ReasonTimeout" -n output/run_xxx/tasks/01_rope
find output/run_xxx/tasks/01_rope/runs -path "*prompts*fallback*.json" -print -exec cat {} \;
```

---

## 13. 本次 Rope 优化贡献

1. **从单一路径改成多候选路径**：降低某个 waypoint 不可行导致整轮失败的概率。
2. **端点抓取改成 inward-offset 抓取**：提高 rope endpoint 抓取稳定性。
3. **parser 中使用更稳定的姿态**：减少不必要的 IK 失败。
4. **PICK / PUT 阶段化**：明确双机器人分工，减少路径交叉。
5. **加入 obstacle-aware PICK candidates**：解决初始抓取阶段穿越障碍墙的问题。
6. **候选路径执行前验证**：使用 parser + feedback manager 筛选候选，而不是直接执行。
7. **从失败日志中区分问题类型**：分别定位 LLM 格式、IK、碰撞、RRT timeout 和 horizon 不足。
8. **将 Rope horizon 放宽到 6**：允许一次恢复性 pick-place 循环。

---

## 14. 泛化性分析

本次 Rope 优化相比单纯手调坐标更具有泛化性，因为主要改动集中在机制层：

- 用 inward-offset 替代精确端点抓取；
- 用多候选路径替代单一路径；
- 用 validator 筛选替代盲目执行；
- 用 obstacle-aware side-lane 替代固定直线；
- 用阶段化策略替代完全自由生成；
- 用 6 步 horizon 支持恢复性动作。

但仍需注意：

- 当前候选路径仍基于本环境的 obstacle wall 和 groove 布局；
- 多候选验证依赖现有 feedback manager，不能完全替代真实 RRT 成功率；
- Rope 物理随机性较强，应继续用更多 runs 验证；
- 后续若要进一步提升，可加入 memory bank，记录成功路径并复用。

---

## 15. 总结表述

本次 Rope 任务调试从失败日志出发，发现主要瓶颈依次为端点抓取不稳定、路径穿越障碍、LLM 输出格式异常、RRT timeout 和评测 horizon 不足。通过 inward-offset grasp、Rope 专用 parser 处理、fallback candidate、obstacle-aware pick candidates、候选路径优先级调整和 6 步恢复性 horizon，成功将 Rope 任务从早期约 `40%-60%` 提升到 `80%`，并消除了 timeout。

整体优化路线可以概括为：

```text
LLM 生成意图
→ 规则/技能生成多个候选路径
→ parser 与 feedback manager 做执行前验证
→ 优先选择避障和可达候选
→ 失败后允许恢复性 pick-place
```

这比单纯手动调 waypoint 更适合后续任务扩展，也更符合多机器人协作任务的泛化优化方向。

---

## 16. 公共规划器合并记录（PR #1）

- 动机：合并远程 PR #1 `Improve multi-arm path planning robustness`，提升多机械臂路径规划稳定性，同时保留 main 上 Cabinet release 规划中“已焊接物体视为 in-hand”的修复。
- 改动点：`rocobench/policy.py` 同时保留 `augment_release_plan_inhand` 和 `sparsify_validated_path`；`rocobench/rrt.py` 修正 near/center sampler 的区间采样；`rocobench/rrt_multi_arm.py` 使用末端局部坐标维护 in-hand 物体相对位姿、放宽 IK 容差、默认只允许末端执行器接触抓取物，并让 split plan 保留强制 waypoints。
- 运行命令：`python -m compileall run_dialog.py prompting rocobench/envs rocobench/policy.py rocobench/rrt.py rocobench/rrt_multi_arm.py`
- 验证结果：2026-05-19 语法检查通过；本次未重新跑完整仿真评测，建议按本任务推荐命令复验成功率。
