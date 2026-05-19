# Sort 任务调试与优化总结

本文记录本次 Sort 任务从失败定位、代码修改、评测验证到泛化性分析的完整过程。Sort 任务最终在 `qwen3.5:27b` 下单轮评测达到 `1/1 = 100%`，核心改进不是单纯让大模型多生成几次，而是围绕任务状态、交接点、抓取姿态和 fallback 机制建立更稳定的执行闭环。

---

## 1. 任务目标与主要难点

Sort 任务要求三个机器人协作，将不同形状的物体放到指定面板：

| 机器人 | 目标物体 | 目标面板 | 可达区域特点 |
|---|---|---|---|
| Alice | `blue_square` | `panel2` | 左侧机器人，适合处理 panel1-panel3 |
| Bob | `pink_polygon` | `panel4` | 中间机器人，适合处理 panel3-panel5 |
| Chad | `yellow_trapezoid` | `panel6` | 右侧机器人，适合处理 panel5-panel7 |

任务难点在于，物体初始位置不一定在目标机器人的可达范围内，因此需要通过 `panel3` 或 `panel5` 进行中转交接。Sort 的本质不是简单分类，而是：

- 识别每个物体当前在哪个 panel；
- 判断目标机器人是否可达；
- 必要时让其他机器人先搬运到中转 panel；
- 避免重复搬运已经完成的物体；
- 保证中转点对下一位机器人可抓取、可规划、可执行。

---

## 2. 初始失败现象

最初使用：

```bash
OLLAMA_MODEL=qwen3.5:9b uv run python evaluator.py --tasks sort --runs 1
```

评测结果为：

```text
sort 0/1 = 0.0%
Timeout Count: 0
```

日志中前三步可执行，但之后 LLM 多次 API error，fallback 反复生成同一个不可执行计划：

```text
NAME Alice ACTION WAIT
NAME Bob ACTION PICK pink_polygon PLACE panel4
NAME Chad ACTION WAIT

IK failed: on Bob (-0.27, 0.73, 0.18)
```

可以看到，主要问题不是模型完全不懂 Sort，而是中转后的物体位置太低或太偏，导致 Bob 后续抓取 `pink_polygon` 时 IK 不稳定。

---

## 3. 模型切换与问题复现

随后切换到更大的模型：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks sort --runs 1
```

结果仍为失败：

```text
sort 0/1 = 0.0%
```

说明失败原因并不是单纯模型能力不足。即使更换更强模型，只要底层交接点、抓取目标和 planner 约束不稳定，仍会生成或执行到 IK/RRT 不可靠的位置。

---

## 4. 关键失败原因分析

本次 Sort 的主要失败原因可以归纳为四类。

### 4.1 panel3 中转点不适合 Bob 后续抓取

日志显示 Alice 把 `pink_polygon` 放到 `panel3` 后，Bob 再从该位置抓取时经常失败：

```text
IK failed: on Bob (-0.28, 0.74, 0.30)
```

这说明 `panel3` 虽然语义上是中转区，但实际物理位置必须兼顾 Alice 放置和 Bob 抓取。原始中转点让物体沉降后处在 Bob 较难达到的位置。

### 4.2 低位物体导致抓取姿态不稳定

交接后的物体高度偏低时，parser 直接使用物体 site 位姿生成抓取目标，容易导致末端姿态和高度不适合 IK 求解。

### 4.3 LLM 响应不稳定

多轮之后出现：

```text
API error, try again
Run 0: Step 3 failed to get a plan from LLM
```

因此不能完全依赖 LLM 每一步都在线规划，必须有 deterministic fallback。

### 4.4 缺少状态化任务调度

Sort 需要记住：

- 哪些物体已经到目标 panel；
- 哪些物体正在中转；
- 下一步应该由谁接手；
- 哪些机器人应该 `WAIT`。

如果只靠 prompt 中的自然语言，模型容易重复执行已经失败或已经完成的动作。

---

## 5. 主要代码修改

### 5.1 增加 evaluator 单任务评测入口

修改 `evaluator.py`，支持直接指定任务、轮数、步数和 timeout：

```bash
uv run python evaluator.py --tasks sort --runs 1
uv run python evaluator.py --tasks sort --runs 5 --tsteps 8
```

这样可以只测 Sort 准确率，而不必一次跑完整六个任务，调试效率明显提高。

### 5.2 增加 fallback-first 调试模式

在 `run_dialog.py` 中加入：

- `--rrt_timeout`
- `--skip_smooth_path`
- `--fallback_first`

其中 `fallback_first` 让系统优先尝试规则 fallback 计划，用于验证物理执行链路是否稳定。如果 fallback 都失败，就说明问题更可能在物理点位、IK、RRT 或 parser，而不是 LLM 推理。

### 5.3 Sort fallback planner

在 `prompting/plan_prompter.py` 中加入 Sort 专用 fallback planner。它根据当前状态选择保守动作：

- 已经在目标面板的物体跳过；
- 不在目标机器人可达范围的物体先送到中转面板；
- 需要交接时让非执行机器人 `WAIT`；
- 优先使用 `panel3` / `panel5` 作为中转点；
- 避免多个机器人同时抢同一个物体。

示例 fallback：

```text
EXECUTE
NAME Alice ACTION WAIT
NAME Bob ACTION PICK blue_square PLACE panel3
NAME Chad ACTION WAIT
```

### 5.4 parser 抓取姿态稳定化

在 `prompting/parser.py` 中，对 Sort 抓取目标做稳定化处理：

- 使用更安全的 top-down grasp pose；
- 对低位物体提升抓取高度；
- 避免直接继承不适合 IK 的物体 site quaternion；
- 对空响应 `None` 做保护，避免 parser 崩溃。

这一步解决的是“计划语义正确，但机械臂抓不到”的问题。

### 5.5 调整 panel3 handoff 位置

在 `rocobench/envs/task_sort.py` 中调整 `panel3` 的中转目标：

```python
if target_name == 'panel3':
    ret[0] += 0.05
    ret[1] -= 0.18
```

修改目的：

- 让 `panel3` 更接近左侧 handoff lane；
- Alice 放置时仍可达；
- Bob 后续抓取时更稳定；
- 减少物体沉降后 Bob 低位抓取失败。

这是本次 Sort 成功的关键物理修正。

### 5.6 HTML 日志鲁棒性修复

在 `prompting/display_utils.py` 中增加对 prompt JSON 的类型过滤，避免异常日志导致 episode HTML 保存失败。

---

## 6. 关键验证过程

### 6.1 失败日志定位

使用：

```bash
tail -200 output/run_xxx/tasks/01_sort/stdout.log
find output/run_xxx/tasks/01_sort/runs/run_0 -path "*prompts*fallback*.json" -print -exec cat {} \;
```

确认失败集中在：

```text
Bob PICK pink_polygon PLACE panel4
IK failed
```

### 6.2 修正后验证

最终运行：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks sort --runs 1
```

结果：

```text
Success Rate: 1/1 = 100.0%
Average Steps: 5.00
Total Time: 80.19s
```

说明 Sort 的核心执行链路已经打通。

---

## 7. 本次 Sort 优化贡献

本次优化主要贡献如下：

1. **将调试入口任务化**：可以直接评测 Sort 准确率。
2. **引入 fallback-first**：把 LLM 不稳定和物理执行不稳定区分开。
3. **建立 Sort 状态化 fallback**：根据当前 panel、目标 panel 和中转 panel 生成保守计划。
4. **修复低位抓取 IK 不稳定**：在 parser 中提升抓取姿态稳定性。
5. **修正 panel3 handoff 位置**：让 Alice 放置和 Bob 接手同时可行。
6. **提高日志可读性和鲁棒性**：便于后续任务继续调试。

---

## 8. 泛化性分析

单次评测达到 100% 不代表完全泛化。若只是针对某个 seed、某个物体初始位置和某条失败日志硬编码动作顺序，就会有过拟合风险。

本次 Sort 修改尽量避免直接写死完整答案，而是优先采用更通用的机制：

- 根据当前观测判断物体所在 panel，而不是固定执行某个顺序；
- 根据机器人可达范围选择中转面板；
- 使用 fallback 作为保守兜底，而不是替代所有规划；
- 修正 handoff 点的物理可达性，而不是只修某一次日志中的坐标；
- 在 parser 中加入低位抓取保护，适用于多个物体而非单个样例。

仍然存在的风险：

- 当前中转策略主要适配本任务固定的 7-panel 布局；
- `panel3` 的偏移值仍然带有任务环境经验；
- 需要更多 runs 和不同 seed 验证是否稳定。

建议后续验证：

```bash
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks sort --runs 5 --tsteps 8
OLLAMA_MODEL=qwen3.5:27b uv run python evaluator.py --tasks sort --runs 10 --tsteps 8
```

---

## 9. 总结表述

本次 Sort 优化从失败日志出发，发现主要瓶颈并非大模型语义理解不足，而是中转交接点和低位抓取导致的 IK 不稳定。通过增加单任务 evaluator、fallback-first 调试模式、Sort 专用 fallback planner、parser 抓取姿态稳定化以及 `panel3` handoff 位置修正，最终使 Sort 在 `qwen3.5:27b` 下达到单轮 `100%` 成功率。

整体思路是将 Sort 从“LLM 直接生成动作”改造为“LLM/规则生成意图，parser 稳定动作，物理中转点保证可执行，失败日志驱动局部修正”的闭环系统。

## 公共规划器合并记录（PR #1）

- 动机：合并远程 PR #1 `Improve multi-arm path planning robustness`，提升多机械臂路径规划稳定性，同时保留 main 上 Cabinet release 规划中“已焊接物体视为 in-hand”的修复。
- 改动点：`rocobench/policy.py` 同时保留 `augment_release_plan_inhand` 和 `sparsify_validated_path`；`rocobench/rrt.py` 修正 near/center sampler 的区间采样；`rocobench/rrt_multi_arm.py` 使用末端局部坐标维护 in-hand 物体相对位姿、放宽 IK 容差、默认只允许末端执行器接触抓取物，并让 split plan 保留强制 waypoints。
- 运行命令：`python -m compileall run_dialog.py prompting rocobench/envs rocobench/policy.py rocobench/rrt.py rocobench/rrt_multi_arm.py`
- 验证结果：2026-05-19 语法检查通过；本次未重新跑完整仿真评测，建议按本任务推荐命令复验成功率。

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
