# code_v1.0 与 code_original 实现对比

## 对比对象

本文对比两个项目：

- `code_v1.0`：`/Users/charly/VScodeProjects/ChuangZhi/code_g11/code_v1.0`
- `code_original`：`/Users/charly/VScodeProjects/ChuangZhi/code_original`

两者都基于 RoCoBench，但技术路线明显不同：

```text
code_original:
  IoA-Lite local communication-state coordinator
  + centralized planner
  + typed feedback / task table / dynamic plan
  + evaluator integration / monitoring / API robustness

code_v1.0:
  task-specific deterministic fallback planners
  + stronger prompts
  + candidate validation
  + parser / env / RRT / policy physical execution fixes
```

一句话概括：`code_original` 更像“研究型协作协议方法”，`code_v1.0` 更像“面向得分的任务专家系统 + 运动规划修复”。

## 总体架构差异

| 维度 | code_original | code_v1.0 |
|---|---|---|
| 主方法 | `comm_mode=ioa_lite` | `comm_mode=plan/chat` + fallback |
| 核心抽象 | IoA-inspired state protocol | task-specific fallback planner |
| 主要新增文件 | `prompting/ioa_lite_prompter.py`、实验 notes、monitor | `prompting/llm_client.py`、各任务 `*_SOLUTION.md` |
| 是否保留 baseline | 新增 `ioa_lite`，baseline 路径独立 | 直接增强 `SingleThreadPrompter`、parser、env、policy |
| 是否修改底层执行 | 基本不改 env/RRT/policy | 大量改 parser/env/RRT/policy |
| 研究叙事 | IoA-Lite 协作状态协议 | 工程故障修复和任务 fallback |
| 风险 | prompt 长、协作协议未必直接提分 | 任务特化强，可能较难形成统一论文贡献 |

## 入口层差异

### code_original

`run_dialog.py` 支持：

- `chat`
- `plan`
- `dialog`
- `ioa_lite`

`ioa_lite` 通过 lazy import 接入 `IoaLitePrompter(SingleThreadPrompter)`，并保留 plan/chat/dialog baseline 路径。`evaluator.py` 被改造成可以显式传入：

- `--comm_mode`
- `--ioa_preset`
- `--monitor_interval`
- `--skip_video`
- `--num_replans`

该项目更重视“正式评测入口可复现”和“不破坏 baseline”。

### code_v1.0

`run_dialog.py` 没有新增 `ioa_lite`。它仍然使用 `SingleThreadPrompter` / `DialogPrompter`，但给 `SingleThreadPrompter` 增加了 fallback 能力。入口层还加入：

- `--fallback_first`
- `--rrt_timeout`
- `--skip_smooth_path`

并在 `main()` 中按任务自动改运行配置：

- `rope` 自动启用 `action_and_path`、split parsed plans、6 steps。
- `pack` 自动启用 `action_and_path`、split parsed plans、至少 12 steps、30s RRT timeout。

这比 `code_original` 更激进，因为它不仅选择方法，还根据任务改变执行预算和底层规划参数。

## LLM 接入差异

### code_original

`prompting/plan_prompter.py` 直接支持 DashScope/OpenAI-compatible/Ollama，并加入：

- 从 `.env` 或 `scripts/dashscope_profile.sh` 读取环境变量。
- 子进程封装 LLM call。
- 单次 API timeout。
- retry/backoff。
- `ROCO_LLM_TOTAL_TIMEOUT_SECONDS` 总预算。
- Ollama `/no_think` 支持。

重点是防止远程 API 卡死，保障 600s 评测预算。

### code_v1.0

`prompting/llm_client.py` 抽象出公共 LLM 客户端：

- 默认 OpenAI-compatible。
- 支持 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OLLAMA_MODEL` 等变量。
- 允许用未提交的 `prompting/openai_client.py` 做个人覆盖。
- 对 Qwen 默认关闭 thinking。

它更通用、更干净，但没有 `code_original` 那种强制子进程 timeout 和 episode budget guard。

## Prompter 与规划逻辑差异

### code_original: IoA-Lite

`IoaLitePrompter` 在原始 centralized planner 上追加结构化状态：

- `[COMM_HEADER]`
- `[OBSERVATION_CONTEXT]`
- `[ROBOT_REGISTRY]`
- `[CONVERSATION_CONTEXT]`
- `[GROUP_STATE]`
- `[TASK_TABLE]`
- `[AWAIT_STATUS]`
- `[DYNAMIC_PLAN]`
- `[VALIDATOR_FEEDBACK]`
- `[LOCAL_TURN_TRACE]`
- `[OUTPUT_CONTRACT]`

它希望用统一的 IoA-inspired schema 改善多机器人协作，而不是为每个任务写完整代码策略。

### code_v1.0: Deterministic Fallback

`SingleThreadPrompter` 内部加入了大量 fallback：

- Sort panel relay planner。
- Cabinet door/object stage planner。
- Rope path candidate planner。
- Sweep MOVE/SWEEP/DUMP planner。
- Sandwich recipe planner。
- Pack item/slot/path planner。

并通过 `validate_fallback_candidates()` 让候选计划先经过 parser 和 feedback 检查。这个机制对成功率很有帮助，因为它能绕过 LLM 格式错误、重复动作和不稳定推理。

本质差异：

```text
code_original 把状态交给 LLM 决策；
code_v1.0 把很多决策直接写成 Python 规则。
```

## Parser 与反馈差异

### code_original

主要目标是鲁棒性：

- 缺失 `EXECUTE` 不 crash。
- 空 parsed actions 不 crash。
- 使用最后一个 `EXECUTE` block。
- parse failure 转成 feedback。

它尽量不改变任务语义和执行层。

### code_v1.0

parser 改动更偏物理执行：

- 支持更复杂的 `action_and_path`。
- rope grasp target 特殊处理。
- pick-and-place 组合动作处理。
- place/put 的 pre-place 和 release waypoint。
- action PATH 强制解析。

这类改动会直接影响 IK/RRT 可行性，属于执行层优化。

## 环境与任务 prompt 差异

### code_original

基本不修改 `rocobench/envs/*`。任务知识主要进入 `IoaLitePrompter.TASK_PROFILES` 和 prompt sections。

优点：

- 更干净，更容易说明没有改变官方任务。
- 方便做方法消融。

缺点：

- 对任务物理细节的控制较弱。
- 遇到 pack/cabinet/sweep 的 terminal-state 和 placement 问题时，单靠 prompt 不够稳。

### code_v1.0

直接修改多个 `rocobench/envs/task_*.py`：

- Sort 增强 panel 状态、目标提示和 task feedback。
- Sweep 增强 MOVE/SWEEP/DUMP 顺序反馈。
- Pack 增加 packed/unpacked、slot occupied/empty、hand status、已打包标记。
- Rope 增加 inward-offset grasp target 和 obstacle/groove 信息。
- Cabinet 强化门和物体规则。

优点是模型和 fallback 能读取更明确的状态；缺点是修改面更大，需要确认没有改变 reward/success 标准。

## 运动规划层差异

### code_original

基本保留原始 RoCoBench RRT/policy，只在 runner 层加 timeout/monitor。它的路线是“高层协作改善，不碰低层执行”。

### code_v1.0

明显修改底层执行：

- `policy.py`：release 阶段保持 in-hand object，sparsify path 后再碰撞验证。
- `rrt.py`：timeout 信息、sampler 调整、可跳过 smoothing。
- `rrt_multi_arm.py`：in-hand 物体相对姿态维护，mandatory waypoint，split planning。

这些修改对 Rope、Pack 等依赖 PATH 的任务很关键，但也让方法贡献混合了“高层规划”和“底层运动规划修复”。

## 任务表现路线差异

### code_original 已知状态

从 notes 看：

- DashScope/API 路线：sort、sweep、sandwich 成功；cabinet、rope、pack 多受 API timeout 或 near-terminal repair loop 影响。
- Ollama 路线：六任务 smoke 通过；local full pilot 中 sort、sandwich、rope 成功，sweep/cabinet/pack 仍需诊断。

失败主要集中在：

- API latency。
- near-terminal 状态判断。
- pack 物体放置状态追踪。
- sweep 最终 success false。

### code_v1.0 已知状态

各 `*_SOLUTION.md` 显示其目标是逐任务提分：

- Cabinet 曾达到 3/3。
- Rope 文档记录提升到 4/5。
- Sort、Pack、Sweep、Sandwich 都有单独调试文档和 fallback 策略。

这些结果更偏局部任务工程验证，而不是统一方法的系统评估。

## 可以互相借鉴的内容

### code_original 可以吸收 code_v1.0 的内容

优先级最高：

1. **Fallback candidate validation**
   - 不一定全盘采用 deterministic planner，但可以把“生成 K 个代码候选，parser/feedback 选第一个可行”作为 IoA-Lite 的 safety layer。

2. **Pack state/slot hints**
   - `packed/unpacked`
   - `slot occupied/empty`
   - `hand status`
   - failed slot avoidance

3. **Sweep deterministic phase hint**
   - MOVE/MOVE -> WAIT/SWEEP -> DUMP/WAIT

4. **Cabinet terminal-state rule**
   - 双门开后，未完成物体必须继续 `PICK obj PLACE coaster`，禁止 all-WAIT。

5. **Rope inward-offset grasp and path candidates**
   - 这属于物理可执行性修复，可能比 prompt 更直接。

需要谨慎吸收：

- 自动提高 `tsteps`。
- 大规模修改 reward/task env。
- 修改 RRT/policy 后直接归因于 IoA-Lite。

### code_v1.0 可以吸收 code_original 的内容

1. **正式 evaluator 参数化**
   - 当前 code_v1.0 没有提交 `evaluator.py`，而 code_original 的评测入口更完整。

2. **API timeout budget guard**
   - 子进程 LLM call 和 total budget 对远程 API 更稳。

3. **统一方法叙事**
   - IoA-Lite 的 task table、robot registry、typed feedback 可以给 code_v1.0 的 fallback 规则提供更清晰的论文解释。

4. **实验记录结构**
   - `EXPERIMENT_NOTES.md`、`experiment_notes/`、`SESSION_HANDOFF.md` 的记录方式更适合长期迭代。

## 风险对比

| 风险 | code_original | code_v1.0 |
|---|---|---|
| 成功率短期冲刺 | 中等，依赖 LLM 和状态提示 | 高，fallback 直接解决失败 |
| 论文方法统一性 | 高 | 中等偏低 |
| 任务泛化性 | 较好 | 较弱 |
| 官方评测边界风险 | 较低 | 中等，需要核对 tsteps/env/policy 改动 |
| 调试可控性 | 中等 | 高 |
| 消融设计 | 清晰：schema/task_table/feedback/fsm/dynamic | 较复杂：每个 fallback 和物理修复都可能是变量 |
| 原版 IoA 相关性 | 较强，是 IoA-inspired | 较弱，主要是 task engineering |

## 综合判断

如果目标是“拿高分”，`code_v1.0` 的工程策略很值得借鉴，尤其是确定性 fallback 和物理路径修复。它直接针对 RoCoBench 的实际失败模式，能快速提高任务成功率。

如果目标是“写出有研究叙事的报告/论文”，`code_original` 的 IoA-Lite 路线更容易形成统一贡献：能力注册、任务表、动态计划、反馈驱动修复和协作状态协议。

最合理的组合路线是：

```text
以 code_original 的 IoA-Lite 作为主方法叙事；
吸收 code_v1.0 的 fallback candidate validation 和关键任务状态 hints；
把底层 RRT/policy 修复作为 engineering stabilization，单独报告；
不要把所有 task-specific fallback 都混称为 IoA-Lite。
```

## 建议下一步

1. 在 `code_original` 中先移植最小风险模块：
   - Pack slot/state hints。
   - Cabinet terminal-state hints。
   - Sweep phase hints。
   - fallback candidate validation 框架。

2. 对 Rope 的底层修复单独评估：
   - inward-offset grasp target。
   - obstacle-aware path candidates。
   - split planning / mandatory waypoint。

3. 保持正式 evaluator 的默认预算：
   - `num_runs=5`
   - `tsteps=10`，除非题目/助教确认可针对 rope/pack 改 horizon。
   - `run_timeout=600`

4. 报告中明确区分：
   - IoA-Lite coordination method。
   - Deterministic fallback safety layer。
   - Task-specific execution fixes。

这样可以同时获得较好的工程表现和较清晰的研究叙事。
