# code_v1.0 技术总结

## 总体定位

`code_v1.0` 是一个面向 RoCoBench 六任务成功率的工程型优化版本。它的核心路线不是复现 IoA 或构建新的多智能体通信协议，而是保留 RoCoBench 原有 `run_dialog.py -> prompter -> parser -> feedback -> RRT -> MuJoCo` 执行链，在此基础上加入大量任务专用规则、确定性 fallback、路径候选和底层运动规划修复。

简化概括：

```text
code_v1.0 =
  task-specific prompt
  + deterministic fallback planner
  + candidate validation by parser/feedback
  + task/env state hints
  + path/RRT/IK stability patches
```

这是一条“以任务通过率为目标”的策略，优先解决 LLM 输出不稳定、IK/RRT 失败、碰撞、重复无效动作和任务终止误判。

## 入口与运行方式

主要入口是 `run_dialog.py`。相比原始 RoCoBench，它新增或强化了：

- `--fallback_first`：在查询 LLM 前优先尝试确定性 fallback。
- `--rrt_timeout`：控制每段 RRT 规划超时。
- `--skip_smooth_path`：跳过路径平滑，避免平滑破坏强制 waypoint。
- 针对 `rope` 自动启用 `action_and_path`、split parsed plans，并把 horizon 调整到 6 步。
- 针对 `pack` 自动启用 `action_and_path`、split parsed plans、较低控制频率、至少 12 步和较短 RRT timeout。

正式成功判定仍在 `run_dialog.py` 中保持：

```text
timeout -> success=False
otherwise success = reward > 0
```

因此该项目主要改变“如何生成和执行计划”，而不是改变最终 reward/success 判定。

## LLM 接入

`prompting/llm_client.py` 是该项目的 LLM 兼容层，默认使用 OpenAI-compatible API：

- `OPENAI_BASE_URL` / `OPENAI_API_BASE` / `PROXY_BASE`
- `OPENAI_API_KEY` / `VLLM_API_KEY`
- `OPENAI_MODEL` / `LLM_MODEL` / `OLLAMA_MODEL`
- 默认模型名为 `Qwen/Qwen3.5-27B`
- 对 Qwen 默认加入 `enable_thinking=False`

它还允许本地创建未提交的 `prompting/openai_client.py` 作为个人覆盖实现。这个设计把公共仓库和个人 API 配置分离，比直接把 DashScope/Ollama 逻辑写死在 prompter 中更通用。

## Prompt 与任务规则

`prompting/plan_prompter.py` 对每个任务都写了专用规划提示：

- `SortOneBlockTask`：明确目标 panel、relay panel、Bob 的桥接角色。
- `CabinetTask`：明确开门、保持门、取杯流程，禁止未完成时全员 `WAIT`。
- `MoveRopeTask`：拆分 PICK/PUT 阶段，要求输出 PATH。
- `SweepTask`：强调 MOVE 同一 cube、WAIT/SWEEP、DUMP 顺序。
- `MakeSandwichTask`：严格 recipe order，一次只允许一个 PUT。
- `PackGroceryTask`：详细规定持物状态、空槽位、路径高度、机器人-物品偏好和串行策略。

这些 prompt 更像“任务专家系统说明”，而不是通用多机器人协作协议。

## 确定性 Fallback Planner

该项目最核心的实现是 `SingleThreadPrompter` 中的 fallback 机制。

主要函数包括：

- `build_fallback_response()`
- `build_fallback_candidates()`
- `validate_fallback_candidates()`
- `build_sort_fallback_response()`
- `build_sweep_fallback_response()`
- `build_sandwich_fallback_response()`
- `build_rope_fallback_candidates()`
- `build_pack_fallback_candidates()`
- `build_cabinet_fallback_response()`

执行逻辑：

1. 如果 `fallback_first=True`，先生成候选 fallback。
2. 每个候选先走 parser。
3. 再走 `FeedbackManager.give_feedback()`。
4. 选择第一个通过 parser 和环境反馈的候选。
5. 如果 LLM 多次失败，仍会在末尾尝试 fallback。

这种方法直接把一部分高层规划从 LLM 移到代码中，减少了模型幻觉和长 prompt 造成的格式失败。

## 各任务主要优化

### Sort

Sort 被建模成 panel 拓扑转运问题。代码中维护：

- cube 固定目标：`blue_square -> panel2`、`pink_polygon -> panel4`、`yellow_trapezoid -> panel6`
- 机器人可达 panel
- panel3/panel5 作为 handoff zone

fallback 会判断 cube 当前 panel、目标 panel、下一步可达 panel，并选择一个机器人执行单步转运，其余机器人 `WAIT`。

### Cabinet

Cabinet 使用确定性阶段机：

1. 门未开：门机器人 `PICK handle`。
2. 已抓 handle：执行 `OPEN handle`。
3. 门已开：门机器人 `WAIT` 保持。
4. 双门打开后：item robot 执行 `PICK mug/cup PLACE coaster`。

这避免了 LLM 在接近完成时输出全员 `WAIT` 或重复不合法动作。

### Rope

Rope 是该项目改动最重的任务之一：

- 在 env 中加入 inward-offset rope grasp target，避免直接抓绳端导致 IK 不稳。
- parser 对 rope 抓取点做专门处理。
- fallback 生成 obstacle-aware pick/put PATH 候选。
- `rrt_multi_arm.py` 支持强制 waypoint 和 split planning。
- `policy.py` 增强 release 时 in-hand 物体处理。

目标是将失败类型从 IK、碰撞、RRT timeout 收敛到可诊断的少数路径失败。

### Sweep

Sweep fallback 是一个固定同步流程：

```text
MOVE/MOVE -> WAIT/SWEEP -> DUMP/WAIT
```

env prompt 和 feedback 也更强地约束 Alice 与 Bob 必须移动到同一 cube，Bob 不能在 Alice 未就位时 sweep。

### Sandwich

Sandwich fallback 基于 recipe order：

- 找到下一个未完成 item。
- 若已有机器人持有该 item，则执行 `PUT item target`。
- 否则选择能到达该 item 的机器人执行 `PICK`。
- 另一机器人 `WAIT`。

该策略牺牲并行性，换取 recipe 约束稳定性。

### Pack Grocery

Pack 使用最多工程规则：

- 每轮优先一个机器人执行，另一个 `WAIT`。
- 持物优先 `PLACE`，空手才 `PICK`。
- 使用固定批次：`bread+milk`、`cereal+soda_can`、`banana+apple`。
- 物品-机器人偏好和可抓取限制。
- 空槽位检查、已失败槽位避让。
- 多候选 fallback，逐个通过 parser/feedback 验证。

这是典型的“LLM 不可靠时，用代码 planner 接管”的实现。

## Parser 与底层执行改动

`prompting/parser.py` 增强了：

- 空响应保护。
- rope grasp pose 特殊处理。
- pick-and-place 组合动作。
- safer pre-place / release waypoint。
- action_and_path 解析。

`rocobench/policy.py` 增强了：

- release 阶段保持 in-hand object 参与规划。
- RRT path sparsify 后再做碰撞验证。
- split plan 支持。

`rocobench/rrt.py` 和 `rocobench/rrt_multi_arm.py` 增强了：

- timeout 信息。
- center/near sampler。
- mandatory waypoint 处理。
- in-hand 物体相对姿态维护。
- IK 容差和 split planning。

## 风险与局限

1. **任务特化强**：许多规则直接绑定六个 RoCoBench 任务，泛化到新任务较弱。
2. **方法叙述不如 IoA-Lite 统一**：它更像一组 task-specific systems fixes，而不是一个统一协作框架。
3. **可能触碰评测预算争议**：例如 pack 自动提升到至少 12 steps，rope 调整到 6 steps；需要和题目/官方默认预算严格核对。
4. **fallback 可能遮蔽 LLM 贡献**：如果 `fallback_first` 大量生效，最终表现更接近代码策略而不是 LLM 多机器人协作。
5. **底层 planner 改动较大**：这能提升成功率，但报告中必须明确说明改的是路径规划/执行层，不应只声称 prompt 优化。

## 总结

`code_v1.0` 是一个强工程优化项目。它的优势是务实、直接、对失败模式响应快；它的劣势是研究抽象较弱、任务特化明显。若目标是短期提高 RoCoBench 成功率，它提供了很多可借鉴模块，尤其是 fallback candidate validation、Pack slot/state tracking、Rope path/IK 修复和 Sweep/Sandwich/Cabinet 的确定性阶段机。
