# Sweep 实训方案

本项目基于开源 RoCo / RoCoBench 框架完成 Sweep 多机器人协同清扫任务。原框架使用大语言模型生成多机器人协作动作，通过文本解析器转换为机器人动作，再由 MuJoCo 环境、反馈模块和 RRT 路径规划器验证并执行。

## 任务理解

Sweep 任务中，两个机器人需要合作清扫桌子上的立方体：

- **机器人配置：**
  - Alice (UR5E + Robotiq) - 持有簸箕
  - Bob (Franka Panda) - 持有扫帚

- **目标立方体：**
  - `red_cube`
  - `green_cube`
  - `blue_cube`

- **任务流程：**
  1. 两个机器人一起移动到同一个立方体旁边
  2. Alice 用簸箕等待，Bob 用扫帚将立方体扫入簸箕
  3. Alice 将簸箕中的立方体倒入垃圾桶

任务成功的关键是严格按照三阶段循环操作：MOVE → SWEEP → DUMP，确保两个机器人同步协作。

## 基线方法

沿用 RoCoBench 的原始流程：

1. 环境通过 `describe_obs()` 给出立方体位置、簸箕位置、垃圾桶位置、机器人末端位姿。
2. LLM 根据任务描述和动作格式输出每个机器人一条动作。
3. `LLMResponseParser` 解析 `MOVE`、`SWEEP`、`DUMP`、`WAIT`。
4. `FeedbackManager` 检查动作约束、同步性、状态验证。
5. `PlannedPathPolicy` 使用 RRT / IK 在 MuJoCo 中规划和执行。

## 改进方向

### 1. Prompt 改进

在 `prompting/plan_prompter.py` 中增强清扫操作提示：

- 明确三阶段循环：MOVE → SWEEP → DUMP。
- Alice 只能 DUMP，Bob 只能 SWEEP。
- 两个机器人必须同时 MOVE 到同一个立方体。
- SWEEP 时 Alice 必须 WAIT 在立方体旁边。
- DUMP 时 Bob 必须 WAIT。
- 如果反馈指出状态错误或动作无效，调整策略。

### 2. 双臂协同机制

为 Sweep 任务设计同步协作策略：

- **阶段一：移动**
  - Alice 和 Bob 同时 MOVE 到同一个立方体
- **阶段二：清扫**
  - Alice WAIT（用簸箕对准立方体）
  - Bob SWEEP（用扫帚将立方体扫入簸箕）
- **阶段三：倾倒**
  - Alice DUMP（将立方体倒入垃圾桶）
  - Bob WAIT
- **协调规则：**
  - 必须完成当前立方体的清扫才能开始下一个
  - 只有当簸箕中有立方体时才能 DUMP
  - 两个机器人必须同步移动到同一个目标

### 3. 路径规划稳定性

新增路径和调试控制：

- `--rrt_timeout`：缩短失败规划等待时间。
- `--skip_smooth_path`：调试阶段跳过路径平滑。
- 簸箕放置高度固定为 0.23
- 扫帚偏移高度固定为 0.432
- 过滤已经进入垃圾桶的立方体

### 4. 大模型适配

公共代码默认通过 `prompting/llm_client.py` 读取 OpenAI-compatible 环境变量；个人需要特殊客户端时，可以在本地创建被 `.gitignore` 忽略的 `prompting/openai_client.py`。

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=Qwen/Qwen3.5-27B
```

## 推荐测试命令

快速调试前 3 步：

```bash
python run_dialog.py --task sweep --comm_mode plan --num_runs 1 --tsteps 3 --num_replans 1 --skip_display --run_name sweep_debug_3 --rrt_timeout 15 --skip_smooth_path
```

完整测试：

```bash
python run_dialog.py --task sweep --comm_mode plan --num_runs 1 --tsteps 10 --num_replans 1 --skip_display --run_name sweep_eval_10 --rrt_timeout 15 --skip_smooth_path
```

查看失败原因：

```bash
cat data/sweep_eval_10/run_0/step_*/prompts/*feedback*.json
cat data/sweep_eval_10/run_0/step_*/prompts/fallback_*.json
```

## 报告表述建议

可以这样描述本项目贡献：

> 本实训基于 RoCoBench 的 Sweep 任务，沿用其 LLM 生成协作计划、环境反馈修正、RRT 路径规划和 MuJoCo 执行验证流程。针对清扫任务的同步性要求、角色分工和状态管理等问题，本文设计了三阶段状态机控制、角色专用动作限制、同步移动约束和状态验证反馈，以提升双机器人协同清扫任务的成功率和效率。

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
