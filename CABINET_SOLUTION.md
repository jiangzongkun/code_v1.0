# Cabinet 实训方案

本项目基于开源 RoCo / RoCoBench 框架完成 Cabinet 多机器人协同柜门操作任务。原框架使用大语言模型生成多机器人协作动作，通过文本解析器转换为机器人动作，再由 MuJoCo 环境、反馈模块和 RRT 路径规划器验证并执行。

## 2026-05-19 失败分析与修复

分析目录：

- `output/run_20260519_021721/tasks/02_cabinet`：5 个 run 中仅 `run_2` 成功，成功率 1/5。
- `output/run_20260519_034509/tasks/02_cabinet`：5 个 run 中仅 `run_2` 成功，成功率 1/5，且有 2 个超时。

共同失败模式不是开门顺序错误。多数失败 run 都按相同的前 4 步执行：Alice/Bob 抓左右门把手，Alice/Bob 打开门，Chad 搬 `mug`，Chad 搬 `cup`。真正的问题在 `PICK cup PLACE cup_coaster` 执行后，`cup` 没有稳定落到 `cup_coaster`，而是偏到桌面边缘或低处，例如历史日志中出现过 `(0.8, 0.3, 0.2)`、`(0.5, 1.1, -0.4)`、`(-0.2, 1.1, -0.4)`。之后任务尚未完成，模型又经常输出全员 `WAIT`，被环境反馈拒绝；或者继续让 Chad 抓 `cup`，但 cup 已经不可达，触发 `Out of reach`，最终耗尽步数或超时。

本次修复面向机制，不针对固定 seed 或固定坐标：

1. `prompting/plan_prompter.py`
   - Cabinet 改回默认 LLM-first：先让模型根据当前场景规划；只有模型多次输出不可解析或不可执行动作时，才调用确定性 fallback 兜底。
   - 将 fallback 的状态机规则提升为高质量 Cabinet prompt：按门是否关闭、机器人是否已抓住门把手、mug/cup 是否在杯垫上来决定下一步，严格要求只输出 `EXECUTE` block。
   - prompt 根据当前柜体/把手位置选择左右侧分工，不针对固定 run 或固定 seed；fallback 也继续根据 `cabinet_pos` 动态选择门机器人和取物机器人。
   - 明确“mug/cup 未完成时不能全员 `WAIT`”、“必须等两扇门都打开后再取物”、“`PICK mug/cup` 必须和 `PLACE coaster` 写在同一个动作里”。

2. `prompting/parser.py`
   - 对 Cabinet 的 `PICK mug/cup PLACE coaster` 增加专用搬运路径：从柜内抓取后先抬高，再水平移动到 coaster 上方，再下降释放。
   - 路径由当前抓取点、目标 coaster 和物体几何半径计算，避免低空直线横移擦碰柜体、桌面或导致杯子释放不稳。
   - Cabinet 放置后不再强制回 home，避免释放瞬间或释放后回撤动作把杯子再次带偏。
   - 对 Cabinet 门把手抓取不再使用固定高度，而是从 MuJoCo 模型中的把手 cylinder 几何生成同一根把手上的候选点，再用正式的碰撞感知 IK 选择可达点。
   - 对 `mug/cup` 抓取不再写固定平移量；从 weld body、物体几何半径、物体垂直范围和当前机器人末端位置生成接近方向候选点，再用同一套 IK/碰撞检查选择候选。

3. `rocobench/policy.py`
   - `policy.py` 是执行层：把 parser 生成的末端目标转换为 IK、RRT 路径、夹爪/吸盘控制和 MuJoCo weld 开关。
   - 修复 release 计划的状态不一致：如果释放阶段发现对应物体的 weld 已经 active，就把该物体作为 in-hand 物体加入规划，让 RRT/碰撞检查按“机器人手里拿着 cup/mug”的真实状态规划。
   - `WAIT` 机器人保持当前关节状态，不再重新对当前末端位姿做 IK，避免等待开门的机器人在取物阶段引入无关 IK 扰动。

4. `rocobench/envs/task_cabinet.py`
   - 修正 Cabinet 可达性反馈：根据柜体在桌面左侧或右侧动态判断 Alice/Bob/Chad 能到达的门把手，避免 prompt 或 fallback 被错误反馈带偏。
   - 保留门把手优先级约束：单独推进非优先门前，必须先抓住或同时抓住优先门，避免某些场景下一扇门先打开后挡住另一侧抓取。

5. `rocobench/rrt_multi_arm.py`
   - 将 IK reset 搜索预算从 20 提高到 40。随机 seed 下某些合法候选点在 20 次 reset 时会偶发失败，但 40 次稳定成功；这是通用搜索预算提升，不依赖 Cabinet 坐标。

验证结果：

```bash
/root/miniconda3/envs/roco/bin/python -m compileall run_dialog.py prompting rocobench/envs rocobench/policy.py
```

结果：通过。

固定 seed 0 的 LLM-first plan mode 已通过完整 Cabinet 流程：

```bash
MUJOCO_GL=egl xvfb-run -a /root/miniconda3/envs/roco/bin/python run_dialog.py --task cabinet --run_name cabinet --data_dir output/direct_plan_proxy --start_id 0 --num_runs 1 --skip_display --comm_mode plan --tsteps 10 --seed 0 --run_timeout 600
```

结果：`output/direct_plan_proxy/cabinet/run_0_seed_0/steps3_success_True.json`，step 0 抓门把手，step 1 开门，step 2 放 `mug`，step 3 放 `cup`。

随机 seed 验证暴露过两个额外问题：

- base seed `927311687` 下的多个 run 在 step 0 失败于 Bob `PICK right_door_handle` IK，失败目标高度约 `0.63-0.66`。修复后把手抓取点由把手几何和 IK 选择，不再固定到某个高度。
- seed `179283730` 下开门后 `mug` 抓取会因为高位目标和 IK 随机 reset 不稳定而失败。修复后 `mug/cup` 抓取点由 weld 点、物体几何和碰撞感知 IK 共同选择，并提升 IK 搜索预算。

随机 seed 复验命令：

```bash
MUJOCO_GL=egl xvfb-run -a /root/miniconda3/envs/roco/bin/python run_dialog.py --task cabinet --run_name cabinet --data_dir output/direct_plan_handle_seed927311691 --start_id 0 --num_runs 1 --skip_display --comm_mode plan --tsteps 10 --seed 927311691 --run_timeout 600
```

结果：`output/direct_plan_handle_seed927311691/cabinet/run_0_seed_927311691/steps3_success_True.json`，用时约 301.53 秒。该场景在修复前会卡在 step 0 的 Bob 右门把手 IK，修复后能完成开门、搬 `mug`、搬 `cup` 全流程。

```bash
MUJOCO_GL=egl xvfb-run -a /root/miniconda3/envs/roco/bin/python run_dialog.py --task cabinet --run_name cabinet --data_dir output/direct_plan_general3_seed179283730 --start_id 0 --num_runs 1 --skip_display --comm_mode plan --tsteps 10 --seed 179283730 --run_timeout 600
```

结果：`output/direct_plan_general3_seed179283730/cabinet/run_0_seed_179283730/steps3_success_True.json`，用时约 431.07 秒。该场景覆盖了随机高把手、开门后 `mug` 高位抓取和 `cup` 放置。

继续按 evaluator 入口运行 3 次随机 seed 复验：

```bash
MUJOCO_GL=egl xvfb-run -a /root/miniconda3/envs/roco/bin/python -c "from evaluator import test_run_dialog; test_run_dialog('cabinet', 3, 'output/single_task_gpu', comm_mode='plan', run_timeout=600)"
```

## 任务理解

Cabinet 任务中，三个机器人需要从柜子中取出杯子并放置到指定位置：

- **机器人配置：**
  - Alice (UR5E + Robotiq)
  - Bob (Franka Panda)
  - Chad (UR5E + Suction)

- **任务目标：**
  - 将 `mug` 放到 `mug_coaster`
  - 将 `cup` 放到 `cup_coaster`

- **前置条件：**
  - 必须先打开左右两扇柜门
  - 打开柜门后需要保持打开状态（一个机器人需要 WAIT 保持）

任务成功的关键是正确的操作顺序：先开门，再取物，最后放置。需要机器人之间的密切配合，避免碰撞。

## 基线方法

沿用 RoCoBench 的原始流程：

1. 环境通过 `describe_obs()` 给出柜子状态、物体位置、杯垫位置、机器人末端位姿和持物状态。
2. LLM 根据任务描述和动作格式输出每个机器人一条动作。
3. `LLMResponseParser` 解析 `PICK`、`OPEN`、`PLACE`、`WAIT`。
4. `FeedbackManager` 检查动作约束、碰撞、可达性。
5. `PlannedPathPolicy` 使用 RRT / IK 在 MuJoCo 中规划和执行。

## 改进方向

### 1. Prompt 改进

在 `prompting/plan_prompter.py` 中增强柜门操作提示：

- 明确操作顺序：先打开柜门，再取物，最后放置。
- 明确开门规则：必须先 PICK 门把手，再 OPEN，然后 WAIT 保持开门状态。
- 指定机器人分工：根据可达性分配开门和取物任务。
- 允许 `WAIT` 动作，用于保持柜门打开或等待其他机器人完成操作。
- 如果反馈指出柜门未打开或物体不可达，则调整策略。

### 2. 三机器人协同机制

为 Cabinet 任务设计协作策略：

- **阶段一：开门**
  - 两个机器人分别打开左右柜门
  - 第三个机器人准备取物
- **阶段二：取物**
  - 一个机器人取出 mug
  - 另一个机器人取出 cup
  - 保持柜门打开
- **阶段三：放置**
  - 将物体放置到正确杯垫上
- **协调规则：**
  - 柜门必须保持打开状态才能取物
  - 只能从柜子内部抓取物体
  - 放置时使用高位路径避免碰撞

### 3. 路径规划稳定性

新增路径和调试控制：

- `--rrt_timeout`：缩短失败规划等待时间。
- `--skip_smooth_path`：调试阶段跳过路径平滑。
- 抓取路径安全高度约 0.5
- 放置路径安全高度约 0.55
- 过滤异常状态的物体（如已取出的物体）

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
python run_dialog.py --task cabinet --comm_mode plan --num_runs 1 --tsteps 3 --num_replans 1 --skip_display --run_name cabinet_debug_3 --rrt_timeout 15 --skip_smooth_path
```

完整测试：

```bash
python run_dialog.py --task cabinet --comm_mode plan --num_runs 1 --tsteps 12 --num_replans 1 --skip_display --run_name cabinet_eval_12 --rrt_timeout 15 --skip_smooth_path
```

查看失败原因：

```bash
cat data/cabinet_eval_12/run_0/step_*/prompts/*feedback*.json
cat data/cabinet_eval_12/run_0/step_*/prompts/fallback_*.json
```

## 报告表述建议

可以这样描述本项目贡献：

> 本实训基于 RoCoBench 的 Cabinet 任务，沿用其 LLM 生成协作计划、环境反馈修正、RRT 路径规划和 MuJoCo 执行验证流程。针对多阶段任务协调、柜门操作顺序和双臂协作碰撞等问题，本文设计了分阶段任务规划、开门-取物-放置的状态机控制、WAIT 保持动作和可达性感知提示，以提升多机器人协同柜门操作任务的成功率和可靠性。

## 代码实现关键点

### 核心代码文件

| 文件 | 功能 |
|------|------|
| `rocobench/envs/task_cabinet.py` | Cabinet任务环境定义 |
| `rocobench/envs/base_env.py` | 基础环境类和通用接口 |
| `rocobench/policy.py` | 策略执行和路径规划 |
| `rocobench/rrt.py` | RRT路径规划算法 |
| `real_world/prompts/feedback.py` | 反馈生成模块 |
| `real_world/prompts/parser.py` | LLM输出解析器 |

### 成功案例配置 (cabinet_test2)

```json
{
  "task": "cabinet",
  "tsteps": 10,
  "comm_mode": "plan",
  "output_mode": "action_only",
  "num_replans": 3,
  "llm_source": "gpt-4",
  "rrt_timeout": 20.0,
  "skip_smooth_path": false,
  "direct_waypoints": 5,
  "use_weld": 1
}
```

### 关键代码实现

#### 1. 任务环境初始化 (`task_cabinet.py`)

```python
class CabinetTask(MujocoSimEnv):
    def __init__(self, filepath="rocobench/envs/task_cabinet.xml", ...):
        self.robot_names = ["ur5e_robotiq", "panda", "ur5e_suction"]
        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob",
            "ur5e_suction": "Chad",
        }
        # 定义柜子位置范围
        CABINET_LEFT_RANGE = (
            np.array([-0.7, 0.5, 0.47]),
            np.array([-0.7, 0.5, 0.7]),
        )
        # 定义杯垫位置
        self.coaster_pos = dict(
            mug_coaster=(x, y, z+0.10),
            cup_coaster=(x, y, z+0.10)
        )
        self.align_threshold = 0.25  # 放置精度阈值
```

**关键参数：**
- `align_threshold = 0.25` - 判断物体是否正确放置到杯垫的距离阈值
- 柜门铰链角度范围：左门 [-2.6, 0]，右门 [0, 2.6]

#### 2. 动作空间定义

```python
CABINET_ACTION_SPACE = """
[Action Options]
1) PICK <handle>.          # 抓取门把手
2) OPEN <handle>.          # 打开柜门
3) PICK <object> PLACE <location>  # 抓取并放置物体
4) WAIT.                   # 保持当前位置
"""
```

#### 3. 状态描述 (`describe_obs`)

```python
def describe_obs(self, obs: EnvState):
    # 描述柜门状态（开/闭）和把手位置
    object_desp += self.describe_cabinet(obs) + "\n"
    # 描述杯子和马克杯状态（在柜内/已放置）
    object_desp += self.describe_cups(obs) + "\n"
    # 描述三个机器人的位置和持物状态
    robot_desp = "\n".join([self.describe_robot_state(obs, name) ...])
```

#### 4. 可达性管理 (`get_agent_prompt`)

```python
def get_agent_prompt(self, obs, agent_name):
    if self.cabinet_pos[0] < 0:  # 柜子在左侧
        if agent_name == "Alice":
            reachables = "left_door_handle, mug, cup"  # Alice负责左侧
        elif agent_name == "Bob":
            reachables = "right_door_handle"  # Bob负责右侧门把手
        elif agent_name == "Chad":
            reachables = "right_door_handle, mug, cup"  # Chad负责右侧取物
```

#### 5. 反馈检查 (`get_task_feedback`)

```python
def get_task_feedback(self, llm_plan, pose_dict):
    # 检查PICK mug/cup时是否包含PLACE
    # 检查可达性（Chad不能到达左侧门把手，Bob不能到达左侧门把手）
    # 检查是否全部WAIT（至少需要一个机器人行动）
```

#### 6. 完成条件 (`get_reward_done`)

```python
def get_reward_done(self, obs: EnvState):
    # mug和cup都在各自杯垫上（距离<0.25）才算成功
    for obj in ["mug", "cup"]:
        obj_pos = self.physics.data.body(obj).xpos
        coaster_pos = self.coaster_pos[f"{obj}_coaster"]
        if np.linalg.norm(obj_pos - coaster_pos) > self.align_threshold:
            return 0, False
    return 1, True
```

### 执行流程

1. **环境初始化** - 随机放置柜子位置，杯子在柜内
2. **Prompt生成** - 描述柜门状态、物体位置、机器人可达范围
3. **LLM规划** - GPT-4生成开门和取物动作
4. **反馈验证** - 检查动作合法性和可达性
5. **路径规划** - RRT算法规划无碰撞路径
6. **动作执行** - 在MuJoCo中执行OPEN/PICK/PLACE/WAIT
7. **状态更新** - 更新柜门角度和物体位置

### 成功执行示例 (cabinet_test2)

**Step 0-1 成功的关键：**
- Alice和Bob分别PICK左右门把手
- Chad保持WAIT状态
- 正确理解柜门需要先被抓取才能打开

**Step 2 成功的关键：**
- Alice和Bob分别OPEN左右柜门
- Chad开始PICK mug并PLACE到mug_coaster
- 使用联合动作（PICK+PLACE）简化操作

**Step 3 成功的关键：**
- Alice和Bob保持WAIT保持柜门打开
- Chad PICK cup并PLACE到cup_coaster
- 完成放置任务

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
