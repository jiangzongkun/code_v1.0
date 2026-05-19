# Sandwich 实训方案

本项目基于开源 RoCo / RoCoBench 框架完成 Sandwich 多机器人协同制作三明治任务。原框架使用大语言模型生成多机器人协作动作，通过文本解析器转换为机器人动作，再由 MuJoCo 环境、反馈模块和 RRT 路径规划器验证并执行。

## 任务理解

Sandwich 任务中，两个机器人需要合作制作三明治：

- **机器人配置：**
  - Chad (UR5E + Suction)
  - Dave (Humanoid)

- **食材列表：**
  - `bread_slice1`, `bread_slice2` (面包片)
  - `bacon`, `cheese`, `tomato`, `cucumber`, `ham`, `beef_patty` (配料)

- **食谱顺序（示例）：**
  - bacon: bread_slice1 → bacon → cheese → tomato → bread_slice2
  - vegetarian: bread_slice1 → cheese → tomato → cucumber → bread_slice2
  - beef_patty: bread_slice1 → beef_patty → cheese → tomato → bread_slice2
  - ham: bread_slice1 → ham → cheese → tomato → cucumber → bread_slice2

- **可达范围限制：**
  - Chad 只能从桌子右侧取物
  - Dave 只能从桌子左侧取物

任务成功的关键是严格按照食谱顺序堆叠食材，并且只有一个机器人可以同时进行放置操作。

## 基线方法

沿用 RoCoBench 的原始流程：

1. 环境通过 `describe_obs()` 给出食材位置、砧板位置、机器人末端位姿和持物状态。
2. LLM 根据任务描述和动作格式输出每个机器人一条动作。
3. `LLMResponseParser` 解析 `PICK`、`PUT`、`WAIT`。
4. `FeedbackManager` 检查动作约束、食谱顺序、碰撞。
5. `PlannedPathPolicy` 使用 RRT / IK 在 MuJoCo 中规划和执行。

## 改进方向

### 1. Prompt 改进

在 `prompting/plan_prompter.py` 中增强三明治制作提示：

- 明确当前食谱顺序。
- 明确每个机器人的可达区域（Chad 右侧，Dave 左侧）。
- 明确一次只能有一个机器人进行 PUT 操作。
- 必须按照食谱顺序堆叠食材。
- bread_slice1 必须先放到砧板上。
- 如果反馈指出顺序错误或物体不可达，调整策略。

### 2. 双臂协同机制

为 Sandwich 任务设计协作策略：

- **阶段一：准备**
  - 将 bread_slice1 放到砧板上
- **阶段二：堆叠**
  - 按照食谱顺序依次放置配料
  - 只有一个机器人进行 PUT，另一个机器人可以 PICK 准备
- **阶段三：完成**
  - 放置 bread_slice2 完成三明治
- **协调规则：**
  - 严格按照食谱顺序操作
  - 一次只能有一个机器人进行放置
  - 已堆叠的食材不能被重新抓取

### 3. 路径规划稳定性

新增路径和调试控制：

- `--rrt_timeout`：缩短失败规划等待时间。
- `--skip_smooth_path`：调试阶段跳过路径平滑。
- 抓取路径安全高度约 0.45
- 放置路径安全高度约 0.5
- 过滤已经堆叠的物体

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
python run_dialog.py --task sandwich --comm_mode plan --num_runs 1 --tsteps 3 --num_replans 1 --skip_display --run_name sandwich_debug_3 --rrt_timeout 15 --skip_smooth_path
```

完整测试：

```bash
python run_dialog.py --task sandwich --comm_mode plan --num_runs 1 --tsteps 12 --num_replans 1 --skip_display --run_name sandwich_eval_12 --rrt_timeout 15 --skip_smooth_path
```

查看失败原因：

```bash
cat data/sandwich_eval_12/run_0/step_*/prompts/*feedback*.json
cat data/sandwich_eval_12/run_0/step_*/prompts/fallback_*.json
```

## 报告表述建议

可以这样描述本项目贡献：

> 本实训基于 RoCoBench 的 Sandwich 任务，沿用其 LLM 生成协作计划、环境反馈修正、RRT 路径规划和 MuJoCo 执行验证流程。针对食谱顺序控制、双臂交替放置和分区取物等问题，本文设计了食谱感知提示词、单臂放置约束、分区可达性管理和顺序验证机制，以提升双机器人协同制作三明治任务的成功率和规范性。

## 代码实现关键点

### 核心代码文件

| 文件 | 功能 |
|------|------|
| `rocobench/envs/task_sandwich.py` | Sandwich任务环境定义 |
| `rocobench/envs/base_env.py` | 基础环境类和通用接口 |
| `rocobench/policy.py` | 策略执行和路径规划 |
| `rocobench/rrt.py` | RRT路径规划算法 |
| `real_world/prompts/feedback.py` | 反馈生成模块 |
| `real_world/prompts/parser.py` | LLM输出解析器 |

### 成功案例配置 (sandwich_test1)

```json
{
  "task": "sandwich",
  "tsteps": 10,
  "comm_mode": "plan",
  "output_mode": "action_only",
  "num_replans": 3,
  "llm_source": "gpt-4",
  "rrt_timeout": 20.0,
  "skip_smooth_path": true,
  "direct_waypoints": 5,
  "use_weld": 1
}
```

### 关键代码实现

#### 1. 任务环境初始化 (`task_sandwich.py`)

```python
class MakeSandwichTask(MujocoSimEnv):
    def __init__(self, filepath="rocobench/envs/task_sandwich.xml", ...):
        self.robot_names = ["ur5e_suction", "humanoid"]
        self.agent_names = ["Chad", "Dave"]
        self.robot_name_map = {
            "ur5e_suction": "Chad",
            "humanoid": "Dave",
        }
        # 定义食谱
        SANDWICH_RECIPES = {
            "bacon": ["bread_slice1", "bacon", "cheese", "tomato", "bread_slice2"],
            "vegetarian": ["bread_slice1", "cheese", "tomato", "cucumber", "bread_slice2"],
            "beef_patty": ["bread_slice1", "beef_patty", "cheese", "tomato", "bread_slice2"],
            "ham": ["bread_slice1", "ham", "cheese", "tomato", "cucumber", "bread_slice2"],
        }
        self.align_threshold = 0.15  # 放置精度阈值
        self.use_prepick = True  # 启用预抓取
        self.use_preplace = True  # 启用预放置
```

**关键参数：**
- `align_threshold = 0.15` - 判断物体是否正确堆叠的距离阈值
- `use_prepick = True`, `use_preplace = True` - 启用预抓取和预放置
- Chad 只能到达桌子右侧，Dave 只能到达桌子左侧

#### 2. 食谱顺序定义

```python
SANDWICH_ACTION_SPACE = """
[Action Options]
1) PICK <obj>, Only PICK if gripper is empty. PICK only the correct next item according to the recipe.
2) PUT <obj1> <obj2>. <obj1> can be one of the foods. <obj2> can be food, cutting_board, or table.
3) WAIT, do nothing.
Only one robot can PUT each round. You must PICK up an item before PUT.
"""
```

#### 3. 状态描述 (`describe_obs`)

```python
def describe_obs(self, obs: EnvState):
    # 描述每个食材的位置和状态
    for food in self.food_items:
        object_desp += self.describe_food_state(obs, food) + "\n"
    # 描述每个机器人的位置和持物状态
    robot_desp = "The robots:\n"
    for robot_name, agent_name in self.robot_name_map.items():
        robot_desp += self.describe_robot_state(obs, robot_name) + "\n"
```

#### 4. 可达性管理 (`get_agent_prompt`)

```python
def get_agent_prompt(self, obs, agent_name):
    # Dave 只能到达桌子左侧，Chad 只能到达桌子右侧
    table_side = "left" if agent_name == "Dave" else "right"
    other_side = "right" if agent_name == "Dave" else "left"
    # 只描述机器人可达侧的食材
    for food in self.food_items:
        desp = self.describe_food_state(obs, food)
        if other_side not in desp:
            food_states.append(desp.replace(table_side, "your"))
```

#### 5. 反馈检查 (`get_task_feedback`)

```python
def get_task_feedback(self, llm_plan, pose_dict):
    # 检查食谱顺序
    for agent_name, action_str in llm_plan.action_strs.items():
        if 'PUT' in action_str:
            idx1 = self.recipe_order.index(obj1)
            if idx1 == 0 and obj2 != 'cutting_board':
                feedback += f"recipe says {obj1} must be put on cutting_board\n"
            elif idx1 > 0:
                idx2 = self.recipe_order.index(obj2)
                if idx2 != idx1 - 1:
                    feedback += f"recipe says {obj1} must be put on {self.recipe_order[idx1-1]}\n"
    # 检查是否同时PUT
    if all(['PUT' in action_str for action_str in llm_plan.action_strs.values()]):
        feedback += "only one robot can PUT at a time\n"
```

#### 6. 完成条件 (`get_reward_done`)

```python
def get_reward_done(self, obs):
    # 检查bread_slice1是否在砧板上
    # 按食谱顺序检查每层食材是否正确堆叠
    for i, item in enumerate(self.recipe_order):
        item_contacts = obs.objects[item].contacts
        if len(item_contacts) == 0:
            return 0, False
        if i == len(self.recipe_order) - 1:
            return 1, True
        next_item = self.recipe_order[i+1]
        if next_item not in item_contacts:
            return 0, False
```

### 执行流程

1. **环境初始化** - 随机放置食材在桌面左右两侧，随机选择食谱
2. **Prompt生成** - 描述食材位置、食谱顺序、机器人可达范围
3. **LLM规划** - GPT-4生成PICK/PUT动作
4. **反馈验证** - 检查食谱顺序、放置约束
5. **路径规划** - RRT算法规划无碰撞路径
6. **动作执行** - 在MuJoCo中执行PICK/PUT
7. **状态更新** - 更新食材堆叠状态

### 成功执行示例 (sandwich_test1)

**食谱 bacon 示例：**
```
bread_slice1 → bacon → cheese → tomato → bread_slice2
```

**执行策略：**
```
Round 1:
- Dave: PICK bread_slice1 (从左侧取面包片)
- Chad: PICK bacon (从右侧取培根)

Round 2:
- Dave: PUT bread_slice1 cutting_board (将面包放到砧板上)
- Chad: WAIT

Round 3:
- Dave: PICK bacon
- Chad: WAIT

Round 4:
- Dave: PUT bacon bread_slice1 (将培根放到面包上)
- Chad: PICK cheese (同时Chad准备奶酪)

Round 5:
- Dave: WAIT
- Chad: PUT cheese bacon (将奶酪放到培根上)

... (继续按照食谱顺序堆叠)
```

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
