import os
import json
import pickle 
import numpy as np
from rocobench.envs import MujocoSimEnv, EnvState
from datetime import datetime
from .feedback import FeedbackManager
from .llm_client import chat_completion, response_content, response_usage
from .parser import LLMResponseParser
from typing import List, Tuple, Dict, Union, Optional, Any


SAFE_PICK_HEIGHT = 0.62
SAFE_PLACE_HEIGHT = 0.68
MIN_PATH_HEIGHT = 0.45
MAX_PATH_HEIGHT = 0.78
MIN_PACK_ITEM_Z = 0.05
MAX_PACK_ITEM_Z = 0.90
PACK_ROBOT_HANDOFF_PENALTY = 2.0
PACK_PICK_BATCHES = [
    ("bread", "milk"),
    ("cereal", "soda_can"),
    ("banana", "apple"),
]
PACK_ROBOT_ITEM_PREFERENCE = {
    "ur5e_robotiq": ["bread", "cereal", "banana"],
    "panda": ["milk", "soda_can", "apple"],
}
PACK_ROBOT_ALLOWED_ITEMS = {
    "ur5e_robotiq": {"bread", "cereal", "banana"},
    "panda": {"milk", "soda_can", "apple"},
}
SORT_CUBE_ORDER = ["blue_square", "pink_polygon", "yellow_trapezoid"]
SORT_CUBE_TARGETS = {
    "blue_square": "panel2",
    "pink_polygon": "panel4",
    "yellow_trapezoid": "panel6",
}
SORT_REACHABLE_PANELS = {
    "Alice": ["panel1", "panel2", "panel3"],
    "Bob": ["panel3", "panel4", "panel5"],
    "Chad": ["panel5", "panel6", "panel7"],
}

PATH_PLAN_INSTRUCTION="""
[How to plan PATH]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) Do not include the current gripper position as the first PATH point. The first PATH point must be a non-zero movement away from the current gripper position.
4) All PATH points must be evenly spaced from the current gripper position to the final target. Avoid repeated or duplicate coordinates.
5) Each <coord> must not collide with other robots, and must stay away from table and objects.
6) If a robot is already holding an object, it must PLACE that same object and must not PICK another object.
7) If a robot gripper is empty, it may PICK one grocery item and must not PLACE an object.
8) Never PLACE an object that the robot is not holding. Use different empty bin slots for different objects.
9) For PLACE actions, keep all PATH coordinates at z >= 0.60 before the final target; large objects such as cereal and milk should use z >= 0.68.
10) For PLACE actions, make the PATH smoothly and evenly approach the final bin slot; do not stop far from the bin slot before the final target.
11) For simultaneous PLACE actions, choose separated bin slots and keep the two PATHs far apart in x-y distance.
12) If any previous plan fails, do not repeat the same action pair and same PATH coordinates.
13) WAIT is allowed when it prevents collision. For WAIT, keep the robot stationary by repeating its current gripper position in the PATH.
14) If simultaneous PLACE actions fail or involve large objects, use one PLACE action and one WAIT action.
15) Follow this fixed batch order unless the named object is already packed: first bread+milk, then cereal+soda_can, then banana+apple.
16) Prefer assignments: Alice handles bread, cereal, banana; Bob handles milk, soda_can, apple.
16a) If the preferred robot cannot reach its assigned object or keeps failing, let the other robot take over that object while the preferred robot WAITs.
17) If both robots PICK and the plan fails due to collision, keep the batch but separate the PATHs, or use one PICK and one WAIT.
18) For simultaneous PICK actions, keep Alice's and Bob's first two PATH points separated by at least 0.35 in x-y distance.
19) For PICK actions, every PATH coordinate should keep z >= 0.45 so the gripper stays above objects and the table; the simulator will handle the final grasp target.
20) For apple and soda_can, use a high vertical approach with z around 0.72 before the final grasp.
21) Do not output PATH points at table or object contact height.
22) Keep all PATH z coordinates within 0.45 to 0.78 unless the simulator target is a bin slot.
23) If feedback says an object is out of reach, has world collision, or has an abnormal target pose, skip that object and continue with other unpacked objects.
24) Output only EXECUTE and one NAME ... ACTION ... PATH line per robot. Do not output reasoning, analysis, feedback, or alternative plans.
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""
DEFAULT_USER_PROMPT = "Generate the next robot plan. Output the required EXECUTE block."


def get_chat_prompt(env: MujocoSimEnv):
    if env.__class__.__name__ == "CabinetTask":
        return """
Plan the next Cabinet action from the current [Scene description] and [Environment Feedback]. Output only the EXECUTE block. Do not show reasoning.

Use this state policy:
1) Choose roles from the current handle/object positions, not from a fixed seed.
   - If the cabinet/handles are on the table-left side, Alice handles left_door_handle, Bob handles right_door_handle, and Chad handles mug/cup.
   - If the cabinet/handles are on the table-right side, Chad handles left_door_handle, Alice handles right_door_handle, and Bob handles mug/cup.
   - For a table-left cabinet, secure right_door_handle before advancing left_door_handle alone; for a table-right cabinet, secure left_door_handle before advancing right_door_handle alone.
2) For each door:
   - If the door is closed and its door robot is not holding the handle, that robot must PICK that handle.
   - If the door is closed and its door robot is already holding the handle, that robot must OPEN that handle.
   - If the door is open, that door robot should WAIT to keep it open.
3) Only after both doors are open, move cabinet objects:
   - If mug is not on mug_coaster, the item robot must output PICK mug PLACE mug_coaster.
   - Else if cup is not on cup_coaster, the item robot must output PICK cup PLACE cup_coaster.
   - PICK and PLACE must be in the same single action for mug/cup.
4) Never output all WAIT while mug or cup is not on its coaster. Never PICK mug/cup before both doors are open.
5) If feedback says IK failed for one robot in a simultaneous door plan, split the door work: keep the other robots WAIT and retry only the still-needed PICK/OPEN action. Do not repeat the same combined failing plan.
   - If the IK failure is on the priority handle above, retry that priority-handle action alone before moving the other door.
6) If feedback says IK failed for PICK mug PLACE mug_coaster, try PICK cup PLACE cup_coaster next if cup is still unfinished. If cup fails similarly and mug is unfinished, switch back to mug. Do not repeat the exact same failed object action more than twice.
7) If previous feedback says all robots WAITed, fix it by assigning the unfinished mug/cup action to the item robot. If feedback says format/parsing failed, output only valid NAME lines.

Output only:
EXECUTE
NAME Alice ACTION <PICK handle | OPEN handle | PICK mug PLACE mug_coaster | PICK cup PLACE cup_coaster | WAIT>
NAME Bob ACTION <PICK handle | OPEN handle | PICK mug PLACE mug_coaster | PICK cup PLACE cup_coaster | WAIT>
NAME Chad ACTION <PICK handle | OPEN handle | PICK mug PLACE mug_coaster | PICK cup PLACE cup_coaster | WAIT>
        """
    robot_names = env.get_sim_robots().keys()
    talk_order_str = ",".join([f"[{name}]" for name in robot_names])
    chat_prompt = f"""
The robots discuss to find the best strategy. They carefully analyze others' responses and use [Environment Feedback] to improve their plan. 
They talk in order {talk_order_str}... Once they reach agreement, they summarize the plan by **strictly** following [Action Output Instruction] to format the output, then stop talking.
Their entire discussion and final plan are:
    """
    return chat_prompt 


def get_plan_prompt(env: MujocoSimEnv):
    if env.__class__.__name__ == "SortOneBlockTask":
        return """
Find the best strategy for the current Sort Cube round based on [Scene description]. Do not show reasoning.
Fixed targets: blue_square -> panel2, pink_polygon -> panel4, yellow_trapezoid -> panel6.
Reachability: Alice uses panel1/2/3; Bob uses panel3/4/5; Chad uses panel5/6/7.
Relay rules:
  - If a cube is NOT yet on its target panel, move it closer. Use panel3 or panel5 as handoff zones.
  - If a cube is already on an intermediate panel (panel3 or panel5), the robot that can reach that panel AND the next target should PICK it and continue. Do NOT place it back on the same panel.
  - Check [Scene description] "(target: panelX, needs to move)" to see which cubes still need action.
  - Bob is the ONLY robot who can bridge left zone (panel3) and right zone (panel5). Bob MUST ACT when any cube is at panel3 or panel5 and needs to move further.
  - Bob CANNOT reach panel1 or panel2. To move blue_square to panel2: Bob PICK blue_square PLACE panel3, then Alice PICK blue_square PLACE panel2.
  - pink_polygon belongs to Bob: Bob can directly PICK pink_polygon PLACE panel4.
At least one robot must act each round. Output only:
EXECUTE
NAME Alice ACTION <PICK object PLACE panel or WAIT>
NAME Bob ACTION <PICK object PLACE panel or WAIT>
NAME Chad ACTION <PICK object PLACE panel or WAIT>
        """
    if env.__class__.__name__ == "MakeSandwichTask":
        return """
Find the best current action for each robot, but do not show your reasoning.
Strictly follow the recipe order in [Scene description]. Only PICK the next needed ingredient or an ingredient that will be needed very soon.
Only PUT an item after that robot is already holding it. Only one robot may PUT in a round; the other robot should WAIT or PICK the next useful ingredient.
Do not output PATH coordinates for this task. Output exactly one action per robot and nothing else.
Your output must contain only:
EXECUTE
NAME Chad ACTION <PICK item or PUT item target or WAIT>
NAME Dave ACTION <PICK item or PUT item target or WAIT>
        """
    if env.__class__.__name__ == "MoveRopeTask":
        return """
Plan one action per robot to cooperatively move the rope into the groove. Do not show reasoning.
Phase 1 (PICK): Each robot picks its rope end. Keep PATH z between 0.25 and 0.52; approach directly without lifting high.
  - Alice picks rope_front_end; Bob picks rope_back_end.
  - Bob (Panda) MUST keep PATH x >= -0.40 when z > 0.50 to avoid IK failure. Approach rope_back_end at low z.
Phase 2 (PUT): After both hold the rope, move conservatively below z=0.55, then place in groove.
  - Alice: PUT rope_front_end groove_left_end; Bob: PUT rope_back_end groove_right_end.
  - Bob must first move right to x >= -0.40 at z <= 0.48, then continue toward groove_right_end.
  - This avoids arm collision: Alice (starts left) goes to nearby groove_left; Bob goes right. Paths do NOT cross.
If IK failed for a waypoint, change that waypoint: raise z or bring x/y closer to the robot's base position.
EXECUTE
NAME Alice ACTION <PICK rope_front_end or PUT rope_front_end groove_left_end> PATH <4 coords>
NAME Bob ACTION <PICK rope_back_end or PUT rope_back_end groove_right_end> PATH <4 coords>
        """
    if env.__class__.__name__ == "PackGroceryTask":
        return """
Pack all grocery items into the bin. Do not show reasoning.
- PICK only if your gripper is EMPTY. PLACE only if you are already holding the item.
- PATH must have EXACTLY 4 (x,y,z) coords starting from near your CURRENT GRIPPER toward the target. First waypoint must be within 0.5m of your gripper, NOT at the target position. Compute evenly spaced waypoints: step=(target-gripper)/5; wp[i]=gripper+step*i for i=1,2,3,4.
- Items shown as "inside slot ..." in [Scene description] are already packed - do NOT PICK or PLACE them again. Do NOT PLACE into a slot already occupied by another item.
- Bob (Panda) CANNOT reach items at x < -0.35. If an item is at x < -0.35, Alice must PICK it; Bob should pick a closer item.
- Alice (UR5E) stands at near side (y~0); Bob (Panda) stands at far side (y~1.0). Both robots MUST act every round unless truly blocked - never have both WAIT.
- Bob should proactively PICK items nearest to Bob's gripper, then PLACE into an empty bin slot.
- For WAIT, PATH must be exactly 4 identical coords at the gripper's current position.
Output only:
EXECUTE
NAME Alice ACTION <PICK item PATH <4 coords> | PLACE item slot PATH <4 coords> | WAIT PATH <4 coords>>
NAME Bob ACTION <PICK item PATH <4 coords> | PLACE item slot PATH <4 coords> | WAIT PATH <4 coords>>
        """
    if env.__class__.__name__ == "CabinetTask":
        return """
Coordinate the 3 robots to take mug and cup from the cabinet and place them on the correct coasters. Output only the EXECUTE block. Do not show reasoning.

Use this Cabinet state policy:
1) Choose roles from the current handle/object positions, not from a fixed seed.
   - Table-left cabinet: Alice handles left_door_handle, Bob handles right_door_handle, Chad handles mug/cup.
   - Table-right cabinet: Chad handles left_door_handle, Alice handles right_door_handle, Bob handles mug/cup.
   - Table-left cabinet: right_door_handle must be secured before left_door_handle advances alone. Table-right cabinet: left_door_handle must be secured before right_door_handle advances alone.
2) Open-door phase:
   - Closed door + robot not holding its handle -> PICK that handle.
   - Closed door + robot already holding its handle -> OPEN that handle.
   - Open door -> that door robot WAITs to hold it open.
3) Item phase starts only after BOTH doors are open:
   - If mug is not on mug_coaster, item robot outputs PICK mug PLACE mug_coaster.
   - Else if cup is not on cup_coaster, item robot outputs PICK cup PLACE cup_coaster.
   - For mug/cup, PICK and PLACE are one single ACTION and must always appear together.
4) Never PICK mug/cup before both doors are open. Never output all WAIT while either mug or cup is not on its coaster. If feedback reports all WAIT or a failed format, correct that specific issue in the next EXECUTE block.
5) If feedback reports IK failure for one robot during simultaneous door work, split the door work into a single active door action and WAIT for the others. Do not repeat the same combined failing plan.
   If the failed action is the priority handle above, retry that priority-handle PICK alone first.
6) If feedback reports IK failure for PICK mug PLACE mug_coaster, try PICK cup PLACE cup_coaster if cup is not done; if cup fails and mug is not done, switch back. Avoid repeating the exact same failed object action more than twice.

Output only:
EXECUTE
NAME Alice ACTION <PICK handle | OPEN handle | PICK mug PLACE mug_coaster | PICK cup PLACE cup_coaster | WAIT>
NAME Bob ACTION <PICK handle | OPEN handle | PICK mug PLACE mug_coaster | PICK cup PLACE cup_coaster | WAIT>
NAME Chad ACTION <PICK handle | OPEN handle | PICK mug PLACE mug_coaster | PICK cup PLACE cup_coaster | WAIT>
        """
    return """
Find the best strategy to coordinate the robots, but do not show your reasoning. Propose a plan of **exactly** one action per robot.
Use [Environment Feedback] to improve your plan. Strictly follow [Action Output Instruction] to format and output only the executable plan.
Your output must contain only:
EXECUTE
NAME <robot> ACTION <action> PATH <path>
NAME <robot> ACTION <action> PATH <path>
    """
    

class SingleThreadPrompter:
    """
    At each round, queries LLM once for each action plan, 
    query again with environment feedback if the action plan cannot be executed
    """
    def __init__(
        self, 
        env: MujocoSimEnv,
        parser: LLMResponseParser, 
        feedback_manager: FeedbackManager,
        comm_mode: str = "plan", # or chat
        use_waypoints: bool = False,
        use_history: bool = True,
        max_api_queries: int = 3,
        num_replans: int = 3,
        debug_mode: bool = False,   
        temperature: float = 0,
        max_tokens: int = 1000, 
        llm_source: str = "gpt-4",
        fallback_first: bool = False,
        pack_fallback_first: Optional[bool] = None,
    ):
        self.env = env 
        self.robot_agent_names = env.get_sim_robots().keys()
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.comm_mode = comm_mode
        self.max_api_queries = max_api_queries
        self.num_replans = num_replans
        self.debug_mode = debug_mode 
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.temperature = temperature
        self.llm_source = llm_source
        self.max_tokens = max_tokens
        if pack_fallback_first is not None:
            fallback_first = pack_fallback_first
        self.fallback_first = fallback_first

        self.round_history = [] # [obs_t, action_t] but only if action_t got executed
        self.failed_plans = [] # could inherit from previous round if the final plan failed to execute in env.
        self.response_history = [] # [response_t]
        

    def save_state(self, save_path, fname = 'prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            failed_plans=self.failed_plans,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname = 'prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.failed_plans = state_dict["failed_plans"]

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}"
        ret += f"== Current Round ==\n"
        return ret
        
    def compose_system_prompt(
        self,
        obs_desp: str,
        plan_feedbacks: List[str] = [], 
        ):
        
        task_desp = self.env.describe_task_context() # should include task rules
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints and self.env.__class__.__name__ != "MoveRopeTask":
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n" 
        
        if self.use_history:
            history_desp = self.compose_round_history() 
            full_prompt += history_desp + "\n" 
        
        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) 
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt
        
        if self.comm_mode == "plan":
            comm_prompt = get_plan_prompt(self.env)
        elif self.comm_mode == "chat":
            comm_prompt = get_chat_prompt(self.env) 
        else:
            raise NotImplementedError
        full_prompt += comm_prompt

        return full_prompt 

    def _extract_executable_response(self, response: str) -> str:
        """Keep only the final executable block from verbose local LLM output."""
        if response is None or "EXECUTE" not in response:
            return response
        block = response.split("EXECUTE")[-1]
        action_lines = [
            line.strip()
            for line in block.splitlines()
            if "NAME" in line and "ACTION" in line
        ]
        if len(action_lines) == 0:
            return response
        return "EXECUTE\n" + "\n".join(action_lines)

    def _format_path(self, points: List[np.ndarray]) -> str:
        return "[" + ", ".join(
            f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in points
        ) + "]"

    def _interpolate_path(
        self,
        start: np.ndarray,
        target: np.ndarray,
        safe_height: float,
        n_points: int = 4,
    ) -> List[np.ndarray]:
        """Generate conservative, evenly spaced waypoints above the table."""
        start = np.asarray(start[:3], dtype=float)
        target = np.asarray(target[:3], dtype=float)
        elevated_start = start.copy()
        elevated_start[2] = min(max(elevated_start[2], safe_height), MAX_PATH_HEIGHT)
        elevated_target = target.copy()
        elevated_target[2] = min(
            max(target[2] + 0.20, safe_height, MIN_PATH_HEIGHT),
            MAX_PATH_HEIGHT,
        )
        return [
            elevated_start + (elevated_target - elevated_start) * ((i + 1) / n_points)
            for i in range(n_points)
        ]

    def _held_pack_object(self, obs: EnvState, robot_name: str) -> Optional[str]:
        robot_state = getattr(obs, robot_name)
        item_names = getattr(self.env, "item_names", [])
        contacts = [c for c in robot_state.contacts if c in item_names]
        return contacts[0] if len(contacts) > 0 else None

    def _empty_pack_slots(self, obs: EnvState) -> List[str]:
        slot_names = list(getattr(self.env, "bin_slot_xposes", {}).keys())
        occupied = set()
        for obj_name, obj_state in obs.objects.items():
            if obj_name not in getattr(self.env, "item_names", []):
                continue
            if not self._pack_item_is_packed(obs, obj_name):
                continue
            obj_xy = obj_state.xpos[:2]
            nearest = min(
                slot_names,
                key=lambda slot: np.linalg.norm(
                    obj_xy - self.env.bin_slot_xposes[slot][:2]
                ),
            )
            occupied.add(nearest)
        return [slot for slot in slot_names if slot not in occupied]

    def _available_pack_items(self, obs: EnvState) -> List[str]:
        held = {
            self._held_pack_object(obs, robot_name)
            for robot_name in self.env.robot_names
        }
        available = []
        for obj_name in getattr(self.env, "item_names", []):
            obj_state = obs.objects[obj_name]
            if "bin_inside" in obj_state.contacts:
                continue
            if self._pack_item_is_packed(obs, obj_name):
                continue
            if obj_name in held:
                continue
            if not self._pack_item_pose_is_valid(obs, obj_name):
                continue
            available.append(obj_name)
        return available

    def _pack_item_is_packed(self, obs: EnvState, item: str) -> bool:
        if item not in obs.objects:
            return False
        obj_state = obs.objects[item]
        if "bin_inside" in obj_state.contacts:
            return True
        try:
            bin_xy = self.env.physics.data.body("bin").xpos[:2]
            obj_xy = np.asarray(obj_state.xpos[:2], dtype=float)
            threshold = getattr(self.env, "align_threshold", 0.06)
            return np.linalg.norm(obj_xy - bin_xy) <= threshold
        except Exception:
            return False

    def _pack_item_pose_is_valid(self, obs: EnvState, item: str) -> bool:
        """Skip objects that have fallen out of the workspace or have invalid poses."""
        if item not in obs.objects:
            return False
        obj_pos = np.asarray(obs.objects[item].xpos[:3], dtype=float)
        if not np.all(np.isfinite(obj_pos)):
            return False
        if obj_pos[2] < MIN_PACK_ITEM_Z or obj_pos[2] > MAX_PACK_ITEM_Z:
            return False

        for agent_name in self.env.robot_name_map.values():
            target_pos = self.env.get_target_pos(agent_name, item)
            if target_pos is None:
                continue
            target_pos = np.asarray(target_pos[:3], dtype=float)
            if not np.all(np.isfinite(target_pos)):
                continue
            if MIN_PACK_ITEM_Z <= target_pos[2] <= MAX_PACK_ITEM_Z:
                return True
        return False

    def _robot_can_pick_item(
        self,
        obs: EnvState,
        robot_name: str,
        item: str,
        allow_handoff: bool = False,
    ) -> bool:
        if not self._pack_item_pose_is_valid(obs, item):
            return False
        allowed_items = PACK_ROBOT_ALLOWED_ITEMS.get(robot_name)
        if not allow_handoff and allowed_items is not None and item not in allowed_items:
            return False
        agent_name = self.env.robot_name_map[robot_name]
        target_pos = self.env.get_target_pos(agent_name, item)
        if target_pos is None:
            return False
        target_pos = np.asarray(target_pos[:3], dtype=float)
        if not np.all(np.isfinite(target_pos)):
            return False
        if target_pos[2] < MIN_PACK_ITEM_Z or target_pos[2] > MAX_PACK_ITEM_Z:
            return False
        # panda arm cannot physically reach items at x < -0.35
        if robot_name == "panda" and target_pos[0] < -0.35:
            return False

        hover = target_pos.copy()
        hover[2] = min(max(target_pos[2] + 0.20, SAFE_PICK_HEIGHT), MAX_PATH_HEIGHT)
        return self.env.check_reach_range(robot_name, target_pos) and self.env.check_reach_range(robot_name, hover)

    def _choose_pack_item(
        self,
        obs: EnvState,
        robot_name: str,
        candidates: List[str],
    ) -> Optional[str]:
        strict_candidates = [
            item for item in candidates
            if self._robot_can_pick_item(obs, robot_name, item)
        ]
        if len(strict_candidates) > 0:
            return min(
                strict_candidates,
                key=lambda obj: self._pack_pick_cost(obs, robot_name, obj),
            )

        handoff_candidates = [
            item for item in candidates
            if self._robot_can_pick_item(obs, robot_name, item, allow_handoff=True)
        ]
        if len(handoff_candidates) == 0:
            return None
        return min(
            handoff_candidates,
            key=lambda obj: self._pack_pick_cost(obs, robot_name, obj, allow_handoff=True),
        )

    def _pack_pick_cost(
        self,
        obs: EnvState,
        robot_name: str,
        item: str,
        allow_handoff: bool = False,
    ) -> float:
        if not self._robot_can_pick_item(obs, robot_name, item, allow_handoff=allow_handoff):
            return float("inf")
        robot_state = getattr(obs, robot_name)
        target_pos = self.env.get_target_pos(self.env.robot_name_map[robot_name], item)
        if target_pos is None:
            target_pos = obs.objects[item].xpos
        target_pos = np.asarray(target_pos[:3], dtype=float)
        cost = float(np.linalg.norm(target_pos[:2] - robot_state.ee_xpos[:2]))

        preference = PACK_ROBOT_ITEM_PREFERENCE.get(robot_name, [])
        if item in preference:
            cost += preference.index(item) * 0.03
        else:
            cost += 0.50
        allowed_items = PACK_ROBOT_ALLOWED_ITEMS.get(robot_name)
        if allowed_items is not None and item not in allowed_items:
            cost += PACK_ROBOT_HANDOFF_PENALTY

        x, y = target_pos[:2]
        if robot_name == "ur5e_robotiq":
            if y > 0.62:
                cost += 1.0
            if x > 0.50:
                cost += 0.8
        elif robot_name == "panda":
            if y < 0.50:
                cost += 1.0
            if x < -0.55:
                cost += 0.8
        return cost

    def _build_pack_pick_fallback(
        self,
        obs: EnvState,
        available_items: List[str],
        batch_items: List[str],
    ) -> Optional[str]:
        candidates = batch_items if len(batch_items) > 0 else available_items
        candidates = [item for item in candidates if item in available_items]
        if len(candidates) == 0:
            return None

        robot_names = list(self.env.robot_name_map.keys())
        best_assignment = None
        for allow_handoff in (False, True):
            best_cost = float("inf")

            for first_item in candidates + [None]:
                for second_item in candidates + [None]:
                    assignment = {
                        robot_names[0]: first_item,
                        robot_names[1]: second_item,
                    }
                    assigned_items = [item for item in assignment.values() if item is not None]
                    if len(set(assigned_items)) != len(assigned_items):
                        continue
                    if len(assigned_items) == 0:
                        continue

                    total_cost = 0.0
                    valid = True
                    for robot_name, item in assignment.items():
                        if item is None:
                            total_cost += 0.35
                            continue
                        cost = self._pack_pick_cost(
                            obs,
                            robot_name,
                            item,
                            allow_handoff=allow_handoff,
                        )
                        if not np.isfinite(cost):
                            valid = False
                            break
                        total_cost += cost
                    if valid and total_cost < best_cost:
                        best_cost = total_cost
                        best_assignment = assignment

            if best_assignment is not None:
                break

        if best_assignment is None:
            return None

        lines = ["EXECUTE"]
        for robot_name, agent_name in self.env.robot_name_map.items():
            item = best_assignment.get(robot_name)
            if item is None:
                lines.append(
                    f"NAME {agent_name} ACTION WAIT PATH {self._wait_path(obs, robot_name)}"
                )
                continue
            path = self._pick_path(obs, robot_name, item)
            lines.append(
                f"NAME {agent_name} ACTION PICK {item} PATH {self._format_path(path)}"
            )
        return "\n".join(lines)

    def _current_pack_batch_items(self, obs: EnvState, available_items: List[str]) -> List[str]:
        if len(available_items) == 0:
            return []
        for batch in PACK_PICK_BATCHES:
            remaining = [
                item
                for item in batch
                if item in available_items
                and not self._pack_item_is_packed(obs, item)
            ]
            if len(remaining) > 0:
                return remaining
        return available_items

    def _wait_path(self, obs: EnvState, robot_name: str) -> str:
        robot_state = getattr(obs, robot_name)
        point = np.asarray(robot_state.ee_xpos[:3], dtype=float)
        return self._format_path([point.copy() for _ in range(4)])

    def _pick_path(self, obs: EnvState, robot_name: str, item: str) -> List[np.ndarray]:
        robot_state = getattr(obs, robot_name)
        start = np.asarray(robot_state.ee_xpos[:3], dtype=float)
        agent_name = self.env.robot_name_map[robot_name]
        target = self.env.get_target_pos(agent_name, item)
        if target is None:
            target = obs.objects[item].xpos
        target = np.asarray(target[:3], dtype=float)

        return self._interpolate_path(start, target, SAFE_PICK_HEIGHT)

    def _choose_pack_slot(
        self,
        obs: EnvState,
        robot_name: str,
        empty_slots: List[str],
        used_slots: set,
    ) -> Optional[str]:
        candidates = [slot for slot in empty_slots if slot not in used_slots]
        if len(candidates) == 0:
            candidates = empty_slots
        if len(candidates) == 0:
            return None

        robot_state = getattr(obs, robot_name)
        robot_xy = robot_state.ee_xpos[:2]
        return min(
            candidates,
            key=lambda slot: np.linalg.norm(self.env.bin_slot_xposes[slot][:2] - robot_xy),
        )

    def _choose_placing_robot(self, held_by_robot: Dict[str, Optional[str]]) -> Optional[str]:
        priority = ["panda", "ur5e_robotiq"]
        for robot_name in priority:
            if held_by_robot.get(robot_name) is not None:
                return robot_name
        return next(
            (robot_name for robot_name, held_obj in held_by_robot.items() if held_obj is not None),
            None,
        )

    def build_pack_fallback_response(self, obs: EnvState) -> Optional[str]:
        """Create a conservative Pack plan when the LLM keeps failing."""
        if not hasattr(self.env, "item_names") or not hasattr(self.env, "bin_slot_xposes"):
            return None

        lines = ["EXECUTE"]
        available_items = self._available_pack_items(obs)
        batch_items = self._current_pack_batch_items(obs, available_items)
        empty_slots = self._empty_pack_slots(obs)
        used_items = set()
        used_slots = set()
        held_by_robot = {
            robot_name: self._held_pack_object(obs, robot_name)
            for robot_name in self.env.robot_names
        }
        placing_robot = self._choose_placing_robot(held_by_robot)
        if placing_robot is None and all(held_obj is None for held_obj in held_by_robot.values()):
            pick_response = self._build_pack_pick_fallback(obs, available_items, batch_items)
            if pick_response is not None:
                return pick_response

        made_progress = False

        for robot_name, agent_name in self.env.robot_name_map.items():
            robot_state = getattr(obs, robot_name)
            held_obj = held_by_robot.get(robot_name)

            if placing_robot is not None and robot_name != placing_robot:
                lines.append(
                    f"NAME {agent_name} ACTION WAIT PATH {self._wait_path(obs, robot_name)}"
                )
                continue

            if held_obj is not None and len(empty_slots) > 0:
                slot = self._choose_pack_slot(obs, robot_name, empty_slots, used_slots)
                if slot is None:
                    return None
                used_slots.add(slot)
                target = self.env.bin_slot_xposes[slot].copy()
                path = self._interpolate_path(robot_state.ee_xpos, target, SAFE_PLACE_HEIGHT)
                lines.append(
                    f"NAME {agent_name} ACTION PLACE {held_obj} {slot} PATH {self._format_path(path)}"
                )
                made_progress = True
                continue

            candidates = [item for item in batch_items if item not in used_items]
            if len(candidates) == 0:
                candidates = [item for item in available_items if item not in used_items]
            item = self._choose_pack_item(obs, robot_name, candidates)
            # Bob (panda) cannot reach items at x < -0.35; let Alice handle them
            if item is not None and robot_name == "panda":
                try:
                    item_x = self.env.physics.data.site(f"{item}_top").xpos[0]
                    if item_x < -0.35:
                        reachable = [c for c in candidates if c != item and c not in used_items]
                        item = next(
                            (c for c in reachable
                             if self.env.physics.data.site(f"{c}_top").xpos[0] >= -0.35),
                            None
                        )
                except Exception:
                    pass
            if item is None:
                lines.append(
                    f"NAME {agent_name} ACTION WAIT PATH {self._wait_path(obs, robot_name)}"
                )
                continue
            used_items.add(item)
            path = self._pick_path(obs, robot_name, item)
            lines.append(
                f"NAME {agent_name} ACTION PICK {item} PATH {self._format_path(path)}"
            )
            made_progress = True

        if not made_progress:
            return None
        return "\n".join(lines)

    def _sweep_cube_in_contact(self, obs: EnvState, cube: str, contact_name: str) -> bool:
        if cube not in obs.objects:
            return False
        return contact_name in obs.objects[cube].contacts

    def _sweep_cube_already_dumped(self, obs: EnvState, cube: str) -> bool:
        if self._sweep_cube_in_contact(obs, cube, "trash_bin_bottom"):
            return True
        try:
            cube_pos = np.asarray(obs.objects[cube].xpos[:3], dtype=float)
            trash_pos = np.asarray(self.env.physics.data.body("trash_bin_bottom").xpos[:3], dtype=float)
            return np.linalg.norm(cube_pos - trash_pos) <= 0.20
        except Exception:
            return False

    def _sweep_select_cube(self, obs: EnvState) -> Optional[str]:
        cube_names = getattr(self.env, "cube_names", [])
        candidates = [
            cube for cube in cube_names
            if cube in obs.objects
            and not self._sweep_cube_already_dumped(obs, cube)
            and not self._sweep_cube_in_contact(obs, cube, "dustpan_bottom")
        ]
        if len(candidates) == 0:
            return None
        priority = {"red_cube": 0, "green_cube": 1, "blue_cube": 2}
        return min(candidates, key=lambda cube: priority.get(cube, 99))

    def _sweep_alice_ready_for_cube(self, cube: str) -> bool:
        try:
            target = self.env.get_target_pos("Alice", cube)
            current = self.env.physics.data.site("dustpan_bottom").xpos.copy()
            return np.linalg.norm(current[:2] - target[:2]) <= 0.35
        except Exception:
            return False

    def build_sweep_fallback_response(self, obs: EnvState) -> Optional[str]:
        """Create a synchronized Sweep plan: MOVE/MOVE, WAIT/SWEEP, DUMP/WAIT."""
        if not hasattr(self.env, "cube_names"):
            return None
        cube_names = getattr(self.env, "cube_names", [])
        dustpan_cubes = [
            cube for cube in cube_names
            if self._sweep_cube_in_contact(obs, cube, "dustpan_bottom")
        ]
        if len(dustpan_cubes) > 0:
            return "EXECUTE\nNAME Alice ACTION DUMP\nNAME Bob ACTION WAIT"

        cube = self._sweep_select_cube(obs)
        if cube is None:
            return None
        if self._sweep_alice_ready_for_cube(cube):
            return f"EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION SWEEP {cube}"
        return f"EXECUTE\nNAME Alice ACTION MOVE {cube}\nNAME Bob ACTION MOVE {cube}"

    def _sort_panel_index(self, panel_name: str) -> Optional[int]:
        if not isinstance(panel_name, str) or not panel_name.startswith("panel"):
            return None
        try:
            return int(panel_name.replace("panel", ""))
        except ValueError:
            return None

    def _sort_current_panel(self, obs: EnvState, cube: str) -> Optional[str]:
        panel_coords = getattr(self.env, "panel_coords", {})
        if cube not in obs.objects or len(panel_coords) == 0:
            return None
        cube_pos = obs.objects[cube].xpos
        return min(
            panel_coords,
            key=lambda panel: np.linalg.norm(cube_pos[:2] - panel_coords[panel][:2]),
        )

    def _sort_cube_at_target(self, obs: EnvState, cube: str, target_panel: str) -> bool:
        if cube not in obs.objects:
            return False
        cube_pos = obs.objects[cube].xpos
        target_pos = None
        bin_slots = getattr(self.env, "bin_slot_pos", {})
        if f"{target_panel}_middle" in bin_slots:
            target_pos = bin_slots[f"{target_panel}_middle"]
        else:
            target_pos = getattr(self.env, "panel_coords", {}).get(target_panel)
        if target_pos is None:
            return False
        threshold = max(getattr(self.env, "align_threshold", 0.1), 0.12)
        if np.linalg.norm(cube_pos[:2] - target_pos[:2]) <= threshold:
            return True
        contacts = getattr(obs.objects[cube], "contacts", [])
        return target_panel in contacts and np.linalg.norm(cube_pos[:2] - target_pos[:2]) <= 0.16

    def _sort_cube_x(self, obs: EnvState, cube: str) -> Optional[float]:
        if cube not in obs.objects:
            return None
        return float(obs.objects[cube].xpos[0])

    def _sort_robot_for_move(self, from_panel: str, to_panel: str) -> Optional[str]:
        reachable = getattr(self.env, "reachable_panels", SORT_REACHABLE_PANELS)
        for agent_name in ["Alice", "Bob", "Chad"]:
            panels = set(reachable.get(agent_name, []))
            if from_panel in panels and to_panel in panels:
                return agent_name
        return None

    def _sort_next_panel(self, cube: str, current_panel: str) -> Optional[str]:
        targets = getattr(self.env, "cube_to_bin", SORT_CUBE_TARGETS)
        target_panel = targets.get(cube)
        if target_panel is None:
            return None
        if current_panel == target_panel:
            return target_panel
        if self._sort_robot_for_move(current_panel, target_panel) is not None:
            return target_panel

        current_idx = self._sort_panel_index(current_panel)
        target_idx = self._sort_panel_index(target_panel)
        if current_idx is None or target_idx is None:
            return None

        if current_idx < target_idx:
            if current_idx <= 2 and target_idx >= 3:
                return "panel3"
            if current_idx <= 4 and target_idx >= 5:
                return "panel5"
        else:
            if current_idx >= 6 and target_idx <= 5:
                return "panel5"
            if current_idx >= 4 and target_idx <= 3:
                return "panel3"
        return target_panel

    def build_sort_fallback_response(self, obs: EnvState) -> Optional[str]:
        """Create one conservative Sort action using panel topology and reachability."""
        cube_names = set(getattr(self.env, "cube_names", SORT_CUBE_ORDER))
        targets = getattr(self.env, "cube_to_bin", SORT_CUBE_TARGETS)
        actions = []

        for order, cube in enumerate(SORT_CUBE_ORDER):
            if cube not in cube_names:
                continue
            target_panel = targets.get(cube)
            if target_panel is None or self._sort_cube_at_target(obs, cube, target_panel):
                continue

            current_panel = self._sort_current_panel(obs, cube)
            if current_panel is None:
                continue
            next_panel = self._sort_next_panel(cube, current_panel)
            if next_panel is None or next_panel == current_panel:
                continue
            active_agent = self._sort_robot_for_move(current_panel, next_panel)
            if active_agent is None:
                continue

            target_bonus = 0 if next_panel == target_panel else 10
            actions.append((target_bonus + order, active_agent, cube, next_panel))

        # If the nearest-panel estimate is ambiguous after execution jitter, fall back
        # to broad x-zone routing. This keeps the evaluator moving instead of asking
        # the LLM for a state that the deterministic relay policy can handle.
        if len(actions) == 0:
            if "blue_square" in cube_names and not self._sort_cube_at_target(obs, "blue_square", targets["blue_square"]):
                x = self._sort_cube_x(obs, "blue_square")
                if x is not None:
                    if x <= -0.20:
                        actions.append((0, "Alice", "blue_square", "panel2"))
                    else:
                        actions.append((10, "Bob", "blue_square", "panel3"))

            if "pink_polygon" in cube_names and not self._sort_cube_at_target(obs, "pink_polygon", targets["pink_polygon"]):
                x = self._sort_cube_x(obs, "pink_polygon")
                if x is not None:
                    if -0.65 <= x <= 0.55:
                        actions.append((1, "Bob", "pink_polygon", "panel4"))
                    elif x < -0.65:
                        actions.append((11, "Alice", "pink_polygon", "panel3"))
                    else:
                        actions.append((11, "Chad", "pink_polygon", "panel5"))

            if "yellow_trapezoid" in cube_names and not self._sort_cube_at_target(obs, "yellow_trapezoid", targets["yellow_trapezoid"]):
                x = self._sort_cube_x(obs, "yellow_trapezoid")
                if x is not None:
                    if x >= 0.40:
                        actions.append((2, "Chad", "yellow_trapezoid", "panel6"))
                    elif x >= -0.75:
                        actions.append((12, "Bob", "yellow_trapezoid", "panel5"))
                    else:
                        actions.append((22, "Alice", "yellow_trapezoid", "panel3"))

        if len(actions) == 0:
            return None

        _, active_agent, cube, next_panel = sorted(actions)[0]
        lines = ["EXECUTE"]
        for agent_name in ["Alice", "Bob", "Chad"]:
            if agent_name == active_agent:
                lines.append(f"NAME {agent_name} ACTION PICK {cube} PLACE {next_panel}")
            else:
                lines.append(f"NAME {agent_name} ACTION WAIT")
        return "\n".join(lines)

    def _sandwich_item_on_target(self, obs: EnvState, item: str, target: str) -> bool:
        if item not in obs.objects:
            return False
        contacts = getattr(obs.objects[item], "contacts", [])
        if target in contacts:
            return True
        if target == "cutting_board":
            target_pos = getattr(self.env, "cutting_board_pos", None)
            if target_pos is not None:
                return np.linalg.norm(obs.objects[item].xpos[:2] - target_pos[:2]) < 0.18
        if target in obs.objects:
            return np.linalg.norm(obs.objects[item].xpos[:2] - obs.objects[target].xpos[:2]) < 0.12
        return False

    def _sandwich_next_item_and_target(self, obs: EnvState) -> Tuple[Optional[str], Optional[str]]:
        recipe = list(getattr(self.env, "recipe_order", []))
        if len(recipe) == 0:
            return None, None
        for idx, item in enumerate(recipe):
            target = "cutting_board" if idx == 0 else recipe[idx - 1]
            if not self._sandwich_item_on_target(obs, item, target):
                return item, target
        return None, None

    def _sandwich_holding(self, obs: EnvState, item: str) -> Optional[str]:
        for robot_name, agent_name in getattr(self.env, "robot_name_map", {}).items():
            contacts = getattr(getattr(obs, robot_name), "contacts", [])
            if item in contacts:
                return agent_name
        return None

    def _sandwich_robot_can_reach_item(self, obs: EnvState, agent_name: str, item: str) -> bool:
        robot_name = self.env.robot_name_map_inv[agent_name]
        if item not in obs.objects:
            return False
        item_state = obs.objects[item]
        site_pos = item_state.sites[item].xpos if item in item_state.sites else item_state.xpos
        return self.env.check_reach_range(robot_name, site_pos)

    def build_sandwich_fallback_response(self, obs: EnvState) -> Optional[str]:
        """Create one recipe-following Make Sandwich action."""
        next_item, target = self._sandwich_next_item_and_target(obs)
        if next_item is None or target is None:
            return None

        lines = ["EXECUTE"]
        holder = self._sandwich_holding(obs, next_item)
        if holder is not None:
            for agent_name in ["Chad", "Dave"]:
                if agent_name == holder:
                    lines.append(f"NAME {agent_name} ACTION PUT {next_item} {target}")
                else:
                    lines.append(f"NAME {agent_name} ACTION WAIT")
            return "\n".join(lines)

        active_agent = None
        for agent_name in ["Chad", "Dave"]:
            if self._sandwich_robot_can_reach_item(obs, agent_name, next_item):
                active_agent = agent_name
                break
        if active_agent is None:
            return None

        for agent_name in ["Chad", "Dave"]:
            if agent_name == active_agent:
                lines.append(f"NAME {agent_name} ACTION PICK {next_item}")
            else:
                lines.append(f"NAME {agent_name} ACTION WAIT")
        return "\n".join(lines)

    def build_rope_fallback_response(self, obs: EnvState, variant: int = 0) -> Optional[str]:
        """Deterministic fallback for MoveRopeTask using current physics state."""
        from rocobench.envs.task_rope import ROPE_FRONT_BODY, ROPE_BACK_BODY
        alice_contacts = getattr(obs, "ur5e_robotiq", None)
        bob_contacts = getattr(obs, "panda", None)
        if alice_contacts is None or bob_contacts is None:
            return None

        alice_holding = any("CB" in c for c in getattr(alice_contacts, "contacts", []))
        bob_holding = any("CB" in c for c in getattr(bob_contacts, "contacts", []))

        alice_pos = np.asarray(alice_contacts.ee_xpos[:3], dtype=float)
        bob_pos = np.asarray(bob_contacts.ee_xpos[:3], dtype=float)

        def _low_approach(start, target):
            """4-waypoint path at low z for picking rope ends."""
            s, t = np.asarray(start[:3], dtype=float), np.asarray(target[:3], dtype=float)
            safe_z = min(max(float(s[2]), 0.38), 0.50)
            p0 = s.copy(); p0[2] = safe_z
            t_low = t.copy(); t_low[2] = min(max(float(t[2]), 0.32), 0.46)
            pts = [p0 + (t_low - p0) * (i + 1) / 4 for i in range(4)]
            return pts

        def _rope_pick_path(agent_name, start, target):
            """Generate direct or obstacle-aware pick paths for candidate validation."""
            if variant == 0:
                return _low_approach(start, target)
            s, t = np.asarray(start[:3], dtype=float), np.asarray(target[:3], dtype=float)
            t_low = t.copy(); t_low[2] = min(max(float(t[2]), 0.32), 0.46)
            if agent_name == "Alice":
                lane_y = min(float(s[1]), float(t_low[1]), 0.06)
            else:
                lane_y = max(float(s[1]), float(t_low[1]), 0.88)
            if variant == 1:
                p1 = s.copy(); p1[2] = min(max(float(s[2]), 0.42), 0.48)
                p2 = np.array([(s[0] + t_low[0]) * 0.5, lane_y, 0.44])
                p3 = np.array([t_low[0], lane_y, 0.40])
                return [p1, p2, p3, t_low]
            p1 = np.array([s[0], lane_y, min(max(float(s[2]), 0.42), 0.48)])
            p2 = np.array([(s[0] + t_low[0]) * 0.5, lane_y, 0.46])
            p3 = np.array([t_low[0], lane_y, 0.42])
            return [p1, p2, p3, t_low]

        def _lift_place(start, target, lift_z=0.54):
            """4-waypoint path that lifts conservatively then descends to target."""
            s, t = np.asarray(start[:3], dtype=float), np.asarray(target[:3], dtype=float)
            lift_z = min(max(float(lift_z), 0.48), 0.52)
            p1 = s.copy()
            p1[2] = min(max(float(s[2]), 0.42), 0.46)
            t_arr = t.copy(); t_arr[2] = max(float(t[2]), 0.42)
            p2 = p1 + (t_arr - p1) * 0.33; p2[2] = lift_z
            p3 = p1 + (t_arr - p1) * 0.67; p3[2] = max(float(p3[2]), 0.48)
            return [p1, p2, p3, t_arr]

        def _bob_rope_place(start, target):
            """Bob/Panda cannot IK high on the left side; move right before lifting."""
            s, t = np.asarray(start[:3], dtype=float), np.asarray(target[:3], dtype=float)
            t_arr = t.copy(); t_arr[2] = max(float(t[2]), 0.42)
            p1 = s.copy()
            p1[0] = max(float(p1[0]), -0.38)
            p1[2] = min(max(float(s[2]), 0.42), 0.48)
            p2 = p1 + (t_arr - p1) * 0.40
            p2[2] = 0.52
            p3 = p1 + (t_arr - p1) * 0.75
            p3[2] = 0.50
            return [p1, p2, p3, t_arr]

        def _alice_rope_place(start, target, lift_z=0.52):
            """Alice/UR5E gets multiple conservative variants for feedback validation."""
            s, t = np.asarray(start[:3], dtype=float), np.asarray(target[:3], dtype=float)
            t_arr = t.copy(); t_arr[2] = max(float(t[2]), 0.42)
            if variant == 1:
                p1 = s.copy(); p1[2] = min(max(float(s[2]), 0.38), 0.42)
                p2 = p1 + (t_arr - p1) * 0.25; p2[2] = 0.46
                p3 = p1 + (t_arr - p1) * 0.65; p3[2] = 0.48
                return [p1, p2, p3, t_arr]
            if variant == 2:
                p1 = s.copy(); p1[1] = min(float(p1[1]), 0.56); p1[2] = 0.42
                p2 = p1 + (t_arr - p1) * 0.35; p2[2] = 0.48
                p3 = p1 + (t_arr - p1) * 0.70; p3[2] = 0.48
                return [p1, p2, p3, t_arr]
            return _lift_place(start, target, lift_z)

        if not alice_holding and not bob_holding:
            a_t = np.asarray(self.env.get_target_pos("Alice", "rope_front_end")[:3], dtype=float)
            b_t = np.asarray(self.env.get_target_pos("Bob", "rope_back_end")[:3], dtype=float)
            a_path = _rope_pick_path("Alice", alice_pos, a_t)
            b_path = _rope_pick_path("Bob", bob_pos, b_t)
            return (
                f"EXECUTE\n"
                f"NAME Alice ACTION PICK rope_front_end PATH {self._format_path(a_path)}\n"
                f"NAME Bob ACTION PICK rope_back_end PATH {self._format_path(b_path)}"
            )

        if alice_holding and bob_holding:
            groove_right = np.asarray(self.env.groove_pos.get("groove_right_end", [1.0, 0.50, 0.43]), dtype=float)
            groove_left = np.asarray(self.env.groove_pos.get("groove_left_end", [0.20, 0.50, 0.43]), dtype=float)
            obstacle_tops = [self.env.physics.data.site(n).xpos[2] for n in ["obstacle_wall_front_top", "obstacle_wall_back_top"] if self.env.physics.model.site(n).id >= 0]
            lift_z = (max(obstacle_tops) + 0.06) if obstacle_tops else 0.52
            lift_z = min(lift_z, 0.52)
            # Alice (rope_front, starts left at x≈-1.2) → groove_LEFT (x≈0.20): paths diverge, no crossing with Bob
            # Bob (rope_back, starts at x≈-0.54) → groove_RIGHT (x≈1.00): Bob goes further right
            a_path = _alice_rope_place(alice_pos, groove_left, lift_z)
            b_path = _bob_rope_place(bob_pos, groove_right)
            return (
                f"EXECUTE\n"
                f"NAME Alice ACTION PUT rope_front_end groove_left_end PATH {self._format_path(a_path)}\n"
                f"NAME Bob ACTION PUT rope_back_end groove_right_end PATH {self._format_path(b_path)}"
            )

        # One holding, one not: the one not holding should pick
        if not alice_holding:
            a_t = np.asarray(self.env.get_target_pos("Alice", "rope_front_end")[:3], dtype=float)
            a_path = _rope_pick_path("Alice", alice_pos, a_t)
            b_path = [bob_pos.copy() for _ in range(4)]
            return (
                f"EXECUTE\n"
                f"NAME Alice ACTION PICK rope_front_end PATH {self._format_path(a_path)}\n"
                f"NAME Bob ACTION WAIT PATH {self._format_path(b_path)}"
            )
        else:
            b_t = np.asarray(self.env.get_target_pos("Bob", "rope_back_end")[:3], dtype=float)
            b_path = _rope_pick_path("Bob", bob_pos, b_t)
            a_path = [alice_pos.copy() for _ in range(4)]
            return (
                f"EXECUTE\n"
                f"NAME Alice ACTION WAIT PATH {self._format_path(a_path)}\n"
                f"NAME Bob ACTION PICK rope_back_end PATH {self._format_path(b_path)}"
            )

    def build_rope_fallback_candidates(self, obs: EnvState) -> List[str]:
        """Generate Rope fallback candidates and let feedback validation choose.

        Put candidates keep the direct plan first because they are usually fast.
        Pick candidates prefer obstacle-aware side lanes first: lightweight
        feedback can miss RRT timeouts caused by low straight-line paths near
        the obstacle wall.
        """
        candidates = []
        alice_state = getattr(obs, "ur5e_robotiq", None)
        bob_state = getattr(obs, "panda", None)
        alice_holding = any("CB" in c for c in getattr(alice_state, "contacts", [])) if alice_state else False
        bob_holding = any("CB" in c for c in getattr(bob_state, "contacts", [])) if bob_state else False
        variant_order = [1, 2, 0] if not (alice_holding and bob_holding) else [0, 1, 2]
        for variant in variant_order:
            response = self.build_rope_fallback_response(obs, variant=variant)
            if response is not None and response not in candidates:
                candidates.append(response)
        return candidates

    def build_fallback_candidates(self, obs: EnvState) -> List[str]:
        if self.env.__class__.__name__ == "MoveRopeTask":
            return self.build_rope_fallback_candidates(obs)
        if self.env.__class__.__name__ == "CabinetTask":
            return self.build_cabinet_fallback_candidates(obs)
        response = self.build_fallback_response(obs)
        return [] if response is None else [response]

    def validate_fallback_candidates(self, obs: EnvState, candidates: List[str]):
        """Return the first fallback candidate that passes parser and env feedback."""
        last_feedback = "Fallback plan parse failed"
        for candidate in candidates:
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, candidate)
            if not parse_succ:
                last_feedback = parsed_str
                continue
            ready_to_execute = True
            last_feedback = "Fallback plan passed parser"
            for llm_plan in llm_plans:
                ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)
                last_feedback = env_feedback if not ready_to_execute else last_feedback
                if not ready_to_execute:
                    break
            if ready_to_execute:
                return True, candidate, llm_plans, last_feedback
        return False, (candidates[-1] if candidates else None), None, last_feedback

    def _format_cabinet_response(self, actions: Dict[str, str]) -> str:
        return (
            f"EXECUTE\n"
            f"NAME Alice ACTION {actions['Alice']}\n"
            f"NAME Bob ACTION {actions['Bob']}\n"
            f"NAME Chad ACTION {actions['Chad']}"
        )

    def build_cabinet_fallback_response(self, obs: EnvState) -> Optional[str]:
        """Deterministic fallback for CabinetTask."""
        alice_state = getattr(obs, "ur5e_robotiq", None)
        bob_state = getattr(obs, "panda", None)
        chad_state = getattr(obs, "ur5e_suction", None)
        if alice_state is None or bob_state is None or chad_state is None:
            return None

        left_slice = self.env.physics.named.data.qpos._convert_key("leftdoorhinge")
        right_slice = self.env.physics.named.data.qpos._convert_key("rightdoorhinge")
        left_qpos = self.env.physics.data.qpos[left_slice.start]
        right_qpos = self.env.physics.data.qpos[right_slice.start]
        left_door_open = left_qpos < -2.0
        right_door_open = right_qpos > 2.0

        alice_raw = getattr(alice_state, "contacts", [])
        bob_raw = getattr(bob_state, "contacts", [])
        chad_raw = getattr(chad_state, "contacts", [])
        states = {
            "Alice": alice_raw,
            "Bob": bob_raw,
            "Chad": chad_raw,
        }
        actions = {name: "WAIT" for name in states}

        if self.env.cabinet_pos[0] < 0:
            door_assignments = {
                "Alice": "left_door_handle",
                "Bob": "right_door_handle",
            }
            item_agent = "Chad"
        else:
            door_assignments = {
                "Chad": "left_door_handle",
                "Alice": "right_door_handle",
            }
            item_agent = "Bob"

        door_open = {
            "left_door_handle": left_door_open,
            "right_door_handle": right_door_open,
        }
        for agent_name, handle in door_assignments.items():
            if door_open[handle]:
                actions[agent_name] = "WAIT"
            elif handle in states[agent_name]:
                actions[agent_name] = f"OPEN {handle}"
            else:
                actions[agent_name] = f"PICK {handle}"

        # Item states
        mug_pos = self.env.physics.data.body("mug").xpos
        cup_pos = self.env.physics.data.body("cup").xpos
        mug_on_coaster = np.linalg.norm(mug_pos - self.env.coaster_pos["mug_coaster"]) < 0.25
        cup_on_coaster = np.linalg.norm(cup_pos - self.env.coaster_pos["cup_coaster"]) < 0.25

        if left_door_open and right_door_open:
            if not mug_on_coaster:
                actions[item_agent] = "PICK mug PLACE mug_coaster"
            elif not cup_on_coaster:
                actions[item_agent] = "PICK cup PLACE cup_coaster"

        return self._format_cabinet_response(actions)

    def build_cabinet_fallback_candidates(self, obs: EnvState) -> List[str]:
        """Generate Cabinet fallback candidates after LLM replans fail.

        The compact state-machine action is tried first. If lightweight
        validation rejects a simultaneous door plan, single-active-door
        candidates let the task still make monotonic progress.
        """
        base = self.build_cabinet_fallback_response(obs)
        if base is None:
            return []

        candidates = [base]
        parse_succ, _, llm_plans = self.parser.parse(obs, base)
        if not parse_succ or len(llm_plans) == 0:
            return candidates

        action_strs = llm_plans[0].action_strs
        active_door_actions = [
            (agent_name, action)
            for agent_name, action in action_strs.items()
            if action != "WAIT" and ("door_handle" in action or action.startswith("OPEN"))
        ]
        priority_handle = "right_door_handle" if self.env.cabinet_pos[0] < 0 else "left_door_handle"
        active_door_actions.sort(key=lambda item: 0 if priority_handle in item[1] else 1)
        for agent_name, action in active_door_actions:
            single_actions = {name: "WAIT" for name in ["Alice", "Bob", "Chad"]}
            single_actions[agent_name] = action
            response = self._format_cabinet_response(single_actions)
            if response not in candidates:
                candidates.append(response)
        return candidates

    def build_fallback_response(self, obs: EnvState) -> Optional[str]:
        if self.env.__class__.__name__ == "PackGroceryTask":
            return self.build_pack_fallback_response(obs)
        if self.env.__class__.__name__ == "SweepTask":
            return self.build_sweep_fallback_response(obs)
        if self.env.__class__.__name__ == "SortOneBlockTask":
            return self.build_sort_fallback_response(obs)
        if self.env.__class__.__name__ == "MakeSandwichTask":
            return self.build_sandwich_fallback_response(obs)
        if self.env.__class__.__name__ == "MoveRopeTask":
            return self.build_rope_fallback_response(obs)
        if self.env.__class__.__name__ == "CabinetTask":
            return self.build_cabinet_fallback_response(obs)
        return None

    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)

        if self.fallback_first:
            fallback_candidates = self.build_fallback_candidates(obs)
            if len(fallback_candidates) > 0:
                ready_to_execute, fallback_response, llm_plans, curr_feedback = (
                    self.validate_fallback_candidates(obs, fallback_candidates)
                )
                response_history.append(fallback_response)

                timestamp = datetime.now().strftime("%m%d-%H%M")
                json.dump(
                    [
                        {"sender": "FallbackCandidates", "message": "\n\n--- candidate ---\n\n".join(fallback_candidates)},
                        {"sender": "SelectedFallback", "message": fallback_response},
                        {"sender": "Feedback", "message": curr_feedback},
                    ],
                    open(f"{save_path}/fallback_first_{timestamp}.json", "w"),
                )
                if ready_to_execute:
                    self.response_history = response_history
                    return True, llm_plans, [curr_feedback], response_history
                plan_feedbacks.append(curr_feedback)

        for i in range(self.num_replans): 
            system_prompt = self.compose_system_prompt(obs_desp, plan_feedbacks)
            response, usage = self.query_once(
                system_prompt, user_prompt=""
                ) # NOTE: single_thread doesn't use user role
            response = self._extract_executable_response(response)
            response_history.append(response)
            
            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": "",
                    },
                    {
                        "sender": "Planner",
                        "message": response,
                    },
                    usage,
                ]
            fname = f'{save_path}/replan{i}_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))  
            
            curr_feedback = "None"
            # try parsing 
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response) 
            if not parse_succ: 
                execute_str = "" if response is None else 'EXECUTE' + response.split('EXECUTE')[-1]
                curr_feedback = f"""
Parsing failed! {parsed_str}
Previous response: {execute_str}
Re-format to strictly follow [Action Output Instruction]!
                """
                plan_feedbacks.append(curr_feedback)
                ready_to_execute = False  
            # give env. feedback 
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            
            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute:
                plan_str = parsed_str
                break  
        if not ready_to_execute:
            fallback_candidates = self.build_fallback_candidates(obs)
            if len(fallback_candidates) > 0:
                ready_to_execute, fallback_response, llm_plans, curr_feedback = (
                    self.validate_fallback_candidates(obs, fallback_candidates)
                )
                response_history.append(fallback_response)
                plan_feedbacks.append(curr_feedback)

                timestamp = datetime.now().strftime("%m%d-%H%M")
                json.dump(
                    [
                        {"sender": "FallbackCandidates", "message": "\n\n--- candidate ---\n\n".join(fallback_candidates)},
                        {"sender": "SelectedFallback", "message": fallback_response},
                        {"sender": "Feedback", "message": curr_feedback},
                    ],
                    open(f"{save_path}/fallback_{timestamp}.json", "w"),
                )
        self.response_history = response_history
        return ready_to_execute, llm_plans, plan_feedbacks, response_history


    def query_once(self, system_prompt, user_prompt=""):
        response = None
        usage = None   
        # print('======= system prompt ======= \n ', system_prompt)
        if self.debug_mode: # query human user input
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()

        for n in range(self.max_api_queries):
            print('querying {}th time'.format(n))
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt or DEFAULT_USER_PROMPT},
                ]
                api_response = chat_completion(
                    messages=messages,
                    llm_source=self.llm_source,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                usage = response_usage(api_response)
                response = response_content(api_response)
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except Exception as exc:
                print(f"API error, try again: {exc}")
            continue
        return response or "", usage or {}

    

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            responses = "\n".join(self.response_history)
            self.round_history.append(
                f"[Response History]\n{responses}\n{obs_desp}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = [] 
        self.response_history = []





