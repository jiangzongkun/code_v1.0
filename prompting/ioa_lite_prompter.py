import re
import math
import textwrap
import uuid
from typing import Any, Dict, List, Optional

from rocobench.envs import EnvState

from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .plan_prompter import SingleThreadPrompter


TASK_PROFILES: Dict[str, Dict[str, str]] = {
    "SortOneBlockTask": {
        "name": "sort",
        "stage_order": "choose misplaced cube -> transfer if needed -> place target panel",
        "sync_policy": "asynchronous only for different cubes and non-conflicting panels",
        "history_policy": "retain last 3 compact episode summaries",
        "active_team_rule": "select robots that can reach the cube, target, or transfer panel",
        "completion_criteria": "the selected cube reaches its valid target panel",
        "repair_priority": "invalid target panel, unreachable cube/panel, duplicate cube manipulation",
        "task_rules": (
            "Do not confuse robot names with cube ownership. Alice's target cube is blue_square -> panel2; "
            "Bob's target cube is pink_polygon -> panel4; Chad's target cube is yellow_trapezoid -> panel6. "
            "Legal PLACE targets are: blue_square only panel2, panel3, or panel5; "
            "pink_polygon only panel4, panel3, or panel5; yellow_trapezoid only panel6, panel3, or panel5. "
            "panel3 and panel5 are transfer handoff panels only. Never place a cube on another cube's final panel."
        ),
    },
    "CabinetTask": {
        "name": "cabinet",
        "stage_order": "open door -> hold/clear access -> extract object -> place object",
        "sync_policy": "synchronous when door state gates object extraction",
        "history_policy": "reset raw turn trace per stage; retain task table",
        "active_team_rule": "select door holders and retriever according to stage",
        "completion_criteria": "door dependency is satisfied and target object progresses",
        "repair_priority": "premature extraction, door reachability, all-wait action",
    },
    "MoveRopeTask": {
        "name": "rope",
        "stage_order": "assign endpoints -> lift -> cross obstacle -> place endpoints",
        "sync_policy": "always synchronous",
        "history_policy": "reset raw trace per phase; preserve endpoint ownership invariant",
        "active_team_rule": "both endpoint owners must act together unless explicitly waiting",
        "completion_criteria": "rope endpoints move through the current phase without endpoint swap",
        "repair_priority": "endpoint swap, bad path height, collision, IK failure",
    },
    "SweepTask": {
        "name": "sweep",
        "stage_order": "position support -> sweep target cube -> dump/reposition",
        "sync_policy": "synchronous for alignment and sweep, sequential for dump",
        "history_policy": "retain last 3 compact episode summaries",
        "active_team_rule": "select sweeper and dustpan/bucket holder for the same cube",
        "completion_criteria": "target cube is swept into support and dump progresses when ready",
        "repair_priority": "sweeper before support readiness, invalid MOVE target",
    },
    "MakeSandwichTask": {
        "name": "sandwich",
        "stage_order": "identify next ingredient -> pick if needed -> put on correct base",
        "sync_policy": "sequential; only one PUT at a time",
        "history_policy": "retain last 3 compact recipe/stack summaries",
        "active_team_rule": "select robot that can reach the next recipe ingredient and stack",
        "completion_criteria": "next recipe ingredient is placed on the required base",
        "repair_priority": "wrong recipe order, wrong base object, picking stacked objects",
    },
    "PackGroceryTask": {
        "name": "pack",
        "stage_order": "choose object -> choose empty bin slot -> place -> update occupancy",
        "sync_policy": "mostly sequential near bin, async only for distant safe picks",
        "history_policy": "retain compact occupancy/holding state; drop raw dialogue",
        "active_team_rule": "select robot with safe path to object and empty bin slot",
        "completion_criteria": "selected object is packed into an empty safe slot",
        "repair_priority": "near-bin collision, occupied slot, path failure, invalid pick/place",
    },
}

PRESET_LEVELS = {
    "schema": 0,
    "task_table": 1,
    "feedback": 2,
    "fsm": 3,
    "dynamic": 4,
    "full": 4,
}


class IoaLitePrompter(SingleThreadPrompter):
    """
    IoA-inspired local coordinator that preserves the SingleThreadPrompter
    execution loop while adding compact communication-state prompt sections.
    """

    def __init__(
        self,
        env,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager,
        ioa_preset: str = "full",
        **kwargs,
    ):
        super().__init__(env=env, parser=parser, feedback_manager=feedback_manager, **kwargs)
        # IoA-Lite uses compact task/state memory instead of raw LLM response history.
        self.use_history = False
        if ioa_preset not in PRESET_LEVELS:
            raise ValueError(f"Unknown ioa_preset={ioa_preset}. Valid presets: {sorted(PRESET_LEVELS)}")
        self.ioa_preset = ioa_preset
        self.comm_id = uuid.uuid4().hex[:10]
        self.turn_id = 0
        self.task_table: Dict[str, Dict[str, Any]] = {}
        self.await_status: Dict[str, Any] = {
            "awaiting_robots": [],
            "responded_robots": [],
            "resume_condition": "validator_result_available",
        }
        self.dynamic_plan = "No dynamic plan has been formed yet."
        self.robot_registry: Dict[str, Dict[str, Any]] = {}
        self.conversation_context: List[str] = []
        self.typed_feedback: Dict[str, str] = {
            "last_failure_type": "NONE",
            "failed_robot": "",
            "failed_action": "",
            "cause": "",
            "repair_instruction": "",
        }
        self.last_plan_feedbacks: List[str] = []
        self.current_obs: Optional[EnvState] = None

    def _level(self) -> int:
        return PRESET_LEVELS[self.ioa_preset]

    def _enabled(self, section: str) -> bool:
        requirements = {
            "task_table": 1,
            "feedback": 2,
            "fsm": 3,
            "dynamic": 4,
        }
        return self._level() >= requirements[section]

    def _profile(self) -> Dict[str, str]:
        return TASK_PROFILES.get(
            self.env.__class__.__name__,
            {
                "name": self.env.__class__.__name__,
                "stage_order": "observe -> choose feasible subgoal -> execute one action per robot",
                "sync_policy": "choose synchronous, asynchronous, or sequential based on collision risk",
                "history_policy": "retain last 3 compact summaries",
                "active_team_rule": "select robots with valid actions and set others to WAIT",
                "completion_criteria": "the selected subgoal makes task progress",
                "repair_priority": "parser, task rule, reachability, IK, collision, path failure",
                "task_rules": "Follow the task prompt and validator feedback exactly.",
            },
        )

    def compose_system_prompt(self, obs_desp: str, plan_feedbacks: List[str] = []):
        base_prompt = super().compose_system_prompt(obs_desp, plan_feedbacks)
        sections = [
            "\n[IoA-Lite Local Communication-State Protocol]\n"
            "Use the following state as coordination guidance only. The final executable output must still follow "
            "the RoCoBench action format exactly.\n",
            self._compose_comm_header(),
            self._compose_observation_context(obs_desp),
            self._compose_robot_registry(),
            self._compose_conversation_context(),
            self._compose_group_state(),
        ]
        if self._enabled("task_table"):
            sections.append(self._compose_task_table())
            sections.append(self._compose_await_status())
        if self._enabled("dynamic"):
            sections.append(self._compose_dynamic_plan())
        if self._enabled("feedback"):
            sections.append(self._compose_validator_feedback())
        if self._enabled("fsm"):
            sections.append(self._compose_local_turn_trace())
        sections.append(self._compose_completion_check())
        sections.append(self._compose_output_contract())
        return base_prompt + "\n" + "\n".join(sections)

    def prompt_one_round(self, obs: EnvState, save_path: str = ""):
        self.current_obs = obs
        self.turn_id += 1
        ready_to_execute, llm_plans, plan_feedbacks, response_history = super().prompt_one_round(
            obs, save_path=save_path
        )
        self.last_plan_feedbacks = plan_feedbacks
        self._update_typed_feedback(plan_feedbacks)
        self._update_await_status_from_plans(llm_plans if ready_to_execute else [])
        return ready_to_execute, llm_plans, plan_feedbacks, response_history

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        super().post_execute_update(obs_desp, execute_success, parsed_plan)
        if not execute_success:
            self.typed_feedback = {
                "last_failure_type": "EXECUTION_FAIL",
                "failed_robot": "",
                "failed_action": self._shorten(parsed_plan, 160),
                "cause": "RRT or simulator execution failed and the environment was rewound.",
                "repair_instruction": "Choose a safer action, add safer waypoints if required, or assign collision-prone robots to WAIT.",
            }
        self._append_task_row(parsed_plan, execute_success)
        self._append_conversation_summary(parsed_plan, execute_success)
        if self._enabled("dynamic"):
            self._update_dynamic_plan(parsed_plan, execute_success)

    def post_episode_update(self):
        super().post_episode_update()
        self.comm_id = uuid.uuid4().hex[:10]
        self.turn_id = 0
        self.task_table = {}
        self.await_status = {
            "awaiting_robots": [],
            "responded_robots": [],
            "resume_condition": "validator_result_available",
        }
        self.dynamic_plan = "No dynamic plan has been formed yet."
        self.conversation_context = []
        self.typed_feedback = {
            "last_failure_type": "NONE",
            "failed_robot": "",
            "failed_action": "",
            "cause": "",
            "repair_instruction": "",
        }
        self.last_plan_feedbacks = []
        self.current_obs = None

    def _compose_comm_header(self) -> str:
        robots = list(self.env.robot_name_map.values())
        active = self.await_status.get("awaiting_robots") or robots
        waiting = [name for name in robots if name not in active]
        profile = self._profile()
        return textwrap.dedent(
            f"""
            [COMM_HEADER]
            comm_id: {self.comm_id}
            turn_id: {self.turn_id}
            max_turns: {self.num_replans}
            current_turn: {self.turn_id}
            current_state: DISCUSSION
            message_type: discussion
            sender: Coordinator
            group_id: {profile['name']}
            next_speaker: Coordinator
            active_robots: {', '.join(active) if active else 'None'}
            waiting_robots: {', '.join(waiting) if waiting else 'None'}
            """
        ).strip()

    def _compose_observation_context(self, obs_desp: str) -> str:
        per_robot = []
        if self.current_obs is not None:
            for agent_name in self.env.robot_name_map.values():
                try:
                    prompt = self.env.get_agent_prompt(self.current_obs, agent_name)
                except TypeError:
                    try:
                        prompt = self.env.get_agent_prompt(agent_name)
                    except Exception:
                        prompt = "See global scene summary."
                except Exception:
                    prompt = "See global scene summary."
                per_robot.append(f"{agent_name}: {self._shorten(prompt, 700)}")
        return textwrap.dedent(
            f"""
            [OBSERVATION_CONTEXT]
            global_scene_summary:
            {self._indent(self._shorten(obs_desp, 1200))}
            per_robot_observation:
            {self._indent(chr(10).join(per_robot) if per_robot else 'Unavailable.')}
            observation_source: env.describe_obs plus per-robot env.get_agent_prompt when available
            """
        ).strip()

    def _compose_robot_registry(self) -> str:
        self.robot_registry = self._build_robot_registry()
        rows = []
        for agent_name, entry in self.robot_registry.items():
            rows.append(
                f"{agent_name} | robot={entry['robot_name']} | role={entry['role']} | "
                f"reachable_objects={entry['reachable_objects']} | "
                f"reachable_regions={entry['reachable_regions']} | "
                f"holding={entry['holding']} | visible_objects={entry['visible_objects']} | "
                f"preferred_role={entry['preferred_role']} | "
                f"can_collaborate_with={entry['can_collaborate_with']} | "
                f"action_space={entry['action_space']} | constraints={entry['constraints']}"
            )
        return "[ROBOT_REGISTRY]\n" + "\n".join(rows)

    def _build_robot_registry(self) -> Dict[str, Dict[str, str]]:
        action_prompt = self._shorten(self.env.get_action_prompt(), 420)
        registry = {}
        for robot_name, agent_name in self.env.robot_name_map.items():
            agent_prompt = self._agent_prompt_for(agent_name)
            visible_objects = self._visible_objects_for(agent_prompt)
            reachable_regions = self._reachable_regions_for(agent_name, robot_name, agent_prompt)
            reachable_objects = self._reachable_objects_for(agent_name, reachable_regions, visible_objects)
            collaborators = [name for name in self.env.robot_name_map.values() if name != agent_name]
            preferred_role = self._preferred_role_for(agent_name, robot_name)
            registry[agent_name] = {
                "robot_name": robot_name,
                "role": preferred_role,
                "reachable_objects": self._join_values(reachable_objects),
                "reachable_regions": self._join_values(reachable_regions),
                "action_space": action_prompt,
                "holding": self._join_values(self._holding_for(robot_name)),
                "visible_objects": self._join_values(visible_objects),
                "constraints": "Follow task prompt, reachability, current holding state, and collision feedback.",
                "preferred_role": preferred_role,
                "can_collaborate_with": self._join_values(collaborators),
            }
        return registry

    def _agent_prompt_for(self, agent_name: str) -> str:
        if self.current_obs is None:
            return ""
        try:
            return self.env.get_agent_prompt(self.current_obs, agent_name)
        except TypeError:
            try:
                return self.env.get_agent_prompt(agent_name)
            except Exception:
                return ""
        except Exception:
            return ""

    def _holding_for(self, robot_name: str) -> List[str]:
        if self.current_obs is None:
            return []
        robot_state = getattr(self.current_obs, robot_name, None)
        contacts = getattr(robot_state, "contacts", None)
        if not contacts:
            return []
        return sorted([str(contact) for contact in contacts])

    def _visible_objects_for(self, agent_prompt: str) -> List[str]:
        if self.current_obs is None:
            return []
        object_names = sorted(self.current_obs.objects.keys())
        if not agent_prompt:
            return object_names
        visible = []
        for object_name in object_names:
            if re.search(rf"\b{re.escape(object_name)}\b", agent_prompt):
                visible.append(object_name)
        return visible or object_names

    def _reachable_regions_for(self, agent_name: str, robot_name: str, agent_prompt: str) -> List[str]:
        if hasattr(self.env, "reachable_panels"):
            panels = getattr(self.env, "reachable_panels", {}).get(agent_name, [])
            if panels:
                return list(panels)

        match = re.search(r"can only reach ([^\.\n]+)", agent_prompt)
        if match:
            return [item.strip() for item in match.group(1).split(",") if item.strip()]

        task_name = self._profile()["name"]
        if task_name == "sandwich":
            if agent_name == "Chad" or robot_name == "ur5e_suction":
                return ["right_table_side", "cutting_board"]
            return ["left_table_side", "cutting_board"]
        if task_name == "pack":
            slots = sorted(getattr(self.env, "bin_slot_xposes", {}).keys())
            return slots or ["bin", "object_table"]
        if task_name == "sweep":
            return ["dustpan_broom_workspace", "trash_bin"]
        if task_name == "rope":
            return ["rope_endpoints", "obstacle_clearance_path"]
        return ["See per-robot observation."]

    def _reachable_objects_for(
        self,
        agent_name: str,
        reachable_regions: List[str],
        visible_objects: List[str],
    ) -> List[str]:
        if self.current_obs is None:
            return []

        if hasattr(self.env, "reachable_panels"):
            if hasattr(self.env, "get_cube_panel"):
                reachable_by_panel = []
                region_set = set(reachable_regions)
                for object_name in self.current_obs.objects.keys():
                    try:
                        current_panel = self.env.get_cube_panel(self.current_obs, object_name)
                    except Exception:
                        current_panel = ""
                    if current_panel in region_set:
                        reachable_by_panel.append(object_name)
                if reachable_by_panel:
                    return sorted(reachable_by_panel)

            reachable = []
            region_set = set(reachable_regions)
            for object_name, object_state in self.current_obs.objects.items():
                contacts = set(getattr(object_state, "contacts", set()))
                if contacts.intersection(region_set):
                    reachable.append(object_name)
            if reachable:
                return sorted(reachable)

        profile_name = self._profile()["name"]
        if profile_name == "cabinet":
            object_names = set(self.current_obs.objects.keys())
            return sorted([item for item in reachable_regions if item in object_names])
        if profile_name == "pack" and hasattr(self.env, "item_names"):
            return [name for name in getattr(self.env, "item_names", []) if name in visible_objects]
        if profile_name == "sandwich" and hasattr(self.env, "food_items"):
            return [name for name in getattr(self.env, "food_items", []) if name in visible_objects]
        return visible_objects

    def _preferred_role_for(self, agent_name: str, robot_name: str) -> str:
        profile_name = self._profile()["name"]
        if profile_name == "sweep":
            return "dustpan_holder" if agent_name == "Alice" or robot_name == "ur5e_robotiq" else "sweeper"
        if profile_name == "rope":
            return "rope_endpoint_owner"
        if profile_name == "cabinet":
            return "door_or_object_actor_by_stage"
        if profile_name == "sort":
            return "cube_sorter_or_transfer_helper"
        if profile_name == "sandwich":
            return "ingredient_picker_or_stack_helper"
        if profile_name == "pack":
            return "grocery_picker_or_bin_placer"
        return "stage_dependent_actor"

    def _compose_conversation_context(self) -> str:
        recent = "\n".join(self.conversation_context[-3:]) if self.conversation_context else "No compact history yet."
        return textwrap.dedent(
            f"""
            [CONVERSATION_CONTEXT]
            goal: solve the current {self._profile()['name']} task under RoCoBench parser and simulator constraints
            recent_turn_summaries:
            {self._indent(recent)}
            completed_task_summaries: see TASK_TABLE rows with status=completed
            latest_validator_feedback: {self.typed_feedback.get('last_failure_type', 'NONE')}
            history_policy: {self._profile()['history_policy']}
            """
        ).strip()

    def _compose_group_state(self) -> str:
        profile = self._profile()
        return textwrap.dedent(
            f"""
            [GROUP_STATE]
            task: {profile['name']}
            stage: {profile['stage_order']}
            sync_policy: {profile['sync_policy']}
            available_team: {', '.join(self.env.robot_name_map.values())}
            active_team: {', '.join(self.await_status.get('awaiting_robots') or self.env.robot_name_map.values())}
            waiting_team: assign WAIT to robots not needed for the current feasible subgoal
            selection_basis: reachability | action_space | current_holding | collision_risk | stage_role
            task_specific_rules: {profile.get('task_rules', 'Follow task prompt and validator feedback exactly.')}
            current_legal_progress_actions:
            {self._indent(self._compose_current_action_hints(profile['name']))}
            shared_goal: maximize task progress without parser, task, IK, path, or collision failure
            current_subgoal: choose the next feasible action set for this round
            blocked_reason: {self.typed_feedback.get('cause') or 'None'}
            """
        ).strip()

    def _compose_task_table(self) -> str:
        if not self.task_table:
            return "[TASK_TABLE]\nid | task_abstract | task_desc | assignee | status | depends_on | trigger | completion_criteria | last_result\n(empty)"
        rows = ["id | task_abstract | task_desc | assignee | status | depends_on | trigger | completion_criteria | last_result"]
        for task_id, row in self.task_table.items():
            rows.append(
                " | ".join(
                    [
                        task_id,
                        row.get("task_abstract", ""),
                        row.get("task_desc", ""),
                        row.get("assignee", ""),
                        row.get("status", ""),
                        row.get("depends_on", ""),
                        row.get("trigger", ""),
                        row.get("completion_criteria", ""),
                        row.get("last_result", ""),
                    ]
                )
            )
        return "[TASK_TABLE]\n" + "\n".join(rows)

    def _compose_await_status(self) -> str:
        return textwrap.dedent(
            f"""
            [AWAIT_STATUS]
            awaiting_robots: {', '.join(self.await_status.get('awaiting_robots', [])) or 'None'}
            responded_robots: {', '.join(self.await_status.get('responded_robots', [])) or 'None'}
            resume_condition: {self.await_status.get('resume_condition', 'validator_result_available')}
            """
        ).strip()

    def _compose_dynamic_plan(self) -> str:
        profile = self._profile()
        return textwrap.dedent(
            f"""
            [DYNAMIC_PLAN]
            enabled: true
            global_plan: {self.dynamic_plan}
            next_expected_effect: {profile['completion_criteria']}
            objects_done: infer from completed TASK_TABLE rows and scene state
            objects_remaining: infer from scene state and task profile
            updated_plan: update only when dependency, collision, or stage strategy changes
            """
        ).strip()

    def _compose_validator_feedback(self) -> str:
        return textwrap.dedent(
            f"""
            [VALIDATOR_FEEDBACK]
            last_failure_type: {self.typed_feedback.get('last_failure_type', 'NONE')}
            failed_robot: {self.typed_feedback.get('failed_robot', '')}
            failed_action: {self.typed_feedback.get('failed_action', '')}
            cause: {self.typed_feedback.get('cause', '')}
            repair_instruction: {self.typed_feedback.get('repair_instruction', '')}
            """
        ).strip()

    def _compose_local_turn_trace(self) -> str:
        robots = list(self.env.robot_name_map.values())
        claims = "\n".join([f"{name}_claim: choose an action only if it is feasible for {name}; otherwise WAIT." for name in robots])
        return textwrap.dedent(
            f"""
            [LOCAL_TURN_TRACE]
            {claims}
            Coordinator_decision: choose the smallest feasible active team for the current subgoal.
            """
        ).strip()

    def _compose_sort_action_hints(self) -> str:
        if self.current_obs is None or not hasattr(self.env, "cube_to_bin"):
            return "Unavailable."

        reachable_panels = getattr(self.env, "reachable_panels", {})
        transfer_panels = ["panel3", "panel5"]
        actions = []
        for cube_name, target_panel in getattr(self.env, "cube_to_bin", {}).items():
            try:
                current_panel = self.env.get_cube_panel(self.current_obs, cube_name)
            except Exception:
                current_panel = ""
            if not current_panel or current_panel == target_panel:
                continue

            current_idx = self._panel_index(current_panel)
            target_idx = self._panel_index(target_panel)
            for agent_name, panels in reachable_panels.items():
                panel_set = set(panels)
                if current_panel not in panel_set:
                    continue

                destinations = []
                if target_panel in panel_set:
                    destinations.append(target_panel)
                for transfer_panel in transfer_panels:
                    transfer_idx = self._panel_index(transfer_panel)
                    is_between = (
                        min(current_idx, target_idx)
                        <= transfer_idx
                        <= max(current_idx, target_idx)
                    )
                    if (
                        transfer_panel in panel_set
                        and transfer_panel != current_panel
                        and is_between
                    ):
                        destinations.append(transfer_panel)

                for dest in destinations:
                    actions.append(f"NAME {agent_name} ACTION PICK {cube_name} PLACE {dest}")

        if not actions:
            return "No non-WAIT progress action inferred; use validator feedback and WAIT for inactive robots."
        return "\n".join(
            [
                "Choose active actions only from this list, plus WAIT for inactive robots:",
                *actions,
                "Do not invent a different PLACE panel for sort.",
            ]
        )

    def _compose_current_action_hints(self, profile_name: str) -> str:
        if profile_name == "sort":
            return self._compose_sort_action_hints()
        if profile_name == "cabinet":
            return self._compose_cabinet_action_hints()
        if profile_name == "sweep":
            return self._compose_sweep_action_hints()
        return "Use task prompt and validator feedback."

    def _compose_completion_check(self) -> str:
        profile_name = self._profile()["name"]
        if profile_name == "sweep":
            return self._compose_sweep_completion_check()
        if profile_name == "cabinet":
            return self._compose_cabinet_completion_check()
        return textwrap.dedent(
            """
            [COMPLETION_CHECK]
            environment_done_now: unknown
            all_required_items_done: infer from scene description and task-specific success criteria
            unfinished_items: unknown
            terminal_rule: Before outputting EXECUTE, verify every required object/state is complete. If any required item is unfinished, all-WAIT is invalid.
            next_required_progress: choose a feasible action for the first unfinished item and assign WAIT only to inactive robots.
            all_wait_policy: invalid unless the task is already complete according to the environment criteria
            """
        ).strip()

    def _compose_sweep_completion_check(self) -> str:
        status = self._sweep_completion_status()
        if not status["available"]:
            return "[COMPLETION_CHECK]\nenvironment_done_now: unknown\nterminal_rule: Sweep is complete only when every cube is inside trash_bin, not merely inside dustpan.\nall_wait_policy: invalid while any cube is outside trash_bin."
        return textwrap.dedent(
            f"""
            [COMPLETION_CHECK]
            environment_done_now: {str(status['all_done']).lower()}
            all_required_items_done: {str(status['all_done']).lower()}
            evaluator_rule: every cube body must be within 0.200 of trash_bin_bottom
            item_status:
            {self._indent(chr(10).join(status['rows']))}
            unfinished_items: {self._join_values(status['unfinished'])}
            terminal_rule: A cube inside dustpan is NOT done; Alice must DUMP it into trash_bin. A cube on table is NOT done; both robots must coordinate MOVE/SWEEP on the same cube.
            next_required_progress: {status['next_progress']}
            all_wait_policy: invalid while unfinished_items is non-empty
            """
        ).strip()

    def _compose_sweep_action_hints(self) -> str:
        status = self._sweep_completion_status()
        if not status["available"]:
            return "Sweep completion unavailable. Use task prompt and validator feedback."
        if status["all_done"]:
            return "All cubes satisfy the trash-bin completion threshold; no further sweep action should be needed."
        return "\n".join(
            [
                "Use the completion check before selecting actions.",
                status["next_progress"],
                "Do not output all-WAIT while any cube is unfinished.",
            ]
        )

    def _sweep_completion_status(self) -> Dict[str, Any]:
        if self.current_obs is None:
            return {"available": False}
        try:
            trash_bin_xpos = self.env.physics.data.body("trash_bin_bottom").xpos
            cube_names = list(getattr(self.env, "cube_names", []))
        except Exception:
            return {"available": False}
        rows = []
        unfinished = []
        in_dustpan = []
        on_table = []
        threshold = 0.2
        for cube_name in cube_names:
            try:
                cube_xpos = self.env.physics.data.body(cube_name).xpos
                distance = self._distance(cube_xpos, trash_bin_xpos)
                object_state = getattr(self.current_obs, "objects", {}).get(cube_name)
                contacts = list(getattr(object_state, "contacts", []) or [])
            except Exception:
                rows.append(f"{cube_name}: UNKNOWN")
                unfinished.append(cube_name)
                continue
            if distance <= threshold:
                state = "DONE_IN_TRASH_BIN"
            elif "dustpan_bottom" in contacts:
                state = "IN_DUSTPAN_NOT_DONE"
                unfinished.append(cube_name)
                in_dustpan.append(cube_name)
            else:
                state = "ON_TABLE_OR_NOT_IN_TRASH_BIN"
                unfinished.append(cube_name)
                on_table.append(cube_name)
            rows.append(
                f"{cube_name}: {state}, distance_to_trash_bin_bottom={distance:.3f}, contacts={self._join_values(contacts)}"
            )
        if not unfinished:
            next_progress = "All cubes satisfy the evaluator threshold; the environment should terminate without another all-WAIT repair."
        elif in_dustpan:
            next_cube = in_dustpan[0]
            next_table = next((cube for cube in on_table if cube != next_cube), "")
            bob_action = f"Bob may MOVE {next_table}" if next_table else "Bob should WAIT"
            next_progress = f"{next_cube} is in dustpan but not done. Use NAME Alice ACTION DUMP; {bob_action}."
        else:
            next_cube = unfinished[0]
            next_progress = (
                f"Focus both robots on {next_cube}: use NAME Alice ACTION MOVE {next_cube} and NAME Bob ACTION MOVE {next_cube}; "
                f"if both tools are already aligned with {next_cube}, use NAME Alice ACTION WAIT and NAME Bob ACTION SWEEP {next_cube}."
            )
        return {
            "available": True,
            "all_done": len(unfinished) == 0,
            "rows": rows,
            "unfinished": unfinished,
            "in_dustpan": in_dustpan,
            "on_table": on_table,
            "next_progress": next_progress,
        }

    def _compose_cabinet_completion_check(self) -> str:
        status = self._cabinet_completion_status()
        if not status["available"]:
            return "[COMPLETION_CHECK]\nenvironment_done_now: unknown\nterminal_rule: Cabinet is complete only when both mug and cup are on their matching coasters.\nall_wait_policy: invalid while either object is unfinished."
        return textwrap.dedent(
            f"""
            [COMPLETION_CHECK]
            environment_done_now: {str(status['all_done']).lower()}
            all_required_items_done: {str(status['all_done']).lower()}
            evaluator_rule: mug and cup must each be within align_threshold={status['threshold']:.3f} of their own coaster
            item_status:
            {self._indent(chr(10).join(status['rows']))}
            unfinished_items: {self._join_values(status['unfinished'])}
            closed_doors: {self._join_values(status['closed_doors'])}
            terminal_rule: Do not treat a near-coaster object as done unless distance_to_coaster <= align_threshold. Do not repeat moving an object already DONE_ON_COASTER.
            next_required_progress: {status['next_progress']}
            all_wait_policy: invalid while unfinished_items is non-empty
            """
        ).strip()

    def _cabinet_completion_status(self) -> Dict[str, Any]:
        if self.current_obs is None:
            return {"available": False}
        try:
            threshold = float(getattr(self.env, "align_threshold", 0.25))
            cabinet_pos = self.env.physics.data.body("cabinet").xpos
        except Exception:
            return {"available": False}
        rows = []
        unfinished = []
        closed_doors = [
            door for door in ("left", "right") if not self._cabinet_door_is_open(door)
        ]
        for object_name in ("mug", "cup"):
            try:
                obj_pos = self.env.physics.data.body(object_name).xpos
                coaster_pos = self.env.coaster_pos[f"{object_name}_coaster"]
                distance = self._distance(obj_pos, coaster_pos)
                cabinet_distance = self._distance(obj_pos, cabinet_pos)
            except Exception:
                rows.append(f"{object_name}: UNKNOWN")
                unfinished.append(object_name)
                continue
            if distance <= threshold:
                state = "DONE_ON_COASTER"
            elif cabinet_distance < 0.35:
                state = "INSIDE_CABINET_NOT_DONE"
                unfinished.append(object_name)
            else:
                state = "OUTSIDE_NOT_ON_COASTER"
                unfinished.append(object_name)
            rows.append(
                f"{object_name}: {state}, distance_to_coaster={distance:.3f}, distance_to_cabinet={cabinet_distance:.3f}"
            )
        if not unfinished:
            next_progress = "Both mug and cup satisfy the evaluator threshold; the environment should terminate without another all-WAIT repair."
        elif closed_doors:
            next_progress = "Open and hold closed cabinet doors before extracting unfinished objects; all-WAIT is invalid."
        else:
            target = unfinished[0]
            next_progress = (
                f"Choose a free reachable robot for NAME <robot> ACTION PICK {target} PLACE {target}_coaster; "
                "door holders and inactive robots should WAIT. Do not move objects already DONE_ON_COASTER."
            )
        return {
            "available": True,
            "threshold": threshold,
            "all_done": len(unfinished) == 0,
            "rows": rows,
            "unfinished": unfinished,
            "closed_doors": closed_doors,
            "next_progress": next_progress,
        }

    def _compose_cabinet_action_hints(self) -> str:
        if self.current_obs is None:
            return "Unavailable."

        closed_doors = [
            door for door in ("left", "right") if not self._cabinet_door_is_open(door)
        ]
        if closed_doors:
            actions = []
            for door in closed_doors:
                handle = f"{door}_door_handle"
                holder = self._cabinet_holder_for(handle)
                if holder:
                    actions.append(f"NAME {holder} ACTION OPEN {handle}")
                    continue
                for agent_name in self.env.robot_name_map.values():
                    if self._cabinet_agent_can_reach(agent_name, handle) and not self._agent_contacts(agent_name):
                        actions.append(f"NAME {agent_name} ACTION PICK {handle}")
                        break
            if actions:
                return "\n".join(
                    [
                        "Door dependency is still active. Choose these progress actions and WAIT for inactive robots:",
                        *actions,
                    ]
                )

        remaining = []
        for object_name in ("mug", "cup"):
            done, distance = self._cabinet_object_on_coaster(object_name)
            if not done:
                suffix = "" if distance is None else f" (distance_to_coaster={distance:.3f})"
                remaining.append((object_name, suffix))

        if not remaining:
            return "Both mug and cup are within coaster threshold; the environment should finish before another all-WAIT prompt."

        actions = []
        for object_name, suffix in remaining:
            for agent_name in self.env.robot_name_map.values():
                if not self._cabinet_agent_can_reach(agent_name, object_name):
                    continue
                if self._agent_contacts(agent_name):
                    continue
                actions.append(
                    f"NAME {agent_name} ACTION PICK {object_name} PLACE {object_name}_coaster{suffix}"
                )
        if not actions:
            return (
                "An object is not on its coaster, but no free reachable robot is inferred. "
                "Keep door holders WAIT and choose the reachable free object actor if available."
            )
        return "\n".join(
            [
                "Objects not explicitly on their coaster remain unfinished. Choose one of these progress actions and WAIT for door holders/inactive robots:",
                *actions,
            ]
        )

    def _cabinet_object_on_coaster(self, object_name: str) -> tuple:
        try:
            obj_pos = self.env.physics.data.body(object_name).xpos
            coaster_pos = self.env.coaster_pos[f"{object_name}_coaster"]
            threshold = getattr(self.env, "align_threshold", 0.12)
            distance = math.sqrt(
                sum((float(obj_pos[i]) - float(coaster_pos[i])) ** 2 for i in range(3))
            )
            return distance <= threshold, distance
        except Exception:
            try:
                description = self.env.describe_cups(self.current_obs, include_coords=False).lower()
            except Exception:
                description = ""
            return f"{object_name} is on its coaster" in description, None

    def _cabinet_door_is_open(self, door: str) -> bool:
        try:
            joint_name = f"{door}doorhinge"
            qpos_slice = self.env.physics.named.data.qpos._convert_key(joint_name)
            qpos = float(self.env.physics.data.qpos[qpos_slice.start])
        except Exception:
            return True
        if door == "left":
            return qpos <= -2.0
        return qpos >= 2.0

    def _cabinet_holder_for(self, target: str) -> str:
        for agent_name in self.env.robot_name_map.values():
            if target in self._agent_contacts(agent_name):
                return agent_name
        return ""

    def _cabinet_agent_can_reach(self, agent_name: str, target: str) -> bool:
        prompt = self._agent_prompt_for(agent_name)
        match = re.search(r"can only reach ([^\.\n]+)", prompt)
        if not match:
            return True
        reachable = [item.strip() for item in match.group(1).split(",")]
        return target in reachable

    def _agent_contacts(self, agent_name: str) -> List[str]:
        if self.current_obs is None:
            return []
        robot_name = getattr(self.env, "robot_name_map_inv", {}).get(agent_name)
        if not robot_name:
            return []
        robot_state = getattr(self.current_obs, robot_name, None)
        contacts = getattr(robot_state, "contacts", None)
        return [str(contact) for contact in contacts] if contacts else []

    def _compose_output_contract(self) -> str:
        robots = ", ".join(self.env.robot_name_map.values())
        if self.use_waypoints:
            action_template = "NAME <robot> ACTION <action> PATH <path>"
            wait_template = "WAIT PATH <current gripper path>"
            path_rule = "Every robot line, including WAIT, must include PATH."
        else:
            action_template = "NAME <robot> ACTION <action>"
            wait_template = "WAIT"
            path_rule = "Do not output PATH coordinates for this task."
        return textwrap.dedent(
            f"""
            [OUTPUT_CONTRACT]
            Return the final executable plan exactly in the native RoCoBench format:
            EXECUTE
            {action_template}
            ...
            Required robots: {robots}
            Each required robot must appear exactly once. Inactive robots must output {wait_template}.
            {path_rule}
            If COMPLETION_CHECK lists unfinished_items, do not output all-WAIT.
            Do not output JSON after EXECUTE. Do not omit any robot.
            """
        ).strip()

    def _update_typed_feedback(self, feedbacks: List[str]):
        merged = "\n".join([f for f in feedbacks if f and f != "None"]).strip()
        if not merged:
            self.typed_feedback = {
                "last_failure_type": "NONE",
                "failed_robot": "",
                "failed_action": "",
                "cause": "",
                "repair_instruction": "",
            }
            return
        lower = merged.lower()
        failure_type = "TASK_RULE_FAIL"
        repair = "Revise the plan to satisfy task constraints and output one valid action per robot."
        if "parsing failed" in lower or "failed to parse" in lower or "does not contain" in lower:
            failure_type = "PARSE_FAIL"
            repair = "Reformat strictly as EXECUTE plus exactly one NAME/ACTION line per robot."
        elif "reachability failed" in lower or "out of reach" in lower:
            failure_type = "REACH_FAIL"
            repair = "Assign the action to a robot that can reach the target, or use a valid intermediate step."
        elif "ik failed" in lower:
            failure_type = "IK_FAIL"
            repair = "Choose a more feasible target pose or simpler waypoint path."
        elif "collision detected" in lower or "collided object pairs" in lower:
            failure_type = "COLLISION_FAIL"
            repair = "Separate robot motions, assign one robot WAIT, or choose safer waypoints."
        elif "path feedback" in lower or "waypoint" in lower:
            failure_type = "PATH_FAIL"
            repair = "Use smoother, evenly spaced waypoints and avoid collision-prone regions."
        elif "task constraints" in lower:
            failure_type = "TASK_RULE_FAIL"
        self.typed_feedback = {
            "last_failure_type": failure_type,
            "failed_robot": self._extract_robot_name(merged),
            "failed_action": "",
            "cause": self._shorten(merged, 500),
            "repair_instruction": repair,
        }

    def _update_await_status_from_plans(self, llm_plans):
        if not llm_plans:
            self.await_status = {
                "awaiting_robots": [],
                "responded_robots": [],
                "resume_condition": "repair_after_validator_feedback",
            }
            return
        proposal = getattr(llm_plans[0], "parsed_proposal", "")
        active = self._active_robots_from_plan(proposal)
        self.await_status = {
            "awaiting_robots": active,
            "responded_robots": active,
            "resume_condition": "validator_result_available",
        }

    def _append_task_row(self, parsed_plan: str, execute_success: bool):
        if not self._enabled("task_table"):
            return
        task_id = f"T{len(self.task_table) + 1}"
        active = self._active_robots_from_plan(parsed_plan)
        profile = self._profile()
        status = "completed" if execute_success else "failed"
        self.task_table[task_id] = {
            "task_abstract": self._shorten(profile["stage_order"].split("->")[0].strip(), 80),
            "task_desc": self._shorten(parsed_plan.replace("\n", " "), 180),
            "assignee": ",".join(active) if active else "none",
            "status": status,
            "depends_on": "",
            "trigger": "simulator_feedback",
            "completion_criteria": self._shorten(profile["completion_criteria"], 120),
            "last_result": status if execute_success else self.typed_feedback.get("last_failure_type", "failed"),
        }

    def _append_conversation_summary(self, parsed_plan: str, execute_success: bool):
        status = "success" if execute_success else "failed"
        summary = f"Turn {self.turn_id}: {status}; {self._shorten(parsed_plan.replace(chr(10), ' '), 220)}"
        self.conversation_context.append(summary)
        limit = 3
        profile_name = self._profile()["name"]
        if profile_name in {"cabinet", "rope"}:
            limit = 1
        self.conversation_context = self.conversation_context[-limit:]

    def _update_dynamic_plan(self, parsed_plan: str, execute_success: bool):
        status = "completed" if execute_success else f"needs repair: {self.typed_feedback.get('last_failure_type')}"
        self.dynamic_plan = (
            f"Last action {status}. Continue with profile stage order: {self._profile()['stage_order']}. "
            f"Repair priority: {self._profile()['repair_priority']}."
        )

    def _active_robots_from_plan(self, parsed_plan: str) -> List[str]:
        active = []
        for agent_name in self.env.robot_name_map.values():
            pattern = rf"(?:NAME\s+)?{re.escape(agent_name)}\s*(?:ACTION|:)\s+([^\n]+)"
            match = re.search(pattern, parsed_plan or "", re.IGNORECASE)
            if match and "WAIT" not in match.group(1).upper():
                active.append(agent_name)
        return active

    def _extract_robot_name(self, text: str) -> str:
        for agent_name in self.env.robot_name_map.values():
            if agent_name in text:
                return agent_name
        return ""

    @staticmethod
    def _join_values(values: List[Any], empty: str = "None", limit: int = 12) -> str:
        if not values:
            return empty
        unique = []
        for value in values:
            text = str(value)
            if text not in unique:
                unique.append(text)
        shown = unique[:limit]
        suffix = f", ...(+{len(unique) - limit})" if len(unique) > limit else ""
        return ", ".join(shown) + suffix

    @staticmethod
    def _shorten(text: Any, limit: int) -> str:
        text = str(text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _panel_index(panel_name: str) -> int:
        match = re.search(r"panel(\d+)", str(panel_name))
        return int(match.group(1)) if match else 0

    @staticmethod
    def _distance(pos_a, pos_b) -> float:
        return math.sqrt(sum((float(pos_a[i]) - float(pos_b[i])) ** 2 for i in range(3)))

    @staticmethod
    def _indent(text: str, prefix: str = "  ") -> str:
        return "\n".join(prefix + line for line in str(text).splitlines())
