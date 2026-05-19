import numpy as np
import logging
from time import time
from typing import Callable, List, Optional, Tuple, Union, Dict, Set
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from pydantic import dataclasses, validator
import matplotlib.pyplot as plt

from rocobench.envs import SimAction, EnvState, SimRobot
from rocobench.envs.env_utils import Pose
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.subtask_plan import LLMPathPlan


class PlannedPathPolicy:
    """
    Takes in a series of LLM-proposed plans, i.e. a path of desired ee poses for each robot and where should it grasp/resealse objects
    Use these plans to compute the desired joint position waypoints via IK
    Use MultiArmRRT to plan and interpolate between the waypoints
    By default, each RRT planning step only cares about going from start to end without arms colliding,
    so the intermediate GPT-proposed waypoints may get skipped in the final motion plan.
    The plan for each robot may end with a target object to grasp.
    Note the assumption: each LLMPathPlan can grasp at most one object per robot, so a pick-and-place motion would need two LLMPathPlan's to complete. 
    """
    def __init__(
        self,
        physics,
        robots: Dict[str, SimRobot],
        path_plan: LLMPathPlan,  
        control_freq: int = 20,
        close_loop: bool = False,
        use_weld: bool = True,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        graspable_object_names: Optional[Union[Dict[str, str], List[str]]] = None,
        check_relative_pose: bool = False,
        allowed_collision_pairs: Optional[List[Tuple[int, int]]] = None,
        plan_splitted: bool = False,
        timeout: int = 200,
    ):
        self.robot_names = robots.keys()
        self.robots = robots
        physics = physics.copy(share_model=True)
        self.graspable_object_names = graspable_object_names
        self.rrt_planner = MultiArmRRT(
            physics=physics,
            robots=robots,
            graspable_object_names=graspable_object_names,
            allowed_collision_pairs=allowed_collision_pairs,
            inhand_object_info=path_plan.get_inhand_obj_info(physics),
            )
        self.robots = robots
        self.rrt_plan_results = None
        self.path_plan = path_plan 
        self.control_freq = control_freq
        
        self.close_loop = close_loop # need to re-plan if close_loop is True
        self.check_relative_pose = check_relative_pose
        
        # check if target object and site are valid
        self.use_weld = use_weld # need to touch model.eq_active
        self.tograsp = self.parse_llm_plan_for_grasp(physics, path_plan)  
        self.inhand = path_plan.get_inhand_ids(physics).copy()
        self.grasp_allowed = path_plan.get_allowed_collision_ids(physics).copy()
        self.allowed_collision_pairs = allowed_collision_pairs
        self.ik_goal_top_k = 3
        self.ik_per_robot_top_k = 3
        self.enable_dry_run = False
        self.dry_run_max_actions = 80
        self.dry_run_forward_steps = 8
        self.parse_llm_plan_to_qpos(
            physics, path_plan, update=True
            )
        self.action_buffer = []
        self.action_idx = 0
        self.skip_direct_path = skip_direct_path # enforces the planner to go through the valid waypoints 
        self.skip_smooth_path = skip_smooth_path # skip smoothing the path, useful for debugging
        self.plan_splitted = plan_splitted # if True, the plan is splitted into two parts, one for each robot
        self.timeout = timeout # timeout for each planning step, in number of planning steps

    def sparsify_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Downsample by actual joint motion instead of a fixed stride."""
        path_ls = list(path)
        if len(path_ls) <= 4:
            return path_ls
        selected = [path_ls[0]]
        last = path_ls[0]
        max_joint_step = 0.18
        for qpos in path_ls[1:-1]:
            if np.max(np.abs(qpos - last)) >= max_joint_step:
                selected.append(qpos)
                last = qpos
        tail = path_ls[-3:]
        for qpos in tail:
            if len(selected) == 0 or not np.allclose(selected[-1], qpos):
                selected.append(qpos)
        return selected

    def validate_sparse_path(self, path: List[np.ndarray], physics) -> bool:
        """Check shortcut segments introduced by sparsification."""
        path_ls = list(path)
        if len(path_ls) <= 1:
            return True
        for q1, q2 in zip(path_ls[:-1], path_ls[1:]):
            if not self.rrt_planner.is_motion_valid(
                q1,
                q2,
                physics=physics,
                allow_grasp=True,
                check_grasp_ids=self.grasp_allowed,
                check_relative_pose=self.check_relative_pose,
            ):
                return False
        return True

    def sparsify_validated_path(self, path: List[np.ndarray], physics) -> Optional[List[np.ndarray]]:
        sparse_path = self.sparsify_path(path)
        if self.validate_sparse_path(sparse_path, physics=physics):
            return sparse_path
        logging.warning("Sparse path validation failed; trying dense RRT path")
        dense_path = list(path)
        if self.validate_sparse_path(dense_path, physics=physics):
            return dense_path
        logging.warning("Dense RRT path also failed validation under current physics")
        return None


    def ik_ee_poses_to_qpos(self, physics, ee_poses: Dict[str, Pose]) -> Dict[str, np.ndarray]:
        """
        Computes the joint positions for each robot to achieve the desired ee poses
        """
        for _name in self.robot_names:
            assert _name in ee_poses.keys(), f"missing robot name {_name} in ee_poses"
        full_qpos_result = physics.data.qpos.copy()
        qpos_target_dict = self.rrt_planner.inverse_kinematics_all(
            physics=physics,
            ee_poses=ee_poses,
            allow_grasp=True, 
            check_grasp_ids=self.inhand,
            check_relative_pose=self.check_relative_pose,
        ) # qpos for the robots only
        for _name, ik_result in qpos_target_dict.items():
            if ik_result is not None:
                robot_qpos, qpos_idxs = ik_result[0], ik_result[1] 
                full_qpos_result[qpos_idxs] = robot_qpos
        return qpos_target_dict, full_qpos_result

    def ik_ee_poses_to_qpos_candidates(
        self,
        physics,
        ee_poses: Dict[str, Pose],
    ) -> List[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """
        Computes several valid joint-space targets for the same EE target poses.
        The first candidate preserves the old single-target behavior.
        """
        for _name in self.robot_names:
            assert _name in ee_poses.keys(), f"missing robot name {_name} in ee_poses"
        reference_qpos = physics.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos]
        candidates = self.rrt_planner.inverse_kinematics_all_candidates(
            physics=physics,
            ee_poses=ee_poses,
            allow_grasp=True,
            check_grasp_ids=self.inhand,
            check_relative_pose=self.check_relative_pose,
            num_candidates=self.ik_goal_top_k,
            per_robot_top_k=self.ik_per_robot_top_k,
            max_combined_candidates=self.ik_goal_top_k,
            reference_qpos=reference_qpos,
        )
        if len(candidates) == 0:
            qpos_target_dict, full_qpos_result = self.ik_ee_poses_to_qpos(physics, ee_poses)
            if any(ik_result is None for ik_result in qpos_target_dict.values()):
                return []
            return [(qpos_target_dict, full_qpos_result)]

        return [
            (qpos_target_dict, full_qpos_result)
            for qpos_target_dict, full_qpos_result, _score in candidates
        ]

    def parse_llm_plan_to_qpos(
        self, 
        physics, 
        path_plan: LLMPathPlan, 
        verbose: bool = False,
        update: bool = False,
    ) -> Tuple[np.ndarray]:
        """
        Assumes the paths for each robot are the same length, and some might end with a grasp/release.
        returns the target qpos after computing IK on all the goal/waypoint poses 
        """
        target_candidates = self.ik_ee_poses_to_qpos_candidates(
            physics,
            path_plan.ee_target_poses,
        )
        assert len(target_candidates) > 0, "failed to compute IK candidates for target poses"
        qpos_target_dict, full_qpos_target = target_candidates[0]
        for _name, ik_result in qpos_target_dict.items():
            assert ik_result is not None, f"failed to compute IK for {_name}" 
        joint_qpos_target = full_qpos_target[self.rrt_planner.all_joint_idxs_in_qpos]
        full_qpos_target_candidates = [full_qpos for _, full_qpos in target_candidates]
        joints_qpos_target_candidates = [
            full_qpos[self.rrt_planner.all_joint_idxs_in_qpos]
            for full_qpos in full_qpos_target_candidates
        ]
 
        # target_qpos are NOT allowed to be IK-insolvable, but waypoints might be
        ee_waypoints_list = path_plan.ee_waypoints_list 
        waypoints_full_qpos = []
        for tstep, ee_poses in enumerate(ee_waypoints_list):  
            attempt_qpos_dict, attempt_full_qpos = self.ik_ee_poses_to_qpos(
                physics, ee_poses
                ) 

            if all([ik_result is not None for ik_result in attempt_qpos_dict.values()]): 
                waypoints_full_qpos.append(attempt_full_qpos)
        logging.debug(f"Given {len(ee_waypoints_list)} waypoints, found {len(waypoints_full_qpos)} valid waypoints via IK")
        joints_qpos_waypoints = [
            qpos[self.rrt_planner.all_joint_idxs_in_qpos] for qpos in waypoints_full_qpos
            ]
        if verbose:
            logging.debug(f"found {len(waypoints_full_qpos)} valid waypoints via IK")
        if update:
            self.full_qpos_target = full_qpos_target
            self.joints_qpos_target = joint_qpos_target
            self.full_qpos_target_candidates = full_qpos_target_candidates
            self.joints_qpos_target_candidates = joints_qpos_target_candidates
            self.waypoints_full_qpos = waypoints_full_qpos
            self.joints_qpos_waypoints = joints_qpos_waypoints
        return full_qpos_target, waypoints_full_qpos, joint_qpos_target, joints_qpos_waypoints

    def parse_llm_plan_for_grasp(self, physics, path_plan: LLMPathPlan) -> Dict[str, Tuple[str, str, int]]:
        """ parses each object to grasp/release """
        tograsp = dict()
        for robot_name, obj in path_plan.tograsp.items():
            tograsp[robot_name] = None 
            if obj is not None:
                # make sure the object is in the physics
                obj_name, obj_site_name = obj[0], obj[1]
                grasp = obj[2] # 1 or 0
                if 'rope' in obj_name:
                    weld_body_name = self.robots[robot_name].weld_body_name
                    # special case for rope task 
                    if 'front' in obj_name: 
                        weld_name = f'rope_front_end_{weld_body_name}'    
                        body_name = 'CB0'

                    elif 'back' in obj_name:
                        weld_name = f'rope_back_end_{weld_body_name}'
                        body_name = 'CB24'

                    else:
                        raise ValueError(f"Unexpected rope object name: {obj_name}")
                    
                    weld_id = physics.named.model.eq_active._convert_key(weld_name)
                    tograsp[robot_name] = dict(
                        obj_name=obj_name,
                        grasp_site_name=body_name,
                        grasp_val=grasp,
                        weld_id=weld_id,
                        weld_name=weld_name,
                        )
                    continue

                try:
                    obj_body = physics.model.body(obj_name)
                except:
                    raise ValueError(f"object {obj_name} not in physics")
                try:
                    obj_site = physics.data.site(obj_site_name)
                except:
                    raise ValueError(f"object site {obj_site} not in physics")
                tograsp[robot_name] = dict(
                    obj_name=obj_name, 
                    grasp_site_name=obj_site_name, 
                    grasp_val=grasp,
                    )
                if self.use_weld:
                    weld_body_name = self.robots[robot_name].weld_body_name
                    weld_name = f"{obj_site_name}_{weld_body_name}" # e.g. apple_top_rhand
                    try:
                        enabled = physics.named.model.eq_active[weld_name] 
                        weld_id = physics.named.model.eq_active._convert_key(weld_name)
                        tograsp[robot_name]["weld_id"] = weld_id # change to weld id!
                        tograsp[robot_name]["weld_name"] = weld_name
                    except KeyError:
                        raise ValueError(f"{weld_name} not found in eq_active")
                    
        return tograsp 

    def get_goal_candidates(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        full_goal_candidates = getattr(
            self,
            "full_qpos_target_candidates",
            [self.full_qpos_target],
        )
        joint_goal_candidates = getattr(
            self,
            "joints_qpos_target_candidates",
            [self.joints_qpos_target],
        )
        seen = []
        goal_candidates = []
        for full_goal, joint_goal in zip(full_goal_candidates, joint_goal_candidates):
            if any(np.allclose(joint_goal, prev, atol=1e-4) for prev in seen):
                continue
            seen.append(joint_goal)
            goal_candidates.append((full_goal, joint_goal))
        if len(goal_candidates) == 0:
            goal_candidates = [(self.full_qpos_target, self.joints_qpos_target)]
        return goal_candidates

    def has_rope_interaction(self) -> bool:
        for info_dict in [self.path_plan.tograsp, self.path_plan.inhand]:
            for info in info_dict.values():
                if info is not None and ("rope" in str(info[0]).lower() or "cb" in str(info[0]).lower()):
                    return True
        action_text = " ".join(self.path_plan.action_strs.values()).lower()
        return "rope" in action_text

    def is_pick_place_like(self) -> bool:
        if self.has_rope_interaction():
            return False
        has_grasp_change = any(info is not None for info in self.tograsp.values())
        has_inhand = any(len(ids) > 0 for ids in self.inhand.values())
        action_text = " ".join(self.path_plan.action_strs.values()).lower()
        keywords = ("pick", "place", "put", "grasp", "release", "drop")
        return has_grasp_change or has_inhand or any(keyword in action_text for keyword in keywords)

    def moving_robot_names(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        threshold: float = 1e-4,
    ) -> List[str]:
        active = []
        for robot_name, joint_slice in self.rrt_planner.robot_joint_slices.items():
            joint_delta = np.abs(goal_qpos[joint_slice] - start_qpos[joint_slice])
            if len(joint_delta) > 0 and np.max(joint_delta) > threshold:
                active.append(robot_name)
        return active

    def strip_duplicate_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for qpos in path:
            if len(result) == 0 or not np.allclose(result[-1], qpos, atol=1e-5):
                result.append(qpos)
        return result

    def try_direct_path(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        physics,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        if self.skip_direct_path or len(self.joints_qpos_waypoints) > 0:
            return None, "ReasonDirectSkipped_time0_iter0"
        if not self.rrt_planner.is_state_valid(
            start_qpos,
            physics=physics,
            allow_grasp=True,
            check_grasp_ids=self.grasp_allowed,
            check_relative_pose=self.check_relative_pose,
        ):
            return None, "ReasonCollisionAtStart_time0_iter0"
        if not self.rrt_planner.is_state_valid(
            goal_qpos,
            physics=physics,
            allow_grasp=True,
            check_grasp_ids=self.grasp_allowed,
            check_relative_pose=self.check_relative_pose,
        ):
            return None, "ReasonCollisionAtGoal_time0_iter0"
        if not self.rrt_planner.is_motion_valid(
            start_qpos,
            goal_qpos,
            physics=physics,
            allow_grasp=True,
            check_grasp_ids=self.grasp_allowed,
            check_relative_pose=self.check_relative_pose,
        ):
            return None, "ReasonDirectMotionInvalid_time0_iter0"
        path = [start_qpos] + self.rrt_planner.extend_ee_l2(
            start_qpos,
            goal_qpos,
            physics=physics,
        )
        return path, "ReasonDirectSelector_time0_iter1"

    def try_pick_place_template(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        physics,
        lift_height: float = 0.08,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        if len(self.joints_qpos_waypoints) > 0 or self.check_relative_pose:
            return None, "ReasonTemplateSkippedWaypointsOrConstraint_time0_iter0"
        if not self.is_pick_place_like():
            return None, "ReasonTemplateSkippedAction_time0_iter0"

        moving = set(self.moving_robot_names(start_qpos, goal_qpos))
        if len(moving) == 0:
            return None, "ReasonTemplateSkippedNoMotion_time0_iter0"

        current_poses = self.rrt_planner.forward_kinematics_all(
            q=start_qpos,
            physics=physics,
            return_ee_pose=True,
        )
        if current_poses is None:
            return None, "ReasonTemplateNoCurrentPose_time0_iter0"

        lift_current_poses = {}
        lift_target_poses = {}
        for robot_name in self.robot_names:
            current_pose = current_poses[robot_name]
            target_pose = self.path_plan.ee_target_poses[robot_name]
            if robot_name in moving:
                current_pos = current_pose.position.copy()
                current_pos[2] += lift_height
                target_pos = target_pose.position.copy()
                target_pos[2] += lift_height
                lift_current_poses[robot_name] = Pose(
                    position=current_pos,
                    orientation=current_pose.orientation.copy(),
                )
                lift_target_poses[robot_name] = Pose(
                    position=target_pos,
                    orientation=target_pose.orientation.copy(),
                )
            else:
                lift_current_poses[robot_name] = Pose(
                    position=current_pose.position.copy(),
                    orientation=current_pose.orientation.copy(),
                )
                lift_target_poses[robot_name] = Pose(
                    position=current_pose.position.copy(),
                    orientation=current_pose.orientation.copy(),
                )

        lift_current_dict, lift_current_full = self.ik_ee_poses_to_qpos(
            physics,
            lift_current_poses,
        )
        if any(ik_result is None for ik_result in lift_current_dict.values()):
            return None, "ReasonTemplateCurrentLiftIKFailed_time0_iter0"

        lift_target_dict, lift_target_full = self.ik_ee_poses_to_qpos(
            physics,
            lift_target_poses,
        )
        if any(ik_result is None for ik_result in lift_target_dict.values()):
            return None, "ReasonTemplateTargetLiftIKFailed_time0_iter0"

        path = self.strip_duplicate_path([
            start_qpos,
            lift_current_full[self.rrt_planner.all_joint_idxs_in_qpos],
            lift_target_full[self.rrt_planner.all_joint_idxs_in_qpos],
            goal_qpos,
        ])
        if len(path) <= 1:
            return None, "ReasonTemplateSkippedNoMotion_time0_iter0"
        if not self.validate_sparse_path(path, physics=physics):
            return None, "ReasonTemplateMotionInvalid_time0_iter0"
        return path, "ReasonPickPlaceTemplate_time0_iter3"

    def is_low_coupling_plan(self, start_qpos: np.ndarray, goal_qpos: np.ndarray) -> bool:
        if len(self.joints_qpos_waypoints) > 0 or self.check_relative_pose or self.has_rope_interaction():
            return False
        return len(self.moving_robot_names(start_qpos, goal_qpos)) == 1

    def try_prioritized_plan(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        physics,
        timeout: float,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        if not self.is_low_coupling_plan(start_qpos, goal_qpos):
            return None, "ReasonPrioritizedSkippedCoupled_time0_iter0"
        return self.rrt_planner.plan_prioritized(
            start_qpos=start_qpos,
            goal_qpos=goal_qpos,
            skip_endpoint_collision_check=0,
            allow_grasp=True,
            check_grasp_ids=self.grasp_allowed,
            skip_direct_path=self.skip_direct_path,
            skip_smooth_path=self.skip_smooth_path,
            check_relative_pose=self.check_relative_pose,
            timeout=timeout,
            physics=physics,
        )

    def run_dry_run(
        self,
        physics,
        actions: List[SimAction],
    ) -> bool:
        if not self.enable_dry_run:
            return True
        physics_cp = physics.copy(share_model=False)
        for action_idx, action in enumerate(actions[: self.dry_run_max_actions]):
            if len(action.ctrl_idxs) == 0:
                continue
            for _ in range(self.dry_run_forward_steps):
                physics_cp.data.ctrl[action.ctrl_idxs] = action.ctrl_vals
                if action.eq_active_idxs is not None and len(action.eq_active_idxs) > 0:
                    physics_cp.model.eq_active[action.eq_active_idxs] = action.eq_active_vals
                physics_cp.step()
                robot_qpos = physics_cp.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos]
                if self.rrt_planner.check_collision(
                    robot_qpos=robot_qpos,
                    physics=physics_cp,
                    allow_grasp=True,
                    check_grasp_ids=self.grasp_allowed,
                    check_relative_pose=self.check_relative_pose,
                ):
                    logging.warning(f"Dry-run collision at action {action_idx}")
                    return False
        return True

    def plan_qpos(self, physics):
        start_qpos = physics.data.qpos.copy()
        joints_start_qpos = start_qpos[self.rrt_planner.all_joint_idxs_in_qpos] 
        
        # physics_cp = physics.copy(share_model=True)
        # for qpos in self.joints_qpos_waypoints + [self.joints_qpos_target]:
        #     # physics_cp.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos] = qpos
        #     physics_cp = self.rrt_planner.forward_kinematics_all(
        #        q=qpos, physics=physics_cp, return_ee_pose=False,
        #     )
            
        #     img_arr = physics_cp.render(
        #         camera_id='teaser', height=400, width=600,
        #     )
        #     physics_cp.data.qpos[:] = start_qpos
        #     physics_cp.forward()
        #     img_arr = np.concatenate([img_arr, physics_cp.render(
        #         camera_id='teaser', height=400, width=600,
        #         )], axis=1)
        #     plt.imshow(img_arr)
        #     plt.show() 
        #     # dist = np.linalg.norm(
        #     # physics_cp.data.site('robotiq_ee').xpos - physics_cp.data.site('obstacle_wall_front_top').xpos
        #     # )
        #     # print(f"dist: {dist}")
        #     # print(physics_cp.model.contacts)
        # breakpoint()

        plan_fn = self.rrt_planner.plan_splitted if self.plan_splitted else self.rrt_planner.plan
        goal_candidates = self.get_goal_candidates()
        start_time = time()
        last_path = (None, "ReasonNoIKGoalCandidate_time0_iter0")
        _, last_joint_goal = goal_candidates[0]
        for candidate_idx, (full_goal, joint_goal) in enumerate(goal_candidates):
            path, reason = self.try_direct_path(
                joints_start_qpos,
                joint_goal,
                physics,
            )
            if path is not None:
                path_ls = self.sparsify_validated_path(path, physics=physics)
                if path_ls is not None:
                    self.full_qpos_target = full_goal
                    self.joints_qpos_target = joint_goal
                    return path_ls, f"{reason}_PlannerDirect_IKCandidate{candidate_idx}"
                last_path = (None, f"{reason}_ValidationFailed")
            else:
                last_path = (None, reason)

            path, reason = self.try_pick_place_template(
                joints_start_qpos,
                joint_goal,
                physics,
            )
            if path is not None:
                path_ls = self.sparsify_validated_path(path, physics=physics)
                if path_ls is not None:
                    self.full_qpos_target = full_goal
                    self.joints_qpos_target = joint_goal
                    return path_ls, f"{reason}_PlannerTemplate_IKCandidate{candidate_idx}"
                last_path = (None, f"{reason}_ValidationFailed")
            else:
                last_path = (None, reason)

            remaining_timeout = max(1e-6, self.timeout - (time() - start_time))
            prioritized_timeout = min(remaining_timeout, max(1e-6, self.timeout * 0.35))
            path, reason = self.try_prioritized_plan(
                joints_start_qpos,
                joint_goal,
                physics,
                timeout=prioritized_timeout,
            )
            if path is not None:
                path_ls = self.sparsify_validated_path(path, physics=physics)
                if path_ls is not None:
                    self.full_qpos_target = full_goal
                    self.joints_qpos_target = joint_goal
                    return path_ls, f"{reason}_PlannerPrioritized_IKCandidate{candidate_idx}"
                last_path = (None, f"{reason}_ValidationFailed")
            else:
                last_path = (None, reason)

            remaining_timeout = max(1e-6, self.timeout - (time() - start_time))
            rrt_path = plan_fn(
                start_qpos=joints_start_qpos,
                goal_qpos=joint_goal,
                skip_endpoint_collision_check=0,
                init_samples=self.joints_qpos_waypoints[::-1], # NOTE: reverse waypoints
                allow_grasp=True,
                check_grasp_ids=self.grasp_allowed,
                skip_direct_path=self.skip_direct_path,
                skip_smooth_path=self.skip_smooth_path,
                check_relative_pose=self.check_relative_pose,
                timeout=remaining_timeout,
                physics=physics,
            )
            if rrt_path[0] is not None:
                path_ls = self.sparsify_validated_path(rrt_path[0], physics=physics)
                if path_ls is not None:
                    self.full_qpos_target = full_goal
                    self.joints_qpos_target = joint_goal
                    reason = f"{rrt_path[1]}_PlannerRRT_IKCandidate{candidate_idx}"
                    return path_ls, reason
                last_path = (None, f"{rrt_path[1]}_ValidationFailed")
            else:
                last_path = rrt_path
            last_joint_goal = joint_goal

        logging.warning(f"failed to find a path, reason: {last_path[1]}")
        physics_cp = physics.copy(share_model=True)
        physics_cp.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos] = last_joint_goal
        qpos_str = " ".join(physics_cp.data.qpos.astype(str))
        logging.warning(f"<key name='rrt_fail' qpos='{qpos_str}'/>")
        # physics_cp.forward()
        # img_arr = physics_cp.render(
        # camera_id='teaser', height=400, width=400,
        # )
        # physics_cp.data.qpos[:] = start_qpos
        # physics_cp.forward()
        # img_arr = np.concatenate([img_arr, physics_cp.render(
        #     camera_id='teaser', height=400, width=600,
        #     )], axis=1)
        # plt.imshow(img_arr)
        # plt.show()
        # breakpoint()
        return None, last_path[1]
    
    def map_qpos_to_ctrl(self, physics, qpos: np.ndarray, include_inhand: bool = True) -> Dict[str, np.ndarray]:
        ctrl_idxs = []
        qpos_idxs = []
        for robot_name, robot in self.robots.items():
            # _vals, _idxs = robot.map_qpos_to_joint_ctrl(qpos)
            ctrl_idxs.extend(robot.joint_idxs_in_ctrl)
        assert len(ctrl_idxs) == len(qpos), "qpos and ctrl do not match"
        ctrl_vals = qpos.copy().tolist()
        qpos_target = qpos.copy()

        
        for robot_name, robot in self.robots.items():
            if include_inhand and len(self.inhand[robot_name]) > 0:
                # robot should keep grasping the object
                grasp_ctrl_val = robot.get_grasp_ctrl_val(grasp=1) # single number
                ctrl_vals.append(grasp_ctrl_val)
                ctrl_idxs.append(robot.grasp_idx)
                # print(f"robot {robot_name} keeps grasping {self.inhand[robot_name]}: ctrl_val={grasp_ctrl_val} ctrl_idx={robot.grasp_idx}")

            qpos_idxs.extend(
                robot.joint_idxs_in_qpos
            ) 
        
        return dict(
            ctrl_idxs=np.array(ctrl_idxs),
            ctrl_vals=np.array(ctrl_vals),
            # NOTE: setting qpos-target makes motion jitter a lot
            qpos_idxs=np.array(qpos_idxs),
            qpos_target=qpos_target,
            )

    def get_grasp_action(
        self,
        physics, 
        qpos,
    ) -> List[SimAction]:

        joint_ctrls = self.map_qpos_to_ctrl(physics, qpos)
        eq_active_idxs = []
        eq_active_vals = []

        target_ee_poses = self.rrt_planner.forward_kinematics_all(
            physics=physics.copy(share_model=True),
            q=qpos,
            return_ee_pose=True,
        )
        grasp_idxs, grasp_vals = [], []
        for robot_name, obj_info in self.tograsp.items(): 
            if obj_info is not None: 
                grasp_val = obj_info["grasp_val"]
                if 'rope' in obj_info['obj_name']:  
                    grasp_vals.append(
                        self.robots[robot_name].get_grasp_ctrl_val(grasp=(grasp_val > 0))
                        )
                    grasp_idxs.append(
                        self.robots[robot_name].grasp_idx
                    )
                    if self.use_weld and obj_info.get("weld_id", None) is not None:
                        # both adhesion and eq_active is turned on
                        weld_id = obj_info["weld_id"]
                        weld_name = obj_info["weld_name"]
                        eq_active_idxs.append(weld_id)
                        assert int(grasp_val) in [int(0), int(1)], f"grasp_val should be integer 0 or 1 when using weld"
                        eq_active_vals.append(int(grasp_val))
    
                    continue 
 
                obj_site = obj_info["grasp_site_name"]
                site_xpos = physics.data.site(obj_site).xpos
                if grasp_val > 0:
                    pose = target_ee_poses[robot_name]
                    robot_ee_pos = pose.position 
                    dist = np.linalg.norm(site_xpos - robot_ee_pos)
                    if dist > 0.1:
                        logging.warning(f"robot {robot_name} end effector distance: {dist} is too far from object {obj_info['obj_name']}")
            
                grasp_idxs.append(
                    self.robots[robot_name].grasp_idx
                )  
                grasp_ctrl_val = self.robots[robot_name].get_grasp_ctrl_val(grasp=(grasp_val > 0))
                grasp_vals.append(grasp_ctrl_val)
                # print(f'seting grasp of robot {robot_name} to {grasp_val}')
                if self.use_weld and obj_info.get("weld_id", None) is not None:
                    # both adhesion and eq_active is turned on
                    weld_id = obj_info["weld_id"]
                    weld_name = obj_info["weld_name"]
                    eq_active_idxs.append(weld_id)
                    assert int(grasp_val) in [int(0), int(1)], f"grasp_val should be integer 0 or 1 when using weld"
                    eq_active_vals.append(int(grasp_val))
 
        if len(grasp_idxs) > 0:
            joint_ctrls["ctrl_idxs"] = np.concatenate(
                [joint_ctrls["ctrl_idxs"], np.array(grasp_idxs)]
            )
            joint_ctrls["ctrl_vals"] = np.concatenate(
                [joint_ctrls["ctrl_vals"], np.array(grasp_vals)]
            )
            joint_ctrls['eq_active_idxs'] = np.array(eq_active_idxs)
            joint_ctrls['eq_active_vals'] = np.array(eq_active_vals)

        return [SimAction(**joint_ctrls)]
    
    def plan_home(
        self,
        physics, 
        start_qpos,
    ) -> List[SimAction]:
        # TODO: try arm returns home but base stays fixed?
        need_plan = False
        home_qpos = self.full_qpos_target.copy()
        all_qpos_idxs = self.rrt_planner.all_joint_idxs_in_qpos
        home_qpos[all_qpos_idxs] = start_qpos
        
        for agent_name, return_home in self.path_plan.return_home.items():
            if return_home:
                need_plan = True
                qpos_idxs = self.robots[agent_name].joint_idxs_in_qpos
                robot_qpos = self.robots[agent_name].get_home_qpos()
                home_qpos[qpos_idxs] = robot_qpos
       
        if not need_plan:
            return []
        
        goal_qpos = home_qpos[all_qpos_idxs]
        # TODO: handle return home with object already dropped
        path = self.rrt_planner.plan(
            start_qpos=start_qpos,
            goal_qpos=goal_qpos,
            skip_endpoint_collision_check=1,
            init_samples=[],
            allow_grasp=True, 
            check_grasp_ids=self.grasp_allowed,
            skip_direct_path=self.skip_direct_path,
            skip_smooth_path=self.skip_smooth_path,
            check_relative_pose=self.check_relative_pose,
            physics=physics,
        )
        if path[0] is None:
            logging.warning(f"Failed to find a path to return to Home, reason: {path[1]}")
            physics_cp = physics.copy(share_model=True)
            physics_cp.data.qpos[self.rrt_planner.all_joint_idxs_in_qpos]  = goal_qpos
            qpos_str = " ".join(physics_cp.data.qpos.astype(str))
            logging.warning(f"<key name='rrt_return_home_fail' qpos='{qpos_str}'/>")
            # breakpoint()
            return []
        else:
            logging.debug("Found a path to return to Home")
            path_ls = self.sparsify_validated_path(path[0], physics=physics)
            if path_ls is None:
                logging.warning("Return-home path failed validation under current physics")
                return []
            actions = []
            for qpos in path_ls:
                kwargs = self.map_qpos_to_ctrl(physics, qpos, include_inhand=False) # avoid gripper keep grasping after placing
                actions.append(SimAction(**kwargs))
            return actions
        
                
    def plan(self, env) -> Tuple[bool, str]:
        """
        plan a series of actions for each robot
        """  
        physics = getattr(env, "physics", env)
        path_ls, reason = self.plan_qpos(physics)
        self.rrt_plan_results = path_ls 
        if path_ls is None:
            return False, reason
        actions = []
        for qpos in path_ls:
            kwargs = self.map_qpos_to_ctrl(physics, qpos)
            actions.append(SimAction(**kwargs))
        
        end_qpos = path_ls[-1] 
        grasp_actions = self.get_grasp_action(physics, end_qpos)
        actions.extend(grasp_actions) 
        actions.extend(
            self.plan_home(physics, end_qpos)
        )  
        if not self.run_dry_run(physics, actions):
            return False, f"{reason}_DryRunFailed"
        self.action_buffer = actions  
        self.action_idx = 0
        return True, reason
    
    @property
    def plan_exhausted(self) -> bool:
        return self.action_idx == len(self.action_buffer)
    
    @property
    def num_actions(self) -> int:
        return len(self.action_buffer)

    def act(self, obs: EnvState, physics) -> SimAction:
        if self.close_loop:
            replanned, reason = self.plan(physics)
            if not replanned:
                logging.warning("replanning failed, using previous plan")
        else:
            assert len(self.action_buffer) != 0, "action buffer is empty, cal plan_qpos first"
        action = self.action_buffer[self.action_idx]
        self.action_idx += 1
        return action 
