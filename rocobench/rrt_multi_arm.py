import logging
import numpy as np
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from transforms3d import euler, quaternions
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.utils.transformations import mat_to_quat, quat_to_euler, euler_to_quat 

from rocobench.rrt import direct_path, smooth_path, birrt, NearJointsUniformSampler, CenterWaypointsUniformSampler
from rocobench.envs import SimRobot 
from rocobench.envs.env_utils import Pose


class MultiArmRRT:
    """ Stores the info for a group of arms and plan all the combined joints together """
    def __init__(
        self,
        physics,
        robots: Optional[Dict[str, SimRobot]] = None,
        robot_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: int = 0,
        graspable_object_names: Optional[Union[Dict[str, str], List[str]]] = None,
        allowed_collision_pairs: Optional[List[Tuple[int, int]]] = None,
        inhand_object_info: Optional[Dict[str, Tuple]] = None,
    ):
        self.robots = {} if robots is None else robots
        robot_configs = {} if robot_configs is None else robot_configs
        if len(self.robots) == 0:
            assert len(robot_configs) > 0, "No robot config is passed in"
            logging.warning(
                "Warning: No robot is passed in, will use robot_configs to create robots"
            )
        
            for robot_name, robot_config in robot_configs.items():
                self.robots[robot_name] = SimRobot(physics, **robot_config)
        self.physics = physics 
        self.np_random = np.random.RandomState(seed)
        
        self.all_joint_names = []
        self.all_joint_ranges = []
        self.all_joint_idxs_in_qpos = []
        self.all_collision_link_names = []
        self.inhand_object_info = dict()
        self.robot_joint_slices = {}

        cursor = 0
        for name, robot in self.robots.items():
            width = len(robot.joint_idxs_in_qpos)
            self.robot_joint_slices[name] = slice(cursor, cursor + width)
            cursor += width
            self.all_joint_names.extend(
                robot.ik_joint_names
            ) 
            self.all_joint_idxs_in_qpos.extend(
                robot.joint_idxs_in_qpos
            )
            self.all_joint_ranges.extend(
                robot.joint_ranges
            )
            self.all_collision_link_names.extend(
                robot.collision_link_names
            )
        
        self.set_inhand_info(physics, inhand_object_info)


        self.joint_minmax = np.array([jrange for jrange in self.all_joint_ranges])
        self.joint_ranges = self.joint_minmax[:, 1] - self.joint_minmax[:, 0]

        # assign a list of allowed grasp ids to each robot
        graspable_name_dict = dict()
        for robot_name in self.robots.keys():
            if type(graspable_object_names) is dict:
                assert robot_name in graspable_object_names, f"robot_name: {robot_name} not in graspable_object_names"
                graspable_name_dict[robot_name] = graspable_object_names[robot_name]

            elif type(graspable_object_names) is list:
                graspable_name_dict[robot_name] = graspable_object_names

        self.allowed_collision_pairs = [] if allowed_collision_pairs is None else allowed_collision_pairs
        self.set_ungraspable(graspable_name_dict)
    
    def set_inhand_info(self, physics, inhand_object_info: Optional[Dict[str, Tuple]] = None):
        """ Set the inhand object info """
        self.inhand_object_info = dict()
        if inhand_object_info is not None:
            for name, robot in self.robots.items():
                self.inhand_object_info[name] = None
                
                obj_info = inhand_object_info.get(name, None)
                
                if obj_info is not None:
                    if 'rope' in obj_info[0] or 'CB' in obj_info[0]:
                        continue
                    assert len(obj_info) == 3, f"inhand obj info: {obj_info} should be a tuple of (obj_body_name, obj_site_name, obj_joint_name)"
                    body_name, site_name, joint_name = obj_info
                    try:
                        mjsite = physics.data.site(site_name)
                        qpos_slice = physics.named.data.qpos._convert_key(joint_name) 
                    except: 
                        raise ValueError(
                            f"site_name: {site_name} joint_name {joint_name} not found in mujoco model"
                        )
                    self.inhand_object_info[name] = (body_name, site_name, joint_name, (qpos_slice.start, qpos_slice.stop))
        return 

    
    def set_ungraspable(
        self, 
        graspable_object_dict: Optional[Dict[str, List[str]]]
    ):
        """ Find all sim objects that are not graspable """
        
        all_bodies = []
        for i in range(self.physics.model.nbody):
            all_bodies.append(self.physics.model.body(i))

        # all robot link bodies are ungraspable:
        ungraspable_ids = [0]  # world
        for name, robot in self.robots.items():
            ungraspable_ids.extend(
                robot.collision_link_ids
            )
        # append all children of ungraspable body
        ungraspable_ids += [
            body.id for body in all_bodies if body.rootid[0] in ungraspable_ids
        ]
 
        if graspable_object_dict is None or len(graspable_object_dict) == 0:
            graspable = set(
                [body.id for body in all_bodies if body.id not in ungraspable_ids]
            )
            ungraspable = set(ungraspable_ids)
            self.graspable_body_ids = {name: graspable for name in self.robots.keys()}
            self.ungraspable_body_ids = {name: ungraspable for name in self.robots.keys()}
        else: 
            # in addition to robots, everything else would be ungraspable if not in this list of graspable objects
            self.graspable_body_ids = {}
            self.ungraspable_body_ids = {}
            for robot_name, graspable_object_names in graspable_object_dict.items():
                graspable_ids = [
                    body.id for body in all_bodies if body.name in graspable_object_names
                ]
                graspable_ids += [
                    body.id for body in all_bodies if body.rootid[0] in graspable_ids
                ]
                self.graspable_body_ids[robot_name] = set(graspable_ids)
                robot_ungraspable = ungraspable_ids.copy()
                robot_ungraspable += [
                    body.id for body in all_bodies if body.rootid[0] not in graspable_ids
                ]
                self.ungraspable_body_ids[robot_name] = set(robot_ungraspable)
            # breakpoint()

    def forward_kinematics_all(
        self,
        q: np.ndarray,
        physics = None,
        return_ee_pose: bool = False,
    ) -> Optional[Dict[str, Pose]]:
        if physics is None:
            physics = self.physics.copy(share_model=True)
        physics = physics.copy(share_model=True)
        
        # transform inhand objects!
        obj_transforms = dict()
        for robot_name, obj_info in self.inhand_object_info.items():
            gripper_pose = self.robots[robot_name].get_ee_pose(physics)
            if obj_info is not None:
                body_name, site_name, joint_name, (start, end) = obj_info
                obj_quat = mat_to_quat(
                    physics.data.site(site_name).xmat.reshape((3, 3))
                )
                obj_pos = physics.data.site(site_name).xpos
                rel_rot = quaternions.qmult( 
                    quaternions.qinverse(
                        gripper_pose.orientation
                        ),
                    obj_quat,
                    )
                ee_rot = quaternions.quat2mat(gripper_pose.orientation)
                rel_pos = ee_rot.T @ (obj_pos - gripper_pose.position)
                obj_transforms[robot_name] = (rel_pos, rel_rot)
            else:
                obj_transforms[robot_name] = None
        
        physics.data.qpos[self.all_joint_idxs_in_qpos] = q
        physics.forward()

        ee_poses = {}
        for robot_name, robot in self.robots.items():
            ee_poses[robot_name] = robot.get_ee_pose(physics)
        
        # also transform inhand objects!
        for robot_name, obj_info in self.inhand_object_info.items():
            if obj_info is not None:
                body_name, site_name, joint_name, (start, end) = obj_info
                rel_pos, rel_rot = obj_transforms[robot_name] 
                new_ee_pos = ee_poses[robot_name].position
                new_ee_quat = ee_poses[robot_name].orientation 
                new_ee_rot = quaternions.quat2mat(new_ee_quat)
                target_pos = new_ee_pos + new_ee_rot @ rel_pos
                target_quat = quaternions.qmult(new_ee_quat, rel_rot) 
                result = self.solve_ik(
                    physics,
                    site_name,
                    target_pos,
                    target_quat,
                    joint_names=[joint_name], 
                    max_steps=300,
                    inplace=0,   
                    )
                if result is not None:
                    new_obj_qpos = result.qpos[start:end]
                    physics.data.qpos[start:end] = new_obj_qpos
                    physics.forward()
        if return_ee_pose:
            return ee_poses
        # physics.step(10) # to make sure the physics is stable
        return physics # a copy of the original physics object 

    
    def check_joint_range(
        self, 
        physics,
        joint_names,
        qpos_idxs,
        ik_result,
        allow_err=0.03,
    ) -> bool:
        _lower, _upper = physics.named.model.jnt_range[joint_names].T
        qpos = ik_result.qpos[qpos_idxs]
        assert len(qpos) == len(_lower) == len(_upper), f"Shape mismatch: qpos: {qpos}, _lower: {_lower}, _upper: {_upper}"
        for i, name in enumerate(joint_names):
            if qpos[i] < _lower[i] - allow_err or qpos[i] > _upper[i] + allow_err:
                # print(f"Joint {name} out of range: {_lower[i]} < {qpos[i]} < {_upper[i]}")
                return False 
        return True

    def solve_ik(
        self,
        physics,
        site_name,
        target_pos,
        target_quat,
        joint_names, 
        tol=1e-6,
        max_steps=300,
        max_resets=20,
        inplace=True, 
        max_range_steps=0,
        qpos_idxs=None,
        allow_grasp=True,
        check_grasp_ids=None,
        check_relative_pose=False
    ):
        physics_cp = physics.copy(share_model=True)
        accepted_result = None
        
        def reset_fn(physics):
            model = physics.named.model 
            _lower, _upper = model.jnt_range[joint_names].T
            
            curr_qpos = physics.named.data.qpos[joint_names]
            # deltas = (_upper - _lower) / 2
            # new_qpos = self.np_random.uniform(low=_lower, high=_upper)
            new_qpos = self.np_random.uniform(low=curr_qpos-0.5, high=curr_qpos + 0.5)
            new_qpos = np.clip(new_qpos, _lower, _upper)
            physics.named.data.qpos[joint_names] = new_qpos
            physics.forward()

        for i in range(max_resets):
            # print(f"Resetting IK {i}")
            if i > 0:
                reset_fn(physics_cp)
                
            result = qpos_from_site_pose(
                physics=physics_cp,
                site_name=site_name,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=joint_names,
                tol=tol,
                max_steps=max_steps,
                inplace=True,
            )
            need_reset = False
            if result.success:
                in_range = True 
                collided = False
                if qpos_idxs is not None:
                    in_range = self.check_joint_range(physics_cp, joint_names, qpos_idxs, result)
                    ik_qpos = result.qpos.copy()
                    _low, _high = physics_cp.named.model.jnt_range[joint_names].T
                    ik_qpos[qpos_idxs] = np.clip(
                        ik_qpos[qpos_idxs], _low, _high
                    )
                    ik_qpos = ik_qpos[self.all_joint_idxs_in_qpos]
                    # print('checking collision on IK result: step {}'.format(i))
                    collided = self.check_collision(
                        physics=physics_cp,
                        robot_qpos=ik_qpos,
                        check_grasp_ids=check_grasp_ids,
                        allow_grasp=allow_grasp,
                        check_relative_pose=check_relative_pose,
                        )

                need_reset = (not in_range) or collided

            else:
                need_reset = True
            if not need_reset:
                accepted_result = result
                break
        # img = physics_cp.render(camera_id='teaser', height=400, width=400)
        # plt.imshow(img)
        # plt.show()

        return accepted_result

    def score_ik_qpos(
        self,
        qpos: np.ndarray,
        reference_qpos: Optional[np.ndarray] = None,
    ) -> float:
        qpos = np.asarray(qpos)
        if reference_qpos is None:
            reference_qpos = self.physics.data.qpos[self.all_joint_idxs_in_qpos]
        elif len(reference_qpos) != len(self.all_joint_idxs_in_qpos):
            reference_qpos = reference_qpos[self.all_joint_idxs_in_qpos]
        reference_qpos = np.asarray(reference_qpos)

        delta = np.abs(qpos - reference_qpos)
        range_safe = np.maximum(self.joint_ranges, 1e-6)
        normalized_delta = delta / range_safe
        low, high = self.joint_minmax[:, 0], self.joint_minmax[:, 1]
        normalized_margin = np.minimum(qpos - low, high - qpos) / range_safe
        limit_penalty = np.mean(np.maximum(0.0, 0.1 - normalized_margin))
        return (
            float(np.max(normalized_delta))
            + 0.25 * float(np.linalg.norm(normalized_delta))
            + 0.5 * float(limit_penalty)
        )

    def solve_ik_candidates(
        self,
        physics,
        site_name,
        target_pos,
        target_quat,
        joint_names,
        tol=1e-6,
        max_steps=300,
        max_resets=24,
        qpos_idxs=None,
        allow_grasp=True,
        check_grasp_ids=None,
        check_relative_pose=False,
        num_candidates: int = 4,
        reference_qpos: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, np.ndarray]]:
        physics_cp = physics.copy(share_model=True)
        candidates = []

        def reset_fn(physics):
            model = physics.named.model
            _lower, _upper = model.jnt_range[joint_names].T
            curr_qpos = physics.named.data.qpos[joint_names]
            new_qpos = self.np_random.uniform(low=curr_qpos - 0.5, high=curr_qpos + 0.5)
            physics.named.data.qpos[joint_names] = np.clip(new_qpos, _lower, _upper)
            physics.forward()

        for i in range(max_resets):
            if i > 0:
                reset_fn(physics_cp)

            result = qpos_from_site_pose(
                physics=physics_cp,
                site_name=site_name,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=joint_names,
                tol=tol,
                max_steps=max_steps,
                inplace=True,
            )
            if not result.success or qpos_idxs is None:
                continue

            if not self.check_joint_range(physics_cp, joint_names, qpos_idxs, result):
                continue

            candidate_full_qpos = result.qpos.copy()
            _low, _high = physics_cp.named.model.jnt_range[joint_names].T
            candidate_full_qpos[qpos_idxs] = np.clip(
                candidate_full_qpos[qpos_idxs],
                _low,
                _high,
            )
            candidate_joint_qpos = candidate_full_qpos[self.all_joint_idxs_in_qpos]
            if self.check_collision(
                physics=physics_cp,
                robot_qpos=candidate_joint_qpos,
                check_grasp_ids=check_grasp_ids,
                allow_grasp=allow_grasp,
                check_relative_pose=check_relative_pose,
            ):
                continue

            robot_qpos = candidate_full_qpos[qpos_idxs].copy()
            if any(np.allclose(robot_qpos, existing[1], atol=1e-4) for existing in candidates):
                continue
            score = self.score_ik_qpos(candidate_joint_qpos, reference_qpos=reference_qpos)
            candidates.append((score, robot_qpos))

        candidates.sort(key=lambda item: item[0])
        return candidates[:num_candidates]

    def inverse_kinematics_all(
        self,
        physics,
        ee_poses: Dict[str, Pose],
        inplace=False, 
        allow_grasp=True, 
        check_grasp_ids=None,
        check_relative_pose=False,
    ) -> Dict[str, Union[None, np.ndarray]]:

        if physics is None:
            physics = self.physics
        physics = physics.copy(share_model=True)
        results = dict() 
        for robot_name, target_ee in ee_poses.items():
            assert robot_name in self.robots, f"robot_name: {robot_name} not in self.robots"
            robot = self.robots[robot_name]
            pos = target_ee.position 
            quat = target_ee.orientation
            if robot.use_ee_rest_quat:
                quat = quaternions.qmult(
                    quat, robot.ee_rest_quat
                )
            # print(robot.ee_site_name, pos, quat, robot.joint_names)
            qpos_idxs = robot.joint_idxs_in_qpos     
            result = self.solve_ik(
                physics=physics,
                site_name=robot.ee_site_name,
                target_pos=pos,
                target_quat=quat,
                joint_names=robot.ik_joint_names,
                tol=1e-6,
                max_steps=300,
                inplace=inplace,  
                qpos_idxs=qpos_idxs,
                allow_grasp=allow_grasp, 
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
            )
            if result is not None:
                result_qpos = result.qpos[qpos_idxs].copy()
                _lower, _upper = physics.named.model.jnt_range[robot.ik_joint_names].T
                result_qpos = np.clip(result_qpos, _lower, _upper)
                results[robot_name] = (result_qpos, qpos_idxs) 
            else:
                results[robot_name] = None
        return results      

    def inverse_kinematics_all_candidates(
        self,
        physics,
        ee_poses: Dict[str, Pose],
        allow_grasp=True,
        check_grasp_ids=None,
        check_relative_pose=False,
        num_candidates: int = 3,
        per_robot_top_k: int = 3,
        max_combined_candidates: int = 3,
        reference_qpos: Optional[np.ndarray] = None,
    ) -> List[Tuple[Dict[str, Tuple[np.ndarray, List[int]]], np.ndarray, float]]:
        if physics is None:
            physics = self.physics
        physics = physics.copy(share_model=True)
        if reference_qpos is None:
            reference_qpos = physics.data.qpos[self.all_joint_idxs_in_qpos]
        elif len(reference_qpos) != len(self.all_joint_idxs_in_qpos):
            reference_qpos = reference_qpos[self.all_joint_idxs_in_qpos]

        per_robot_candidates = {}
        for robot_name, target_ee in ee_poses.items():
            assert robot_name in self.robots, f"robot_name: {robot_name} not in self.robots"
            robot = self.robots[robot_name]
            quat = target_ee.orientation
            if robot.use_ee_rest_quat:
                quat = quaternions.qmult(quat, robot.ee_rest_quat)
            candidates = self.solve_ik_candidates(
                physics=physics,
                site_name=robot.ee_site_name,
                target_pos=target_ee.position,
                target_quat=quat,
                joint_names=robot.ik_joint_names,
                tol=1e-6,
                max_steps=300,
                qpos_idxs=robot.joint_idxs_in_qpos,
                allow_grasp=allow_grasp,
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
                num_candidates=max(num_candidates, per_robot_top_k),
                reference_qpos=reference_qpos,
            )
            if len(candidates) == 0:
                return []
            per_robot_candidates[robot_name] = [
                (score, qpos, robot.joint_idxs_in_qpos)
                for score, qpos in candidates[:per_robot_top_k]
            ]

        beam = [(physics.data.qpos.copy(), {}, 0.0)]
        for robot_name in ee_poses.keys():
            next_beam = []
            for full_qpos, result_dict, base_score in beam:
                for score, robot_qpos, qpos_idxs in per_robot_candidates[robot_name]:
                    candidate_full_qpos = full_qpos.copy()
                    candidate_full_qpos[qpos_idxs] = robot_qpos
                    candidate_joint_qpos = candidate_full_qpos[self.all_joint_idxs_in_qpos]
                    if not self.is_state_valid(
                        candidate_joint_qpos,
                        physics=physics,
                        allow_grasp=allow_grasp,
                        check_grasp_ids=check_grasp_ids,
                        check_relative_pose=check_relative_pose,
                    ):
                        continue
                    combined_results = dict(result_dict)
                    combined_results[robot_name] = (robot_qpos.copy(), qpos_idxs)
                    combined_score = (
                        base_score
                        + score
                        + self.score_ik_qpos(candidate_joint_qpos, reference_qpos=reference_qpos)
                    )
                    next_beam.append((candidate_full_qpos, combined_results, combined_score))
            if len(next_beam) == 0:
                return []
            beam = sorted(next_beam, key=lambda item: item[2])[:max_combined_candidates]

        return [
            (result_dict, full_qpos, score)
            for full_qpos, result_dict, score in beam[:max_combined_candidates]
        ]


    def ee_l2_distance(
        self, 
        q1: np.ndarray, 
        q2: np.ndarray, 
        orientation_factor: float = 0.2,
        physics=None,
    ) -> float: 
        pose1s = self.forward_kinematics_all(q1, physics=physics, return_ee_pose=True) # {robotA: Pose1, robotB: Pose1}
        pose2s = self.forward_kinematics_all(q2, physics=physics, return_ee_pose=True) # {robotA: Pose2, robotB: Pose2}
        assert pose1s is not None and pose2s is not None
        dist = 0

        # compute pair-wise distance between each robot's Pose1 and Pose2
        for robot_name in pose1s.keys():
            pose1 = pose1s[robot_name]
            pose2 = pose2s[robot_name]
            dist += pose1.distance(pose2, orientation_factor=orientation_factor)
        return dist

    def compute_motion_steps(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        ee_resolution: float = 0.006,
        joint_resolution: float = 0.08,
        max_steps: int = 200,
        physics=None,
    ) -> int:
        ee_dist = self.ee_l2_distance(q1, q2, physics=physics)
        joint_dist = float(np.max(np.abs(q2 - q1))) if len(q1) > 0 else 0.0
        n_ee = int(np.ceil(ee_dist / ee_resolution)) if ee_dist > 0 else 0
        n_joint = int(np.ceil(joint_dist / joint_resolution)) if joint_dist > 0 else 0
        n_steps = max(1, n_ee, n_joint)

        if any(obj_info is not None for obj_info in self.inhand_object_info.values()):
            n_steps *= 2

        moving_robots = 0
        cursor = 0
        for robot in self.robots.values():
            width = len(robot.joint_idxs_in_qpos)
            joint_delta = np.abs(q2[cursor:cursor + width] - q1[cursor:cursor + width])
            if len(joint_delta) > 0 and np.max(joint_delta) > 1e-5:
                moving_robots += 1
            cursor += width
        if moving_robots > 1:
            n_steps = int(np.ceil(n_steps * 1.5))

        return max(1, min(max_steps, n_steps))

    def extend_ee_l2(
        self, 
        q1: np.ndarray, 
        q2: np.ndarray, 
        resolution: float = 0.006,
        joint_resolution: float = 0.08,
        adaptive: bool = True,
        physics=None,
    ) -> List[np.ndarray]:
        if np.allclose(q1, q2):
            return []
        if adaptive:
            n_steps = self.compute_motion_steps(
                q1,
                q2,
                ee_resolution=resolution,
                joint_resolution=joint_resolution,
                physics=physics,
            )
        else:
            dist = self.ee_l2_distance(q1, q2, physics=physics)
            n_steps = max(1, int(np.ceil(dist / resolution)))
        return [q1 + (q2 - q1) * (i / n_steps) for i in range(1, n_steps + 1)]

    def allow_collision_pairs(
        self,
        physics: Any,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, List]] = None,
    ) -> Set[FrozenSet[int]]:
        
        """ Get the allowed collision pairs """ 
        allowed = set()
        for robot_name, robot in self.robots.items():
            # add the robot's set to the allowed set:
            allowed.update( 
                robot.ee_link_pairs
            )
        for id_pair in self.allowed_collision_pairs:
            allowed.add(frozenset([id_pair[0], id_pair[1]]))

        if allow_grasp:
            # if the robot is in contact with some allowed objects, allow the collision  
            for robot_name, robot in self.robots.items():
                assert robot_name in self.graspable_body_ids, f"Robot {robot_name} not found in graspable_body_ids"
                graspable_ids = self.graspable_body_ids[robot_name]
                
                if check_grasp_ids is not None:
                    assert robot_name in check_grasp_ids, f"Robot {robot_name} not found in check_grasp_ids"
                    # only find the desired grasp_id to allow collision
                    graspable_ids = check_grasp_ids[robot_name]
                
                for ee_id in robot.ee_link_body_ids:
                    for _id in graspable_ids: 
                        allowed.add(
                            frozenset([ee_id, _id])
                            ) 
                # Task envs can still explicitly allow broader contacts via
                # get_allowed_collision_pairs(); default grasp allowance is EE-only.
        return allowed 

    def get_collided_links(
        self,
        qpos: Optional[np.ndarray] = None,
        physics = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, List]] = None,
        verbose: bool = False,
        show: bool = False,
    ) -> List[str]:
        """ Get the collided links """ 
        if physics is None:
            physics = self.physics.copy(share_model=True)    
        physics = self.forward_kinematics_all(physics=physics, q=qpos, return_ee_pose=False) 
        
        robot_collison_ids = [physics.model.body(link_name).id for link_name in self.all_collision_link_names]
        # NOTE: cant allow grasped object to collide with other objects in the env
        allowed_collisions = self.allow_collision_pairs(
            physics, allow_grasp=allow_grasp, check_grasp_ids=check_grasp_ids
            ) # robot-to-object 
        collided_id1 = physics.model.geom_bodyid[physics.data.contact.geom1].copy()
        collided_id2 = physics.model.geom_bodyid[physics.data.contact.geom2].copy() 
        

        if len(allowed_collisions) > 0:
            undesired_mask = np.ones_like(collided_id1).astype(bool)
            for idx in range(len(collided_id1)):
                body1 = collided_id1[idx]
                body2 = collided_id2[idx]
                if frozenset([body1, body2]) in allowed_collisions:
                    undesired_mask[idx] = False
            collided_id1 = collided_id1[undesired_mask]
            collided_id2 = collided_id2[undesired_mask]
        
        # TODO: if an object is being grasped, don't allow it to collide with other objects
        # if allow_grasp and check_grasp_ids is not None:
        #     
        #     for robot_name, grasp_id in check_grasp_ids.items():
        #         graspable_ids = check_grasp_ids[robot_name]
        #         robot_collison_ids.extend(graspable_ids)
        all_pairs = set(zip(collided_id1, collided_id2))
        bad_pairs = set()
        for pair in all_pairs:
            # if pair[0] in robot_collison_ids or pair[1] in robot_collison_ids:
            root1 = physics.model.body(pair[0]).rootid 
            root2 = physics.model.body(pair[1]).rootid
            bad_pairs.add(
                (physics.model.body(root1).name, physics.model.body(root2).name)
            )
            # if (pair[0] == 62 and pair[1] == 64) or (pair[0] == 64 and pair[1] == 62):
            #     breakpoint()
        all_ids = set(collided_id1).union(set(collided_id2)) # could contain both object-to-object, robot-to-robot, etc
       
        # undesired_ids = set(robot_collison_ids).intersection(all_ids) 
        undesired_ids = all_ids
        
        # dist = np.linalg.norm(
        #     physics.data.site('robotiq_ee').xpos  - physics.data.site('panda_ee').xpos
        #     )
        # if dist > 0.8 or dist < 0.6:
        #     bad_pairs.add((dist, dist))

        # if a link is on robot AND it's in contact with something Not in the allowed_collisions
        # if np.linalg.norm(physics.data.body('red_cube').xpos  - physics.data.body('dustpan').xpos) < 0.1:
        # if 54 in collided_id1 or 54 in collided_id2:
        if len(undesired_ids) > 0 and show:
            logging.info(bad_pairs)
            img_arr = np.concatenate(
                [
                     physics.render(camera_id=i, height=400, width=400,) for i in range(3)
                ]
                , axis=1
            )
            plt.imshow(img_arr)
            plt.show()
            
            qpos_str = " ".join(physics.data.qpos.astype(str))
            logging.info(f"<key name='rrt_check' qpos='{qpos_str}'/>")
           
        return bad_pairs
    
    def check_relative_pose(
        self, 
        qpos: Optional[np.ndarray] = None,
        physics = None,
    ):  
        # get ee poses from qpos?
        poses_dict = self.forward_kinematics_all(q=qpos, physics=physics, return_ee_pose=True) # {robotA: Pose1, robotB: Pose1}
        if poses_dict is None or "Alice" not in poses_dict or "Bob" not in poses_dict:
            return True
        alice_quat = np.array([7.07106781e-01, 1.73613722e-16, 1.69292055e-16, 7.07106781e-01])
        bob_quat = np.array([7.07106781e-01, 1.73613722e-16, 1.69292055e-16, 7.07106781e-01])
        rot_align = np.allclose(alice_quat, poses_dict['Alice'].orientation) and \
            np.allclose(bob_quat, poses_dict['Bob'].orientation)

        dist = np.linalg.norm(poses_dict["Alice"].position - poses_dict["Bob"].position)
        dist_align = 0.1 <= dist <= 0.4
        # print("===== dist", dist, dist_align)
        # print("===== rot_align", rot_align,  poses_dict['Alice'].orientation, poses_dict['Bob'].orientation)
        return 1 and dist_align
             

    def check_collision(
        self,
        robot_qpos: Optional[np.ndarray] = None,
        physics = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        verbose: bool = False,
        check_relative_pose: bool = False,
        show: bool = False,
    ) -> bool: 
        
        if check_relative_pose:
            passed = self.check_relative_pose(qpos=robot_qpos, physics=physics)
            if not passed:
                return True
        collided_links = self.get_collided_links(
            qpos=robot_qpos, 
            physics=physics,
            allow_grasp=allow_grasp,           
            check_grasp_ids=check_grasp_ids,
            verbose=verbose,
            show=show,
        ) 
        # if len(collided_links) > 0: 
        #     print("collided_link_ids", collided_links)
        #     for link_id in collided_links:
        #         link_name = self.physics.model.body(link_id).name
        #         print("collided_link_name", link_name)
        #     return True
        # print("collided_link_ids", collided_links)
        # for i in collided_links:
        #     print(self.physics.model.body(i).name)
        # if len(collided_links) > 0: 
        #     physics.named.data.qpos[self.all_joint_names] = robot_qpos
        #     physics.forward()
        #     img = physics.render(camera_id='teaser', height=400, width=400,)
        #     plt.imshow(img)
        #     plt.show()
        #     breakpoint()
        bad = len(collided_links) > 0
        
        # breakpoint()
        return bad 

    def is_state_valid(
        self,
        q: np.ndarray,
        physics=None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        check_relative_pose: bool = False,
        show: bool = False,
    ) -> bool:
        if physics is None:
            physics = self.physics
        return not self.check_collision(
            robot_qpos=q,
            physics=physics,
            allow_grasp=allow_grasp,
            check_grasp_ids=check_grasp_ids,
            check_relative_pose=check_relative_pose,
            show=show,
        )

    def is_motion_valid(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        physics=None,
        resolution: float = 0.006,
        **kwargs,
    ) -> bool:
        for q in self.extend_ee_l2(q1, q2, resolution=resolution, physics=physics):
            if not self.is_state_valid(q, physics=physics, **kwargs):
                return False
        return True

    def normalize_joint_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos)
        if len(qpos) == len(self.all_joint_idxs_in_qpos):
            return qpos
        return qpos[self.all_joint_idxs_in_qpos]

    def parse_rrt_info(self, info: str) -> Tuple[str, float, int]:
        try:
            duration = float(info.split("time")[1].split("_")[0])
            iteration = int(info.split("iter")[1].split("_")[0])
            reason = info.split("Reason")[1].split("_")[0]
            return reason, duration, iteration
        except Exception:
            logging.warning(f"Failed to parse RRT info string: {info}")
            return "Unknown", 0.0, 0

    def get_active_robot_names(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        threshold: float = 1e-4,
    ) -> List[str]:
        active = []
        for robot_name, joint_slice in self.robot_joint_slices.items():
            joint_delta = np.abs(goal_qpos[joint_slice] - start_qpos[joint_slice])
            if len(joint_delta) > 0 and np.max(joint_delta) > threshold:
                active.append(robot_name)
        return active

    def plan_active_subset(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        active_robot_names: List[str],
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False,
        physics=None,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        planning_physics = self.physics if physics is None else physics
        start_qpos = self.normalize_joint_qpos(start_qpos)
        goal_qpos = self.normalize_joint_qpos(goal_qpos)

        active_positions = set()
        for robot_name in active_robot_names:
            joint_slice = self.robot_joint_slices[robot_name]
            active_positions.update(range(joint_slice.start, joint_slice.stop))

        min_values = self.joint_minmax[:, 0].copy()
        max_values = self.joint_minmax[:, 1].copy()
        for idx in range(len(start_qpos)):
            if idx not in active_positions:
                min_values[idx] = start_qpos[idx]
                max_values[idx] = start_qpos[idx]

        def collision_fn(q: np.ndarray, show: bool = False):
            return self.check_collision(
                robot_qpos=q,
                physics=planning_physics,
                allow_grasp=allow_grasp,
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
                show=show,
            )

        def motion_validator(q1: np.ndarray, q2: np.ndarray):
            return self.is_motion_valid(
                q1,
                q2,
                physics=planning_physics,
                allow_grasp=allow_grasp,
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
            )

        def distance_fn(q1: np.ndarray, q2: np.ndarray):
            return self.ee_l2_distance(q1, q2, physics=planning_physics)

        def extend_fn(q1: np.ndarray, q2: np.ndarray):
            return self.extend_ee_l2(q1, q2, physics=planning_physics)

        if not skip_endpoint_collision_check:
            if collision_fn(start_qpos, show=0):
                return None, "ReasonCollisionAtStart_time0_iter0"
            if collision_fn(goal_qpos, show=0):
                return None, "ReasonCollisionAtGoal_time0_iter0"

        path, info = birrt(
            start_conf=start_qpos,
            goal_conf=goal_qpos,
            distance_fn=distance_fn,
            sample_fn=CenterWaypointsUniformSampler(
                bias=0.05,
                start_conf=start_qpos,
                goal_conf=goal_qpos,
                numpy_random=self.np_random,
                min_values=min_values,
                max_values=max_values,
                init_samples=[],
            ),
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            iterations=500,
            smooth_iterations=80,
            timeout=timeout,
            greedy=True,
            np_random=self.np_random,
            smooth_extend_fn=extend_fn,
            skip_direct_path=skip_direct_path,
            skip_smooth_path=skip_smooth_path,
            motion_validator=motion_validator,
        )
        if path is None:
            return None, f"RRT failed: {info}"
        return path, f"RRT succeeded: {info}"

    def plan_prioritized(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False,
        physics=None,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        planning_physics = self.physics if physics is None else physics
        if check_relative_pose:
            return None, "ReasonPrioritizedSkippedCoupledConstraint_time0_iter0"
        start_qpos = self.normalize_joint_qpos(start_qpos)
        goal_qpos = self.normalize_joint_qpos(goal_qpos)

        active_robot_names = self.get_active_robot_names(start_qpos, goal_qpos)
        if len(active_robot_names) == 0:
            return [start_qpos], "ReasonPrioritizedNoMotion_time0_iter0"

        def robot_priority(robot_name: str):
            joint_slice = self.robot_joint_slices[robot_name]
            delta = np.max(np.abs(goal_qpos[joint_slice] - start_qpos[joint_slice]))
            has_object = len(check_grasp_ids.get(robot_name, [])) > 0 if check_grasp_ids else False
            return (0 if has_object else 1, -float(delta))

        ordered_robot_names = sorted(active_robot_names, key=robot_priority)
        current = start_qpos.copy()
        all_paths = []
        start_time = time()

        for robot_name in ordered_robot_names:
            segment_goal = current.copy()
            joint_slice = self.robot_joint_slices[robot_name]
            segment_goal[joint_slice] = goal_qpos[joint_slice]
            remaining_timeout = max(1e-6, timeout - (time() - start_time))
            segment_path, info = self.plan_active_subset(
                start_qpos=current,
                goal_qpos=segment_goal,
                active_robot_names=[robot_name],
                allow_grasp=allow_grasp,
                check_grasp_ids=check_grasp_ids,
                skip_endpoint_collision_check=skip_endpoint_collision_check,
                skip_direct_path=skip_direct_path,
                skip_smooth_path=skip_smooth_path,
                timeout=remaining_timeout,
                check_relative_pose=check_relative_pose,
                physics=planning_physics,
            )
            if segment_path is None:
                duration = float(time() - start_time)
                return None, f"ReasonPrioritizedFailed_{robot_name}_time{duration}_iter0_{info}"
            if len(all_paths) > 0:
                all_paths.extend(segment_path[1:])
            else:
                all_paths.extend(segment_path)
            current = segment_goal

        duration = float(time() - start_time)
        return all_paths, f"ReasonPrioritizedSuccess_time{duration}_iter{len(ordered_robot_names)}"


    
    def plan(
        self, 
        start_qpos: np.ndarray,  # can be either full length or just the desired qpos for the joints 
        goal_qpos: np.ndarray,
        init_samples: Optional[List[np.ndarray]] = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False,
        physics=None,
    ) -> Tuple[Optional[List[np.ndarray]], str]:

        planning_physics = self.physics if physics is None else physics
        start_qpos = self.normalize_joint_qpos(start_qpos)
        goal_qpos = self.normalize_joint_qpos(goal_qpos)
        if len(start_qpos) != len(goal_qpos):
            return None, "RRT failed: start and goal configs have different lengths."
  
        def collision_fn(q: np.ndarray, show: bool = False):
            return self.check_collision(
                robot_qpos=q,
                physics=planning_physics,
                allow_grasp=allow_grasp,           
                check_grasp_ids=check_grasp_ids,  
                check_relative_pose=check_relative_pose,
                show=show,
                # detect_grasp=False, TODO?
            )
        def motion_validator(q1: np.ndarray, q2: np.ndarray):
            return self.is_motion_valid(
                q1,
                q2,
                physics=planning_physics,
                allow_grasp=allow_grasp,
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
            )
        def distance_fn(q1: np.ndarray, q2: np.ndarray):
            return self.ee_l2_distance(q1, q2, physics=planning_physics)
        def extend_fn(q1: np.ndarray, q2: np.ndarray):
            return self.extend_ee_l2(q1, q2, physics=planning_physics)
        if not skip_endpoint_collision_check:
            if collision_fn(start_qpos, show=0):
                # print("RRT failed: start qpos in collision.")
                return None, f"ReasonCollisionAtStart_time0_iter0"
            elif collision_fn(goal_qpos, show=0): 
                # print("RRT failed: goal qpos in collision.")
                return None, "ReasonCollisionAtGoal_time0_iter0"
        paths, info = birrt(
                start_conf=start_qpos,
                goal_conf=goal_qpos,
                distance_fn=distance_fn,
                sample_fn=CenterWaypointsUniformSampler(
                    bias=0.05,
                    start_conf=start_qpos,
                    goal_conf=goal_qpos,
                    numpy_random=self.np_random,
                    min_values=self.joint_minmax[:, 0],
                    max_values=self.joint_minmax[:, 1],
                    init_samples=init_samples,
                ),
                extend_fn=extend_fn,
                collision_fn=collision_fn,
                iterations=800,
                smooth_iterations=200,
                timeout=timeout,
                greedy=True,
                np_random=self.np_random,
                smooth_extend_fn=extend_fn,
                skip_direct_path=skip_direct_path,
                skip_smooth_path=skip_smooth_path, # enable to make sure it passes through the valid init_samples 
                motion_validator=motion_validator,
            )
        if paths is None:
            return None, f"RRT failed: {info}"
        return paths, f"RRT succeeded: {info}"
 
    def plan_splitted(
        self, 
        start_qpos: np.ndarray,  # can be either full length or just the desired qpos for the joints 
        goal_qpos: np.ndarray,
        init_samples: Optional[List[np.ndarray]] = None,
        allow_grasp: bool = False,
        check_grasp_ids: Optional[Dict[str, int]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False,
        physics=None,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
       
        planning_physics = self.physics if physics is None else physics
        start_qpos = self.normalize_joint_qpos(start_qpos)
        goal_qpos = self.normalize_joint_qpos(goal_qpos)
        init_samples = [] if init_samples is None else [
            self.normalize_joint_qpos(qpos) for qpos in init_samples
        ]
        all_paths, all_info = [], []
        duration = 0 
        iteration = 0
        original_waypoint_count = len(init_samples)
        waypoint_reason = ""
        def collision_fn(q: np.ndarray, show: bool = False):
            return self.check_collision(
                robot_qpos=q,
                physics=planning_physics,
                allow_grasp=allow_grasp,           
                check_grasp_ids=check_grasp_ids,  
                check_relative_pose=check_relative_pose,
                show=show,
                # detect_grasp=False, TODO?
            )
        def motion_validator(q1: np.ndarray, q2: np.ndarray):
            return self.is_motion_valid(
                q1,
                q2,
                physics=planning_physics,
                allow_grasp=allow_grasp,
                check_grasp_ids=check_grasp_ids,
                check_relative_pose=check_relative_pose,
            )
        def distance_fn(q1: np.ndarray, q2: np.ndarray):
            return self.ee_l2_distance(q1, q2, physics=planning_physics)
        def extend_fn(q1: np.ndarray, q2: np.ndarray):
            return self.extend_ee_l2(q1, q2, physics=planning_physics)
        
        if not skip_endpoint_collision_check:
            if collision_fn(goal_qpos, show=0): 
                logging.info("RRT failed: goal qpos in collision.")
                return None, "ReasonCollisionAtGoal_time0_iter0"
            
            valid_init_samples = []
            invalid_waypoint_idxs = []
            for i, interm_goal_qpos in enumerate(init_samples):
                if not collision_fn(interm_goal_qpos, show=0): 
                    valid_init_samples.append(interm_goal_qpos)
                else:
                    invalid_waypoint_idxs.append(i)
                # return None, "RRT failed: goal qpos in collision."
                # omit this waypoint and try planning with pruned init_sample 
            logging.debug(f"Given waypoints: {len(init_samples)}, valid: {len(valid_init_samples)} points")
            init_samples = valid_init_samples
            if original_waypoint_count > 0 and len(init_samples) == 0:
                return None, f"ReasonAllWaypointsInvalid_total{original_waypoint_count}_time0_iter0"
            if len(invalid_waypoint_idxs) > 0:
                waypoint_reason = (
                    f"_PrunedInvalidWaypoints_valid{len(init_samples)}"
                    f"_total{original_waypoint_count}"
                )

        has_mandatory_waypoints = len(init_samples) > 0
        # If valid LLM/procedural waypoints exist, treat them as mandatory
        # segment goals. A global direct path can skip lift/corridor semantics.
        if not skip_direct_path and not has_mandatory_waypoints:
            start_time = time()
            path = direct_path(
                start_qpos,
                goal_qpos,
                extend_fn,
                collision_fn,
                motion_validator=motion_validator,
            )
            if path is not None:
                return path, f"ReasonDirect_time{time() - start_time}_iter1"

        for i, interm_goal_qpos in enumerate(init_samples[::-1] + [goal_qpos]):
            interm_start_qpos = start_qpos if i == 0 else init_samples[::-1][i-1]
            logging.debug(f"planning interm_start_qpos {i}")
            if len(interm_start_qpos) != len(interm_goal_qpos):
                return None, "RRT failed: start and goal configs have different lengths."
            if len(interm_start_qpos) != len(self.all_joint_idxs_in_qpos):
                interm_start_qpos = interm_start_qpos[self.all_joint_idxs_in_qpos]
            if len(interm_goal_qpos) != len(self.all_joint_idxs_in_qpos):
                interm_goal_qpos = interm_goal_qpos[self.all_joint_idxs_in_qpos]
  
        
            if not skip_endpoint_collision_check:
                if collision_fn(interm_start_qpos):
                    return None, f"ReasonCollisionAtStart_time0_iter0"
                elif collision_fn(interm_goal_qpos): 
                    return None, f"ReasonCollisionAtGoal_time0_iter0"
                    
            segment_timeout = max(1e-6, timeout - duration)
            # Keep smoothing centralized: segment-level below for mandatory
            # waypoints, global smoothing below for ordinary split planning.
            segment_skip_smooth = True
            paths, info = birrt(
                    start_conf=interm_start_qpos,
                    goal_conf=interm_goal_qpos,
                    distance_fn=distance_fn,
                    sample_fn=CenterWaypointsUniformSampler(
                        bias=0.05,
                        start_conf=interm_start_qpos,
                        goal_conf=interm_goal_qpos,
                        numpy_random=self.np_random,
                        min_values=self.joint_minmax[:, 0],
                        max_values=self.joint_minmax[:, 1],
                        init_samples=[],
                    ),
                    extend_fn=extend_fn,
                    collision_fn=collision_fn,
                    iterations=800,
                    smooth_iterations=200,
                    timeout=segment_timeout,
                    greedy=True,
                    np_random=self.np_random,
                    smooth_extend_fn=extend_fn,
                    skip_direct_path=skip_direct_path,
                    skip_smooth_path=segment_skip_smooth,
                    motion_validator=motion_validator,
                ) 
            reason, sub_duration, sub_iteration = self.parse_rrt_info(info)
                
            if paths is None: 
                return None, f"Reason{reason}_time{sub_duration}_iter{sub_iteration}{waypoint_reason}"
            if has_mandatory_waypoints and not skip_smooth_path:
                paths = smooth_path(
                    path=paths,
                    extend_fn=extend_fn,
                    collision_fn=collision_fn,
                    np_random=self.np_random,
                    iterations=50,
                    motion_validator=motion_validator,
                )
                info += "_segment_smoothed"
            if len(all_paths) > 0:
                all_paths.extend(paths[1:])
            else:
                all_paths.extend(paths)
            all_info.append(info)
            duration += sub_duration
            iteration += sub_iteration
        
        if skip_smooth_path or has_mandatory_waypoints:
            return all_paths, f"ReasonSuccess_time{duration}_iter{iteration}{waypoint_reason}"
        
        logging.debug('begin smoothing')
        smoothed_paths = smooth_path(
            path=all_paths,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            np_random=self.np_random,
            iterations=50,
            motion_validator=motion_validator,
        )
        logging.debug('done smoothing')
        return smoothed_paths, f"ReasonSmoothed_time{duration}_iter{iteration}{waypoint_reason}"
