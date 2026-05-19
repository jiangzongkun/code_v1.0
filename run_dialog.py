import os
import pickle
import json
import numpy as np
import logging
import time
import threading
from datetime import datetime
from glob import glob
from natsort import natsorted
from copy import deepcopy
import argparse
from typing import List, Tuple, Dict, Union, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt

from rocobench.envs import SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask, MakeSandwichTask, PackGroceryTask, MujocoSimEnv, SimRobot, visualize_voxel_scene
from rocobench import PlannedPathPolicy, LLMPathPlan, MultiArmRRT
from prompting import LLMResponseParser, FeedbackManager, DialogPrompter, SingleThreadPrompter, save_episode_html

# print out logging.info
logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)

TASK_NAME_MAP = {
    "sort": SortOneBlockTask,
    "cabinet": CabinetTask,
    "rope": MoveRopeTask,
    "sweep": SweepTask,
    "sandwich": MakeSandwichTask,
    "pack": PackGroceryTask,
}


class RunMonitor:
    """Lightweight progress heartbeat for long-running task runs."""

    def __init__(self, enabled: bool, interval: float, log_path: Optional[str] = None):
        self.enabled = enabled and interval > 0
        self.interval = max(float(interval), 1.0) if self.enabled else 0.0
        self.log_path = log_path
        self.run_id = None
        self.phase = "not_started"
        self.details: Dict[str, Any] = {}
        self.start_time = 0.0
        self.phase_start_time = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def start(self, run_id: int):
        if not self.enabled:
            return
        now = time.time()
        with self._state_lock:
            self.run_id = run_id
            self.start_time = now
            self.phase_start_time = now
            self.phase = "started"
            self.details = {}
        self._emit("started")
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def update(self, phase: str, **details):
        if not self.enabled:
            return
        now = time.time()
        with self._state_lock:
            self.phase = phase
            self.details = {key: value for key, value in details.items() if value is not None}
            self.phase_start_time = now
        self._emit("phase")

    def finish(self, **details):
        if not self.enabled:
            return
        self.update("finished", **details)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _heartbeat_loop(self):
        while not self._stop_event.wait(self.interval):
            self._emit("heartbeat")

    def _emit(self, event: str):
        now = time.time()
        with self._state_lock:
            elapsed = now - self.start_time if self.start_time else 0.0
            phase_elapsed = now - self.phase_start_time if self.phase_start_time else 0.0
            detail_text = self._format_details(self.details)
            line = (
                f"[RunMonitor {datetime.now().strftime('%H:%M:%S')}] "
                f"event={event} run={self.run_id} phase={self.phase} "
                f"elapsed={elapsed:.1f}s phase_elapsed={phase_elapsed:.1f}s"
            )
            if detail_text:
                line += f" {detail_text}"
        with self._write_lock:
            print(line, flush=True)
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")

    @staticmethod
    def _format_details(details: Dict[str, Any]) -> str:
        if not details:
            return ""
        parts = []
        for key, value in details.items():
            text = str(value).replace("\n", " ")
            if len(text) > 160:
                text = text[:157] + "..."
            parts.append(f"{key}={text}")
        return " ".join(parts)


class LLMRunner:
    def __init__(
        self,
        env: MujocoSimEnv,
        robots: Dict[str, SimRobot],
        max_runner_steps: int = 50,
        video_format: str = "mp4",
        num_runs: int = 1,
        verbose: bool =False,
        np_seed: int = 0,
        start_seed: int = 0,
        run_name: str = "run",
        data_dir: str = "data",
        overwrite: bool = False,
        llm_output_mode="action_only", # "action_only" or "action_and_path"
        llm_comm_mode="chat",
        llm_num_replans=1,
        give_env_feedback=True,
        skip_display=False,
        skip_video=False,
        policy_kwargs: Dict[str, Any] = dict(control_freq=50),
        direct_waypoints: int = 0,
        max_failed_waypoints: int = 0,
        debug_mode: bool = False,
        split_parsed_plans: bool = False,
        use_history: bool = False,
        use_feedback: bool = False,
        temperature: float = 0.0,
        llm_source: str = "gpt4",
        run_timeout: float = 600,  # Default 10 minutes timeout
        ioa_preset: str = "full",
        monitor_interval: float = 0.0,
        fallback_first: bool = False,
        ):
        self.env = env
        self.env.reset()
        self.robots = robots
        self.run_timeout = run_timeout
        self.monitor_interval = monitor_interval
        self.robot_agent_names = list(robots.keys()) # ['Alice', etc.]
        self.data_dir = data_dir
        self.run_name = run_name
        run_dir = os.path.join(self.data_dir, self.run_name)
        os.makedirs(run_dir, exist_ok=overwrite)
        self.run_dir = run_dir
        self.verbose = verbose
        self.np_seed = np_seed
        self.start_seed = start_seed
        self.num_runs = num_runs
        self.overwrite = overwrite
        self.direct_waypoints = direct_waypoints
        self.max_failed_waypoints = max_failed_waypoints
        self.max_runner_steps = max_runner_steps
        self.give_env_feedback = give_env_feedback
        self.use_history = use_history
        self.use_feedback = use_feedback

        self.llm_output_mode = llm_output_mode
        self.debug_mode = debug_mode # useful for debug


        self.llm_num_replans = llm_num_replans
        self.llm_comm_mode = llm_comm_mode
        self.response_keywords = ['NAME', 'ACTION']
        if llm_output_mode == "action_and_path":
            self.response_keywords.append('PATH')
        self.planner = MultiArmRRT(
            self.env.physics,
            robots=robots,
            graspable_object_names=self.env.get_graspable_objects(),
            allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
        )
        self.policy_kwargs = policy_kwargs
        self.video_format = video_format
        self.skip_display = skip_display
        self.skip_video = skip_video
        self.save_video = not skip_video
        if hasattr(self.env, "record_video"):
            self.env.record_video = self.save_video
        self.split_parsed_plans = split_parsed_plans
        self.temperature = temperature
        self.fallback_first = fallback_first
        self.parser = LLMResponseParser(
            self.env,
            llm_output_mode,
            self.env.robot_name_map,
            self.response_keywords,
            self.direct_waypoints,
            use_prepick=self.env.use_prepick,
            use_preplace=self.env.use_preplace, # NOTE: should be custom defined in each task env
            split_parsed_plans=False, # self.split_parsed_plans,
        )
        self.feedback_manager = FeedbackManager(
            env=self.env,
            planner=self.planner,
            llm_output_mode=self.llm_output_mode,
            robot_name_map=self.env.robot_name_map,
            step_std_threshold=self.env.waypoint_std_threshold,
            max_failed_waypoints=self.max_failed_waypoints,
        )
        if llm_comm_mode in ["plan", "chat"]:
            logging.warning(f'Using SingleThreadPrompter for {llm_comm_mode} mode')
            self.prompter = SingleThreadPrompter(
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=1024,
                debug_mode=self.debug_mode,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                num_replans=self.llm_num_replans,
                comm_mode=llm_comm_mode,
                temperature=self.temperature,
                llm_source=llm_source,
                fallback_first=self.fallback_first,
            )

        elif llm_comm_mode == "ioa_lite":
            logging.warning(f'Using IoaLitePrompter for {llm_comm_mode} mode with preset={ioa_preset}')
            from prompting.ioa_lite_prompter import IoaLitePrompter

            self.prompter = IoaLitePrompter(
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=1024,
                debug_mode=self.debug_mode,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                num_replans=self.llm_num_replans,
                comm_mode="plan",
                temperature=self.temperature,
                llm_source=llm_source,
                fallback_first=self.fallback_first,
                ioa_preset=ioa_preset,
            )

        elif llm_comm_mode == "dialog":
            self.prompter = DialogPrompter(
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=512,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
            )
        else:
            raise ValueError(f"Unsupported llm_comm_mode: {llm_comm_mode}")


    def display_plan(self, plan: LLMPathPlan, save_name = "vis_plan", save_dir = None):
        """ Display the plan in the open3d viewer """ 
        env = deepcopy(self.env)
        env.physics.data.qpos[:] = self.env.physics.data.qpos[:].copy()
        env.physics.forward()
        env.render_point_cloud = True
        obs = env.get_obs()
        path_ls = plan.path_3d_list
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{save_name}.jpg")
        visualize_voxel_scene(
            obs.scene,
            path_pts=path_ls,
            save_img=(save_dir is not None),
            img_path=save_path
            )
        

    def one_run(self, run_id: int = 0, start_step: int = 0, skip_reset = False, prev_llm_plans = [], prev_response = None, prev_actions = None):
        """ uses planner """
        # Record start time for timeout detection
        run_start_time = time.time()
        save_dir = os.path.join(self.run_dir, f"run_{run_id}")
        os.makedirs(save_dir, exist_ok=self.overwrite)
        monitor = RunMonitor(
            enabled=self.monitor_interval > 0,
            interval=self.monitor_interval,
            log_path=os.path.join(save_dir, "run_monitor.log"),
        )
        monitor.start(run_id)
        monitor.update("seed_and_reset", start_step=start_step, skip_reset=skip_reset)
        
        self.env.seed(np_seed=run_id)
        if not skip_reset:
            self.env.reset(reload=True) # NOTE: need to do this to reset the model.eq_active vals
        env = self.env
        physics = env.physics
        success = False

        done = False
        reward = 0
        monitor.update("get_initial_observation")
        obs = env.get_obs()
        timed_out = False
        
        for step in range(start_step, start_step + self.max_runner_steps):
            # Check if timeout exceeded
            elapsed_time = time.time() - run_start_time
            if elapsed_time > self.run_timeout:
                print(f"Run {run_id} TIMED OUT after {elapsed_time:.2f} seconds (limit: {self.run_timeout}s)")
                timed_out = True
                break

            monitor.update("step_start", step=step, elapsed=f"{elapsed_time:.1f}s")
            step_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=self.overwrite)
            prompt_path = os.path.join(step_dir, "prompts")
            os.makedirs(prompt_path, exist_ok=self.overwrite)

            monitor.update("save_initial_state", step=step)
            sim_data = env.save_intermediate_state()
            data_fname = f"{step_dir}/env_init.pkl"
            with open(data_fname, "wb") as f:
                pickle.dump(sim_data, f)


            if step == start_step and len(prev_llm_plans) > 0:
                ready_to_execute = 1
                current_llm_plan = prev_llm_plans
                response = ""
                prompt_breakdown = dict()

            elif step == start_step and prev_actions is not None:
                ready_to_execute = 1
                current_llm_plan = prev_llm_plans
                response = ""
                prompt_breakdown = dict()

            else:
                monitor.update(
                    "llm_prompt_start",
                    step=step,
                    comm_mode=self.llm_comm_mode,
                    replans=self.llm_num_replans,
                )
                try:
                    ready_to_execute, current_llm_plan, response, prompt_breakdown = self.prompter.prompt_one_round(
                        obs,
                        save_path=prompt_path,
                        # prev_response=(prev_response['response'] if step == start_step and prev_response is not None else None)
                        )
                except Exception as exc:
                    monitor.update("llm_prompt_error", step=step, error=repr(exc))
                    monitor.finish(success=False, timed_out=False, step=step, error=repr(exc))
                    raise
                monitor.update(
                    "llm_prompt_done",
                    step=step,
                    ready_to_execute=ready_to_execute,
                    plan_count=(len(current_llm_plan) if current_llm_plan is not None else 0),
                )
                if not ready_to_execute or current_llm_plan is None:
                    print(f"Run {run_id}: Step {step} failed to get a plan from LLM. Move on to next step.")
                    continue

                if not self.skip_display:
                    monitor.update("display_llm_plan_start", step=step, plan_count=len(current_llm_plan))
                    for i, plan in enumerate(current_llm_plan):
                        self.display_plan(plan, save_name=f"vis_llm_plan_{i}", save_dir=step_dir)
                    monitor.update("display_llm_plan_done", step=step)


                monitor.update("save_llm_plan_start", step=step, plan_count=len(current_llm_plan))
                for i, plan in enumerate(current_llm_plan):
                    save_fname = os.path.join(step_dir, f"llm_plan_{i}.pkl")
                    with open(save_fname, "wb") as f:
                        pickle.dump(plan, f)
                monitor.update("save_llm_plan_done", step=step)


            logging.info(f"Step: {step} LLM plan parsed, begin RRT planning ")
            # try execute this plan, if one of the plan failed, rewind the env to before the first plan was executed!
            rewind_env = False

            for i, plan in enumerate(current_llm_plan):
                print('tograsp:', plan.tograsp, 'inhand:', plan.inhand, plan.action_strs)
                monitor.update("rrt_policy_init", step=step, plan_index=i)
                policy = PlannedPathPolicy(
                    physics=env.physics,
                    robots=self.robots,
                    path_plan=plan,
                    graspable_object_names=self.env.get_graspable_objects(),
                    allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                    plan_splitted=self.split_parsed_plans,
                    **self.policy_kwargs,
                )

                num_sim_steps = 0
                if prev_actions is not None:
                    for sim_action in prev_actions:
                        # env.physics.model.eq_active[52:] = 0
                        # env.physics.forward() # DEBUG
                        obs, reward, done, info = env.step(sim_action, verbose=False)
                        num_sim_steps += 1
                else:
                    # breakpoint()
                    monitor.update("rrt_plan_start", step=step, plan_index=i)
                    plan_success, reason = policy.plan(env)
                    monitor.update(
                        "rrt_plan_done",
                        step=step,
                        plan_index=i,
                        plan_success=plan_success,
                        reason=reason,
                    )
                    logging.info(f"Stesp: {step} Plan success: {plan_success}, reason: {reason}")
                    if plan_success:
                        logging.info(f"Execute the plan for {len(policy.action_buffer)} steps")

                        monitor.update(
                            "save_rrt_plan_start",
                            step=step,
                            plan_index=i,
                            action_steps=len(policy.action_buffer),
                        )
                        plan_fname = os.path.join(step_dir, f"rrt_plan_{i}.pkl")
                        plans = policy.rrt_plan_results
                        with open(plan_fname, "wb") as f:
                            pickle.dump(plans, f)

                        actions_fname = f"{step_dir}/actions_{i}.pkl"
                        with open(actions_fname, "wb") as f:
                            pickle.dump(policy.action_buffer, f)
                        monitor.update("save_rrt_plan_done", step=step, plan_index=i)

                        while not policy.plan_exhausted:
                            sim_action = policy.act(obs, env.physics)
                            obs, reward, done, info = env.step(sim_action, verbose=False)
                            num_sim_steps += 1
                            if num_sim_steps == 1 or num_sim_steps % 250 == 0:
                                monitor.update(
                                    "execute_plan",
                                    step=step,
                                    plan_index=i,
                                    sim_steps=num_sim_steps,
                                    done=done,
                                    reward=reward,
                                )

                if num_sim_steps > 0 and self.save_video:
                    vid_name = f"{step_dir}/execute.mp4"
                    monitor.update("video_export_start", step=step, plan_index=i, sim_steps=num_sim_steps)
                    env.export_render_to_video(vid_name, out_type=self.video_format,  fps=50)
                    monitor.update("video_export_done", step=step, plan_index=i, video=vid_name)
                    print(f'Plans all executed! Video sample saved to {vid_name}')

                elif num_sim_steps > 0:
                    monitor.update("video_export_skipped", step=step, plan_index=i, sim_steps=num_sim_steps)
                    print(f"Plans all executed! Video export skipped for step {step}, plan {i}.")

                else:
                    print(f"Plan {i} failed to execute.")
                    rewind_env = True
                    break

            if rewind_env:
                print("Rewinding the environment to before the first plan was executed.")
                monitor.update("rewind_environment", step=step)
                env.load_saved_state(sim_data)

            else:
                monitor.update("save_end_state", step=step)
                sim_data = env.save_intermediate_state()

            data_fname = f"{step_dir}/env_end.pkl"
            with open(data_fname, "wb") as f:
                pickle.dump(sim_data, f)

            monitor.update("post_execute_update", step=step, execute_success=(not rewind_env))
            self.prompter.post_execute_update(
                obs_desp="", # TODO
                execute_success=(not rewind_env),
                parsed_plan=current_llm_plan[0].get_action_desp()
            )
            monitor.update("post_execute_update_done", step=step)

            if done:
                break

        # If timeout, force set as failure
        if timed_out:
            success = False
        else:
            success = reward > 0
        
        elapsed_time = time.time() - run_start_time
        json.dump(
            dict(step=step, success=success, timed_out=timed_out, elapsed_time=elapsed_time),
            open(f"{save_dir}/steps{step}_success_{success}.json", "w"),
        )
        
        if timed_out:
            print(f"Run {run_id} FAILED due to timeout after {elapsed_time:.2f}s")
        else:
            print("Run finished after {} timesteps in {:.2f}s".format(step, elapsed_time))
        monitor.update("post_episode_update", success=success, timed_out=timed_out, step=step)
        self.prompter.post_episode_update()
        monitor.update("save_episode_html", success=success, timed_out=timed_out, step=step)
        save_episode_html(
            save_dir,
            html_fname=f"steps{step}_success_{success}",
            video_fname="execute.mp4",
            include_video=self.save_video,
            sender_keys=["Alice", "Bob", "Chad", "Dave", "Planner", "Feedback", "Action"],
            )
        monitor.finish(success=success, timed_out=timed_out, step=step, elapsed=f"{elapsed_time:.2f}s")
        print(f"Episode html saved to {save_dir}")


    def run(self, args):
        start_id = 0 if args.start_id == -1 else args.start_id
        if args.cont:
            logging.info("Continuing from previous run")
            load_run = glob(os.path.join(self.data_dir, args.load_run_name, f"run_{args.load_run_id}"))
            if len(load_run) == 0:
                raise ValueError(f"Cannot find run {args.load_run_id} in {args.load_run_name}")
                exit()
            load_run = load_run[0]
            # find the latest steps
            step_dirs = natsorted(
                glob(os.path.join(load_run, "step_*"))
            )
            if len(step_dirs) == 0:
                raise ValueError(f"Cannot find any steps in {load_run}")
                exit()
            latest_step = step_dirs[-1]
            env_init_fname = os.path.join(latest_step, "env_init.pkl")
            with open(env_init_fname, "rb") as f:
                saved_data = pickle.load(f)
                self.env.load_saved_state(saved_data)

            print(f"==== Loading back Run {args.load_run_id} ====")
            next_step = int(latest_step.split("/")[-1].split("_")[-1])
            prev_llm_plans = []
            prev_plans = natsorted(
                    glob(os.path.join(latest_step, "llm_plan_*pkl"))
                    )
            if len(prev_plans) > 0:
                prev_llm_plans = [pickle.load(open(fname, "rb")) for fname in prev_plans]

            prev_response = None
            prev_responses = natsorted(
                    glob(os.path.join(latest_step, "prompts", "*response.json"))
                    )
            if len(prev_responses) > 0:
                prev_response = json.load(open(prev_responses[-1], "rb"))

            prev_actions = None
            fname = os.path.join(latest_step, "actions.pkl")
            if os.path.exists(fname):
                prev_actions = pickle.load(open(fname, "rb"))

            self.one_run(
                args.load_run_id,
                start_step=next_step,
                skip_reset=True,
                prev_llm_plans=prev_llm_plans,
                prev_response=prev_response,
                prev_actions=prev_actions
                )
            start_id = args.load_run_id + 1
        existing_runs = glob(os.path.join(self.data_dir, args.run_name, "run_*"))
        if args.start_id == -1 and len(existing_runs) > 0:
            existing_run_ids = [int(run.split("_")[-1]) for run in existing_runs]
            start_id = max(existing_run_ids) + 1
        for run_id in range(start_id, start_id + self.num_runs):
            print(f"==== Run {run_id} starts ====")
            self.one_run(run_id)

def main(args):
    assert args.task in TASK_NAME_MAP.keys(), f"Task {args.task} not supported"
    bootstrap_monitor = RunMonitor(
        enabled=getattr(args, "monitor_interval", 0.0) > 0,
        interval=getattr(args, "monitor_interval", 0.0),
        log_path=os.path.join(args.data_dir, args.run_name, "bootstrap_monitor.log"),
    )
    if bootstrap_monitor.log_path:
        os.makedirs(os.path.dirname(bootstrap_monitor.log_path), exist_ok=True)
    bootstrap_monitor.start(-1)
    bootstrap_monitor.update("main_start", task=args.task, comm_mode=args.comm_mode)
    env_cl = TASK_NAME_MAP[args.task]
    if args.task == 'rope':
        args.output_mode = 'action_and_path'
        args.split_parsed_plans = True
        logging.warning("MoveRopeTask requires split parsed plans\n")

        args.control_freq = 20
        args.max_failed_waypoints = 0
        logging.warning("MopeRope requires max failed waypoints 0\n")
        if not args.no_feedback:
            args.tsteps = 6
            logging.warning("MoveRope uses 6 tsteps to allow one recovery pick-place cycle\n")

    elif args.task == 'pack':
        args.output_mode = 'action_and_path'
        args.control_freq = 10
        args.split_parsed_plans = True
        args.max_failed_waypoints = 0
        args.direct_waypoints = 0
        args.tsteps = max(args.tsteps, 12)
        args.rrt_timeout = min(args.rrt_timeout, 30)
        logging.warning("PackGroceryTask uses split parsed plans, serial fallback, at least 12 tsteps, and 30s RRT timeout\n")

    render_freq = 600
    if args.control_freq == 15:
        render_freq = 1200
    elif args.control_freq == 10:
        render_freq = 2000
    elif args.control_freq == 5:
        render_freq = 3000
    bootstrap_monitor.update("create_env_start", task=args.task, render_freq=render_freq)
    env = env_cl(
        render_freq=render_freq,
        image_hw=(400,400),
        sim_forward_steps=300,
        error_freq=30,
        error_threshold=1e-5,
        randomize_init=True,
        render_point_cloud=0,
        record_video=(not args.skip_video),
        render_cameras=["face_panda","face_ur5e","teaser",],
        one_obj_each=True,
        np_seed=args.seed,
    )
    bootstrap_monitor.update("create_env_done", env=env.__class__.__name__)
    bootstrap_monitor.update("get_sim_robots_start")
    robots = env.get_sim_robots()
    bootstrap_monitor.update("get_sim_robots_done", robots=",".join(robots.keys()))
    if args.no_feedback:
        assert args.num_replans == 1, "no feedback mode requires num_replans=1 but longer -tsteps"


    # save args into a json file
    args_dict = vars(args)
    args_dict["env"] = env.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m_%H%M")
    fname = os.path.join(args.data_dir, args.run_name, f"args_{timestamp}.json")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    json.dump(args_dict, open(fname, "w"), indent=2)
    bootstrap_monitor.update("init_runner_start", data_dir=args.data_dir, run_name=args.run_name)
    runner = LLMRunner(
        env=env,
        data_dir=args.data_dir,
        robots=robots,
        max_runner_steps=args.tsteps,
        num_runs=args.num_runs,
        run_name=args.run_name,
        overwrite=True,
        skip_display=args.skip_display,
        skip_video=args.skip_video,
        llm_output_mode=args.output_mode, # "action_only" or "action_and_path"
        llm_comm_mode=args.comm_mode, # "chat" or "plan"
        llm_num_replans=args.num_replans,
        policy_kwargs=dict(
            control_freq=args.control_freq,
            use_weld=args.use_weld,
            skip_direct_path=0,
            skip_smooth_path=args.skip_smooth_path,
            check_relative_pose=args.rel_pose,
            timeout=args.rrt_timeout,
        ),
        direct_waypoints=args.direct_waypoints,
        max_failed_waypoints=args.max_failed_waypoints,
        debug_mode=args.debug_mode,
        split_parsed_plans=args.split_parsed_plans,
        use_history=(not args.no_history),
        use_feedback=(not args.no_feedback),
        temperature=args.temperature,
        llm_source=args.llm_source,
        run_timeout=args.run_timeout,
        ioa_preset=args.ioa_preset,
        monitor_interval=args.monitor_interval,
        fallback_first=args.fallback_first,
    )
    bootstrap_monitor.update("init_runner_done")
    bootstrap_monitor.update("runner_run_start")
    try:
        runner.run(args)
    except Exception as exc:
        bootstrap_monitor.update("runner_run_error", error=repr(exc))
        bootstrap_monitor.finish(success=False, error=repr(exc))
        raise
    bootstrap_monitor.finish(success=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--temperature", "-temp", type=float, default=0)
    parser.add_argument("--start_id", "-sid", type=int, default=-1)
    parser.add_argument("--num_runs", '-nruns', type=int, default=1)
    parser.add_argument("--run_name", "-rn", type=str, default="cabinet")
    parser.add_argument("--tsteps", "-t", type=int, default=10)
    parser.add_argument("--task", type=str, default="cabinet")
    parser.add_argument("--output_mode", type=str, default="action_only", choices=["action_only", "action_and_path"])
    parser.add_argument("--comm_mode", type=str, default="chat", choices=["chat", "plan", "dialog", "ioa_lite"])
    parser.add_argument(
        "--ioa_preset",
        type=str,
        default="full",
        choices=["schema", "task_table", "feedback", "fsm", "dynamic", "full"],
        help="IoA-Lite feature preset; only used when --comm_mode ioa_lite",
    )
    parser.add_argument("--control_freq", "-cf", type=int, default=15)
    parser.add_argument("--skip_display", "-sd", action="store_true")
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip camera recording and video export for faster evaluation runs.",
    )
    parser.add_argument("--direct_waypoints", "-dw", type=int, default=5)
    parser.add_argument("--num_replans", "-nr", type=int, default=5)
    parser.add_argument("--cont", "-c", action="store_true")
    parser.add_argument("--load_run_name", "-lr", type=str, default="sort_task")
    parser.add_argument("--load_run_id", "-ld", type=int, default=0)
    parser.add_argument("--max_failed_waypoints", "-max", type=int, default=1)
    parser.add_argument("--debug_mode", "-i", action="store_true")
    parser.add_argument("--use_weld", "-w", type=int, default=1)
    parser.add_argument("--rel_pose", "-rp", action="store_true")
    parser.add_argument("--split_parsed_plans", "-sp", action="store_true")
    parser.add_argument("--no_history", "-nh", action="store_true")
    parser.add_argument("--no_feedback", "-nf", action="store_true")
    parser.add_argument("--llm_source", "-llm", type=str, default="llama3.3:latest") # You can choose one model here.
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument("--run_timeout", "-rt", type=float, default=600, help="Timeout for each run in seconds (default: 600s = 10min)")
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=0.0,
        help="Print and save run heartbeat logs every N seconds; 0 disables monitoring.",
    )
    parser.add_argument("--rrt_timeout", type=int, default=200, help="RRT timeout for each planning segment in iterations")
    parser.add_argument("--skip_smooth_path", action="store_true", help="Skip RRT path smoothing to speed up evaluation")
    parser.add_argument("--fallback_first", action="store_true", help="Try deterministic task fallback before querying the LLM")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    main(args)
