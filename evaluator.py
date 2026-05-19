import subprocess
import os
import sys
from glob import glob
import json
import time
import argparse
import re
from datetime import datetime
from utils import Colors, log_subprocess_result, next_task_log_dir, relpath, terminal_log, to_text, write_run_summary, write_task_summary

# Default timeout settings (seconds) for each task, can be customized per task
DEFAULT_RUN_TIMEOUTS = {
    "sort": 600,
    "cabinet": 600,
    "rope": 600,
    "sweep": 600,
    "sandwich": 600,
    "pack": 600,
}

DEFAULT_TASKS = ["sort", "cabinet", "rope", "sweep", "sandwich", "pack"]
RESULT_JSON_RE = re.compile(r"^steps(?P<step>\d+)_success_(?P<success>True|False)\.json$")


def find_run_result_json(run_dir: str) -> str:
    """Return the final per-run result json written by run_dialog.py."""
    candidates = []
    for path in glob(os.path.join(run_dir, "steps*_success_*.json")):
        match = RESULT_JSON_RE.match(os.path.basename(path))
        if not match:
            continue
        candidates.append((int(match.group("step")), os.path.getmtime(path), path))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def test_run_dialog(
    task: str,
    num_runs: int,
    output_dir: str,
    seed: int = 0,
    run_timeout: float = None,
    comm_mode: str = "ioa_lite",
    ioa_preset: str = "full",
    skip_video: bool = True,
    monitor_interval: float = 30.0,
    num_replans: int = 5,
    tsteps: int = 10,
    llm_source: str = None,
    fallback_first: bool = False,
    rrt_timeout: int = 200,
    skip_smooth_path: bool = False,
):
    """
    Test and run dialog tasks
    
    Args:
        task: Task name
        num_runs: Number of runs
        output_dir: Output directory
        seed: Random seed
        run_timeout: Timeout for single run (seconds). If None, use value from DEFAULT_RUN_TIMEOUTS, default 60s
        comm_mode: run_dialog communication mode. Defaults to the IoA-Lite scoring path.
        ioa_preset: IoA-Lite feature preset when comm_mode is ioa_lite.
        skip_video: Whether to skip per-step video export for speed.
        monitor_interval: Heartbeat interval passed through to run_dialog.
        num_replans: Number of LLM replans per step.
        tsteps: Maximum task steps per run.
        llm_source: Optional model name fallback; environment variables still take precedence in run_dialog.
        fallback_first: Whether to try deterministic fallback before querying the LLM.
        rrt_timeout: RRT timeout for each planning segment in iterations.
        skip_smooth_path: Whether to skip RRT path smoothing.
    """
    # If timeout not specified, use task default or global default of 60s
    if run_timeout is None:
        run_timeout = DEFAULT_RUN_TIMEOUTS.get(task, 60)
    
    print("\n" + Colors.CYAN + Colors.BOLD + f"▶ Starting Task: {task.upper()}" + Colors.ENDC)
    print(
        Colors.CYAN
        + (
            f"  Configuration: {num_runs} runs, {run_timeout}s timeout per run, "
            f"comm_mode={comm_mode}, ioa_preset={ioa_preset}, "
            f"num_replans={num_replans}, tsteps={tsteps}, skip_video={skip_video}, "
            f"fallback_first={fallback_first}, rrt_timeout={rrt_timeout}, "
            f"skip_smooth_path={skip_smooth_path}"
        )
        + Colors.ENDC
    )
    
    task_log_dir = next_task_log_dir(task)
    if task_log_dir is not None:
        task_output_dir = task_log_dir
        run_name = "runs"
        artifacts_dir = os.path.join(task_output_dir, run_name)
        os.makedirs(artifacts_dir, exist_ok=True)
        print(Colors.CYAN + f"  Task artifacts: {relpath(artifacts_dir)}" + Colors.ENDC)
    else:
        task_output_dir = output_dir
        run_name = task
        artifacts_dir = os.path.join(task_output_dir, run_name)

    # Record number of runs before execution, to only count runs from this execution
    existing_runs = glob(os.path.join(task_output_dir, run_name, 'run_*'))
    start_run_count = len(existing_runs)
    if start_run_count > 0:
        existing_run_ids = [int(run.split("_")[-1]) for run in existing_runs]
        next_run_id = max(existing_run_ids) + 1
    else:
        next_run_id = 0
    
    # Record start time
    start_time = time.time()
    
    # Calculate total timeout: number of runs * timeout per run, with some buffer (20%)
    total_timeout = num_runs * run_timeout * 1.2
    
    command = [
        sys.executable,
        'run_dialog.py',
        '--task', task,
        '--run_name', run_name,
        '--data_dir', task_output_dir,
        '--start_id', str(-1),
        '--num_runs', str(num_runs),
        '--skip_display',
        '--tsteps', str(tsteps),
        '--seed', str(seed),
        '--run_timeout', str(run_timeout),
        '--comm_mode', comm_mode,
        '--ioa_preset', ioa_preset,
        '--num_replans', str(num_replans),
        '--rrt_timeout', str(rrt_timeout),
    ]
    if skip_video:
        command.append('--skip_video')
    if fallback_first:
        command.append('--fallback_first')
    if skip_smooth_path:
        command.append('--skip_smooth_path')
    if monitor_interval and monitor_interval > 0:
        command.extend(['--monitor_interval', str(monitor_interval)])
    if llm_source:
        command.extend(['--llm_source', llm_source])

    subprocess_started_at = datetime.now()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=total_timeout)
    except subprocess.TimeoutExpired as e:
        print(Colors.RED + f"\n✗ Subprocess timed out after {total_timeout:.0f}s" + Colors.ENDC)
        result = subprocess.CompletedProcess(
            args=e.args,
            returncode=-1,
            stdout=to_text(e.stdout),
            stderr=f"Subprocess timeout after {total_timeout:.0f}s\n" + to_text(e.stderr)
        )
    subprocess_finished_at = datetime.now()

    task_log_dir = log_subprocess_result(
        task,
        result,
        command,
        started_at=subprocess_started_at,
        finished_at=subprocess_finished_at,
        task_dir=task_log_dir,
    )
    
    total_time = time.time() - start_time
    
    success_cnt = 0
    total_cnt = 0
    total_steps = 0
    timeout_cnt = 0
    result_jsons = {}

    # Only count runs from this execution (num_runs starting from next_run_id)
    current_run_ids = list(range(next_run_id, next_run_id + num_runs))
    
    for run_id in current_run_ids:
        run_dir = os.path.join(task_output_dir, run_name, f'run_{run_id}')
        if not os.path.exists(run_dir):
            print(f"Warning: Run {run_id} directory not found")
            continue
            
        json_dir = find_run_result_json(run_dir)
        if not json_dir:
            print(f"Warning: Run {run_id} has no final steps<N>_success_<bool>.json result file")
            continue
        result_jsons[run_id] = relpath(json_dir)

        with open(json_dir, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            success = json_data.get('success', False)
            steps = json_data.get('step', 0)
            timed_out = json_data.get('timed_out', False)
            
            if timed_out:
                timeout_cnt += 1
            
            if success:
                success_cnt += 1
                total_steps += steps
        
        total_cnt += 1

    if total_cnt < num_runs:
        print(Colors.YELLOW + f"Warning: Task {task} completed {total_cnt}/{num_runs} runs" + Colors.ENDC)

    print("\n" + Colors.BOLD + f"▶ RESULTS FOR TASK: {task.upper()}" + Colors.ENDC)
    
    # Success metrics
    success_pct = 100 * success_cnt / total_cnt if total_cnt > 0 else 0
    if success_pct >= 80:
        color = Colors.GREEN
    elif success_pct >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    print("Success Rate:  " + color + f"{success_cnt}/{total_cnt} ({success_pct:.1f}%)" + Colors.ENDC)
    
    # Timeout metrics
    if timeout_cnt > 0:
        print("Timeout Count: " + Colors.YELLOW + f"{timeout_cnt}/{total_cnt} ({100*timeout_cnt/total_cnt:.1f}%)" + Colors.ENDC)
    else:
        print(f"Timeout Count: {timeout_cnt}/{total_cnt}")
    
    # Steps metrics
    if success_cnt > 0:
        avg_steps = total_steps / success_cnt
        print("Average Steps: " + Colors.CYAN + f"{avg_steps:.2f}" + Colors.ENDC + " (successful runs only)")
    else:
        print("Average Steps: N/A (no successful runs)")
    
    # Time metrics
    print("Total Time:    " + Colors.BLUE + f"{total_time:.2f}s ({total_time/60:.2f} minutes)" + Colors.ENDC)
    
    # Return code
    if result.returncode == 0:
        print("Return Code:   " + Colors.GREEN + f"{result.returncode}" + Colors.ENDC + " (Success)")
    else:
        print("Return Code:   " + Colors.RED + f"{result.returncode}" + Colors.ENDC + " (Error)")
    
    # Error output if any
    if result.stderr and result.stderr.strip():
        print(Colors.YELLOW + "\nStderr Output:" + Colors.ENDC)
        stderr_lines = result.stderr[:500].split('\n')
        for line in stderr_lines[:5]:  # Show first 5 lines
            if line.strip():
                print(f"   {line}")
        if len(result.stderr) > 500:
            print("   ... (truncated)")
    
    print()
    
    summary = {
        'task': task,
        'success_rate': success_cnt / total_cnt if total_cnt > 0 else 0,
        'success_count': success_cnt,
        'total_count': total_cnt,
        'timeout_count': timeout_cnt,
        'avg_steps': total_steps / success_cnt if success_cnt > 0 else 0,
        'total_time': total_time,
        'returncode': result.returncode,
        'run_ids': current_run_ids,
        'result_jsons': result_jsons,
        'num_runs': num_runs,
        'seed': seed,
        'run_timeout': run_timeout,
        'comm_mode': comm_mode,
        'ioa_preset': ioa_preset,
        'skip_video': skip_video,
        'monitor_interval': monitor_interval,
        'num_replans': num_replans,
        'tsteps': tsteps,
        'fallback_first': fallback_first,
        'rrt_timeout': rrt_timeout,
        'skip_smooth_path': skip_smooth_path,
    }
    if task_log_dir is not None:
        summary['log_dir'] = relpath(task_log_dir)
        summary['artifacts_dir'] = relpath(artifacts_dir)
        summary['stdout_log'] = relpath(os.path.join(task_log_dir, "stdout.log"))
        summary['stderr_log'] = relpath(os.path.join(task_log_dir, "stderr.log"))

    write_task_summary(task_log_dir, summary)

    return summary

def parse_args():
    parser = argparse.ArgumentParser(description="Run RoCoBench evaluation batches.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--output_dir", "--data_dir", dest="output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_timeout",
        type=float,
        default=None,
        help="Timeout per run in seconds. Defaults to the task-specific table.",
    )
    parser.add_argument(
        "--comm_mode",
        type=str,
        default="ioa_lite",
        choices=["chat", "plan", "dialog", "ioa_lite"],
        help="Communication mode passed to run_dialog.py. Default is the IoA-Lite scoring path.",
    )
    parser.add_argument(
        "--ioa_preset",
        type=str,
        default="full",
        choices=["schema", "task_table", "feedback", "fsm", "dynamic", "full"],
        help="IoA-Lite feature preset passed to run_dialog.py.",
    )
    parser.add_argument("--num_replans", type=int, default=5)
    parser.add_argument("--tsteps", type=int, default=10)
    parser.add_argument("--monitor_interval", type=float, default=30.0)
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Export videos. By default evaluation skips video for speed.",
    )
    parser.add_argument(
        "--llm_source",
        type=str,
        default=None,
        help="Optional fallback model name passed to run_dialog.py.",
    )
    parser.add_argument("--fallback_first", action="store_true")
    parser.add_argument("--rrt_timeout", type=int, default=200)
    parser.add_argument("--skip_smooth_path", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import time as time_module

    args = parse_args()

    with terminal_log(args.output_dir):
        begin_time = time_module.time()

        results = []

        for task in args.tasks:
            results.append(
                test_run_dialog(
                    task,
                    args.num_runs,
                    args.output_dir,
                    seed=args.seed,
                    run_timeout=args.run_timeout,
                    comm_mode=args.comm_mode,
                    ioa_preset=args.ioa_preset,
                    skip_video=(not args.save_video),
                    monitor_interval=args.monitor_interval,
                    num_replans=args.num_replans,
                    tsteps=args.tsteps,
                    llm_source=args.llm_source,
                    fallback_first=args.fallback_first,
                    rrt_timeout=args.rrt_timeout,
                    skip_smooth_path=args.skip_smooth_path,
                )
            )

        end_time = time_module.time()
        total_elapsed = end_time - begin_time

        print("\n" + "=" * 80)
        print(Colors.BOLD + Colors.CYAN + " " * 30 + "FINAL SUMMARY" + Colors.ENDC)
        print("=" * 80)

        # Header
        print(Colors.BOLD + f"{'Task':<12} {'Success':<12} {'Rate':<10} {'Timeouts':<10} {'Avg Steps':<12} {'Time':<10}" + Colors.ENDC)
        print("─" * 80)

        # Results table
        for result in results:
            task_name = result['task']
            success_str = f"{result['success_count']}/{result['total_count']}"
            rate = result['success_rate'] * 100

            # Color code based on success rate
            if rate >= 80:
                rate_color = Colors.GREEN
            elif rate >= 50:
                rate_color = Colors.YELLOW
            else:
                rate_color = Colors.RED

            rate_str = f"{rate:.1f}%"

            # Color timeouts if any
            if result['timeout_count'] > 0:
                timeout_str = Colors.YELLOW + f"{result['timeout_count']}" + Colors.ENDC
                timeout_padding = " " * (10 - len(str(result['timeout_count'])))
            else:
                timeout_str = f"{result['timeout_count']}"
                timeout_padding = " " * (10 - len(str(result['timeout_count'])))

            avg_steps_str = f"{result['avg_steps']:.2f}" if result['avg_steps'] > 0 else "N/A"
            avg_steps_padding = " " * (12 - len(avg_steps_str))

            time_val = f"{result['total_time']:.1f}s"
            time_str = Colors.BLUE + time_val + Colors.ENDC
            time_padding = " " * (10 - len(time_val))

            print(f"{task_name:<12} "
                  f"{success_str:<12} "
                  f"{rate_color}{rate_str:<10}{Colors.ENDC} "
                  f"{timeout_str}{timeout_padding} "
                  f"{Colors.CYAN}{avg_steps_str}{Colors.ENDC}{avg_steps_padding} "
                  f"{time_str}{time_padding}")

        print("─" * 80)
        print(Colors.BOLD + "Total Execution Time: " + Colors.ENDC + Colors.BLUE + f"{total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)" + Colors.ENDC)
        print("=" * 80 + "\n")

        write_run_summary(results, total_elapsed)
