"""Benchmark a policy's per-step latency and optionally render it to video.

This measures the wall-clock cost of one control step (observation, policy
inference, motor command, simulation step) and compares it against the
policy's ``control_dt``. It is meant to answer "does this machine keep up
with real-time control?" on the target hardware (e.g. a Jetson).

When ``--vis render`` is used the MuJoCo offscreen renderer records every
step and writes ``eval.mp4`` at the end, matching ``run_policy.py --vis render``.

Typical usage:
    MUJOCO_GL=egl python3 toddlerbot/tools/bench_policy.py \
        --policy replay --path motion/push_up_2xc.lz4 --steps 900 --vis render
"""

import argparse
import json
import os
import time

import cv2
import mujoco
import numpy as np
import numpy.typing as npt

from toddlerbot.policies.run_policy import get_policy_class
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot

PHASES = ["obs", "inference", "set_action", "sim_step"]
CAMERAS = ["perspective", "side", "top", "front"]


def _stats(samples_ms: npt.NDArray[np.float64]) -> dict[str, float]:
    """Return p50/p95/p99/max/mean in milliseconds."""
    return {
        "p50": float(np.percentile(samples_ms, 50)),
        "p95": float(np.percentile(samples_ms, 95)),
        "p99": float(np.percentile(samples_ms, 99)),
        "max": float(np.max(samples_ms)),
        "mean": float(np.mean(samples_ms)),
    }


def _print_stats(name: str, stats: dict[str, float]) -> None:
    print(
        f"  {name:<12} "
        f"p50={stats['p50']:7.2f}  p95={stats['p95']:7.2f}  "
        f"p99={stats['p99']:7.2f}  max={stats['max']:7.2f}  "
        f"mean={stats['mean']:7.2f} ms"
    )


def _skip_prep(policy) -> None:
    """Put the policy on its steady-state path before timing.

    Policies spend their first ``prep_duration`` interpolating from the
    initial pose; timing that phase would never exercise inference. This
    mirrors the state run_policy.py sets when it skips the prep phase.
    """
    policy.is_prepared = True
    policy.prep_duration = 0.0
    policy.time_start = 0.0
    if hasattr(policy, "step_curr"):
        policy.step_curr = 0


def _write_video_streaming(
    sim: MuJoCoSim, out_path: str, dt: float, render_every: int
) -> None:
    """Write the recorded frames as a 2x2 camera montage, streaming to disk.

    ``MuJoCoRenderer.save_recording`` builds every frame for all four cameras
    in memory before encoding, which OOMs an 8 GB Jetson. Rendering one frame
    at a time keeps memory constant.
    """
    vis = sim.visualizer
    assert vis is not None
    frames = zip(vis.qpos_data[::render_every], vis.qvel_data[::render_every])

    writer: cv2.VideoWriter | None = None
    try:
        for qpos, qvel in frames:
            data = mujoco.MjData(sim.model)
            data.qpos, data.qvel = qpos, qvel
            mujoco.mj_forward(sim.model, data)

            images = []
            for camera in CAMERAS:
                vis.renderer.update_scene(
                    data, camera=camera, scene_option=vis.scene_option
                )
                images.append(vis.renderer.render())

            grid = np.vstack([np.hstack(images[0:2]), np.hstack(images[2:4])])
            bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    1.0 / (dt * render_every),
                    (width, height),
                )
            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()


def run_benchmark(args: argparse.Namespace) -> dict:
    robot = Robot(args.robot)

    # Build the simulator first so the policy can start from the measured
    # initial motor positions, exactly like run_policy.py.
    sim = MuJoCoSim(
        robot,
        vis_type=args.vis,
        fixed_base="fixed" in args.policy,
    )
    init_motor_pos = sim.get_observation().motor_pos

    PolicyClass = get_policy_class(args.policy.replace("_fixed", ""))
    kwargs = {
        "name": args.policy,
        "robot": robot,
        "init_motor_pos": init_motor_pos,
    }
    if args.path:
        kwargs["path"] = args.path
    if args.command:
        kwargs["fixed_command"] = np.array(args.command.split(), dtype=np.float32)
    policy = PolicyClass(**kwargs)

    control_dt = float(policy.control_dt)
    budget_ms = control_dt * 1000.0

    # Skip the prep interpolation so every measured step runs the actual
    # policy (including the ONNX inference for neural policies).
    _skip_prep(policy)

    print(f"\nBenchmark: robot={args.robot} policy={args.policy} vis={args.vis}")
    print(
        f"control_dt={budget_ms:.2f} ms  steps={args.steps}  "
        f"warmup={args.warmup}"
    )

    def _step(obs_time: float) -> list[float]:
        step_start = time.perf_counter()

        obs = sim.get_observation()
        obs.time = obs_time
        t_obs = time.perf_counter()

        _, motor_target = policy.step(obs, sim)
        t_inference = time.perf_counter()

        sim.set_motor_target(motor_target)
        t_set_action = time.perf_counter()

        sim.step()
        t_sim_step = time.perf_counter()

        return [
            (t_obs - step_start) * 1000.0,
            (t_inference - t_obs) * 1000.0,
            (t_set_action - t_inference) * 1000.0,
            (t_sim_step - t_set_action) * 1000.0,
        ]

    # Warm up (first ONNX call allocates; caches get populated).
    for i in range(args.warmup):
        _step(i * control_dt)

    # Restart the policy from the beginning of its trajectory and drop the
    # warm-up frames so the recorded video covers only the measured run.
    if hasattr(policy, "step_curr"):
        policy.step_curr = 0
    if args.vis == "render" and getattr(sim, "visualizer", None) is not None:
        sim.visualizer.qpos_data.clear()
        sim.visualizer.qvel_data.clear()

    samples: list[list[float]] = []
    for i in range(args.steps):
        samples.append(_step(i * control_dt))

    arr = np.asarray(samples, dtype=np.float64)
    phases = {name: _stats(arr[:, i]) for i, name in enumerate(PHASES)}
    total = _stats(arr.sum(axis=1))

    print("\nPer-phase latency (ms):")
    for name in PHASES:
        _print_stats(name, phases[name])
    _print_stats("total", total)

    achieved_hz = 1000.0 / total["mean"] if total["mean"] > 0 else 0.0
    print(f"\nAchieved loop rate (mean): {achieved_hz:.1f} Hz")

    inference_ok = phases["inference"]["p99"] <= budget_ms
    total_ok = total["p99"] <= budget_ms
    print(
        f"Verdict: inference p99 {'fits' if inference_ok else 'EXCEEDS'} "
        f"the {budget_ms:.1f} ms budget; "
        f"loop p99 {'fits' if total_ok else 'EXCEEDS'} it."
    )
    if not total_ok:
        print(
            "  -> the control loop cannot hold control_dt on this machine; "
            "expect degraded real-time behaviour."
        )

    result = {
        "robot": args.robot,
        "policy": args.policy,
        "path": args.path,
        "vis": args.vis,
        "steps": args.steps,
        "control_dt_ms": budget_ms,
        "achieved_hz": achieved_hz,
        "phases_ms": phases,
        "total_ms": total,
        "inference_fits_budget": inference_ok,
        "loop_fits_budget": total_ok,
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "bench.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {json_path}")

    if args.vis == "render":
        video_name = "eval.mp4"
        video_path = os.path.join(args.out, video_name)
        render_every = 2
        print(f"Rendering {args.steps} frames to {video_path} ...")
        _write_video_streaming(sim, video_path, control_dt, render_every)
        print("  video written.")
        result["video"] = video_path
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    sim.close()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark policy step latency.")
    parser.add_argument("--robot", type=str, default="toddlerbot_2xc")
    parser.add_argument("--policy", type=str, default="replay")
    parser.add_argument(
        "--path", type=str, default="", help="Policy checkpoint or motion path."
    )
    parser.add_argument(
        "--command",
        type=str,
        default="",
        help="Space-separated fixed command for RL policies (e.g. walk).",
    )
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--vis",
        type=str,
        default="render",
        choices=["render", "none"],
        help="'render' also writes eval.mp4; 'none' only measures.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory (default: results/bench_<policy>_<timestamp>).",
    )
    args = parser.parse_args()

    if not args.out:
        time_str = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join("results", f"bench_{args.policy}_{time_str}")

    run_benchmark(args)


if __name__ == "__main__":
    main()
