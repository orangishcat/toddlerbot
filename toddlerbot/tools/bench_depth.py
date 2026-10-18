"""Benchmark the FoundationStereo TensorRT depth engine on this machine.

Measures the wall-clock cost of one depth frame (rectification + TensorRT
inference + disparity->depth) and reports percentiles and FPS, so you can
decide whether the engine is fast enough for the control loop on a given
Jetson.

Timing mirrors ``DepthEstimatorFoundationStereo.get_depth`` but calls the
internal steps directly to avoid the ``@profile()`` line-profiler overhead
that ``get_depth`` carries in the real pipeline. GPU work is bracketed with
``torch.cuda.synchronize()`` so the numbers are real GPU time, not just the
async enqueue time.

Typical usage (no cameras):
    MUJOCO_GL=egl python3 toddlerbot/tools/bench_depth.py \
        --engine toddlerbot/depth/models/<engine>.engine \
        --steps 200 --warmup 20

Or with a folder of stereo pairs ``{i}_original_left.jpg``/``{i}_original_right.jpg``:
    python3 toddlerbot/tools/bench_depth.py --engine <engine> --images results/depth_blake_debug
"""

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from toddlerbot.depth.depth_estimator_foundation_stereo import (
    DepthEstimatorFoundationStereo,
)


def _load_images(images_dir: str | None, width: int, height: int):
    """Return a list of (left, right) BGR uint8 images."""
    if images_dir:
        left_paths = sorted(glob.glob(os.path.join(images_dir, "*_original_left.jpg")))
        if not left_paths:
            raise FileNotFoundError(f"No *_original_left.jpg under {images_dir}")
        pairs = []
        for lp in left_paths:
            rp = lp.replace("_original_left.jpg", "_original_right.jpg")
            if not os.path.isfile(rp):
                continue
            left = cv2.imread(lp)
            right = cv2.imread(rp)
            if left is None or right is None:
                continue
            pairs.append((left, right))
        if not pairs:
            raise FileNotFoundError(f"No valid stereo pairs under {images_dir}")
        print(f"Loaded {len(pairs)} stereo pairs from {images_dir}")
        return pairs

    # Synthetic frames at calibration resolution.
    rng = np.random.default_rng(0)
    left = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    right = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    print(f"Using synthetic {width}x{height} frames")
    return [(left, right)]


def _percentiles(samples_ms):
    a = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def run(args: argparse.Namespace) -> dict:
    estimator = DepthEstimatorFoundationStereo(
        calib_params_path=args.calib_params,
        rec_params_path=args.rec_params,
        engine_path=args.engine,
        calib_width=args.calib_width,
        calib_height=args.calib_height,
    )
    print(f"Engine input shape: {estimator.width}x{estimator.height}")
    print(f"model_type={estimator.model_type}")

    pairs = _load_images(args.images, args.calib_width, args.calib_height)

    def one_frame(idx: int):
        left, right = pairs[idx % len(pairs)]

        t0 = time.perf_counter()
        rect_l, rect_r = estimator._process_images(left, right)
        torch.cuda.synchronize()
        t_rect = time.perf_counter()

        disparity = estimator._infer(rect_l, rect_r)  # syncs internally
        t_infer = time.perf_counter()

        depth = estimator.fx_times_baseline / (disparity + 0)
        t_post = time.perf_counter()

        return [
            (t_rect - t0) * 1000.0,
            (t_infer - t_rect) * 1000.0,
            (t_post - t_infer) * 1000.0,
        ], depth

    print(f"\nWarming up ({args.warmup} frames)...")
    for i in range(args.warmup):
        one_frame(i)

    print(f"Benchmarking {args.steps} frames...")
    rows = []
    for i in range(args.steps):
        phase, _ = one_frame(i)
        rows.append(phase)

    arr = np.asarray(rows, dtype=np.float64)
    names = ["rectify", "infer", "post"]
    phases = {n: _percentiles(arr[:, i]) for i, n in enumerate(names)}
    total = _percentiles(arr.sum(axis=1))
    fps = 1000.0 / total["mean"] if total["mean"] > 0 else 0.0

    print("\nPer-phase latency (ms):")
    for n in names:
        s = phases[n]
        print(
            f"  {n:<9} p50={s['p50']:7.2f}  p95={s['p95']:7.2f}  "
            f"p99={s['p99']:7.2f}  max={s['max']:7.2f}  mean={s['mean']:7.2f}"
        )
    s = total
    print(
        f"  {'total':<9} p50={s['p50']:7.2f}  p95={s['p95']:7.2f}  "
        f"p99={s['p99']:7.2f}  max={s['max']:7.2f}  mean={s['mean']:7.2f}"
    )
    print(f"\nThroughput (mean): {fps:.1f} FPS")

    control_dt_ms = args.control_dt * 1000.0
    print(
        f"Control loop budget {control_dt_ms:.1f} ms -> "
        f"{'depth can keep up with every control step' if total['p99'] <= control_dt_ms else 'depth is slower than the control step (runs as a separate, conflated stream)'}"
    )

    result = {
        "engine": args.engine,
        "images": args.images,
        "steps": args.steps,
        "warmup": args.warmup,
        "engine_input": [estimator.width, estimator.height],
        "model_type": estimator.model_type,
        "phases_ms": phases,
        "total_ms": total,
        "fps": fps,
        "control_dt_ms": control_dt_ms,
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "bench_depth.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {json_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the depth engine.")
    parser.add_argument("--engine", type=str, required=True,
                        help="Path to the .engine/.plan/.onnx depth model.")
    parser.add_argument("--calib-params", type=str,
                        default=os.path.join("toddlerbot", "depth", "params", "calibration.pkl"))
    parser.add_argument("--rec-params", type=str,
                        default=os.path.join("toddlerbot", "depth", "params", "rectification.npz"))
    parser.add_argument("--calib-width", type=int, default=640)
    parser.add_argument("--calib-height", type=int, default=480)
    parser.add_argument("--images", type=str, default="",
                        help="Folder of {i}_original_left/right.jpg. Omit for synthetic frames.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--control-dt", type=float, default=0.02,
                        help="Control period to compare against (s).")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if not args.out:
        args.out = os.path.join("results", f"bench_depth_{time.strftime('%Y%m%d_%H%M%S')}")

    run(args)


if __name__ == "__main__":
    main()
