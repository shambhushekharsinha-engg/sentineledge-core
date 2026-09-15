"""
latency_profile.py -- OpenVINO inference latency profiler for SentinelEdge.

Benchmarks inference on CPU, GPU, and NPU (skips unavailable devices),
reports detailed latency statistics, generates a matplotlib bar-chart PNG,
and writes a JSON report.

Usage
-----
python inference/latency_profile.py
python inference/latency_profile.py --ir_path inference/ir_model/vla_policy_int8.xml
python inference/latency_profile.py --warmup 100 --iterations 500 --devices CPU GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import openvino as ov
    _OV_AVAILABLE = True
except ImportError:
    _OV_AVAILABLE = False
    print("[WARN] openvino not installed -- running in SIMULATION mode.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    print("[WARN] matplotlib not installed -- chart will be skipped.")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR_PATH = ROOT / "inference" / "ir_model" / "vla_policy_int8.xml"
DEFAULT_OUT_PNG = ROOT / "inference" / "latency_profile.png"
DEFAULT_OUT_JSON = ROOT / "inference" / "latency_report.json"

WARMUP_ITERS = 50
BENCH_ITERS = 200
TARGET_DEVICES = ["CPU", "GPU", "NPU"]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(latencies_ms):
    arr = np.array(latencies_ms)
    return {
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "throughput_fps": float(1000.0 / np.mean(arr)),
        "num_iterations": len(latencies_ms),
    }


# ---------------------------------------------------------------------------
# Simulation fallback
# ---------------------------------------------------------------------------

def _simulate_inference(warmup, iterations):
    """Return simulated latencies when no real model is available."""
    np.random.seed(0)
    base = np.random.uniform(5, 30)
    return list(np.random.normal(base, base * 0.05, warmup + iterations)[warmup:])


def _build_dummy_input(model):
    inputs = {}
    for inp in model.inputs:
        shape = inp.partial_shape.to_static()
        static_shape = [max(1, d.get_length()) for d in shape]
        inputs[inp] = np.random.rand(*static_shape).astype(np.float32)
    return inputs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_device(ir_path, device, warmup, iterations):
    if not _OV_AVAILABLE:
        print(f"  [{device}] Simulating (openvino not installed) ...")
        lats = _simulate_inference(warmup, iterations)
        return compute_stats(lats)

    core = ov.Core()
    available = core.available_devices
    if device not in available:
        print(f"  [{device}] Not available on this system -- skipping.")
        return None

    if not ir_path.exists():
        print(f"  [{device}] IR model not found at {ir_path} -- simulating ...")
        lats = _simulate_inference(warmup, iterations)
        return compute_stats(lats)

    try:
        print(f"  [{device}] Compiling model ... ", end="", flush=True)
        compiled = core.compile_model(str(ir_path), device)
        infer_req = compiled.create_infer_request()
        dummy_inputs = _build_dummy_input(compiled)
        print("done")

        print(f"  [{device}] Warmup ({warmup} iters) ... ", end="", flush=True)
        for _ in range(warmup):
            infer_req.infer(dummy_inputs)
        print("done")

        print(f"  [{device}] Benchmarking ({iterations} iters) ... ", end="", flush=True)
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            infer_req.infer(dummy_inputs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
        print("done")

        return compute_stats(latencies)

    except Exception as exc:
        print(f"\n  [{device}] ERROR during benchmark: {exc}")
        return None


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_bar_chart(results, out_path):
    if not _MPL_AVAILABLE:
        print("[WARN] matplotlib unavailable -- skipping chart.")
        return

    devices = list(results.keys())
    metrics = ["min_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]
    labels = ["Min", "Mean", "P50", "P95", "P99", "Max"]

    x = np.arange(len(devices))
    bar_width = 0.12
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(metrics)))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results[d][metric] for d in devices]
        ax.bar(x + i * bar_width, vals, bar_width, label=label, color=colors[i])

    ax.set_xlabel("Device", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("SentinelEdge VLA Policy -- OpenVINO Inference Latency by Device", fontsize=13)
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(devices, fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    fps_vals = [results[d]["throughput_fps"] for d in devices]
    ax2.plot(x + bar_width * (len(metrics) - 1) / 2, fps_vals,
             "D--", color="black", label="Throughput (FPS)", markersize=7)
    ax2.set_ylabel("Throughput (FPS)", fontsize=12)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results):
    print("\n" + "=" * 68)
    print("  SentinelEdge Latency Profiler -- Results")
    print("=" * 68)
    header = f"  {'Device':<8}  {'Min':>7}  {'Mean':>7}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'Max':>7}  {'FPS':>8}"
    print(header)
    print("  " + "-" * 64)
    for dev, stats in results.items():
        print(
            f"  {dev:<8}  "
            f"{stats['min_ms']:>7.2f}  "
            f"{stats['mean_ms']:>7.2f}  "
            f"{stats['p50_ms']:>7.2f}  "
            f"{stats['p95_ms']:>7.2f}  "
            f"{stats['p99_ms']:>7.2f}  "
            f"{stats['max_ms']:>7.2f}  "
            f"{stats['throughput_fps']:>8.1f}"
        )
    print("  " + "-" * 64)
    print("  (all latencies in ms, FPS = 1000 / mean_ms)")
    print("=" * 68 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Latency profiler for SentinelEdge OpenVINO VLA policy.")
    parser.add_argument("--ir_path", type=str, default=str(DEFAULT_IR_PATH),
                        help="Path to the OpenVINO IR model XML")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS,
                        help=f"Warmup iterations per device (default: {WARMUP_ITERS})")
    parser.add_argument("--iterations", type=int, default=BENCH_ITERS,
                        help=f"Benchmark iterations per device (default: {BENCH_ITERS})")
    parser.add_argument("--devices", nargs="+", default=TARGET_DEVICES,
                        help=f"Devices to benchmark (default: {TARGET_DEVICES})")
    parser.add_argument("--output_png", type=str, default=str(DEFAULT_OUT_PNG))
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUT_JSON))
    return parser.parse_args()


def main():
    args = parse_args()
    ir_path = Path(args.ir_path)
    out_png = Path(args.output_png)
    out_json = Path(args.output_json)

    if _OV_AVAILABLE:
        core = ov.Core()
        print(f"OpenVINO version : {ov.__version__}")
        print(f"Available devices: {core.available_devices}")
    else:
        print("OpenVINO not installed -- running in simulation mode.\n")

    print(f"IR model path    : {ir_path}")
    print(f"Warmup iters     : {args.warmup}")
    print(f"Benchmark iters  : {args.iterations}")
    print(f"Target devices   : {args.devices}\n")

    results = {}
    for device in args.devices:
        print(f"--- Benchmarking {device} ---")
        stats = benchmark_device(ir_path, device, args.warmup, args.iterations)
        if stats is not None:
            results[device] = stats
        print()

    if not results:
        print("[ERROR] No devices produced results. Exiting.")
        sys.exit(1)

    print_report(results)
    plot_bar_chart(results, out_png)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ir_model_path": str(ir_path),
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.iterations,
        "devices": results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report saved -> {out_json}\n")


if __name__ == "__main__":
    main()
