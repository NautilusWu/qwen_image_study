"""
Qwen-Image vLLM-Omni BF16 Single-Request Baseline
==================================================

Purpose
-------
Measure the single-request inference performance of Qwen/Qwen-Image
running with vLLM-Omni on NVIDIA DGX Spark.

This experiment establishes the BF16 no-CFG baseline used as the
reference point for subsequent step-scaling, CFG, concurrency,
batching, and quantization experiments.

Experimental definition
-----------------------
- Model: Qwen/Qwen-Image
- Backend: vLLM-Omni
- Precision: BF16
- Resolution: 1024x1024
- Images per request: 1
- Inference steps: 50
- Seed: 42
- True CFG scale: 1.0 (CFG disabled)
- Server max_num_seqs: 1
- Client concurrency: 1

The number of warmup and measured runs can be changed from the command line.

Examples
--------
Default:
    python3 scripts/qwen_image_vllm_omni_bf16_single_request_baseline.py

Run 10 measured generations:
    python3 scripts/qwen_image_vllm_omni_bf16_single_request_baseline.py \
        --measured-runs 10

Use 2 warmup generations and 10 measured generations:
    python3 scripts/qwen_image_vllm_omni_bf16_single_request_baseline.py \
        --warmup-runs 2 \
        --measured-runs 10

Metrics
-------
For every measured request:
- Client-observed end-to-end wall latency
- Server queue wait time
- Server diffusion-stage generation time
- Server-reported peak memory

Summary:
- Mean
- Median
- Standard deviation
- Minimum
- Maximum

Output
------
/outputs/qwen_image_vllm_omni_bf16_single_request_baseline/<timestamp>/

Files:
- config.json
- raw.jsonl
- baseline_latency.csv
- summary.json
- representative.png
"""

import argparse
import base64
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests


# ============================================================
# FIXED BASELINE CONFIGURATION
# ============================================================

SERVER_URL = "http://localhost:8091/v1/images/generations"
HEALTH_URL = "http://localhost:8091/health"

BASE_OUTPUT_DIR = Path(
    "/outputs/qwen_image_vllm_omni_bf16_single_request_baseline"
)

MODEL = "Qwen/Qwen-Image"

PROMPT = (
    "A red sports car parked on a quiet city street at sunset, "
    "with realistic lighting, detailed buildings, and reflections "
    "on the wet pavement."
)

WIDTH = 1024
HEIGHT = 1024

NUM_INFERENCE_STEPS = 50
SEED = 42
TRUE_CFG_SCALE = 1.0

IMAGES_PER_REQUEST = 1
SERVER_MAX_NUM_SEQS = 1
CLIENT_CONCURRENCY = 1


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qwen-Image BF16 single-request inference "
            "through vLLM-Omni."
        )
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup generations to discard. Default: 1",
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help="Number of measured generations. Default: 5",
    )

    return parser.parse_args()


# ============================================================
# SERVER CHECK
# ============================================================

def check_server():
    response = requests.get(
        HEALTH_URL,
        timeout=10,
    )

    response.raise_for_status()

    print("Server health check: PASS")


# ============================================================
# INFERENCE REQUEST
# ============================================================

def send_request():
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "size": f"{WIDTH}x{HEIGHT}",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "seed": SEED,
        "true_cfg_scale": TRUE_CFG_SCALE,
    }

    start_time = time.perf_counter()

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=1800,
    )

    end_time = time.perf_counter()

    response.raise_for_status()

    wall_latency_s = end_time - start_time

    return response.json(), wall_latency_s


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metrics(response_json, wall_latency_s):
    metrics = response_json.get(
        "metrics",
        {},
    )

    stage_durations = metrics.get(
        "stage_durations",
        {},
    )

    return {
        "wall_latency_s": wall_latency_s,
        "queue_wait_ms": stage_durations.get("queue_wait_ms"),
        "stage_0_gen_ms": stage_durations.get("stage_0_gen_ms"),
        "peak_memory_mb": metrics.get("peak_memory_mb"),
    }


# ============================================================
# IMAGE SAVING
# ============================================================

def save_image(response_json, output_path):
    image_b64 = response_json["data"][0]["b64_json"]

    image_bytes = base64.b64decode(
        image_b64
    )

    output_path.write_bytes(
        image_bytes
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.warmup_runs < 0:
        raise ValueError(
            "--warmup-runs must be >= 0"
        )

    if args.measured_runs < 1:
        raise ValueError(
            "--measured-runs must be >= 1"
        )

    # --------------------------------------------------------
    # Unique output directory
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        BASE_OUTPUT_DIR / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print()
    print("==============================================")
    print("Qwen-Image vLLM-Omni BF16 Baseline")
    print("==============================================")
    print()
    print(f"Model:             {MODEL}")
    print(f"Resolution:        {WIDTH}x{HEIGHT}")
    print(f"Steps:             {NUM_INFERENCE_STEPS}")
    print(f"Seed:              {SEED}")
    print(f"True CFG scale:    {TRUE_CFG_SCALE}")
    print(f"Images/request:    {IMAGES_PER_REQUEST}")
    print(f"max_num_seqs:      {SERVER_MAX_NUM_SEQS}")
    print(f"Concurrency:       {CLIENT_CONCURRENCY}")
    print(f"Warmup runs:       {args.warmup_runs}")
    print(f"Measured runs:     {args.measured_runs}")
    print(f"Output directory:  {output_dir}")
    print()

    # --------------------------------------------------------
    # Save experiment configuration
    # --------------------------------------------------------

    config = {
        "experiment": (
            "Qwen-Image vLLM-Omni BF16 "
            "Single-Request Baseline"
        ),
        "model": MODEL,
        "prompt": PROMPT,
        "backend": "vLLM-Omni",
        "precision": "BF16",
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "seed": SEED,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "cfg_enabled": False,
        "images_per_request": IMAGES_PER_REQUEST,
        "server_max_num_seqs": SERVER_MAX_NUM_SEQS,
        "client_concurrency": CLIENT_CONCURRENCY,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
    }

    config_path = (
        output_dir / "config.json"
    )

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Server health
    # --------------------------------------------------------

    check_server()

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print("=== Warmup ===")

    for run_id in range(
        1,
        args.warmup_runs + 1,
    ):
        _, wall_latency_s = send_request()

        print(
            f"Warmup {run_id:02d}: "
            f"{wall_latency_s:.3f} s"
        )

    # --------------------------------------------------------
    # Measured generations
    # --------------------------------------------------------

    print()
    print("=== Measured Runs ===")

    results = []

    representative_response = None

    for run_id in range(
        1,
        args.measured_runs + 1,
    ):
        response_json, wall_latency_s = (
            send_request()
        )

        metrics = extract_metrics(
            response_json=response_json,
            wall_latency_s=wall_latency_s,
        )

        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
            "true_cfg_scale": TRUE_CFG_SCALE,
            **metrics,
        }

        results.append(
            record
        )

        if representative_response is None:
            representative_response = (
                response_json
            )

        stage_0_s = (
            metrics["stage_0_gen_ms"] / 1000
            if metrics["stage_0_gen_ms"] is not None
            else float("nan")
        )

        queue_ms = (
            metrics["queue_wait_ms"]
            if metrics["queue_wait_ms"] is not None
            else float("nan")
        )

        peak_memory_mb = (
            metrics["peak_memory_mb"]
            if metrics["peak_memory_mb"] is not None
            else float("nan")
        )

        print(
            f"Run {run_id:02d}: "
            f"wall={wall_latency_s:.3f} s | "
            f"stage0={stage_0_s:.3f} s | "
            f"queue={queue_ms:.3f} ms | "
            f"peak_memory={peak_memory_mb:.1f} MB"
        )

    # --------------------------------------------------------
    # Raw JSONL
    # --------------------------------------------------------

    raw_path = (
        output_dir / "raw.jsonl"
    )

    with raw_path.open(
        "w"
    ) as file:
        for record in results:
            file.write(
                json.dumps(record)
                + "\n"
            )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir / "baseline_latency.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    wall_latencies = [
        record["wall_latency_s"]
        for record in results
    ]

    stage_latencies = [
        record["stage_0_gen_ms"] / 1000
        for record in results
        if record["stage_0_gen_ms"] is not None
    ]

    peak_memory_values = [
        record["peak_memory_mb"]
        for record in results
        if record["peak_memory_mb"] is not None
    ]

    wall_std = (
        statistics.stdev(wall_latencies)
        if len(wall_latencies) > 1
        else 0.0
    )

    stage_std = (
        statistics.stdev(stage_latencies)
        if len(stage_latencies) > 1
        else 0.0
    )

    summary = {
        **config,
        "wall_latency_s": {
            "mean": statistics.mean(
                wall_latencies
            ),
            "median": statistics.median(
                wall_latencies
            ),
            "std": wall_std,
            "min": min(
                wall_latencies
            ),
            "max": max(
                wall_latencies
            ),
        },
        "stage_0_generation_s": {
            "mean": statistics.mean(
                stage_latencies
            ),
            "median": statistics.median(
                stage_latencies
            ),
            "std": stage_std,
            "min": min(
                stage_latencies
            ),
            "max": max(
                stage_latencies
            ),
        },
        "peak_memory_mb": {
            "max": max(
                peak_memory_values
            )
            if peak_memory_values
            else None,
        },
    }

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary_path = (
        output_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Representative image
    # --------------------------------------------------------

    representative_path = (
        output_dir / "representative.png"
    )

    save_image(
        representative_response,
        representative_path,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("Summary")
    print("==============================================")

    print(
        "Wall latency:"
    )

    print(
        f"  Mean:   "
        f"{summary['wall_latency_s']['mean']:.3f} s"
    )

    print(
        f"  Median: "
        f"{summary['wall_latency_s']['median']:.3f} s"
    )

    print(
        f"  Std:    "
        f"{summary['wall_latency_s']['std']:.3f} s"
    )

    print(
        f"  Min:    "
        f"{summary['wall_latency_s']['min']:.3f} s"
    )

    print(
        f"  Max:    "
        f"{summary['wall_latency_s']['max']:.3f} s"
    )

    print()

    print(
        "Server stage-0 generation:"
    )

    print(
        f"  Mean:   "
        f"{summary['stage_0_generation_s']['mean']:.3f} s"
    )

    print(
        f"  Median: "
        f"{summary['stage_0_generation_s']['median']:.3f} s"
    )

    print()

    print(
        "Peak memory:"
    )

    print(
        f"  "
        f"{summary['peak_memory_mb']['max']:.1f} MB"
    )

    print()

    print(
        f"Results written to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()