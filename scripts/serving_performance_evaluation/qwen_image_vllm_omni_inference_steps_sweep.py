"""
Qwen-Image vLLM-Omni Inference-Step Scaling Experiment
=======================================================

Purpose
-------
Measure how Qwen/Qwen-Image inference latency scales with the number
of denoising steps when running through vLLM-Omni on NVIDIA DGX Spark.

Only num_inference_steps is changed. All other inference parameters
are fixed to the BF16 no-CFG single-request baseline configuration.

Primary questions
-----------------
1. How does end-to-end latency scale with inference steps?
2. Is the relationship approximately linear?
3. What is the incremental latency per denoising step?
4. What fixed step-independent latency remains?
5. How does reducing the number of steps affect the generated image?

Latency model
-------------
The experiment fits:

    T(N) = a + bN

where:

    N = number of inference / denoising steps
    a = fitted step-independent latency
    b = fitted incremental latency per denoising step

The coefficient of determination R^2 is also reported.

Experimental configuration
--------------------------
Fixed:
- Model: Qwen/Qwen-Image
- Backend: vLLM-Omni
- Precision: BF16
- Resolution: 1024x1024
- Images per request: 1
- Seed: 42
- True CFG scale: 1.0 (CFG disabled)
- Server max_num_seqs: 1
- Client concurrency: 1
- Prompt: same as BF16 baseline

Variable:
- num_inference_steps: 10, 20, 30, 40, 50

Method
------
- One warmup request is performed for each step count by default.
- Measured requests are interleaved across step counts.
- Step execution order is shuffled using a fixed random seed to reduce
  correlation between thermal drift and inference-step count.
- One representative image is saved for every step count.
- Raw per-request measurements are preserved.

Examples
--------
Default:
    python3 scripts/qwen_image_vllm_omni_inference_steps_sweep.py

10 measured runs per step:
    python3 scripts/qwen_image_vllm_omni_inference_steps_sweep.py \
        --measured-runs 10

Disable per-step warmup:
    python3 scripts/qwen_image_vllm_omni_inference_steps_sweep.py \
        --warmup-runs 0

Outputs
-------
/outputs/qwen_image_vllm_omni_inference_steps_sweep/<timestamp>/

Files:
- config.json
- raw.jsonl
- steps_summary.csv
- linear_fit.json
- images/steps_10.png
- images/steps_20.png
- images/steps_30.png
- images/steps_40.png
- images/steps_50.png
"""

import argparse
import base64
import csv
import json
import math
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests


# ============================================================
# FIXED EXPERIMENT CONFIGURATION
# ============================================================

SERVER_URL = "http://localhost:8091/v1/images/generations"
HEALTH_URL = "http://localhost:8091/health"

BASE_OUTPUT_DIR = Path(
    "/outputs/qwen_image_vllm_omni_inference_steps_sweep"
)

MODEL = "Qwen/Qwen-Image"

PROMPT = (
    "A red sports car parked on a quiet city street at sunset, "
    "with realistic lighting, detailed buildings, and reflections "
    "on the wet pavement."
)

WIDTH = 1024
HEIGHT = 1024

INFERENCE_STEPS = [10, 20, 30, 40, 50]

SEED = 42
TRUE_CFG_SCALE = 1.0

IMAGES_PER_REQUEST = 1
SERVER_MAX_NUM_SEQS = 1
CLIENT_CONCURRENCY = 1

# Used only to randomize experiment execution order.
# It does NOT change the Qwen-Image generation seed.
EXPERIMENT_ORDER_SEED = 20260830


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure Qwen-Image inference latency scaling with "
            "denoising-step count through vLLM-Omni."
        )
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help=(
            "Number of warmup requests per step count. "
            "Default: 1"
        ),
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help=(
            "Number of measured requests per step count. "
            "Default: 5"
        ),
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

def send_request(num_inference_steps):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "size": f"{WIDTH}x{HEIGHT}",
        "num_inference_steps": num_inference_steps,
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
# LINEAR REGRESSION
# ============================================================

def fit_linear_model(x_values, y_values):
    """
    Fit:

        y = intercept + slope * x

    using ordinary least squares.

    Returns:
        slope
        intercept
        r_squared
    """

    if len(x_values) != len(y_values):
        raise ValueError(
            "x_values and y_values must have equal length"
        )

    if len(x_values) < 2:
        raise ValueError(
            "At least two points are required for linear regression"
        )

    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )

    denominator = sum(
        (x - x_mean) ** 2
        for x in x_values
    )

    if denominator == 0:
        raise ValueError(
            "Cannot fit regression with identical x values"
        )

    slope = numerator / denominator

    intercept = (
        y_mean
        - slope * x_mean
    )

    predicted = [
        intercept + slope * x
        for x in x_values
    ]

    ss_residual = sum(
        (y - y_hat) ** 2
        for y, y_hat in zip(
            y_values,
            predicted,
        )
    )

    ss_total = sum(
        (y - y_mean) ** 2
        for y in y_values
    )

    if ss_total == 0:
        r_squared = 1.0
    else:
        r_squared = (
            1.0
            - ss_residual / ss_total
        )

    return {
        "slope_s_per_step": slope,
        "intercept_s": intercept,
        "r_squared": r_squared,
    }


# ============================================================
# SUMMARY HELPER
# ============================================================

def summarize_values(values):
    values = list(values)

    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "min": min(values),
        "max": max(values),
    }


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

    image_dir = (
        output_dir / "images"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Console configuration
    # --------------------------------------------------------

    print()
    print("======================================================")
    print("Qwen-Image vLLM-Omni Inference-Step Scaling")
    print("======================================================")
    print()
    print(f"Model:              {MODEL}")
    print(f"Resolution:         {WIDTH}x{HEIGHT}")
    print(f"Steps:              {INFERENCE_STEPS}")
    print(f"Seed:               {SEED}")
    print(f"True CFG scale:     {TRUE_CFG_SCALE}")
    print(f"Images/request:     {IMAGES_PER_REQUEST}")
    print(f"max_num_seqs:       {SERVER_MAX_NUM_SEQS}")
    print(f"Concurrency:        {CLIENT_CONCURRENCY}")
    print(f"Warmups/step:       {args.warmup_runs}")
    print(f"Measured runs/step: {args.measured_runs}")
    print(f"Order seed:         {EXPERIMENT_ORDER_SEED}")
    print(f"Output directory:   {output_dir}")
    print()

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "experiment": (
            "Qwen-Image vLLM-Omni "
            "Inference-Step Scaling"
        ),
        "model": MODEL,
        "prompt": PROMPT,
        "backend": "vLLM-Omni",
        "precision": "BF16",
        "width": WIDTH,
        "height": HEIGHT,
        "inference_steps": INFERENCE_STEPS,
        "seed": SEED,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "cfg_enabled": False,
        "images_per_request": IMAGES_PER_REQUEST,
        "server_max_num_seqs": SERVER_MAX_NUM_SEQS,
        "client_concurrency": CLIENT_CONCURRENCY,
        "warmup_runs_per_step": args.warmup_runs,
        "measured_runs_per_step": args.measured_runs,
        "experiment_order_seed": EXPERIMENT_ORDER_SEED,
    }

    (
        output_dir / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Server health check
    # --------------------------------------------------------

    check_server()

    rng = random.Random(
        EXPERIMENT_ORDER_SEED
    )

    # --------------------------------------------------------
    # Warmups
    # --------------------------------------------------------

    print()
    print("=== Warmup ===")

    for warmup_round in range(
        1,
        args.warmup_runs + 1,
    ):
        warmup_order = (
            INFERENCE_STEPS.copy()
        )

        rng.shuffle(
            warmup_order
        )

        print(
            f"Warmup round {warmup_round}: "
            f"{warmup_order}"
        )

        for steps in warmup_order:
            _, wall_latency_s = (
                send_request(steps)
            )

            print(
                f"  {steps:2d} steps: "
                f"{wall_latency_s:.3f} s"
            )

    # --------------------------------------------------------
    # Measured runs
    # --------------------------------------------------------

    print()
    print("=== Measured Runs ===")

    results = []

    representative_responses = {}

    global_request_id = 0

    for measured_round in range(
        1,
        args.measured_runs + 1,
    ):
        execution_order = (
            INFERENCE_STEPS.copy()
        )

        rng.shuffle(
            execution_order
        )

        print()
        print(
            f"Round {measured_round:02d} order: "
            f"{execution_order}"
        )

        for steps in execution_order:
            global_request_id += 1

            response_json, wall_latency_s = (
                send_request(steps)
            )

            metrics = extract_metrics(
                response_json=response_json,
                wall_latency_s=wall_latency_s,
            )

            record = {
                "request_id": global_request_id,
                "round_id": measured_round,
                "timestamp": datetime.now().isoformat(),
                "num_inference_steps": steps,
                "seed": SEED,
                "true_cfg_scale": TRUE_CFG_SCALE,
                **metrics,
            }

            results.append(
                record
            )

            # Save the first measured result for each step
            # as the representative image.
            if steps not in representative_responses:
                representative_responses[
                    steps
                ] = response_json

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
                f"  {steps:2d} steps | "
                f"wall={wall_latency_s:.3f} s | "
                f"stage0={stage_0_s:.3f} s | "
                f"queue={queue_ms:.3f} ms | "
                f"peak_memory={peak_memory_mb:.1f} MB"
            )

    # --------------------------------------------------------
    # Save raw JSONL
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
    # Save representative images
    # --------------------------------------------------------

    for steps in INFERENCE_STEPS:
        response_json = (
            representative_responses[
                steps
            ]
        )

        image_path = (
            image_dir
            / f"steps_{steps}.png"
        )

        save_image(
            response_json,
            image_path,
        )

    # --------------------------------------------------------
    # Aggregate results by step count
    # --------------------------------------------------------

    summary_rows = []

    for steps in sorted(
        INFERENCE_STEPS
    ):
        step_records = [
            record
            for record in results
            if record["num_inference_steps"] == steps
        ]

        wall_values = [
            record["wall_latency_s"]
            for record in step_records
        ]

        stage_values = [
            record["stage_0_gen_ms"] / 1000
            for record in step_records
            if record["stage_0_gen_ms"] is not None
        ]

        queue_values = [
            record["queue_wait_ms"]
            for record in step_records
            if record["queue_wait_ms"] is not None
        ]

        memory_values = [
            record["peak_memory_mb"]
            for record in step_records
            if record["peak_memory_mb"] is not None
        ]

        wall_summary = summarize_values(
            wall_values
        )

        stage_summary = summarize_values(
            stage_values
        )

        queue_summary = summarize_values(
            queue_values
        )

        summary_rows.append(
            {
                "num_inference_steps": steps,
                "measured_runs": len(step_records),

                "wall_mean_s": wall_summary["mean"],
                "wall_median_s": wall_summary["median"],
                "wall_std_s": wall_summary["std"],
                "wall_min_s": wall_summary["min"],
                "wall_max_s": wall_summary["max"],

                "stage_0_mean_s": stage_summary["mean"],
                "stage_0_median_s": stage_summary["median"],
                "stage_0_std_s": stage_summary["std"],

                "queue_mean_ms": queue_summary["mean"],

                "peak_memory_max_mb": (
                    max(memory_values)
                    if memory_values
                    else None
                ),
            }
        )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_csv_path = (
        output_dir / "steps_summary.csv"
    )

    with summary_csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------------
    # Linear regression
    # --------------------------------------------------------

    regression_x = [
        row["num_inference_steps"]
        for row in summary_rows
    ]

    regression_y_wall = [
        row["wall_mean_s"]
        for row in summary_rows
    ]

    regression_y_stage = [
        row["stage_0_mean_s"]
        for row in summary_rows
    ]

    wall_fit = fit_linear_model(
        regression_x,
        regression_y_wall,
    )

    stage_fit = fit_linear_model(
        regression_x,
        regression_y_stage,
    )

    linear_fit = {
        "model": "T(N) = intercept + slope * N",
        "wall_latency": wall_fit,
        "stage_0_generation": stage_fit,
    }

    (
        output_dir / "linear_fit.json"
    ).write_text(
        json.dumps(
            linear_fit,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("======================================================")
    print("Step Scaling Summary")
    print("======================================================")
    print()

    print(
        "Steps | Mean Wall | Median | Std | Stage-0"
    )

    print(
        "-----------------------------------------------"
    )

    for row in summary_rows:
        print(
            f"{row['num_inference_steps']:5d} | "
            f"{row['wall_mean_s']:9.3f} | "
            f"{row['wall_median_s']:6.3f} | "
            f"{row['wall_std_s']:5.3f} | "
            f"{row['stage_0_mean_s']:7.3f}"
        )

    print()
    print("Wall-latency linear fit:")
    print(
        f"  T(N) = "
        f"{wall_fit['intercept_s']:.4f} "
        f"+ "
        f"{wall_fit['slope_s_per_step']:.4f} * N"
    )

    print(
        f"  Slope:     "
        f"{wall_fit['slope_s_per_step']:.4f} s/step"
    )

    print(
        f"  Intercept: "
        f"{wall_fit['intercept_s']:.4f} s"
    )

    print(
        f"  R^2:       "
        f"{wall_fit['r_squared']:.6f}"
    )

    print()
    print("Stage-0 linear fit:")

    print(
        f"  T(N) = "
        f"{stage_fit['intercept_s']:.4f} "
        f"+ "
        f"{stage_fit['slope_s_per_step']:.4f} * N"
    )

    print(
        f"  R^2: "
        f"{stage_fit['r_squared']:.6f}"
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