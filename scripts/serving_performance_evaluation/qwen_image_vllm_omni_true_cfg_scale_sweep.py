"""
Qwen-Image vLLM-Omni True-CFG Scale Experiment
===============================================

Purpose
-------
Measure the latency and output behavior of Qwen/Qwen-Image as
True Classifier-Free Guidance (True CFG) is enabled and its scale
is varied.

Only true_cfg_scale is varied.

Primary questions
-----------------
1. What is the latency cost of enabling True CFG?
2. Does increasing true_cfg_scale from 2 to 4 to 6 further change
   computational latency?
3. How does True CFG strength affect generated images?

Experimental configuration
--------------------------
Fixed:
- Model: Qwen/Qwen-Image
- Backend: vLLM-Omni
- Precision: BF16
- Resolution: 1024x1024
- Inference steps: 50
- Seed: 42
- Guidance scale: 1.0
- Negative prompt: fixed
- Images per request: 1
- Server max_num_seqs: 1
- Client concurrency: 1
- Prompt: same as BF16 baseline

Variable:
- true_cfg_scale: 1.0, 2.0, 4.0, 6.0

Interpretation:
- true_cfg_scale = 1.0: True CFG disabled
- true_cfg_scale > 1.0: True CFG enabled

Method
------
- One warmup request per CFG scale by default.
- Five measured requests per CFG scale by default.
- CFG scales are interleaved and shuffled across rounds to reduce
  correlation with thermal or clock drift.
- One representative image is saved per CFG scale.

Outputs
-------
/outputs/qwen_image_vllm_omni_true_cfg_scale_sweep/<timestamp>/

Files:
- config.json
- raw.jsonl
- true_cfg_summary.csv
- images/
    cfg_1.png
    cfg_2.png
    cfg_4.png
    cfg_6.png
"""

import argparse
import base64
import csv
import json
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
    "/outputs/qwen_image_vllm_omni_true_cfg_scale_sweep"
)

MODEL = "Qwen/Qwen-Image"

PROMPT = (
    "A red sports car parked on a quiet city street at sunset, "
    "with realistic lighting, detailed buildings, and reflections "
    "on the wet pavement."
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, artifacts"
)

WIDTH = 1024
HEIGHT = 1024

NUM_INFERENCE_STEPS = 50
SEED = 42

# Keep generic guidance fixed so only True CFG is varied.
GUIDANCE_SCALE = 1.0

TRUE_CFG_SCALES = [
    1.0,
    2.0,
    4.0,
    6.0,
]

IMAGES_PER_REQUEST = 1
SERVER_MAX_NUM_SEQS = 1
CLIENT_CONCURRENCY = 1

# Only controls randomized experiment ordering.
EXPERIMENT_ORDER_SEED = 20260830


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the effect of Qwen-Image True CFG scale "
            "on latency and generated output."
        )
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup requests per CFG scale. Default: 1",
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help="Measured requests per CFG scale. Default: 5",
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

def send_request(true_cfg_scale):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "size": f"{WIDTH}x{HEIGHT}",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "seed": SEED,
        "guidance_scale": GUIDANCE_SCALE,
        "true_cfg_scale": true_cfg_scale,
    }

    start_time = time.perf_counter()

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=1800,
    )

    end_time = time.perf_counter()

    response.raise_for_status()

    return (
        response.json(),
        end_time - start_time,
    )


# ============================================================
# METRICS
# ============================================================

def extract_metrics(
    response_json,
    wall_latency_s,
):
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
        "queue_wait_ms": stage_durations.get(
            "queue_wait_ms"
        ),
        "stage_0_gen_ms": stage_durations.get(
            "stage_0_gen_ms"
        ),
        "peak_memory_mb": metrics.get(
            "peak_memory_mb"
        ),
    }


# ============================================================
# IMAGE SAVING
# ============================================================

def save_image(
    response_json,
    output_path,
):
    image_b64 = (
        response_json["data"][0]["b64_json"]
    )

    image_bytes = base64.b64decode(
        image_b64
    )

    output_path.write_bytes(
        image_bytes
    )


# ============================================================
# SUMMARY HELPER
# ============================================================

def summarize(values):
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
    # Output directory
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
    # Configuration
    # --------------------------------------------------------

    config = {
        "experiment": (
            "Qwen-Image vLLM-Omni "
            "True-CFG Scale Sweep"
        ),
        "model": MODEL,
        "backend": "vLLM-Omni",
        "precision": "BF16",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": (
            NUM_INFERENCE_STEPS
        ),
        "seed": SEED,
        "guidance_scale": GUIDANCE_SCALE,
        "true_cfg_scales": TRUE_CFG_SCALES,
        "images_per_request": (
            IMAGES_PER_REQUEST
        ),
        "server_max_num_seqs": (
            SERVER_MAX_NUM_SEQS
        ),
        "client_concurrency": (
            CLIENT_CONCURRENCY
        ),
        "warmup_runs_per_scale": (
            args.warmup_runs
        ),
        "measured_runs_per_scale": (
            args.measured_runs
        ),
        "experiment_order_seed": (
            EXPERIMENT_ORDER_SEED
        ),
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
    # Print configuration
    # --------------------------------------------------------

    print()
    print(
        "======================================================"
    )
    print(
        "Qwen-Image vLLM-Omni True-CFG Scale Sweep"
    )
    print(
        "======================================================"
    )
    print()

    print(f"Model:              {MODEL}")
    print(f"Resolution:         {WIDTH}x{HEIGHT}")
    print(
        f"Inference steps:    "
        f"{NUM_INFERENCE_STEPS}"
    )
    print(f"Seed:               {SEED}")
    print(
        f"Guidance scale:     "
        f"{GUIDANCE_SCALE}"
    )
    print(
        f"True CFG scales:    "
        f"{TRUE_CFG_SCALES}"
    )
    print(
        f"max_num_seqs:       "
        f"{SERVER_MAX_NUM_SEQS}"
    )
    print(
        f"Concurrency:        "
        f"{CLIENT_CONCURRENCY}"
    )
    print(
        f"Warmups/scale:      "
        f"{args.warmup_runs}"
    )
    print(
        f"Measured runs/scale:"
        f" {args.measured_runs}"
    )
    print(
        f"Output directory:   "
        f"{output_dir}"
    )
    print()

    check_server()

    rng = random.Random(
        EXPERIMENT_ORDER_SEED
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print("=== Warmup ===")

    for warmup_round in range(
        1,
        args.warmup_runs + 1,
    ):
        warmup_order = (
            TRUE_CFG_SCALES.copy()
        )

        rng.shuffle(
            warmup_order
        )

        print(
            f"Warmup round "
            f"{warmup_round}: "
            f"{warmup_order}"
        )

        for cfg_scale in warmup_order:
            _, wall_latency_s = (
                send_request(
                    cfg_scale
                )
            )

            print(
                f"  CFG {cfg_scale:.1f}: "
                f"{wall_latency_s:.3f} s"
            )

    # --------------------------------------------------------
    # Measured requests
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
            TRUE_CFG_SCALES.copy()
        )

        rng.shuffle(
            execution_order
        )

        print()
        print(
            f"Round {measured_round:02d} "
            f"order: {execution_order}"
        )

        for cfg_scale in execution_order:
            global_request_id += 1

            (
                response_json,
                wall_latency_s,
            ) = send_request(
                cfg_scale
            )

            metrics = extract_metrics(
                response_json,
                wall_latency_s,
            )

            record = {
                "request_id": (
                    global_request_id
                ),
                "round_id": measured_round,
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "true_cfg_scale": cfg_scale,
                "cfg_enabled": (
                    cfg_scale > 1.0
                ),
                "guidance_scale": (
                    GUIDANCE_SCALE
                ),
                "num_inference_steps": (
                    NUM_INFERENCE_STEPS
                ),
                "seed": SEED,
                **metrics,
            }

            results.append(
                record
            )

            if (
                cfg_scale
                not in representative_responses
            ):
                representative_responses[
                    cfg_scale
                ] = response_json

            stage_s = (
                metrics["stage_0_gen_ms"]
                / 1000
                if metrics[
                    "stage_0_gen_ms"
                ] is not None
                else float("nan")
            )

            print(
                f"  CFG {cfg_scale:.1f} | "
                f"wall={wall_latency_s:.3f} s | "
                f"stage0={stage_s:.3f} s | "
                f"queue="
                f"{metrics['queue_wait_ms']:.3f} ms | "
                f"peak_memory="
                f"{metrics['peak_memory_mb']:.1f} MB"
            )

    # --------------------------------------------------------
    # Raw results
    # --------------------------------------------------------

    with (
        output_dir / "raw.jsonl"
    ).open("w") as file:
        for record in results:
            file.write(
                json.dumps(record)
                + "\n"
            )

    # --------------------------------------------------------
    # Save representative images
    # --------------------------------------------------------

    for cfg_scale in TRUE_CFG_SCALES:
        cfg_label = (
            f"{cfg_scale:g}"
        )

        save_image(
            representative_responses[
                cfg_scale
            ],
            image_dir
            / f"cfg_{cfg_label}.png",
        )

    # --------------------------------------------------------
    # Aggregate by CFG scale
    # --------------------------------------------------------

    summary_rows = []

    for cfg_scale in TRUE_CFG_SCALES:
        cfg_records = [
            record
            for record in results
            if (
                record[
                    "true_cfg_scale"
                ]
                == cfg_scale
            )
        ]

        wall_values = [
            record[
                "wall_latency_s"
            ]
            for record in cfg_records
        ]

        stage_values = [
            record[
                "stage_0_gen_ms"
            ] / 1000
            for record in cfg_records
            if record[
                "stage_0_gen_ms"
            ] is not None
        ]

        queue_values = [
            record[
                "queue_wait_ms"
            ]
            for record in cfg_records
            if record[
                "queue_wait_ms"
            ] is not None
        ]

        memory_values = [
            record[
                "peak_memory_mb"
            ]
            for record in cfg_records
            if record[
                "peak_memory_mb"
            ] is not None
        ]

        wall_summary = summarize(
            wall_values
        )

        stage_summary = summarize(
            stage_values
        )

        queue_summary = summarize(
            queue_values
        )

        summary_rows.append(
            {
                "true_cfg_scale": (
                    cfg_scale
                ),
                "cfg_enabled": (
                    cfg_scale > 1.0
                ),
                "measured_runs": (
                    len(cfg_records)
                ),

                "wall_mean_s": (
                    wall_summary["mean"]
                ),
                "wall_median_s": (
                    wall_summary["median"]
                ),
                "wall_std_s": (
                    wall_summary["std"]
                ),
                "wall_min_s": (
                    wall_summary["min"]
                ),
                "wall_max_s": (
                    wall_summary["max"]
                ),

                "stage_0_mean_s": (
                    stage_summary["mean"]
                ),
                "stage_0_median_s": (
                    stage_summary["median"]
                ),

                "queue_mean_ms": (
                    queue_summary["mean"]
                ),

                "peak_memory_max_mb": (
                    max(memory_values)
                    if memory_values
                    else None
                ),
            }
        )

    # --------------------------------------------------------
    # Relative latency vs CFG=1
    # --------------------------------------------------------

    cfg1_row = next(
        row
        for row in summary_rows
        if row[
            "true_cfg_scale"
        ] == 1.0
    )

    cfg1_latency = (
        cfg1_row[
            "wall_mean_s"
        ]
    )

    for row in summary_rows:
        row[
            "latency_ratio_vs_cfg1"
        ] = (
            row[
                "wall_mean_s"
            ]
            / cfg1_latency
        )

        row[
            "latency_increase_percent"
        ] = (
            (
                row[
                    "wall_mean_s"
                ]
                / cfg1_latency
            )
            - 1.0
        ) * 100.0

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "true_cfg_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(
        "======================================================"
    )
    print(
        "True-CFG Scaling Summary"
    )
    print(
        "======================================================"
    )

    print()
    print(
        "CFG | Mean Wall | Median | Std | "
        "Latency Ratio | Stage-0"
    )
    print(
        "-------------------------------------------------------------"
    )

    for row in summary_rows:
        print(
            f"{row['true_cfg_scale']:3.1f} | "
            f"{row['wall_mean_s']:9.3f} | "
            f"{row['wall_median_s']:6.3f} | "
            f"{row['wall_std_s']:5.3f} | "
            f"{row['latency_ratio_vs_cfg1']:13.3f}x | "
            f"{row['stage_0_mean_s']:7.3f}"
        )

    print()
    print(
        "Relative to CFG=1.0:"
    )

    for row in summary_rows:
        print(
            f"  CFG "
            f"{row['true_cfg_scale']:.1f}: "
            f"{row['latency_ratio_vs_cfg1']:.3f}x "
            f"("
            f"{row['latency_increase_percent']:+.1f}%"
            f")"
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