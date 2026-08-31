"""
Qwen-Image vLLM-Omni Concurrency Benchmark
===========================================

Purpose
-------
Measure serving throughput and request-latency behavior of
Qwen/Qwen-Image under concurrent client load.

The same benchmark workload is reused across all concurrency levels
and server execution modes so that comparisons are controlled and
reproducible.

Server modes
------------
1. serial

    vllm serve Qwen/Qwen-Image \
        --omni \
        --port 8091 \
        --max-num-seqs 1

2. request_batch

    vllm serve Qwen/Qwen-Image \
        --omni \
        --port 8091 \
        --max-num-seqs 8 \
        --request-batch-max-wait-ms 20

3. step_batch

    vllm serve Qwen/Qwen-Image \
        --omni \
        --port 8091 \
        --step-execution \
        --max-num-seqs 8

Experimental configuration
--------------------------
Fixed:
- Model: Qwen/Qwen-Image
- Backend: vLLM-Omni
- Precision: BF16
- Resolution: 1024x1024
- Inference steps: 20
- True CFG scale: 1.0 (CFG disabled)
- Images per request: 1
- Prompt: fixed

Variable:
- Client concurrency: 1, 2, 4, 8
- Server execution mode

Workload seeds
--------------
The same seed sequence is reused for EVERY concurrency level and
EVERY server mode.

For the default 24 measured requests:

    workload_index = 0..23
    seed           = 42..65

Therefore:

    serial / concurrency 1
    serial / concurrency 8
    request_batch / concurrency 8
    step_batch / concurrency 8

all process the same set of 24 generation workloads.

request_id and seed have separate meanings:

- request_id:
    globally unique identifier used only for logging

- workload_index:
    identifies a workload within one benchmark level

- seed:
    BASE_SEED + workload_index

Metrics
-------
- Aggregate throughput (images/s)
- Mean request latency
- P50 latency
- P90 latency
- P95 latency
- Maximum latency
- Mean server queue wait
- P95 server queue wait
- Mean server stage-0 generation time
- Peak server-reported memory

P99 is intentionally not reported because the default sample size
(24 requests per level) is too small for a meaningful P99 estimate.

Output
------
/outputs/qwen_image_vllm_omni_concurrency_benchmark/
    <server_mode>/
        <timestamp>/
            config.json
            raw.jsonl
            concurrency_summary.csv

Examples
--------
Serial smoke test:

    python3 scripts/qwen_image_vllm_omni_concurrency_benchmark.py \
        --server-mode serial \
        --concurrency 1 2 \
        --measured-requests 8

Formal serial run:

    python3 scripts/qwen_image_vllm_omni_concurrency_benchmark.py \
        --server-mode serial \
        --concurrency 1 2 4 8 \
        --measured-requests 24
"""

import argparse
import concurrent.futures
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests


# ============================================================
# FIXED INFERENCE CONFIGURATION
# ============================================================

SERVER_URL = "http://localhost:8091/v1/images/generations"
HEALTH_URL = "http://localhost:8091/health"

BASE_OUTPUT_DIR = Path(
    "/outputs/qwen_image_vllm_omni_concurrency_benchmark"
)

MODEL = "Qwen/Qwen-Image"

PROMPT = (
    "A red sports car parked on a quiet city street at sunset, "
    "with realistic lighting, detailed buildings, and reflections "
    "on the wet pavement."
)

WIDTH = 1024
HEIGHT = 1024

NUM_INFERENCE_STEPS = 20

TRUE_CFG_SCALE = 1.0

IMAGES_PER_REQUEST = 1

CONCURRENCY_LEVELS = [1, 2, 4, 8]

DEFAULT_MEASURED_REQUESTS = 24

# The measured workload for every concurrency level and every
# server mode begins at seed 42.
BASE_SEED = 42


# ============================================================
# SERVER-MODE METADATA
# ============================================================

SERVER_MODE_METADATA = {
    "serial": {
        "expected_max_num_seqs": 1,
        "expected_step_execution": False,
        "expected_request_batch_max_wait_ms": 0,
    },

    "request_batch": {
        "expected_max_num_seqs": 8,
        "expected_step_execution": False,
        "expected_request_batch_max_wait_ms": 20,
    },

    "step_batch": {
        "expected_max_num_seqs": 8,
        "expected_step_execution": True,
        "expected_request_batch_max_wait_ms": None,
    },
}


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qwen-Image serving throughput "
            "and latency under concurrent load."
        )
    )

    parser.add_argument(
        "--server-mode",
        required=True,
        choices=[
            "serial",
            "request_batch",
            "step_batch",
        ],
        help=(
            "Execution mode of the currently running "
            "vLLM-Omni server."
        ),
    )

    parser.add_argument(
        "--measured-requests",
        type=int,
        default=DEFAULT_MEASURED_REQUESTS,
        help=(
            "Measured requests per concurrency level. "
            "Default: 24"
        ),
    )

    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=CONCURRENCY_LEVELS,
        help=(
            "Concurrency levels to test. "
            "Default: 1 2 4 8"
        ),
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_args(args):
    if args.measured_requests < 1:
        raise ValueError(
            "--measured-requests must be >= 1"
        )

    for concurrency in args.concurrency:
        if concurrency < 1:
            raise ValueError(
                "Every concurrency level must be >= 1"
            )

        if (
            args.measured_requests
            % concurrency
            != 0
        ):
            raise ValueError(
                f"--measured-requests="
                f"{args.measured_requests} "
                f"is not divisible by "
                f"concurrency={concurrency}. "
                f"Use a request count that produces "
                f"complete concurrent waves."
            )


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
# PERCENTILE
# ============================================================

def percentile_nearest_rank(
    values,
    percentile,
):
    """
    Calculate a percentile using the nearest-rank method.

    Example:

        P95 rank = ceil(0.95 * N)
    """

    if not values:
        return None

    ordered = sorted(
        values
    )

    rank = math.ceil(
        percentile * len(ordered)
    )

    rank = max(
        1,
        min(
            rank,
            len(ordered),
        ),
    )

    return ordered[
        rank - 1
    ]


# ============================================================
# SINGLE REQUEST
# ============================================================

def send_request(
    request_id,
    workload_index,
    concurrency,
    measured,
):
    """
    Execute one Qwen-Image request.

    Important:
    ----------
    The random seed depends ONLY on workload_index.

    It does NOT depend on:
    - request_id
    - concurrency
    - server mode

    This guarantees that every benchmark configuration receives
    the same workload seed sequence.
    """

    seed = (
        BASE_SEED
        + workload_index
    )

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "size": f"{WIDTH}x{HEIGHT}",
        "num_inference_steps": (
            NUM_INFERENCE_STEPS
        ),
        "seed": seed,
        "true_cfg_scale": (
            TRUE_CFG_SCALE
        ),
    }

    start_wall = (
        time.perf_counter()
    )

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=1800,
    )

    end_wall = (
        time.perf_counter()
    )

    response.raise_for_status()

    response_json = (
        response.json()
    )

    metrics = (
        response_json.get(
            "metrics",
            {},
        )
    )

    stage_durations = (
        metrics.get(
            "stage_durations",
            {},
        )
    )

    return {
        "request_id": request_id,

        "workload_index": (
            workload_index
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "measured": measured,

        "concurrency": concurrency,

        "seed": seed,

        "wall_latency_s": (
            end_wall
            - start_wall
        ),

        "queue_wait_ms": (
            stage_durations.get(
                "queue_wait_ms"
            )
        ),

        "stage_0_gen_ms": (
            stage_durations.get(
                "stage_0_gen_ms"
            )
        ),

        "peak_memory_mb": (
            metrics.get(
                "peak_memory_mb"
            )
        ),
    }


# ============================================================
# CONCURRENT REQUEST SET
# ============================================================

def run_requests(
    total_requests,
    concurrency,
    starting_request_id,
    measured,
):
    """
    Run a fixed workload containing total_requests requests.

    workload_index ALWAYS starts at zero for every benchmark level.

    Therefore, if total_requests=24:

        workload_index = 0..23
        seed           = 42..65

    regardless of concurrency or server mode.
    """

    results = []

    benchmark_start = (
        time.perf_counter()
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = []

        for workload_index in range(
            total_requests
        ):
            request_id = (
                starting_request_id
                + workload_index
            )

            future = executor.submit(
                send_request,
                request_id,
                workload_index,
                concurrency,
                measured,
            )

            futures.append(
                future
            )

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):
            results.append(
                future.result()
            )

    benchmark_end = (
        time.perf_counter()
    )

    elapsed_s = (
        benchmark_end
        - benchmark_start
    )

    return (
        results,
        elapsed_s,
    )


# ============================================================
# STATISTICS
# ============================================================

def summarize_level(
    records,
    elapsed_s,
):
    latencies = [
        record[
            "wall_latency_s"
        ]
        for record in records
    ]

    queue_values = [
        record[
            "queue_wait_ms"
        ]
        for record in records
        if record[
            "queue_wait_ms"
        ] is not None
    ]

    stage_values = [
        record[
            "stage_0_gen_ms"
        ] / 1000
        for record in records
        if record[
            "stage_0_gen_ms"
        ] is not None
    ]

    memory_values = [
        record[
            "peak_memory_mb"
        ]
        for record in records
        if record[
            "peak_memory_mb"
        ] is not None
    ]

    return {
        "completed_requests": (
            len(records)
        ),

        "benchmark_wall_s": (
            elapsed_s
        ),

        "throughput_images_per_s": (
            len(records)
            / elapsed_s
        ),

        "latency_mean_s": (
            statistics.mean(
                latencies
            )
        ),

        "latency_median_s": (
            statistics.median(
                latencies
            )
        ),

        "latency_p90_s": (
            percentile_nearest_rank(
                latencies,
                0.90,
            )
        ),

        "latency_p95_s": (
            percentile_nearest_rank(
                latencies,
                0.95,
            )
        ),

        "latency_max_s": (
            max(
                latencies
            )
        ),

        "queue_mean_ms": (
            statistics.mean(
                queue_values
            )
            if queue_values
            else None
        ),

        "queue_p95_ms": (
            percentile_nearest_rank(
                queue_values,
                0.95,
            )
            if queue_values
            else None
        ),

        "stage_0_mean_s": (
            statistics.mean(
                stage_values
            )
            if stage_values
            else None
        ),

        "peak_memory_max_mb": (
            max(
                memory_values
            )
            if memory_values
            else None
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    validate_args(
        args
    )

    mode_metadata = (
        SERVER_MODE_METADATA[
            args.server_mode
        ]
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        BASE_OUTPUT_DIR
        / args.server_mode
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Console configuration
    # --------------------------------------------------------

    print()
    print(
        "======================================================"
    )
    print(
        "Qwen-Image vLLM-Omni Concurrency Benchmark"
    )
    print(
        "======================================================"
    )
    print()

    print(
        f"Server mode:        "
        f"{args.server_mode}"
    )

    print(
        f"Model:              "
        f"{MODEL}"
    )

    print(
        f"Resolution:         "
        f"{WIDTH}x{HEIGHT}"
    )

    print(
        f"Inference steps:    "
        f"{NUM_INFERENCE_STEPS}"
    )

    print(
        f"True CFG scale:     "
        f"{TRUE_CFG_SCALE}"
    )

    print(
        f"Concurrency levels: "
        f"{args.concurrency}"
    )

    print(
        f"Requests/level:     "
        f"{args.measured_requests}"
    )

    print(
        f"Measured seeds:     "
        f"{BASE_SEED}.."
        f"{BASE_SEED + args.measured_requests - 1}"
    )

    print(
        f"Expected max seqs:  "
        f"{mode_metadata['expected_max_num_seqs']}"
    )

    print(
        f"Expected step mode: "
        f"{mode_metadata['expected_step_execution']}"
    )

    print(
        f"Output directory:   "
        f"{output_dir}"
    )

    print()

    # --------------------------------------------------------
    # Configuration record
    # --------------------------------------------------------

    config = {
        "experiment": (
            "Qwen-Image vLLM-Omni "
            "Concurrency Benchmark"
        ),

        "server_mode": (
            args.server_mode
        ),

        "expected_server_configuration": (
            mode_metadata
        ),

        "model": MODEL,

        "backend": "vLLM-Omni",

        "precision": "BF16",

        "prompt": PROMPT,

        "width": WIDTH,

        "height": HEIGHT,

        "num_inference_steps": (
            NUM_INFERENCE_STEPS
        ),

        "true_cfg_scale": (
            TRUE_CFG_SCALE
        ),

        "cfg_enabled": False,

        "images_per_request": (
            IMAGES_PER_REQUEST
        ),

        "concurrency_levels": (
            args.concurrency
        ),

        "measured_requests_per_level": (
            args.measured_requests
        ),

        "base_seed": BASE_SEED,

        "measured_seed_start": (
            BASE_SEED
        ),

        "measured_seed_end": (
            BASE_SEED
            + args.measured_requests
            - 1
        ),

        "same_workloads_across_levels": (
            True
        ),

        "same_workloads_across_server_modes": (
            True
        ),
    }

    (
        output_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    check_server()

    raw_path = (
        output_dir
        / "raw.jsonl"
    )

    summary_rows = []

    next_request_id = 1

    # --------------------------------------------------------
    # Concurrency sweep
    # --------------------------------------------------------

    for concurrency in (
        args.concurrency
    ):

        print()
        print(
            "------------------------------------------------------"
        )

        print(
            f"Concurrency = "
            f"{concurrency}"
        )

        print(
            "------------------------------------------------------"
        )

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------
        #
        # Perform one complete concurrent wave.
        #
        # The warmup is discarded from all metrics.
        # ----------------------------------------------------

        print(
            f"Warmup wave: "
            f"{concurrency} request(s)"
        )

        (
            _,
            warmup_elapsed,
        ) = run_requests(
            total_requests=concurrency,
            concurrency=concurrency,
            starting_request_id=(
                next_request_id
            ),
            measured=False,
        )

        next_request_id += (
            concurrency
        )

        print(
            f"Warmup completed in "
            f"{warmup_elapsed:.3f} s"
        )

        # ----------------------------------------------------
        # Measured fixed workload
        # ----------------------------------------------------

        print(
            f"Measured workload: "
            f"{args.measured_requests} requests"
        )

        print(
            f"Seeds: "
            f"{BASE_SEED}.."
            f"{BASE_SEED + args.measured_requests - 1}"
        )

        (
            records,
            elapsed_s,
        ) = run_requests(
            total_requests=(
                args.measured_requests
            ),
            concurrency=concurrency,
            starting_request_id=(
                next_request_id
            ),
            measured=True,
        )

        next_request_id += (
            args.measured_requests
        )

        # ----------------------------------------------------
        # Preserve raw results immediately
        # ----------------------------------------------------

        with raw_path.open(
            "a"
        ) as file:

            for record in records:
                record[
                    "server_mode"
                ] = (
                    args.server_mode
                )

                file.write(
                    json.dumps(
                        record
                    )
                    + "\n"
                )

        # ----------------------------------------------------
        # Summarize
        # ----------------------------------------------------

        stats = summarize_level(
            records,
            elapsed_s,
        )

        row = {
            "server_mode": (
                args.server_mode
            ),

            "concurrency": (
                concurrency
            ),

            "seed_start": (
                BASE_SEED
            ),

            "seed_end": (
                BASE_SEED
                + args.measured_requests
                - 1
            ),

            **stats,
        }

        summary_rows.append(
            row
        )

        print()

        print(
            f"Throughput: "
            f"{stats['throughput_images_per_s']:.5f} img/s"
        )

        print(
            f"Mean latency: "
            f"{stats['latency_mean_s']:.3f} s"
        )

        print(
            f"P50: "
            f"{stats['latency_median_s']:.3f} s"
        )

        print(
            f"P90: "
            f"{stats['latency_p90_s']:.3f} s"
        )

        print(
            f"P95: "
            f"{stats['latency_p95_s']:.3f} s"
        )

        print(
            f"Max: "
            f"{stats['latency_max_s']:.3f} s"
        )

        if (
            stats[
                "queue_mean_ms"
            ]
            is not None
        ):
            print(
                f"Mean queue wait: "
                f"{stats['queue_mean_ms']:.3f} ms"
            )

        if (
            stats[
                "peak_memory_max_mb"
            ]
            is not None
        ):
            print(
                f"Peak memory: "
                f"{stats['peak_memory_max_mb']:.1f} MB"
            )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "concurrency_summary.csv"
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
    # Console summary table
    # --------------------------------------------------------

    print()
    print(
        "======================================================"
    )
    print(
        "Concurrency Summary"
    )
    print(
        "======================================================"
    )

    print()
    print(
        "C | Throughput | Mean | "
        "P50 | P90 | P95 | Max"
    )

    print(
        "------------------------------------------------------------"
    )

    for row in summary_rows:
        print(
            f"{row['concurrency']:1d} | "
            f"{row['throughput_images_per_s']:.5f} | "
            f"{row['latency_mean_s']:.2f} | "
            f"{row['latency_median_s']:.2f} | "
            f"{row['latency_p90_s']:.2f} | "
            f"{row['latency_p95_s']:.2f} | "
            f"{row['latency_max_s']:.2f}"
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