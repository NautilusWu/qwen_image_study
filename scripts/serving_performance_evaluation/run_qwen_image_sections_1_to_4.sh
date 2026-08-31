#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Qwen-Image vLLM-Omni Sections 1–4 Master Runner
# ============================================================
#
# Runs:
#   Section 1  BF16 Single-Request Baseline
#   Section 2  Inference-Step Scaling
#   Section 3  True CFG Scaling
#   Section 4A Serial Serving
#   Section 4B Request-Level Batching
#   Section 4C Step-Wise Continuous Batching
#
# Usage:
#   cd /workspace/qwen-image
#   bash scripts/run_qwen_image_sections_1_to_4.sh
# ============================================================

PROJECT_DIR="/workspace/qwen-image"
OUTPUT_ROOT="/outputs/qwen_image_vllm_omni_sections_1_to_4_master"

MODEL="Qwen/Qwen-Image"
HOST="127.0.0.1"
PORT="8091"
HEALTH_URL="http://${HOST}:${PORT}/health"

SERVER_START_TIMEOUT_S=1200
SERVER_STOP_TIMEOUT_S=60
HEALTH_POLL_INTERVAL_S=5

BASELINE_SCRIPT="${PROJECT_DIR}/scripts/serving_performance_evaluation/qwen_image_vllm_omni_bf16_baseline.py"
STEPS_SCRIPT="${PROJECT_DIR}/scripts/serving_performance_evaluation/qwen_image_vllm_omni_inference_steps_sweep.py"
CFG_SCRIPT="${PROJECT_DIR}/scripts/serving_performance_evaluation/qwen_image_vllm_omni_true_cfg_scale_sweep.py"
CONCURRENCY_SCRIPT="${PROJECT_DIR}/scripts/serving_performance_evaluation/qwen_image_vllm_omni_concurrency_benchmark.py"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
MASTER_LOG="${MASTER_OUTPUT_DIR}/master.log"

SERVER_PID=""

mkdir -p "${MASTER_OUTPUT_DIR}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

log() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_command() {
    local name="$1"
    if ! command -v "${name}" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: ${name}" >&2
        exit 1
    fi
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: Required file not found: ${path}" >&2
        exit 1
    fi
}

validate_environment() {
    log "Validating environment"

    require_command python3
    require_command curl
    require_command vllm
    require_command setsid

    require_file "${BASELINE_SCRIPT}"
    require_file "${STEPS_SCRIPT}"
    require_file "${CFG_SCRIPT}"
    require_file "${CONCURRENCY_SCRIPT}"

    [[ -d "${PROJECT_DIR}" ]] || {
        echo "ERROR: Project directory not found: ${PROJECT_DIR}" >&2
        exit 1
    }

    [[ -d "/outputs" ]] || {
        echo "ERROR: /outputs is not mounted." >&2
        exit 1
    }

    cd "${PROJECT_DIR}"
    log "Environment validation: PASS"
}

server_is_healthy() {
    curl -fsS \
        --connect-timeout 2 \
        --max-time 5 \
        "${HEALTH_URL}" \
        >/dev/null 2>&1
}

wait_for_server() {
    local server_pid="$1"
    local description="$2"
    local start_time
    start_time="$(date +%s)"

    log "Waiting for server: ${description}"

    while true; do
        if server_is_healthy; then
            log "Server ready: ${description}"
            return 0
        fi

        if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
            echo "ERROR: Server exited before becoming healthy." >&2
            return 1
        fi

        local now
        now="$(date +%s)"
        if (( now - start_time >= SERVER_START_TIMEOUT_S )); then
            echo "ERROR: Server startup timed out after ${SERVER_START_TIMEOUT_S}s." >&2
            return 1
        fi

        sleep "${HEALTH_POLL_INTERVAL_S}"
    done
}

wait_for_server_down() {
    local start_time
    start_time="$(date +%s)"

    while server_is_healthy; do
        local now
        now="$(date +%s)"
        if (( now - start_time >= SERVER_STOP_TIMEOUT_S )); then
            echo "WARNING: Health endpoint still responds after ${SERVER_STOP_TIMEOUT_S}s." >&2
            return 1
        fi
        sleep 1
    done
}

start_server() {
    local mode="$1"
    shift

    if server_is_healthy; then
        echo "ERROR: A server is already responding at ${HEALTH_URL}." >&2
        echo "Stop the existing server before running this script." >&2
        exit 1
    fi

    local server_log="${MASTER_OUTPUT_DIR}/server_${mode}.log"

    log "Starting vLLM-Omni server: ${mode}"
    log "Server log: ${server_log}"

    setsid vllm serve "${MODEL}" \
        --omni \
        --port "${PORT}" \
        "$@" \
        >"${server_log}" 2>&1 &

    SERVER_PID="$!"
    log "Server PID: ${SERVER_PID}"

    if ! wait_for_server "${SERVER_PID}" "${mode}"; then
        echo "ERROR: Failed to start ${mode} server." >&2
        echo "Last 80 lines of server log:" >&2
        tail -n 80 "${server_log}" >&2 || true
        exit 1
    fi
}

stop_server() {
    if [[ -z "${SERVER_PID}" ]]; then
        return 0
    fi

    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        log "Stopping server process group: ${SERVER_PID}"
        kill -TERM -- "-${SERVER_PID}" >/dev/null 2>&1 || true

        local start_time
        start_time="$(date +%s)"

        while kill -0 "${SERVER_PID}" >/dev/null 2>&1; do
            local now
            now="$(date +%s)"
            if (( now - start_time >= SERVER_STOP_TIMEOUT_S )); then
                log "Server did not exit after ${SERVER_STOP_TIMEOUT_S}s; sending SIGKILL"
                kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 || true
                break
            fi
            sleep 1
        done

        wait "${SERVER_PID}" 2>/dev/null || true
    fi

    SERVER_PID=""
    wait_for_server_down || true
    log "Server stopped"
}

cleanup() {
    local exit_code=$?

    if [[ -n "${SERVER_PID}" ]]; then
        stop_server || true
    fi

    if (( exit_code != 0 )); then
        log "Master run FAILED with exit code ${exit_code}"
    fi

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

run_experiment() {
    local section_name="$1"
    shift

    log "============================================================"
    log "START: ${section_name}"
    log "Command: $*"
    log "============================================================"

    local start_time
    start_time="$(date +%s)"

    "$@"

    local end_time
    end_time="$(date +%s)"
    local elapsed=$(( end_time - start_time ))

    log "PASS: ${section_name}"
    log "Elapsed: ${elapsed} s"
}

write_manifest() {
    cat > "${MASTER_OUTPUT_DIR}/run_manifest.txt" <<MANIFEST
Qwen-Image vLLM-Omni Sections 1–4 Master Run

Timestamp: ${TIMESTAMP}
Project: ${PROJECT_DIR}
Model: ${MODEL}
Server: ${HOST}:${PORT}

Experiments:
Section 1  BF16 Single-Request Baseline
Section 2  Inference-Step Scaling
Section 3  True CFG Scaling
Section 4A Serial Serving
Section 4B Request-Level Batching
Section 4C Step-Wise Continuous Batching

Section 4 fixed workload:
Resolution: 1024x1024
Inference steps: 20
True CFG scale: 1.0
Measured requests per concurrency level: 24
Concurrency levels: 1, 2, 4, 8
Measured seeds: 42..65

Server modes:
A: --max-num-seqs 1
B: --max-num-seqs 8 --request-batch-max-wait-ms 20
C: --step-execution --max-num-seqs 8
MANIFEST
}

main() {
    validate_environment
    write_manifest

    log "Master output directory: ${MASTER_OUTPUT_DIR}"
    log "Starting Sections 1–4 reproducibility run"

    # ========================================================
    # SERVER A: Sections 1, 2, 3, 4A
    # ========================================================

    start_server \
        "serial" \
        --max-num-seqs 1

    run_experiment \
        "Section 1 — BF16 Single-Request Baseline" \
        python3 "${BASELINE_SCRIPT}" \
        --warmup-runs 1 \
        --measured-runs 5

    run_experiment \
        "Section 2 — Inference-Step Scaling" \
        python3 "${STEPS_SCRIPT}" \
        --warmup-runs 1 \
        --measured-runs 5

    run_experiment \
        "Section 3 — True CFG Scaling" \
        python3 "${CFG_SCRIPT}" \
        --warmup-runs 1 \
        --measured-runs 5

    run_experiment \
        "Section 4A — Serial Serving" \
        python3 "${CONCURRENCY_SCRIPT}" \
        --server-mode serial \
        --concurrency 1 2 4 8 \
        --measured-requests 24

    stop_server

    # ========================================================
    # SERVER B: Section 4B
    # ========================================================

    start_server \
        "request_batch" \
        --max-num-seqs 8 \
        --request-batch-max-wait-ms 20

    run_experiment \
        "Section 4B — Request-Level Batching" \
        python3 "${CONCURRENCY_SCRIPT}" \
        --server-mode request_batch \
        --concurrency 1 2 4 8 \
        --measured-requests 24

    stop_server

    # ========================================================
    # SERVER C: Section 4C
    # ========================================================

    start_server \
        "step_batch" \
        --step-execution \
        --max-num-seqs 8

    run_experiment \
        "Section 4C — Step-Wise Continuous Batching" \
        python3 "${CONCURRENCY_SCRIPT}" \
        --server-mode step_batch \
        --concurrency 1 2 4 8 \
        --measured-requests 24

    stop_server

    log "============================================================"
    log "ALL SECTIONS 1–4 COMPLETED SUCCESSFULLY"
    log "============================================================"
    log "Master logs: ${MASTER_OUTPUT_DIR}"
    log "Individual benchmark outputs remain under /outputs."
}

main "$@"