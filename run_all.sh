#!/usr/bin/env bash
# One full measurement session on the pod: the cache-on block (miss, hit), then the
# cache-off block (off), then the analysis. Run it inside tmux so a dropped SSH connection
# doesn't kill it:
#   bash run_all.sh 2>&1 | tee session_$(date +%m%d_%H%M).log
# Grid settings can be overridden from the command line, e.g.  OUT=128 CONC=4,16 bash run_all.sh
# WORKLOADS=workloads/a.jsonl,... also replays held-out workloads on the cache-on server, and
# OFF=0 skips the cache-off block (fine once delta_on is known to be zero).
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
CSV=${CSV:-results.csv}
L=${L:-0,512,1024,2048,4096,8192}
OUT=${OUT:-1}
CONC=${CONC:-1}
MIN_REQUESTS=${MIN_REQUESTS:-40}    # long outputs at c=1 take minutes per 40 requests; 30 s still applies
WORKLOADS=${WORKLOADS:-}
HELDOUT_CSV=${HELDOUT_CSV:-results_heldout.csv}
OFF=${OFF:-1}
BACKEND=${BACKEND:-sglang}          # vllm only for the PrefixBench-style reproduction
PORT=30000

# Everything the report says is held fixed. Both servers get exactly these flags,
# the cache-off one just adds --disable-radix-cache.
SERVER_FLAGS="--model-path $MODEL --dtype bfloat16 --port $PORT --schedule-policy fcfs
              --page-size 1 --chunked-prefill-size 8192"

SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true' EXIT

start_server() {
    local health=health_generate
    if [ "$BACKEND" = vllm ]; then
        # vLLM's nearest match to the pinned flags (its default scheduling is already FCFS).
        # Dev mode is what enables /reset_prefix_cache, which the harness flushes with.
        local off=""
        [ "${1:-}" = --disable-radix-cache ] && off=--no-enable-prefix-caching
        VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" --served-model-name m --dtype bfloat16 --port $PORT \
            --max-num-batched-tokens 8192 --enable-prompt-tokens-details $off \
            > "server_$(date +%H%M%S).log" 2>&1 &
        health=health
    else
        python -m sglang.launch_server $SERVER_FLAGS "$@" > "server_$(date +%H%M%S).log" 2>&1 &
    fi
    SERVER_PID=$!
    until curl -sf "localhost:$PORT/$health" > /dev/null; do
        kill -0 $SERVER_PID 2>/dev/null || { echo "server died while loading, see server_*.log"; exit 1; }
        sleep 5
    done
    sleep 30    # let the GPU settle back to idle after loading and CUDA graph capture
}

stop_server() {
    kill $SERVER_PID
    wait $SERVER_PID || true
    SERVER_PID=""
    sleep 10
}

check_cache() {
    # Nothing gets measured unless every request reports the cached_tokens its condition promises.
    local out
    out=$(python harness.py check --backend "$BACKEND" "$@")
    echo "$out"
    if grep -q FAIL <<< "$out"; then
        echo "cache check failed, stopping before measuring anything"
        exit 1
    fi
}

measure_idle() {
    # Prints the idle lines into the log and hands back just the watts. (Not `tee /dev/stderr`:
    # when stderr is a file, tee reopens it from the start and wipes the log written so far.)
    local out
    out=$(python harness.py idle)
    echo "$out" >&2
    awk '/^idle:/ {print $2}' <<< "$out"
}

# What the report's setup section asks for, saved once per session.
{
    date
    nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.total --format=csv
    nvidia-smi | grep "CUDA Version"
    python -c "import importlib.metadata as m; print('$BACKEND', m.version('$BACKEND'))"
    hostname
} > "environment_$(date +%m%d_%H%M).txt"

sweep() {
    python harness.py sweep --backend "$BACKEND" --url "http://localhost:$PORT" --csv "$CSV" "$@"
}

echo "== cache on =="
start_server
check_cache --url "http://localhost:$PORT" --modes miss,hit --L 8192
W=$(measure_idle)
# Same reference window at the start of every session, for the noise floor and for spotting drift.
sweep --modes miss --L 2048 --out 1 --conc 1 --repeats 3 --tag reference --idle-watts "$W"
sweep --modes miss,hit --L "$L" --out "$OUT" --conc "$CONC" --tag main --idle-watts "$W" --min-requests "$MIN_REQUESTS"
if [ -n "$WORKLOADS" ]; then
    python harness.py replay --backend "$BACKEND" --url "http://localhost:$PORT" --csv "$HELDOUT_CSV" --tag heldout \
        --workloads "$WORKLOADS" --idle-watts "$W"
fi
stop_server

if [ "$OFF" = 1 ]; then
    echo "== cache off =="
    start_server --disable-radix-cache
    check_cache --url "http://localhost:$PORT" --modes off --L 8192
    W=$(measure_idle)    # new server process, so idle is measured again
    sweep --modes off --L "$L" --out "$OUT" --conc "$CONC" --tag main --idle-watts "$W" --min-requests "$MIN_REQUESTS"
    stop_server
fi

python analyze.py "$CSV"
