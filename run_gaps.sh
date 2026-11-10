#!/usr/bin/env bash
# Is the held-out RAG underprediction the GPU's power between requests? One cache-on server,
# everything traced, request shapes taken from the held-out workloads (RAG: ~5.5k-token
# prompt, 21 output tokens; agent: ~3.1k, 73):
#   1. trace check: the reference window with and without --trace, to see the sampling costs nothing
#   2. decay: closed loop at c=1 with a pause after every response, 0 to 5 s
#   3. 2x2: 1 vs 20 cached prefixes x no pause vs 0.7 s (RAG at 1 request/s is busy ~0.3 s of each 1 s)
#   4. traced replays of the RAG and agent workloads at 1 request/s
#   bash run_gaps.sh > gaps.log 2>&1
# Qwen3: MODEL=Qwen/Qwen3-8B WORKLOADS_DIR=workloads_qwen3 CSV=results_gaps_qwen3.csv bash run_gaps.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
CSV=${CSV:-results_gapstest.csv}
WORKLOADS_DIR=${WORKLOADS_DIR:-workloads}
TRACE=${TRACE:-0.02}
PORT=30000

# Same pinned flags as run_all.sh.
SERVER_FLAGS="--model-path $MODEL --dtype bfloat16 --port $PORT --schedule-policy fcfs
              --page-size 1 --chunked-prefill-size 8192"

SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true' EXIT

start_server() {
    python -m sglang.launch_server $SERVER_FLAGS "$@" > "server_$(date +%H%M%S).log" 2>&1 &
    SERVER_PID=$!
    until curl -sf "localhost:$PORT/health_generate" > /dev/null; do
        kill -0 $SERVER_PID 2>/dev/null || { echo "server died while loading, see server_*.log"; exit 1; }
        sleep 5
    done
    sleep 30
}

check_cache() {
    local out
    out=$(python harness.py check "$@")
    echo "$out"
    if grep -q FAIL <<< "$out"; then
        echo "cache check failed, stopping before measuring anything"
        exit 1
    fi
}

measure_idle() {
    local out
    out=$(python harness.py idle)
    echo "$out" >&2
    awk '/^idle:/ {print $2}' <<< "$out"
}

{
    date
    nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.total --format=csv
    nvidia-smi | grep "CUDA Version"
    python -c "import importlib.metadata as m; print('sglang', m.version('sglang'))"
    hostname
} > "environment_$(date +%m%d_%H%M).txt"

sweep() {
    python harness.py sweep --url "http://localhost:$PORT" "$@"
}

echo "== cache on == $(date +%H:%M:%S)"
start_server
check_cache --url "http://localhost:$PORT" --modes miss,hit --L 8192
W=$(measure_idle)

# The usual reference window goes to a CSV of its own, since traced rows have extra columns.
sweep --csv "${CSV%.csv}_ref.csv" --modes miss --L 2048 --out 1 --conc 1 --repeats 3 --tag reference --idle-watts "$W"
sweep --csv "$CSV" --modes miss --L 2048 --out 1 --conc 1 --repeats 3 --tag reference-traced --idle-watts "$W" --trace "$TRACE"
echo "TRACE_CHECK_DONE $(date +%H:%M:%S)"

# Long pauses would make 40-request windows take minutes; 10 requests and 30 s is enough here.
sweep --csv "$CSV" --modes hit --L 5440 --out 21 --think 0,0.1,0.25,0.5,1,2,5 --repeats 3 \
    --min-requests 10 --tag decay-rag --idle-watts "$W" --trace "$TRACE"
sweep --csv "$CSV" --modes hit --L 3040 --out 73 --think 0,0.25,0.5,1,2 --repeats 3 \
    --min-requests 10 --tag decay-agent --idle-watts "$W" --trace "$TRACE"
echo "DECAY_DONE $(date +%H:%M:%S)"

sweep --csv "$CSV" --modes hit --L 5440 --out 21 --think 0,0.7 --prefixes 1,20 --repeats 4 \
    --min-requests 10 --tag 2x2 --idle-watts "$W" --trace "$TRACE"
echo "2X2_DONE $(date +%H:%M:%S)"

python harness.py replay --url "http://localhost:$PORT" --csv "${CSV%.csv}_replay.csv" --tag replay-traced \
    --workloads "$WORKLOADS_DIR/rag.jsonl,$WORKLOADS_DIR/agent.jsonl" --arrival fixed --rate 1 \
    --idle-watts "$W" --trace "$TRACE"
echo "ALL_DONE $(date +%H:%M:%S)"
