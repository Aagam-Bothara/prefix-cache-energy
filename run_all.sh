#!/usr/bin/env bash
# One full measurement session on the pod: the cache-on block (miss, hit), then the
# cache-off block (off), then the analysis. Run it inside tmux so a dropped SSH connection
# doesn't kill it:
#   bash run_all.sh 2>&1 | tee session_$(date +%m%d_%H%M).log
# Grid settings can be overridden from the command line, e.g.  OUT=128 CONC=4,16 bash run_all.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
CSV=${CSV:-results.csv}
L=${L:-0,512,1024,2048,4096,8192}
OUT=${OUT:-1}
CONC=${CONC:-1}
PORT=30000

# Everything the report says is held fixed. Both servers get exactly these flags,
# the cache-off one just adds --disable-radix-cache.
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
    out=$(python harness.py check "$@")
    echo "$out"
    if grep -q FAIL <<< "$out"; then
        echo "cache check failed, stopping before measuring anything"
        exit 1
    fi
}

measure_idle() {
    # Prints the idle lines into the log and hands back just the watts.
    python harness.py idle | tee /dev/stderr | awk '/^idle:/ {print $2}'
}

# What the report's setup section asks for, saved once per session.
{
    date
    nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.total --format=csv
    nvidia-smi | grep "CUDA Version"
    python -c "import sglang; print('sglang', sglang.__version__)"
    hostname
} > "environment_$(date +%m%d_%H%M).txt"

sweep() {
    python harness.py sweep --url "http://localhost:$PORT" --csv "$CSV" "$@"
}

echo "== cache on =="
start_server
check_cache --url "http://localhost:$PORT" --modes miss,hit --L 8192
W=$(measure_idle)
# Same reference window at the start of every session, for the noise floor and for spotting drift.
sweep --modes miss --L 2048 --out 1 --conc 1 --repeats 3 --tag reference --idle-watts "$W"
sweep --modes miss,hit --L "$L" --out "$OUT" --conc "$CONC" --tag main --idle-watts "$W"
stop_server

echo "== cache off =="
start_server --disable-radix-cache
check_cache --url "http://localhost:$PORT" --modes off --L 8192
W=$(measure_idle)    # new server process, so idle is measured again
sweep --modes off --L "$L" --out "$OUT" --conc "$CONC" --tag main --idle-watts "$W"
stop_server

python analyze.py "$CSV"
