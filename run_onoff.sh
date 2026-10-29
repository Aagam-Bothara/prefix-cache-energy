#!/usr/bin/env bash
# Delta_on rerun. In run_all.sh, miss comes from one server process and off from another, so a
# small miss-vs-off gap could just be the restart. Here the cache-on and cache-off servers take
# turns over 8 rounds in ABBA order (on off off on on off off on), each round a fresh server
# with its own idle measurement, so restart effects and slow drift land on both sides equally.
# After the rounds, one more cache-on server repeats the noisy out=512 c=4 cell at short L.
#   bash run_onoff.sh > onoff.log 2>&1
# BACKEND=vllm REPEAT=0 does the same rounds on vLLM, without the cell-4 repeat.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
CSV=${CSV:-results_onoff.csv}
L=${L:-0,2048,8192}
ORDER=${ORDER:-"on off off on on off off on"}
REPEAT=${REPEAT:-1}
BACKEND=${BACKEND:-sglang}
PORT=30000

# Same pinned flags as run_all.sh.
SERVER_FLAGS="--model-path $MODEL --dtype bfloat16 --port $PORT --schedule-policy fcfs
              --page-size 1 --chunked-prefill-size 8192"

SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true' EXIT

start_server() {
    local health=health_generate
    if [ "$BACKEND" = vllm ]; then
        # Same vLLM flags as run_all.sh.
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
    sleep 30
}

stop_server() {
    kill $SERVER_PID
    wait $SERVER_PID || true
    SERVER_PID=""
    sleep 10
}

check_cache() {
    local out
    out=$(python harness.py check --backend "$BACKEND" "$@")
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
    python -c "import importlib.metadata as m; print('$BACKEND', m.version('$BACKEND'))"
    hostname
} > "environment_$(date +%m%d_%H%M).txt"

sweep() {
    python harness.py sweep --backend "$BACKEND" --url "http://localhost:$PORT" --csv "$CSV" "$@"
}

round=0
for side in $ORDER; do
    round=$((round + 1))
    echo "== round $round: cache $side == $(date +%H:%M:%S)"
    if [ "$side" = on ]; then
        start_server
        mode=miss
    else
        start_server --disable-radix-cache
        mode=off
    fi
    check_cache --url "http://localhost:$PORT" --modes $mode --L 8192
    W=$(measure_idle)
    # The round number goes in the tag so the analysis can pair rounds and see restart noise.
    sweep --modes $mode --L "$L" --out 1 --conc 1 --repeats 2 --tag "onoff-r$round" --idle-watts "$W"
    sweep --modes $mode --L "$L" --out 128 --conc 4 --repeats 2 --tag "onoff-r$round" --idle-watts "$W"
    stop_server
done
echo "ONOFF_DONE $(date +%H:%M:%S)"
[ "$REPEAT" = 1 ] || { echo "ALL_DONE $(date +%H:%M:%S)"; exit 0; }

echo "== repeat out=512 c=4 at short L == $(date +%H:%M:%S)"
start_server
check_cache --url "http://localhost:$PORT" --modes miss,hit --L 1024
W=$(measure_idle)
sweep --modes miss,hit --L 0,512,1024 --out 512 --conc 4 --repeats 4 --tag cell4-repeat --idle-watts "$W"
stop_server
echo "ALL_DONE $(date +%H:%M:%S)"
