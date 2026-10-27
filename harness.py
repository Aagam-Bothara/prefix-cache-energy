"""
Energy harness for the CS555 prefix-caching study.

What it measures: average GPU energy per request for one of three conditions,
over a "window" of many requests, with the server's prefix cache in a known state.

  off   server started with --disable-radix-cache. Every request recomputes the prefix.
  miss  cache on, but each request's prefix starts with a token no other request has,
        so nothing can be reused. Same token counts as `hit`, so miss-vs-hit isolates reuse.
  hit   cache on, every request shares the exact same prefix. One warm-up request
        puts it in the cache first and is not counted.

Commands:
  check   send 5 requests per mode and print cached_tokens (run this first on a new server)
  idle    measure idle power with the model loaded
  sweep   run a shuffled grid of windows and append one CSV row per window

Example (cache-on server):  python harness.py sweep --modes miss,hit --L 0,512,2048,8192
Example (cache-off server): python harness.py sweep --modes off --L 0,512,2048,8192
"""
import argparse
import asyncio
import csv
import json
import os
import random
import socket
import time
import urllib.request

import aiohttp

# Prompts are random token IDs rather than text. That way the prefix length is exactly L
# tokens, and no chat template sneaks in shared tokens that would get cached by accident.
# IDs below 100k are ordinary tokens in both the Qwen and Llama vocabularies.
RAND_LO, RAND_HI = 1000, 20000        # filler tokens for prefixes and suffixes
UNIQUE_LO, UNIQUE_HI = 20001, 99999   # one per request, never used as filler so it can't collide


# ----------------------------------------------------------------------------- energy
class NvmlEnergy:
    """Reads the GPU's own cumulative energy counter (millijoules since driver load).

    Reading a cumulative counter at the start and end of a window beats sampling
    power and integrating it, since there's no sampling rate to pick and no gaps.
    """

    def __init__(self, gpu_index):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name  # older bindings return bytes

    def millijoules(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def gpu_state(self):
        # Logged per window because on a rented GPU I can't lock clocks or the power limit,
        # so I need to be able to spot a window that ran throttled or hot after the fact.
        nv, h = self.nv, self.handle
        return {
            "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "power_limit_w": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
            # The SGLang server is one process, so anything beyond that is someone else's.
            "other_procs": max(0, len(nv.nvmlDeviceGetComputeRunningProcesses(h)) - 1),
        }


class MockEnergy:
    """Fake counter served by mock_server.py, so the harness can be tested without a GPU."""

    name = "mock"

    def __init__(self, url):
        self.url = url

    def millijoules(self):
        with urllib.request.urlopen(f"{self.url}/mock_energy") as resp:
            return json.load(resp)["energy_mJ"]

    def gpu_state(self):
        return {"sm_mhz": "", "temp_c": "", "power_limit_w": "", "other_procs": ""}


# ----------------------------------------------------------------------------- prompts
def shared_prefix(L, seed):
    # Seeded by L so `off`, `miss` and `hit` all see the same prefix tokens for a given length.
    rng = random.Random(seed * 100_003 + L)
    return [rng.randint(RAND_LO, RAND_HI) for _ in range(L)]


def build_prompt(mode, prefix, suffix_len, rng, req_id):
    # The radix cache matches from the first token onward, so whatever has to differ between
    # requests must differ at the very first token where it starts. Random filler isn't
    # enough there, since with hundreds of requests per window two of them will pick the same
    # first token by chance and get a 1-token cache hit. A per-request ID token fixes that.
    marker = UNIQUE_LO + req_id
    prompt = list(prefix)
    if mode == "miss" and prompt:
        prompt[0] = marker               # breaks the match at position 0, keeps the length at L
    suffix = [marker] + [rng.randint(RAND_LO, RAND_HI) for _ in range(suffix_len - 1)]
    return prompt + suffix               # in `hit`, the marker stops the match at exactly L


def expected_cached_tokens(mode, L):
    return L if mode == "hit" else 0


# ----------------------------------------------------------------------------- server calls
async def send(session, url, prompt, out_tokens):
    body = {
        "input_ids": prompt,
        "sampling_params": {
            "max_new_tokens": out_tokens,
            "temperature": 0,
            # Without this, a request could stop early at an end-of-sequence token and do
            # less decode work than the condition it is supposed to represent.
            "ignore_eos": True,
        },
    }
    async with session.post(f"{url}/generate", json=body) as resp:
        resp.raise_for_status()
        meta = (await resp.json())["meta_info"]
    if meta["completion_tokens"] != out_tokens:
        raise RuntimeError(f"asked for {out_tokens} output tokens, got {meta['completion_tokens']}")
    return meta["cached_tokens"]


async def flush_cache(session, url):
    # Start every window from an empty cache so one window's leftovers can't turn into
    # the next window's hits.
    async with session.post(f"{url}/flush_cache") as resp:
        resp.raise_for_status()
    await asyncio.sleep(1)


# ----------------------------------------------------------------------------- one window
async def run_window(args, meter, mode, L, out_tokens, conc, rep, idle_w):
    rng = random.Random(time.time_ns())
    prefix = shared_prefix(L, args.seed)
    no_total_timeout = aiohttp.ClientTimeout(total=None, sock_read=900)  # long windows are normal

    # aiohttp allows only 100 open connections by default, which would quietly cap conc at 100.
    async with aiohttp.ClientSession(timeout=no_total_timeout,
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        await flush_cache(session, args.url)

        # Warm-up: one request kept out of the window, in every mode, so no condition starts
        # from cold GPU clocks. In `hit` it's also the request that builds the cache entry.
        # Its cost is logged separately, since what I care about is what a reused token costs.
        e, t = meter.millijoules(), time.time()
        await send(session, args.url, build_prompt(mode, prefix, args.suffix, rng, 0), out_tokens)
        warmup_active_j = round((meter.millijoules() - e) / 1000 - idle_w * (time.time() - t), 4)
        await asyncio.sleep(0.5)

        cached_counts = []
        issued = 0
        state_before = meter.gpu_state()
        e_start, t_start = meter.millijoules(), time.time()

        async def client():
            # Closed loop: each client sends its next request as soon as the last one returns,
            # so exactly `conc` requests are in flight the whole time. The window only ends after
            # both enough requests and enough seconds. Short 1-token windows would otherwise be
            # done in a couple of seconds, which is too short to trust an energy delta.
            nonlocal issued
            while issued < args.min_requests or time.time() - t_start < args.min_seconds:
                issued += 1                  # request IDs start at 1; 0 belongs to the warm-up
                prompt = build_prompt(mode, prefix, args.suffix, rng, issued)
                cached_counts.append(await send(session, args.url, prompt, out_tokens))

        await asyncio.gather(*(client() for _ in range(conc)))
        e_end, t_end = meter.millijoules(), time.time()
        state_after = meter.gpu_state()

    if issued > UNIQUE_HI - UNIQUE_LO:
        raise RuntimeError("window issued more requests than there are unique marker tokens")

    n = len(cached_counts)
    seconds = t_end - t_start
    total_j = (e_end - e_start) / 1000
    # Active energy = what the work added on top of a loaded-but-idle GPU. The idle part
    # depends only on how long the window ran, not on caching, so it's removed.
    active_j = total_j - idle_w * seconds
    expected = expected_cached_tokens(mode, L)

    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),          # tells sessions on different rented machines apart
        "gpu": meter.name,
        "tag": args.tag,
        "mode": mode, "L": L, "suffix": args.suffix, "out": out_tokens, "conc": conc, "rep": rep,
        "n_requests": n,
        # Any request whose cache count isn't what the condition promises makes the window invalid.
        "cache_mismatches": sum(c != expected for c in cached_counts),
        "min_cached": min(cached_counts), "max_cached": max(cached_counts),
        "seconds": round(seconds, 3),
        "total_j": round(total_j, 3),
        "idle_w": round(idle_w, 3),
        "active_j": round(active_j, 3),
        "active_j_per_req": round(active_j / n, 5),
        "total_j_per_req": round(total_j / n, 5),
        "warmup_active_j": warmup_active_j,
        "sm_mhz_start": state_before["sm_mhz"], "sm_mhz_end": state_after["sm_mhz"],
        "temp_start": state_before["temp_c"], "temp_end": state_after["temp_c"],
        "power_limit_w": state_before["power_limit_w"],
        "other_procs": state_after["other_procs"],
    }


def save_row(path, row):
    # Written after every window, so a crash or a pod shutdown loses at most one window.
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def print_row(i, total, row):
    warn = "   <-- cache mismatch, don't use this window" if row["cache_mismatches"] else ""
    print(f"[{i}/{total}] {row['mode']:>4} L={row['L']:<5} out={row['out']:<4} c={row['conc']:<3} "
          f"rep={row['rep']}  n={row['n_requests']:<5} {row['seconds']:>6.1f}s  "
          f"{row['active_j_per_req']:.4f} J/req  cached {row['min_cached']}..{row['max_cached']}{warn}",
          flush=True)


def measure_idle(meter, seconds):
    # Run this with the server up and the model loaded but no requests. That's the baseline
    # I subtract. An empty GPU with no model loaded would read lower and give the wrong number.
    e, t = meter.millijoules(), time.time()
    time.sleep(seconds)
    return (meter.millijoules() - e) / 1000 / (time.time() - t)


# ----------------------------------------------------------------------------- check
async def check(args, modes, L):
    rng = random.Random(1)
    prefix = shared_prefix(L, args.seed)
    async with aiohttp.ClientSession() as session:
        for mode in modes:
            await flush_cache(session, args.url)
            if mode == "hit":
                first = await send(session, args.url, build_prompt(mode, prefix, args.suffix, rng, 0), 1)
                print(f" hit warm-up: cached_tokens = {first} (should be 0)")
            got = [await send(session, args.url, build_prompt(mode, prefix, args.suffix, rng, k), 1)
                   for k in range(1, 6)]
            want = expected_cached_tokens(mode, L)
            verdict = "PASS" if all(g == want for g in got) else "FAIL"
            print(f"{mode:>4} L={L}: cached_tokens = {got}, expected {want} -> {verdict}")


# ----------------------------------------------------------------------------- main
def int_list(text):
    return [int(x) for x in text.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["check", "idle", "sweep"])
    p.add_argument("--url", default="http://localhost:30000")
    p.add_argument("--energy", choices=["nvml", "mock"], default="nvml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--tag", default="", help="label for the rows, e.g. reference, pilot, main")
    p.add_argument("--seed", type=int, default=0, help="fixes the prefixes and the run order")
    p.add_argument("--modes", default="miss,hit", help="miss,hit on a cache-on server; off on a cache-off server")
    p.add_argument("--L", default="0,512,1024,2048,4096,8192", help="prefix lengths")
    p.add_argument("--out", default="1", help="output lengths")
    p.add_argument("--conc", default="1", help="requests in flight")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--suffix", type=int, default=64, help="unique tokens after the prefix")
    p.add_argument("--min-requests", type=int, default=40)
    p.add_argument("--min-seconds", type=float, default=30)
    p.add_argument("--idle-seconds", type=float, default=60)
    p.add_argument("--idle-watts", type=float, help="reuse an idle value measured earlier this session")
    args = p.parse_args()

    meter = MockEnergy(args.url) if args.energy == "mock" else NvmlEnergy(args.gpu)
    modes = args.modes.split(",")

    if args.cmd == "check":
        asyncio.run(check(args, modes, max(int_list(args.L))))
        return

    idle_w = args.idle_watts
    if idle_w is None:
        print(f"measuring idle power for {args.idle_seconds:.0f}s, don't send requests...", flush=True)
        idle_w = measure_idle(meter, args.idle_seconds)
    print(f"idle: {idle_w:.2f} W on {meter.name}", flush=True)
    if args.cmd == "idle":
        return

    windows = [(m, L, o, c, r)
               for m in modes for L in int_list(args.L) for o in int_list(args.out)
               for c in int_list(args.conc) for r in range(args.repeats)
               if not (m == "hit" and L == 0)]       # with no prefix, hit and miss are the same run
    # Shuffled so slow drift (the GPU warming up over an hour, say) spreads across all
    # conditions instead of landing on whichever one happened to run last.
    random.Random(args.seed).shuffle(windows)
    print(f"{len(windows)} windows -> {args.csv}", flush=True)

    for i, (mode, L, out_tokens, conc, rep) in enumerate(windows, 1):
        row = asyncio.run(run_window(args, meter, mode, L, out_tokens, conc, rep, idle_w))
        save_row(args.csv, row)
        print_row(i, len(windows), row)


if __name__ == "__main__":
    main()