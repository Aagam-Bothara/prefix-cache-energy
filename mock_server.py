"""Fake SGLang server for testing harness.py without a GPU.

Mimics /generate (input_ids -> meta_info.cached_tokens), /flush_cache, /health,
with a simple prefix cache, plus /mock_energy: a fake cumulative energy counter
(60 W idle + fixed joules per fresh prefill token / cached token / output token).
Expected D(L) for 1-token outputs is about 1 - CACHED_J/FRESH_J = 0.9.

Usage: python mock_server.py [--disable-radix-cache] [--port 30000]
"""
import argparse, asyncio, time
from aiohttp import web

IDLE_W, FRESH_J, CACHED_J, OUT_J = 60.0, 0.002, 0.0002, 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--disable-radix-cache", action="store_true")
ap.add_argument("--port", type=int, default=30000)
args = ap.parse_args()

state = {"cache": [], "t0": time.time(), "work_J": 0.0}


def match_len(ids):
    if args.disable_radix_cache:
        return 0
    best = 0
    for seq in state["cache"]:
        n = 0
        for a, b in zip(seq, ids):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return min(best, len(ids) - 1)  # SGLang always recomputes at least one token


async def generate(req):
    body = await req.json()
    ids = body["input_ids"]
    out = body["sampling_params"]["max_new_tokens"]
    cached = match_len(ids)
    fresh = len(ids) - cached
    await asyncio.sleep(fresh * 2e-6 + out * 0.002)
    state["work_J"] += fresh * FRESH_J + cached * CACHED_J + out * OUT_J
    if not args.disable_radix_cache:
        state["cache"].append(list(ids))
    return web.json_response({"text": "x", "meta_info": {
        "prompt_tokens": len(ids), "completion_tokens": out, "cached_tokens": cached}})


async def flush(_):
    state["cache"].clear()
    return web.Response(text="Cache flushed.")


async def health(_):
    return web.Response(text="ok")


async def energy(_):
    j = IDLE_W * (time.time() - state["t0"]) + state["work_J"]
    return web.json_response({"energy_mJ": int(j * 1000)})


app = web.Application()
app.add_routes([web.post("/generate", generate), web.post("/flush_cache", flush),
                web.get("/health", health), web.get("/mock_energy", energy)])
web.run_app(app, port=args.port, print=lambda *_: print(f"mock server on :{args.port}, radix_cache={'off' if args.disable_radix_cache else 'on'}"))