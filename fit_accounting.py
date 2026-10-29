"""
RQ2: which accounting rule predicts measured energy best? All variants are Eq. 2 of the proposal,

  E = a * n_fresh + b * n_cached + c * n_out + d * n_out * n_ctx

with n_ctx the average context during decoding (prompt plus half the output so far), and they
only differ in how the prefill part is constrained:

  explicit  a and b both fitted, the model the proposal argues for
  unaware   b = a, a cached token costs as much as a fresh one
  pricing   b = 0.1a, the API-pricing rule
  flops     a*n_fresh + b*n_cached replaced by k * (remaining prefill FLOPs), k fitted

One fit per model and GPU, pooled over every load level, as the proposal says. --load-term
gives c a separate value per concurrency level, since batching makes decode steps cheaper per
request; comparing with and without it is the "is a load term needed" test.

Out-of-sample error: leave one prefix length out (from every load level at once), fit on
the rest, predict the held-out windows. Errors are per-request means of whole windows.

Example: python fit_accounting.py results/a100_2026-10-27/results.csv results/a100_2026-10-27/results_grid.csv
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np

# Shapes from each model's config.json, only used by the FLOP variant:
# (non-embedding parameters, layers, query width = heads * head_dim)
MODELS = {
    "qwen2.5-7b": (6.53e9, 28, 3584),
    "qwen3-8b": (6.95e9, 36, 4096),
}
PARAMS, LAYERS, HIDDEN = MODELS["qwen2.5-7b"]
VARIANTS = ["unaware", "pricing", "flops", "explicit"]


def prefill_flops(n_new, n_cached):
    # 2 FLOPs per parameter per new token, plus attention: each new token's QK^T and AV
    # cover every token before it, cached or not (4 * query width FLOPs per pair, per layer).
    pairs = n_new * n_cached + n_new * (n_new + 1) / 2
    return 2 * PARAMS * n_new + 4 * LAYERS * HIDDEN * pairs


def load(paths, tag):
    rows = []   # one dict per window
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["tag"] != tag or int(r["cache_mismatches"]):
                    continue
                L, suffix, out = int(r["L"]), int(r["suffix"]), int(r["out"])
                cached = L if r["mode"] == "hit" else 0
                rows.append({"L": L, "out": out, "conc": int(r["conc"]),
                             "fresh": L - cached + suffix, "cached": cached,
                             "ctx": L + suffix + (out - 1) / 2,
                             "energy": float(r["active_j_per_req"])})
    return rows


def load_weights(k, concs):
    # How much of each fitted level's c applies at k requests in flight: exact at a fitted
    # level, linear in log(k) between two of them, and clamped outside the fitted range.
    # Open-loop replays have a fractional, changing k, so they need the in-between values.
    if k <= concs[0]:
        return [1.0] + [0.0] * (len(concs) - 1)
    if k >= concs[-1]:
        return [0.0] * (len(concs) - 1) + [1.0]
    weights = [0.0] * len(concs)
    for i in range(len(concs) - 1):
        lo, hi = concs[i], concs[i + 1]
        if lo <= k <= hi:
            f = (np.log(k) - np.log(lo)) / (np.log(hi) - np.log(lo))
            weights[i], weights[i + 1] = 1 - f, f
            return weights


def features(variant, w, concs):
    if variant == "explicit":
        prefill = [w["fresh"], w["cached"]]
    elif variant == "unaware":
        prefill = [w["fresh"] + w["cached"]]
    elif variant == "pricing":
        prefill = [w["fresh"] + 0.1 * w["cached"]]
    else:
        prefill = [prefill_flops(w["fresh"], w["cached"]) / 1e12]
    kv_reads = [w["out"] * w["ctx"]]
    if concs is None:
        return prefill + [w["out"]] + kv_reads
    # Load term: one c per concurrency level. Batching shares the weight reads of a decode
    # step between requests, which is what c stands for. d stays shared, since every request
    # still reads its own KV cache however many others are in the batch.
    return prefill + [w["out"] * x for x in load_weights(w["conc"], concs)] + kv_reads


def load_heldout(csv_path):
    # Window rows from `harness.py replay` and their requests from the JSONL next to them.
    windows = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            windows[r["window_id"]] = {"row": r, "requests": []}
    with open(csv_path.removesuffix(".csv") + "_requests.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if q["window_id"] in windows:
                windows[q["window_id"]]["requests"].append(q)

    for w in windows.values():
        reqs = w["requests"]
        for q in reqs:
            # The request's load is the average number in flight over its own lifetime.
            span = max(q["t_done"] - q["t_send"], 1e-6)
            overlap = sum(max(0.0, min(q["t_done"], o["t_done"]) - max(q["t_send"], o["t_send"]))
                          for o in reqs)
            q.update(fresh=q["prompt"] - q["cached"], ctx=q["prompt"] + (q["out"] - 1) / 2,
                     conc=overlap / span)
    return list(windows.values())


def paired_test(diffs, blocks, rng, n=10_000):
    # Sign-flip permutation test on the mean paired difference. Windows replayed from the same
    # requests (same workload and window, other arrival or rate) are correlated, so their
    # differences are summed into one block and flipped together.
    sums = defaultdict(float)
    for d, b in zip(diffs, blocks):
        sums[b] += d
    s = np.array(list(sums.values()))
    observed = abs(s.mean())
    flips = rng.choice([-1, 1], size=(n, len(s)))
    return float(np.mean(np.abs((flips * s).mean(axis=1)) >= observed - 1e-12))


def heldout_report(train, windows, concs):
    print(f"\nheld-out: {len(windows)} replay windows, models fitted on all {len(train)} synthetic windows")
    groups = sorted({(w["row"]["workload"], w["row"]["arrival"], w["row"]["rate"]) for w in windows})
    errs = {}
    for v in VARIANTS:
        coef = fit(v, train, concs)
        # A window's prediction is the sum of its requests' estimates, compared with the
        # window's measured active energy.
        errs[v] = [(predict(v, coef, w["requests"], concs).sum() - float(w["row"]["active_j"]))
                   / float(w["row"]["active_j"]) for w in windows]

    head = "".join(f"  {g[0][:8]}/{g[1][:4]}/{float(g[2]):g}".rjust(20) for g in groups)
    print(f"  {'variant':<9} {'all':>6}{head}")
    for v in VARIANTS:
        cols = "".join(f"{100 * np.mean([abs(e) for e, w in zip(errs[v], windows) if (w['row']['workload'], w['row']['arrival'], w['row']['rate']) == g]):>19.1f}%"
                       for g in groups)
        print(f"  {v:<9} {100 * np.mean(np.abs(errs[v])):>5.1f}%{cols}")
    hits = [float(w["row"]["hit_rate"]) for w in windows]
    print(f"  (cached share of prompt tokens per window: {100 * min(hits):.0f}-{100 * max(hits):.0f}%)")

    rng = np.random.default_rng(0)
    blocks = [(w["row"]["workload"], w["row"]["window"]) for w in windows]
    print("\npaired test, |error| of explicit vs each baseline (negative = explicit is better)")
    for v in VARIANTS[:-1]:
        diffs = [abs(a) - abs(b) for a, b in zip(errs["explicit"], errs[v])]
        print(f"  explicit - {v:<8} {100 * np.mean(diffs):+6.1f} points   "
              f"p = {paired_test(diffs, blocks, rng):.4f}  ({len(set(blocks))} blocks)")


def fit(variant, data, concs):
    X = np.array([features(variant, w, concs) for w in data], dtype=float)
    y = np.array([w["energy"] for w in data])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict(variant, coef, data, concs):
    return np.array([features(variant, w, concs) for w in data], dtype=float) @ coef


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+")
    p.add_argument("--tag", default="main")
    p.add_argument("--model", choices=sorted(MODELS), default="qwen2.5-7b")
    p.add_argument("--load-term", action="store_true", help="separate decode coefficients per concurrency")
    p.add_argument("--heldout", help="CSV from `harness.py replay`, scored with models fitted on the synthetic CSVs")
    args = p.parse_args()
    global PARAMS, LAYERS, HIDDEN
    PARAMS, LAYERS, HIDDEN = MODELS[args.model]

    data = load(args.csv, args.tag)
    concs = sorted({w["conc"] for w in data}) if args.load_term else None
    levels = sorted({(w["out"], w["conc"]) for w in data})
    lengths = sorted({w["L"] for w in data})
    print(f"{len(data)} windows, load levels (out, conc) {levels}, "
          f"load term {'on' if concs else 'off'}")

    header = "".join(f"  {o}/{c:<6}" for o, c in levels)
    print(f"\nleave-one-L-out error, mean |pred - meas| / meas   (columns: out/conc)")
    print(f"  {'variant':<9} {'all':>6} {'MAE J':>7}{header}")
    for v in VARIANTS:
        pred = {}
        for L in lengths:
            train = [w for w in data if w["L"] != L]
            test = [w for w in data if w["L"] == L]
            for w, e in zip(test, predict(v, fit(v, train, concs), test, concs)):
                pred[id(w)] = e
        rel = lambda ws: 100 * np.mean([abs(pred[id(w)] - w["energy"]) / w["energy"] for w in ws])
        mae = np.mean([abs(pred[id(w)] - w["energy"]) for w in data])
        per_level = "".join(f"  {rel([w for w in data if (w['out'], w['conc']) == lv]):>8.1f}%"
                            for lv in levels)
        print(f"  {v:<9} {rel(data):>5.1f}% {mae:>7.2f}{per_level}")

    coef = fit("explicit", data, concs)
    print(f"\nexplicit, fitted on everything: a {1e3 * coef[0]:.2f} mJ/fresh token, "
          f"b {1e3 * coef[1]:.2f} mJ/cached token (b/a {coef[1] / coef[0]:.3f})")
    names = ["c"] if concs is None else [f"c[conc={k}]" for k in concs]
    names += ["d"]
    print("  " + ", ".join(f"{n} {1e3 * x:.4g} mJ" for n, x in zip(names, coef[2:])))

    if args.heldout:
        heldout_report(data, load_heldout(args.heldout), concs)


if __name__ == "__main__":
    main()
