"""
Is the held-out RAG underprediction the GPU's power between requests, or reuse across many
cached documents? Reads a CSV from `harness.py sweep --think ... --prefixes ...`, and
optionally the power trace next to it (`--trace`).

  gap cost    E(think) - E(0) per request at the same prefix count, with a 95% Welch t-interval,
              and that divided by the pause: the average power above idle while waiting
  hold fit    P and T in "a quiet stretch of g seconds costs P * min(g, T)", the gap term
              fit_accounting.py takes (--gap-w P --gap-s T), from three or more think values
  2x2         prefix-count effect without pauses, pause effect with one prefix, and their
              interaction, when the CSV has exactly two think values and two prefix counts
  decay       from the trace: power above idle during quiet stretches (no request in flight),
              binned by time since the GPU last went quiet. Power comes from differencing
              the energy counter, so it is only as sharp as the counter's update interval.

The decay part also works on `harness.py replay --trace` output, grouped by workload and rate.

Example: python analyze_gaps.py results_gapstest.csv --tag decay-rag --trace
         python analyze_gaps.py results_gapstest.csv --tag 2x2
"""
import argparse
import bisect
import csv
import json
import statistics
from collections import defaultdict

import numpy as np

from analyze import saved, saved_ci

mean = statistics.fmean
BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, float("inf")]


def load_windows(path, tags):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r.get("cache_mismatches") or 0) or (tags and r["tag"] not in tags):
                continue
            rows.append(r)
    return rows


def gap_report(rows):
    cells = defaultdict(list)   # (mode, L, out, conc, prefixes, think) -> active J/request per window
    for r in rows:
        key = (r["mode"], int(r["L"]), int(r["out"]), int(r["conc"]), int(r["prefixes"]), float(r["think"]))
        cells[key].append(float(r["active_j_per_req"]))

    print("gap cost per request, E(think) - E(0) at the same prefix count  [95% CI]")
    for base_key in sorted(k for k in cells if k[5] == 0):
        base = cells[base_key]
        mode, L, out, conc, n = base_key[:5]
        print(f"  {mode} L={L} out={out} c={conc} prefixes={n}: E(0) {mean(base):.3f} J ({len(base)} windows)")
        for key in sorted(k for k in cells if k[:5] == base_key[:5] and k[5] > 0):
            v = cells[key]
            lo, hi = saved_ci(v, base)
            print(f"    think {key[5]:>5g}s  E {mean(v):.3f} J  gap cost {saved(v, base):+.3f} J [{lo:+.3f}, {hi:+.3f}]"
                  f"  = {saved(v, base) / key[5]:+.1f} W over the pause")

    fit_hold(cells)

    thinks = sorted({k[5] for k in cells})
    counts = sorted({k[4] for k in cells})
    if len(thinks) != 2 or len(counts) != 2:
        return
    print(f"\n2x2: prefixes {counts[0]} vs {counts[1]}, think {thinks[0]:g} vs {thinks[1]:g}s  (J/request [95% CI])")
    for shape in sorted({k[:4] for k in cells}):
        get = lambda n, th: cells.get((*shape, n, th))
        a, b, c, d = get(counts[0], thinks[0]), get(counts[1], thinks[0]), get(counts[0], thinks[1]), get(counts[1], thinks[1])
        if not (a and b and c and d):
            continue
        print(f"  {shape[0]} L={shape[1]} out={shape[2]} c={shape[3]}")
        for name, x, y in [("more prefixes, no pause", b, a), ("pause, one prefix", c, a), ("both", d, a)]:
            lo, hi = saved_ci(x, y)
            print(f"    {name:<24} {saved(x, y):+.3f} [{lo:+.3f}, {hi:+.3f}]  ({100 * saved(x, y) / mean(y):+.1f}%)")
        inter = mean(d) - mean(b) - mean(c) + mean(a)
        print(f"    {'interaction':<24} {inter:+.3f}")


def hold_fit(points):
    # Least squares for cost = P * min(pause, T): P in closed form for each T on a grid.
    best = None
    for T in np.arange(0.2, 5.001, 0.01):
        x = np.minimum([p for p, _ in points], T)
        y = np.array([c for _, c in points])
        P = float(x @ y / (x @ x))
        sse = float(((y - P * x) ** 2).sum())
        if best is None or sse < best[2]:
            best = (P, T, sse)
    return best[:2]


def fit_hold(cells):
    # The hold model for the accounting gap term: each quiet stretch of g seconds costs
    # P * min(g, T) joules. One point per window: its J/request minus the mean of the
    # no-pause windows of the same shape. 95% CIs from a bootstrap over windows.
    points = []
    for key, v in cells.items():
        base = cells.get((*key[:5], 0.0))
        if key[5] > 0 and base:
            points += [(key[5], e - mean(base)) for e in v]
    if len({p for p, _ in points}) < 3:
        return
    P, T = hold_fit(points)
    rng = np.random.default_rng(0)
    boot = [hold_fit([points[i] for i in rng.integers(len(points), size=len(points))]) for _ in range(300)]
    (p_lo, t_lo), (p_hi, t_hi) = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    print(f"\nhold fit over {len(points)} windows: P = {P:.1f} W [{p_lo:.1f}, {p_hi:.1f}] for up to "
          f"T = {T:.2f} s [{t_lo:.2f}, {t_hi:.2f}], at most {P * T:.1f} J per gap")
    print(f"  for fit_accounting.py: --gap-w {P:.1f} --gap-s {T:.2f}")


def busy_spans(reqs):
    # Merge request lifetimes into stretches when at least one request is in flight.
    spans = []
    for s, e in sorted((q["t_send"], q["t_done"]) for q in reqs):
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return [s for s, _ in spans], [e for _, e in spans]


def decay_report(csv_path, rows):
    base = csv_path.removesuffix(".csv")
    reqs = defaultdict(list)
    with open(base + "_requests.jsonl") as f:
        for line in f:
            q = json.loads(line)
            reqs[q["window_id"]].append(q)
    trace = defaultdict(list)
    with open(base + "_trace.csv", newline="") as f:
        for p in csv.DictReader(f):
            trace[p["window_id"]].append(p)

    def group(r):
        if "workload" in r:
            return f"{r['workload']} {r['arrival']} {float(r['rate']):g}/s"
        return f"{r['mode']} L={r['L']} out={r['out']} think={float(r['think']):g}s prefixes={r['prefixes']}"

    # group -> bin -> [seconds, joules above idle, power_w * s, power_instant_w * s, sm_mhz * s]
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0]))
    busy = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        wid = r["window_id"]
        if wid not in trace or wid not in reqs:
            continue
        idle_w = float(r["idle_w"])
        starts, ends = busy_spans(reqs[wid])
        pts = sorted(trace[wid], key=lambda p: float(p["t"]))
        for p0, p1 in zip(pts, pts[1:]):
            a, b = float(p0["t"]), float(p1["t"])
            dt = b - a
            if dt <= 0:
                continue
            excess = (float(p1["mj"]) - float(p0["mj"])) / 1000 - idle_w * dt
            i = bisect.bisect_right(starts, b) - 1
            if i < 0:
                continue                    # before the window's first request
            if ends[i] > a:                 # a request was in flight during this interval
                busy[group(r)][0] += dt
                busy[group(r)][1] += excess
                continue
            since = a - ends[i]
            k = bisect.bisect_right(BINS, since) - 1
            cell = acc[group(r)][k]
            cell[0] += dt
            cell[1] += excess
            cell[2] += float(p1["power_w"] or "nan") * dt
            cell[3] += float(p1["power_instant_w"] or "nan") * dt
            cell[4] += float(p1["sm_mhz"] or "nan") * dt

    print("\ndecay: power above idle while no request is in flight, by time since the GPU went quiet")
    for g in sorted(acc):
        quiet_s = sum(c[0] for c in acc[g].values())
        quiet_j = sum(c[1] for c in acc[g].values())
        busy_s, busy_j = busy[g]
        print(f"  {g}: quiet {quiet_s:.1f}s holding {quiet_j:.1f} J above idle; busy {busy_s:.1f}s, {busy_j:.1f} J")
        print(f"    {'since quiet':<14}{'seconds':>8}{'W above idle':>14}{'power_w':>9}{'instant':>9}{'SM MHz':>8}")
        for k in sorted(acc[g]):
            s, j, pw, inst, mhz = acc[g][k]
            label = f"{BINS[k]:g}-{BINS[k + 1]:g}s"
            print(f"    {label:<14}{s:>8.1f}{j / s:>14.1f}{pw / s:>9.1f}{inst / s:>9.1f}{mhz / s:>8.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--trace", action="store_true", help="also read <csv>_trace.csv and <csv>_requests.jsonl")
    p.add_argument("--tag", default="", help="comma-separated tags to keep, e.g. 2x2 (default: all)")
    args = p.parse_args()
    rows = load_windows(args.csv, set(filter(None, args.tag.split(","))))
    if rows and "think" in rows[0]:
        gap_report(rows)
    if args.trace:
        decay_report(args.csv, rows)


if __name__ == "__main__":
    main()
