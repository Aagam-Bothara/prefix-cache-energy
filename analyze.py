"""
Turns the results.csv written by harness.py into the numbers the report needs.

  noise floor  CV of active J/request over the repeated reference windows
  D(L)         (E_miss(L) - E_hit(L)) / (E_miss(L) - E_miss(0)), and J saved per reused token
  delta_on     E_miss(L) - E_off(L): what having the cache on costs when nothing is reused

E is the mean active energy per request over a cell's windows. The intervals are 95%
t-intervals over window means (Welch-Satterthwaite degrees of freedom; delta method for D).
A bootstrap would be too narrow here: 3 windows give only 10 distinct resamples per cell.

Example: python analyze.py results.csv
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict

import scipy.stats

mean = statistics.fmean


def load(path, tag):
    cells = defaultdict(list)   # (mode, L, out, conc) -> active J/request of each window
    dropped = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["tag"] != tag:
                continue
            if int(row["cache_mismatches"]):
                dropped += 1    # the harness already flags these, they never count
                continue
            key = (row["mode"], int(row["L"]), int(row["out"]), int(row["conc"]))
            cells[key].append(float(row["active_j_per_req"]))
    return cells, dropped


def t_interval(est, groups, grads):
    # 95% interval for a statistic of independent cell means, linearized as sum(grad * mean):
    # its variance is sum(grad^2 * s^2 / n), degrees of freedom by Welch-Satterthwaite.
    terms = [g * g * statistics.variance(x) / len(x) for g, x in zip(grads, groups)]
    if not sum(terms):
        return est, est
    df = sum(terms) ** 2 / sum(t * t / (len(x) - 1) for t, x in zip(terms, groups))
    half = scipy.stats.t.ppf(0.975, df) * math.sqrt(sum(terms))
    return est - half, est + half


def discount(miss, hit, miss0):
    return (mean(miss) - mean(hit)) / (mean(miss) - mean(miss0))


def discount_ci(miss, hit, miss0):
    m, h, z = mean(miss), mean(hit), mean(miss0)
    grads = [(h - z) / (m - z) ** 2, -1 / (m - z), (m - h) / (m - z) ** 2]
    return t_interval(discount(miss, hit, miss0), [miss, hit, miss0], grads)


def saved(a, b):
    return mean(a) - mean(b)


def saved_ci(a, b):
    return t_interval(saved(a, b), [a, b], [1, -1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="results.csv")
    p.add_argument("--tag", default="main", help="rows used for D(L) and delta_on")
    p.add_argument("--reference-tag", default="reference", help="rows used for the noise floor")
    args = p.parse_args()

    ref, _ = load(args.csv, args.reference_tag)
    print("noise floor")
    for (mode, L, out, conc), v in sorted(ref.items()):
        cv = f"CV {100 * statistics.stdev(v) / mean(v):.1f}%" if len(v) > 1 else "need 2+ windows for a CV"
        print(f"  {mode} L={L} out={out} c={conc}: {len(v)} windows, {mean(v):.4f} J/req, {cv}")

    cells, dropped = load(args.csv, args.tag)
    print(f"\n{args.tag}: {sum(map(len, cells.values()))} windows used, {dropped} dropped for cache mismatches")

    for out, conc in sorted({(k[2], k[3]) for k in cells}):
        lengths = sorted({k[1] for k in cells if k[2:] == (out, conc)})
        get = lambda mode, L: cells.get((mode, L, out, conc))

        # D(L) needs the L=0 miss cell as its floor, so without it there's nothing to print.
        miss0 = get("miss", 0)
        if miss0:
            print(f"\nD(L), out={out} c={conc}  (E = active J/request, [95% CI])")
            for L in lengths:
                miss, hit = get("miss", L), get("hit", L)
                if L == 0 or not (miss and hit):
                    continue
                d_lo, d_hi = discount_ci(miss, hit, miss0)
                s_lo, s_hi = saved_ci(miss, hit)
                print(f"  L={L:<5} E_miss {mean(miss):.4f}  E_hit {mean(hit):.4f}  "
                      f"D {discount(miss, hit, miss0):.3f} [{d_lo:.3f}, {d_hi:.3f}]  "
                      f"saved/token {saved(miss, hit) / L:.3e} J [{s_lo / L:.3e}, {s_hi / L:.3e}]")

        # Miss and off rows come from two different server runs, so this only shows up
        # once both blocks are in the CSV.
        pairs = [(L, get("miss", L), get("off", L)) for L in lengths]
        pairs = [(L, miss, off) for L, miss, off in pairs if miss and off]
        if pairs:
            print(f"\ndelta_on = E_miss - E_off, out={out} c={conc}")
            for L, miss, off in pairs:
                lo, hi = saved_ci(miss, off)
                print(f"  L={L:<5} {saved(miss, off):+.4f} J/req [{lo:+.4f}, {hi:+.4f}]  "
                      f"({100 * saved(miss, off) / mean(off):+.1f}% of off)")


if __name__ == "__main__":
    main()
