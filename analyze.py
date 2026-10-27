"""
Turns the results.csv written by harness.py into the numbers the report needs.

  noise floor  CV of active J/request over the repeated reference windows
  D(L)         (E_miss(L) - E_hit(L)) / (E_miss(L) - E_miss(0)), and J saved per reused token
  delta_on     E_miss(L) - E_off(L): what having the cache on costs when nothing is reused

E is the mean active energy per request over a cell's windows. The intervals are 95%
bootstrap intervals over windows, so with only 3 windows per cell they're rough.

Example: python analyze.py results.csv
"""
import argparse
import csv
import random
import statistics
from collections import defaultdict

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


def bootstrap(stat, groups, rng, n=10_000):
    # Resample each cell's windows on its own, recompute the statistic, take the middle 95%.
    vals = sorted(stat(*[rng.choices(g, k=len(g)) for g in groups]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def discount(miss, hit, miss0):
    return (mean(miss) - mean(hit)) / (mean(miss) - mean(miss0))


def saved(a, b):
    return mean(a) - mean(b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="results.csv")
    p.add_argument("--tag", default="main", help="rows used for D(L) and delta_on")
    p.add_argument("--reference-tag", default="reference", help="rows used for the noise floor")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

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
                d_lo, d_hi = bootstrap(discount, [miss, hit, miss0], rng)
                s_lo, s_hi = bootstrap(saved, [miss, hit], rng)
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
                lo, hi = bootstrap(saved, [miss, off], rng)
                print(f"  L={L:<5} {saved(miss, off):+.4f} J/req [{lo:+.4f}, {hi:+.4f}]  "
                      f"({100 * saved(miss, off) / mean(off):+.1f}% of off)")


if __name__ == "__main__":
    main()
