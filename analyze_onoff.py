"""
Delta_on from run_onoff.sh, where cache-on (miss) and cache-off (off) servers took turns over
several rounds. Each round is one server process, so the unit of the bootstrap here is a
round, not a window: restart-to-restart noise is exactly what this run is meant to expose.

  restart noise  CV of the per-round means within one mode
  delta_on       mean over on-rounds minus mean over off-rounds, with a 95% bootstrap CI

Example: python analyze_onoff.py results_onoff.csv
"""
import argparse
import csv
import random
import statistics
from collections import defaultdict

from analyze import bootstrap, saved

mean = statistics.fmean


def load_rounds(path, prefix):
    # (mode, L, out, conc) -> {round: [active J/request of each window]}
    cells = defaultdict(lambda: defaultdict(list))
    dropped = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row["tag"].startswith(prefix):
                continue
            if int(row["cache_mismatches"]):
                dropped += 1
                continue
            key = (row["mode"], int(row["L"]), int(row["out"]), int(row["conc"]))
            cells[key][int(row["tag"][len(prefix):])].append(float(row["active_j_per_req"]))
    # Collapse each round to one number, so a round with more windows doesn't count for more.
    return {k: [mean(v) for _, v in sorted(rounds.items())] for k, rounds in cells.items()}, dropped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="results_onoff.csv")
    p.add_argument("--tag-prefix", default="onoff-r")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    cells, dropped = load_rounds(args.csv, args.tag_prefix)
    print(f"{sum(map(len, cells.values()))} round means, {dropped} windows dropped for cache mismatches")

    for out, conc in sorted({(k[2], k[3]) for k in cells}):
        print(f"\nout={out} c={conc}  (E = active J/request, one value per round)")
        for L in sorted({k[1] for k in cells if k[2:] == (out, conc)}):
            miss, off = cells.get(("miss", L, out, conc)), cells.get(("off", L, out, conc))
            if not (miss and off):
                continue
            cv = lambda v: 100 * statistics.stdev(v) / mean(v) if len(v) > 1 else float("nan")
            lo, hi = bootstrap(saved, [miss, off], rng)
            print(f"  L={L:<5} miss {mean(miss):.4f} ({len(miss)} rounds, CV {cv(miss):.1f}%)  "
                  f"off {mean(off):.4f} ({len(off)} rounds, CV {cv(off):.1f}%)  "
                  f"delta_on {saved(miss, off):+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"({100 * saved(miss, off) / mean(off):+.1f}% of off)")


if __name__ == "__main__":
    main()
