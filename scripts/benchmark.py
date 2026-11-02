#!/usr/bin/env python3
"""Accuracy and speed benchmark on synthetic shoots with ground truth.

    python scripts/benchmark.py [--seeds 20] [--events 300]

Bracket columns score exposure brackets + focus stacks (exact frame sets).
Burst columns score burst grouping and, most importantly, confusion between
the two: burst frames swallowed by a bracket, and bracket frames put in a burst.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bracketeering.engine import detect  # noqa: E402
from bracketeering.synth import PROFILES, evaluate, evaluate_bursts, generate_labeled  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--profile", choices=sorted(PROFILES), action="append")
    args = ap.parse_args()
    print(f"{'':22} {'--------- brackets ---------':>34}  {'------------ bursts ------------':>34}")
    print(f"{'profile':22} {'groups':>7} {'exact P':>8} {'exact R':>8} {'pair F1':>8}  {'bursts':>7} {'exact R':>8} "
          f"{'B->brk':>8} {'brk->B':>8} {'shots/s':>9}")
    for name in args.profile or PROFILES:
        tg = pg = hits = 0
        pp = pr = 0.0
        tb = bh = bf = bfb = kf = kfb = 0
        n_shots = 0
        elapsed = 0.0
        for seed in range(args.seeds):
            shots, truth, bursts = generate_labeled(PROFILES[name], args.events, seed)
            t0 = time.perf_counter()
            groups = detect(shots)
            elapsed += time.perf_counter() - t0
            m = evaluate(groups, truth)
            tg, pg, hits = tg + m.truth_groups, pg + m.predicted_groups, hits + m.exact_hits
            pp, pr = pp + m.pair_precision, pr + m.pair_recall
            b = evaluate_bursts(groups, bursts, truth)
            tb, bh = tb + b.truth_bursts, bh + b.exact_hits
            bf, bfb, kf, kfb = bf + b.burst_frames, bfb + b.burst_frames_in_brackets, kf + b.bracket_frames, kfb + b.bracket_frames_in_bursts
            n_shots += len(shots)
        p, r = pp / args.seeds, pr / args.seeds
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"{name:22} {tg:7d} {hits / pg if pg else 1:8.3f} {hits / tg if tg else 1:8.3f} {f1:8.3f}  "
              f"{tb:7d} {bh / tb if tb else 1:8.3f} {bfb / bf if bf else 0:8.2%} {kfb / kf if kf else 0:8.2%} "
              f"{n_shots / elapsed:9.0f}")
    print("\nB->brk: share of burst frames wrongly put in a bracket/stack; brk->B: share of bracket frames put in a burst")


if __name__ == "__main__":
    main()
