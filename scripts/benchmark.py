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
from bracketeering.synth import (  # noqa: E402
    ACTION_PROFILES, PROFILES, ActionMetrics, evaluate, evaluate_action, evaluate_bursts, generate_action, generate_labeled,
)


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
    if not args.profile:
        action(args.seeds, args.events)


def action(seeds: int, events: int) -> None:
    """Burst-focused shoots: continuous drive at 1.5-60 fps, re-presses, taps, interval timer, AEB mixed in."""
    print(f"\n{'action profile':22} {'bursts':>7} {'exact P':>8} {'exact R':>8} {'pair F1':>8} {'merged':>7} "
          f"{'split':>7} {'taps->B':>8} {'brk->B':>7} {'brk R':>6} {'tl R':>6}")
    for name, prof in ACTION_PROFILES.items():
        tot = ActionMetrics()
        for seed in range(seeds):
            shots, truth = generate_action(prof, events, seed)
            tot.add(evaluate_action(detect(shots), truth))
        print(f"{name:22} {tot.truth_bursts:7d} {tot.precision:8.3f} {tot.recall:8.3f} {tot.pair_f1:8.3f} "
              f"{tot.merged / max(tot.predicted_bursts, 1):7.2%} {tot.split / max(tot.truth_bursts, 1):7.2%} "
              f"{tot.false_frames / max(tot.not_burst_frames, 1):8.2%} "
              f"{tot.bracket_frames_in_bursts / max(tot.bracket_frames, 1):7.2%} "
              f"{tot.brackets_exact / max(tot.brackets, 1):6.3f} {tot.timelapses_exact / max(tot.timelapses, 1):6.3f}")
    print("merged: predicted bursts joining 2+ real bursts; split: real bursts broken apart; "
          "taps->B: quick separate presses / interval-timer frames wrongly put in a burst; tl R: timelapses found exactly")


if __name__ == "__main__":
    main()
