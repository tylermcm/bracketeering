"""Synthetic shoot generator with ground truth, for benchmarking and tests.

It simulates realistic sessions: AEB brackets in various firing orders and
drive modes, back-to-back brackets, user-driven manual brackets, single shots,
auto-exposure bursts with metering jitter, focus stacks, deleted frames and
cameras that lack sub-second timestamps or don't write ExposureMode=2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from .engine import known_orders
from .model import FileMeta, Group, Shot

_NOMINAL_SHUTTERS = [
    30, 25, 20, 15, 13, 10, 8, 6, 5, 4, 3.2, 2.5, 2, 1.6, 1.3, 1, 0.8, 0.6, 0.5, 0.4, 0.3,
    *(1 / d for d in (4, 5, 6, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 160, 200, 250,
                      320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400, 8000)),
]
_LOG_SHUTTERS = [(math.log2(s), s) for s in _NOMINAL_SHUTTERS]


def nominal_shutter(log2_t: float) -> float:
    return min(_LOG_SHUTTERS, key=lambda x: abs(x[0] - log2_t))[1]


@dataclass(slots=True)
class Profile:
    name: str
    make: str = "Canon"
    model: str = "EOS R5"
    reports_aeb: bool = True  # writes ExposureMode=2 during AEB
    subsec: bool = True
    bias_tracks: bool = True  # ExposureBiasValue includes the bracket offset
    sizes: tuple[int, ...] = (3, 5, 7)
    steps: tuple[float, ...] = (2 / 3, 1.0, 2.0)
    order: str = "0,-,+ (centre first)"
    p_bracket: float = 0.5
    p_burst: float = 0.15
    p_focus: float = 0.05
    p_manual: float = 0.0
    p_back_to_back: float = 0.25
    p_deleted: float = 0.03
    fixed_settings: bool = True  # same bracket config for the whole shoot


PROFILES = {
    "canon-aeb": Profile("canon-aeb"),
    "sony-no-subsec": Profile("sony-no-subsec", make="SONY", model="ILCE-7M4", subsec=False,
                              order="0,-,+ (centre first)", sizes=(3, 5)),
    "no-bracket-metadata": Profile("no-bracket-metadata", make="FUJIFILM", model="X-T5", reports_aeb=False,
                                 bias_tracks=False, sizes=(3, 5, 7, 9), order="-,0,+ (ascending)"),
    "nikon-mixed-settings": Profile("nikon-mixed-settings", make="NIKON CORPORATION", model="NIKON Z 8",
                                    sizes=(2, 3, 5, 7, 9), steps=(0.33, 0.67, 1.0, 2.0, 3.0), fixed_settings=False),
    "manual-brackets": Profile("manual-brackets", make="Canon", model="EOS 6D", reports_aeb=False,
                               bias_tracks=False, p_manual=0.5, p_bracket=0.2, sizes=(3, 5)),
    "worst-case": Profile("worst-case", make="Panasonic", model="DC-S5", reports_aeb=False, subsec=False,
                          bias_tracks=False, sizes=(3, 5, 7), p_burst=0.3, p_back_to_back=0.5,
                          fixed_settings=False, order="-,0,+ (ascending)"),
}


def _fit_base(base: float, size: int, step: float) -> float:
    """Keep a bracket inside the shutter range and on the 1/3-stop metering grid."""
    half_span = (size - 1) / 2 * step + 0.2
    base = min(max(base, math.log2(1 / 8000) + half_span), math.log2(30) - half_span)
    return round(base * 3) / 3


def _order_for(profile: Profile, n: int, rng: random.Random) -> tuple[int, ...]:
    orders = {v: k for k, v in known_orders(n).items()}
    if profile.order in orders:
        return orders[profile.order]
    return tuple(range(n))


def generate(profile: Profile, n_events: int = 200, seed: int = 0) -> tuple[list[Shot], list[set[str]]]:
    """Return (shots, truth) where truth is a list of sets of filenames forming real groups."""
    rng = random.Random(seed)
    t = 1_700_000_000.0 + rng.random() * 1e6
    counter = rng.randint(1, 9000)
    shots: list[Shot] = []
    truth: list[set[str]] = []
    aperture, focal = rng.choice([4.0, 5.6, 8.0, 11.0]), rng.choice([16.0, 24.0, 35.0, 50.0])
    fixed = (rng.choice(profile.sizes), rng.choice(profile.steps))
    while (fixed[0] - 1) * fixed[1] > 12:
        fixed = (rng.choice(profile.sizes), rng.choice(profile.steps))

    def emit(exp_t: float, iso: float, mode: int, bias: float, fdist: float | None = None, program: int = 3) -> str:
        nonlocal counter
        counter = counter % 9999 + 1
        name = f"IMG_{counter:04d}.CR3"
        ts = t if profile.subsec else math.floor(t)
        m = FileMeta(
            path=Path(name), make=profile.make, model=profile.model, serial="123456", lens="RF24-70mm F2.8",
            capture_time=ts, time_precise=profile.subsec, exposure_time=exp_t, f_number=aperture, iso=iso,
            exposure_bias=bias, exposure_mode=mode, exposure_program=program, focal_length=focal,
            white_balance=0, flash_fired=False, focus_distance=fdist,
        )
        shots.append(Shot([m]))
        return name

    def bracket(base: float, size: int, step: float, gap_range: tuple[float, float]) -> None:
        nonlocal t
        ranks = _order_for(profile, size, rng)
        centre = (size - 1) / 2
        base = _fit_base(base, size, step)
        names = []
        delete = rng.random() < profile.p_deleted and size >= 3
        drop = rng.randrange(size) if delete else -1
        for i, r in enumerate(ranks):
            offset = (r - centre) * step
            exp_t = nominal_shutter(base + offset)
            if i != drop:
                names.append(emit(exp_t, 100, 2 if profile.reports_aeb else 0, offset if profile.bias_tracks else 0.0))
            t += exp_t + rng.uniform(*gap_range)
        if len(names) >= 2:
            truth.append(set(names))

    for _ in range(n_events):
        if rng.random() < 0.3:  # move to a new scene / location
            aperture, focal = rng.choice([2.8, 4.0, 5.6, 8.0, 11.0]), rng.choice([16.0, 24.0, 35.0, 50.0, 70.0])
            t += rng.uniform(20, 600)
        else:
            t += rng.uniform(1.5, 15)
        base = rng.uniform(-11, -1)  # log2 of shutter time at ISO 100
        r = rng.random()
        if r < profile.p_bracket:
            size, step = fixed if profile.fixed_settings else (rng.choice(profile.sizes), rng.choice(profile.steps))
            while (size - 1) * step > 12:  # cameras cap the total bracket span
                step = rng.choice([x for x in profile.steps if x < step] or [min(profile.steps)])
            continuous = rng.random() < 0.5
            gaps = (0.05, 0.2) if continuous else (0.3, 1.2)
            repeats = 1
            if continuous and rng.random() < profile.p_back_to_back:
                repeats = rng.randint(2, 3)
            for _ in range(repeats):
                bracket(base, size, step, gaps)
                base += rng.uniform(-0.15, 0.15) if rng.random() < 0.7 else rng.choice([-2, 2])
        elif r < profile.p_bracket + profile.p_manual:
            size = rng.choice((3, 5))
            step = rng.choice((1.0, 2.0))
            base = _fit_base(base, size, step)
            names = []
            for i in range(size):
                exp_t = nominal_shutter(base + (i - (size - 1) / 2) * step)
                names.append(emit(exp_t, 100, 1, 0.0, program=1))
                t += exp_t + rng.uniform(1.2, 4.0)
            truth.append(set(names))
        elif r < profile.p_bracket + profile.p_manual + profile.p_burst:
            for _ in range(rng.randint(4, 15)):  # AE burst with metering jitter
                exp_t = nominal_shutter(base + rng.gauss(0, 0.35))
                emit(exp_t, 100, 0, 0.0)
                t += exp_t + rng.uniform(0.05, 0.15)
                base += rng.gauss(0, 0.15)
        elif r < profile.p_bracket + profile.p_manual + profile.p_burst + profile.p_focus:
            n = rng.randint(8, 30)
            exp_t = nominal_shutter(base)
            d0 = rng.uniform(0.2, 1.0)
            names = []
            for i in range(n):
                names.append(emit(exp_t, 100, 0, 0.0, fdist=round(d0 * (1.12 ** i), 2)))
                t += exp_t + rng.uniform(0.1, 0.4)
            truth.append(set(names))
        else:  # single shot, sometimes a quick handheld pair with an exposure tweak
            emit(nominal_shutter(base), 100, 0, 0.0)
            if rng.random() < 0.2:
                t += rng.uniform(0.6, 2.5)
                emit(nominal_shutter(base + rng.choice([-1, -2 / 3, 2 / 3, 1])), 100, 0, rng.choice([-1.0, 1.0]))
    return shots, truth


@dataclass(slots=True)
class Metrics:
    truth_groups: int
    predicted_groups: int
    exact_hits: int
    pair_precision: float
    pair_recall: float

    @property
    def group_precision(self) -> float:
        return self.exact_hits / self.predicted_groups if self.predicted_groups else 1.0

    @property
    def group_recall(self) -> float:
        return self.exact_hits / self.truth_groups if self.truth_groups else 1.0

    @property
    def pair_f1(self) -> float:
        p, r = self.pair_precision, self.pair_recall
        return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(groups: list[Group], truth: list[set[str]]) -> Metrics:
    pred = [frozenset(s.name for s in g.shots) for g in groups if g.kind != "single"]
    true = [frozenset(t) for t in truth]
    hits = len(set(pred) & set(true))

    def pairs(gs):
        out = set()
        for g in gs:
            items = sorted(g)
            out.update((a, b) for i, a in enumerate(items) for b in items[i + 1 :])
        return out

    pp, tp = pairs(pred), pairs(true)
    inter = len(pp & tp)
    return Metrics(len(true), len(pred), hits, inter / len(pp) if pp else 1.0, inter / len(tp) if tp else 1.0)
