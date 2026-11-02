"""Synthetic shoot generator with ground truth, for benchmarking and tests.

It simulates realistic sessions: AEB brackets in various firing orders and
drive modes, back-to-back brackets, user-driven manual brackets, single shots,
auto-exposure bursts with metering jitter, focus stacks, deleted frames and
cameras that lack sub-second timestamps or don't write ExposureMode=2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
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
    """Return (shots, truth): truth lists the real brackets and focus stacks as sets of filenames."""
    shots, truth, _ = generate_labeled(profile, n_events, seed)
    return shots, truth


def generate_labeled(profile: Profile, n_events: int = 200, seed: int = 0) -> tuple[list[Shot], list[set[str]], list[set[str]]]:
    """Return (shots, brackets_and_stacks, bursts) with ground truth as sets of filenames."""
    rng = random.Random(seed)
    t = 1_700_000_000.0 + rng.random() * 1e6
    counter = rng.randint(1, 9000)
    shots: list[Shot] = []
    truth: list[set[str]] = []
    bursts: list[set[str]] = []
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
            burst = set()
            fps, overhead = rng.choice((5, 8, 10, 12)), rng.uniform(0.04, 0.1)
            for _ in range(rng.randint(4, 15)):  # continuous-drive burst; AE meters each frame
                exp_t = nominal_shutter(base + rng.gauss(0, 0.35))
                burst.add(emit(exp_t, 100, 0, 0.0))
                # the drive fires at its rate, or exposure + fixed overhead when the shutter is slow
                t += max(1 / fps, exp_t + overhead) * (1 + rng.gauss(0, 0.03))
                base += rng.gauss(0, 0.15)
            bursts.append(burst)
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
    return shots, truth, bursts


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
    """Score bracket and focus-stack groups against ground truth (bursts are scored separately)."""
    pred = [frozenset(s.name for s in g.shots) for g in groups if g.kind in ("exposure", "focus")]
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


@dataclass(slots=True)
class BurstMetrics:
    truth_bursts: int
    exact_hits: int
    burst_frames: int
    burst_frames_in_brackets: int  # burst frames wrongly grouped as a bracket / stack
    bracket_frames: int
    bracket_frames_in_bursts: int  # bracket frames wrongly grouped as a burst

    @property
    def recall(self) -> float:
        return self.exact_hits / self.truth_bursts if self.truth_bursts else 1.0


def evaluate_bursts(groups: list[Group], bursts: list[set[str]], brackets: list[set[str]]) -> BurstMetrics:
    kind_of = {s.name: g.kind for g in groups for s in g.shots}
    pred = {frozenset(s.name for s in g.shots) for g in groups if g.kind == "burst"}
    burst_names = set().union(*bursts) if bursts else set()
    bracket_names = set().union(*brackets) if brackets else set()
    return BurstMetrics(
        truth_bursts=len(bursts),
        exact_hits=sum(frozenset(b) in pred for b in bursts),
        burst_frames=len(burst_names),
        burst_frames_in_brackets=sum(kind_of.get(n) in ("exposure", "focus") for n in burst_names),
        bracket_frames=len(bracket_names),
        bracket_frames_in_bursts=sum(kind_of.get(n) == "burst" for n in bracket_names),
    )


# --------------------------------------------------------------------------------------
# Action / event shooting: the burst-focused simulator
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class ActionProfile:
    name: str
    subsec: bool = True
    drive_meta: bool = False  # drive mode + burst sequence numbers (exiftool backend)
    hashes: bool = False  # simulate --pixels preview hashes (enables subject-change pans)
    fps: tuple[float, ...] = (3, 5, 8, 10, 12, 15, 20, 30)
    p_burst: float = 0.35
    p_repress: float = 0.35  # a burst is followed by another press after a short pause
    repress_pause: tuple[float, float] = (0.25, 1.5)
    p_taps: float = 0.15
    p_slow: float = 0.07
    p_bracket: float = 0.08
    p_interval: float = 0.03
    p_hiccup: float = 0.04
    p_zoom: float = 0.1
    p_pan: float = 0.0


ACTION_PROFILES = {
    "action": ActionProfile("action"),
    "action-drive-meta": ActionProfile("action-drive-meta", drive_meta=True),
    "action-pixels": ActionProfile("action-pixels", hashes=True, p_pan=0.12),
    "action-no-subsec": ActionProfile("action-no-subsec", subsec=False),
    "action-no-subsec-meta": ActionProfile("action-no-subsec-meta", subsec=False, drive_meta=True),
    "action-worst": ActionProfile("action-worst", subsec=False, p_repress=0.6, repress_pause=(0.2, 0.8),
                                  p_taps=0.25, p_hiccup=0.08),
}


@dataclass(slots=True)
class ActionTruth:
    brackets: list[set[str]]
    bursts: list[set[str]]
    not_bursts: set[str]  # frames that must not be in any burst (taps, interval timer)
    timelapses: list[set[str]] = field(default_factory=list)


def generate_action(profile: ActionProfile, n_events: int = 200, seed: int = 0) -> tuple[list[Shot], ActionTruth]:
    from .pixels import PixelStats

    rng = random.Random(seed)
    t = 1_700_000_000.0 + rng.random() * 1e6
    counter = rng.randint(1, 9000)
    shots: list[Shot] = []
    truth = ActionTruth([], [], set())
    focal = 200.0
    subject = rng.getrandbits(64)
    base = math.log2(1 / 1000)

    def frame_hash() -> int:
        h = subject
        for _ in range(rng.randint(0, 5)):
            h ^= 1 << rng.randrange(64)
        return h

    def emit(exp_t: float, *, mode: int = 0, bias: float = 0.0, drive: str = "single", seq: int | None = None) -> str:
        nonlocal counter
        counter = counter % 9999 + 1
        name = f"IMG_{counter:04d}.CR3"
        m = FileMeta(
            path=Path(name), make="Canon", model="EOS R3", serial="42", lens="RF100-500mm",
            capture_time=t if profile.subsec else math.floor(t), time_precise=profile.subsec,
            exposure_time=exp_t, f_number=5.6, iso=800, exposure_bias=bias, exposure_mode=mode,
            exposure_program=3, focal_length=focal, white_balance=0, flash_fired=False,
            drive_mode=drive if profile.drive_meta else None,
            sequence_number=seq if profile.drive_meta else None,
        )
        s = Shot([m])
        if profile.hashes:
            s.pixels = PixelStats(frame_hash(), 0.0, 0.0, 0.4, "ok")
        shots.append(s)
        return name

    def burst(fps: float, n: int) -> None:
        nonlocal t, base, focal, subject
        cadence = 1 / fps
        buffer_depth = rng.randint(15, 80)
        slow = 1.0
        current: list[str] = []
        pan_at = rng.randrange(2, n - 1) if profile.hashes and n >= 5 and rng.random() < profile.p_pan else -1
        zoom_at = rng.randrange(1, n) if n >= 3 and rng.random() < profile.p_zoom else -1
        for i in range(n):
            if i == pan_at:  # swing to a new subject mid-burst: still one press, so still one burst
                subject = rng.getrandbits(64)
            if i == zoom_at:
                focal = round(focal * rng.choice([0.8, 1.25]), 1)
            base += rng.gauss(0, 0.08)
            exp_t = nominal_shutter(base)
            current.append(emit(exp_t, drive="continuous", seq=i + 1))
            if i >= buffer_depth:
                slow = max(slow, rng.uniform(2.0, 4.0))
            iv = cadence * slow * (1 + rng.gauss(0, 0.03))
            if rng.random() < profile.p_hiccup:
                iv *= rng.uniform(1.4, 2.5)
            t += max(iv, exp_t + 0.002)
        if len(current) >= 2:
            truth.bursts.append(set(current))

    def burst_length() -> int:
        r = rng.random()
        return rng.randint(2, 4) if r < 0.25 else rng.randint(5, 12) if r < 0.7 else rng.randint(13, 40) if r < 0.95 else rng.randint(41, 120)

    for _ in range(n_events):
        if rng.random() < 0.3:
            t += rng.uniform(20, 600)
            focal = rng.choice([100.0, 200.0, 300.0, 400.0, 500.0])
            subject = rng.getrandbits(64)
            base = rng.uniform(math.log2(1 / 4000), math.log2(1 / 250))
        else:
            t += rng.uniform(1.5, 15)
            if rng.random() < 0.3:
                subject = rng.getrandbits(64)
        r = rng.random()
        p = profile
        if r < p.p_burst:
            fps = rng.choice(p.fps)
            burst(fps, burst_length())
            while rng.random() < p.p_repress:  # release and press again
                t += rng.uniform(*p.repress_pause)
                burst(fps, burst_length())
        elif r < p.p_burst + p.p_taps:
            for _ in range(rng.randint(2, 5)):  # quick separate presses
                truth.not_bursts.add(emit(nominal_shutter(base + rng.gauss(0, 0.1))))
                t += rng.uniform(0.35, 1.6)
        elif r < p.p_burst + p.p_taps + p.p_slow:
            burst(rng.uniform(1.5, 3.0), rng.randint(3, 10))
        elif r < p.p_burst + p.p_taps + p.p_slow + p.p_bracket:
            names = []
            size = rng.choice((3, 5))
            b0 = _fit_base(base, size, 1.0)
            for i, o in enumerate((0, -1, 1, -2, 2)[:size]):
                exp_t = nominal_shutter(b0 + o)
                names.append(emit(exp_t, mode=2, bias=float(o), drive="continuous", seq=i + 1))
                t += exp_t + rng.uniform(0.05, 0.15)
            truth.brackets.append(set(names))
            if rng.random() < 0.5:
                t += rng.uniform(0.3, 1.0)
                burst(rng.choice(p.fps), burst_length())
        elif r < p.p_burst + p.p_taps + p.p_slow + p.p_bracket + p.p_interval:
            every = rng.uniform(1.0, 30.0) if rng.random() < 0.5 else rng.uniform(1.0, 4.0)
            seq = set()
            for _ in range(rng.randint(5, 15)):  # interval timer: regular, but not a burst
                seq.add(emit(nominal_shutter(base), drive="single"))
                t += every
            truth.not_bursts |= seq
            truth.timelapses.append(seq)
        else:
            emit(nominal_shutter(base))
    return shots, truth


@dataclass(slots=True)
class ActionMetrics:
    truth_bursts: int = 0
    predicted_bursts: int = 0
    exact: int = 0
    pair_tp: int = 0
    pair_pred: int = 0
    pair_true: int = 0
    merged: int = 0  # predicted bursts spanning frames of 2+ true bursts
    split: int = 0  # true bursts spread over 2+ predicted groups
    false_frames: int = 0  # tap / interval-timer frames placed in a burst
    not_burst_frames: int = 0
    bracket_frames_in_bursts: int = 0
    bracket_frames: int = 0
    brackets_exact: int = 0
    brackets: int = 0
    timelapses_exact: int = 0
    timelapses: int = 0

    def add(self, o: "ActionMetrics") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(o, f))

    @property
    def precision(self) -> float:
        return self.exact / self.predicted_bursts if self.predicted_bursts else 1.0

    @property
    def recall(self) -> float:
        return self.exact / self.truth_bursts if self.truth_bursts else 1.0

    @property
    def pair_f1(self) -> float:
        p = self.pair_tp / self.pair_pred if self.pair_pred else 1.0
        r = self.pair_tp / self.pair_true if self.pair_true else 1.0
        return 2 * p * r / (p + r) if p + r else 0.0


def evaluate_action(groups: list[Group], truth: ActionTruth) -> ActionMetrics:
    def pairs(sets):
        out = set()
        for g in sets:
            items = sorted(g)
            out.update((a, b) for i, a in enumerate(items) for b in items[i + 1 :])
        return out

    pred = [frozenset(s.name for s in g.shots) for g in groups if g.kind == "burst"]
    true = [frozenset(b) for b in truth.bursts]
    owner = {n: i for i, b in enumerate(true) for n in b}
    group_of = {s.name: gi for gi, g in enumerate(groups) for s in g.shots}
    pp, tp = pairs(pred), pairs(true)
    m = ActionMetrics(
        truth_bursts=len(true), predicted_bursts=len(pred), exact=len(set(pred) & set(true)),
        pair_tp=len(pp & tp), pair_pred=len(pp), pair_true=len(tp),
        merged=sum(len({owner[n] for n in p if n in owner}) > 1 for p in pred),
        split=sum(len({group_of[n] for n in b}) > 1 for b in true),
        not_burst_frames=len(truth.not_bursts),
    )
    in_burst = {n for p in pred for n in p}
    m.false_frames = len(truth.not_bursts & in_burst)
    bracket_names = set().union(*truth.brackets) if truth.brackets else set()
    m.bracket_frames = len(bracket_names)
    m.bracket_frames_in_bursts = len(bracket_names & in_burst)
    pred_brackets = {frozenset(s.name for s in g.shots) for g in groups if g.kind == "exposure"}
    m.brackets = len(truth.brackets)
    m.brackets_exact = sum(frozenset(b) in pred_brackets for b in truth.brackets)
    pred_tl = {frozenset(s.name for s in g.shots) for g in groups if g.kind == "timelapse"}
    m.timelapses = len(truth.timelapses)
    m.timelapses_exact = sum(frozenset(t) in pred_tl for t in truth.timelapses)
    return m
