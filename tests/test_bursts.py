"""Burst and timelapse behaviour: each test pins one real-world shooting pattern."""

import pytest

from bracketeering.engine import EngineConfig, detect
from bracketeering.synth import ACTION_PROFILES, ActionMetrics, evaluate_action, generate_action
from helpers import Roll, names


def drive(r: Roll, n: int, interval: float, **kw) -> list[str]:
    """Continuous-drive frames `interval` s apart (start to start)."""
    out = []
    for _ in range(n):
        out.append(r.shot(1 / 1000, gap=interval - 1 / 1000, mode=0, **kw))
    return out


def test_continuous_drive_is_one_burst():
    r = Roll()
    b = drive(r, 12, 0.1)
    g = detect(r.shots)
    assert names(g, "burst") == [b]
    assert g[0].confidence > 0.95
    assert "10.0 fps" in g[0].evidence[0]


def test_re_press_after_short_pause_is_a_new_burst():
    r = Roll()
    b1 = drive(r, 8, 0.1)
    r.t += 0.5  # release and press again
    b2 = drive(r, 8, 0.1)
    assert names(detect(r.shots), "burst") == [b1, b2]


def test_af_hiccup_does_not_split():
    r = Roll()
    b = drive(r, 5, 0.1) + drive(r, 1, 0.22) + drive(r, 6, 0.1)  # one interval ~2x slower
    assert names(detect(r.shots), "burst") == [b]


def test_buffer_slowdown_does_not_split():
    r = Roll()
    b = drive(r, 20, 0.05) + drive(r, 15, 0.2)  # 20 fps until the buffer fills, then 5 fps
    g = detect(r.shots)
    assert names(g, "burst") == [b]
    assert "pace changed" in g[0].evidence[0]


def test_human_taps_are_not_a_burst():
    r = Roll()
    for gap in (0.6, 1.1, 0.45, 1.4):  # separate presses: irregular, slower than a drive
        r.shot(1 / 500, gap=gap, mode=0)
    r.shot(1 / 500, mode=0)
    assert all(x.kind == "single" for x in detect(r.shots))


def test_slow_continuous_drive_is_a_burst():
    r = Roll()
    b = drive(r, 8, 0.4)  # 2.5 fps, metronomic
    assert names(detect(r.shots), "burst") == [b]


def test_long_exposures_in_continuous_drive():
    r = Roll()
    b = []
    for exp in (0.5, 0.25, 0.5, 0.4, 0.3, 0.5, 0.25):  # slow shutter: interval = exposure + fixed overhead
        b.append(r.shot(exp, gap=0.08, mode=0))
    assert names(detect(r.shots), "burst") == [b]


def test_interval_timer_is_a_timelapse_not_a_burst():
    r = Roll()
    t = [r.shot(1 / 250, gap=2.0 - 1 / 250, mode=0) for _ in range(10)]
    g = detect(r.shots)
    assert names(g, "burst") == []
    assert names(g, "timelapse") == [t]


def test_sequence_number_reset_splits_identical_cadence():
    r = Roll()
    b1 = [r.shot(1 / 1000, gap=0.1, mode=0, drive_mode="continuous", sequence_number=i + 1) for i in range(5)]
    b2 = [r.shot(1 / 1000, gap=0.1, mode=0, drive_mode="continuous", sequence_number=i + 1) for i in range(5)]
    assert names(detect(r.shots), "burst") == [b1, b2]


def test_single_drive_frames_are_not_a_burst():
    r = Roll()
    for _ in range(4):
        r.shot(1 / 1000, gap=0.3, mode=0, drive_mode="single")
    assert all(x.kind == "single" for x in detect(r.shots))


def test_zoom_inside_a_burst_is_allowed():
    r = Roll()
    b = drive(r, 6, 0.1, focal=200.0) + drive(r, 6, 0.1, focal=250.0)
    assert names(detect(r.shots), "burst") == [b]


def test_burst_without_subsecond_timestamps():
    r = Roll(subsec=False)
    r.t = float(int(r.t))
    b = drive(r, 9, 0.12)
    assert names(detect(r.shots), "burst") == [b]


def test_subject_change_splits_with_pixels():
    from bracketeering.pixels import PixelStats
    r = Roll()
    b1 = drive(r, 6, 0.1)
    b2 = drive(r, 6, 0.1)
    for i, s in enumerate(r.shots):
        h = 0x0F0F0F0F0F0F0F0F if i < 6 else 0xF0F0F0F0F0F0F0F0  # previews: 64 bits apart
        s.pixels = PixelStats(h, 0.0, 0.0, 0.4, "ok")
    assert names(detect(r.shots), "burst") == [b1, b2]


def test_bracket_inside_a_burst_session_stays_a_bracket():
    r = Roll()
    pre = drive(r, 6, 0.1)
    r.t += 1.0
    brk = r.bracket(1 / 500, [0, -1, 1], gap=0.08)
    r.t += 1.0
    post = drive(r, 6, 0.1)
    g = detect(r.shots)
    assert names(g, "exposure") == [brk]
    assert names(g, "burst") == [pre, post]


def test_bursts_can_be_disabled():
    r = Roll()
    drive(r, 6, 0.1)
    assert all(x.kind == "single" for x in detect(r.shots, EngineConfig(detect_bursts=False)))


@pytest.mark.parametrize("profile,min_p,min_r", [
    ("action", 0.92, 0.88), ("action-drive-meta", 0.95, 0.97), ("action-pixels", 0.92, 0.87),
    ("action-no-subsec-meta", 0.98, 0.96),
])
def test_action_accuracy(profile, min_p, min_r):
    tot = ActionMetrics()
    for seed in range(3):
        shots, truth = generate_action(ACTION_PROFILES[profile], 200, seed + 3000)
        tot.add(evaluate_action(detect(shots), truth))
    assert tot.precision >= min_p and tot.recall >= min_r, (tot.precision, tot.recall)
    assert tot.false_frames / max(tot.not_burst_frames, 1) < 0.03  # taps / timer frames kept out
    assert tot.bracket_frames_in_bursts == 0
