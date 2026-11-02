import math

import pytest

from bracketeering.engine import EngineConfig, detect, known_orders
from bracketeering.synth import PROFILES, evaluate, generate
from helpers import Roll, names


def test_single_bracket_centre_first():
    r = Roll()
    b = r.bracket(1 / 125, [0, -1, 1])
    g = detect(r.shots)
    assert names(g, "exposure") == [b]
    assert g[0].ev_offsets == [0.0, -1.0, 1.0]
    assert g[0].step_ev == 1.0
    assert g[0].confidence > 0.95


@pytest.mark.parametrize("offsets", [[0, -1, 1, -2, 2], [-2, -1, 0, 1, 2], [0, -2, 2], [-3, -1.5, 0, 1.5, 3, 4.5, -4.5],
                                     [0, -1 / 3, 1 / 3], [0, -2 / 3, 2 / 3, -4 / 3, 4 / 3]])
def test_bracket_shapes(offsets):
    r = Roll()
    b = r.bracket(1 / 60, offsets)
    assert names(detect(r.shots), "exposure") == [b]


@pytest.mark.parametrize("base_f", [2.8, 5.6, 8.0, 11, 13])
@pytest.mark.parametrize("offsets", [[0, -1 / 3, 1 / 3], [0, -2 / 3, 2 / 3, -4 / 3, 4 / 3], [0, -1, 1]])
def test_shutter_priority_aperture_bracket(base_f, offsets):
    """Tv/S-mode AEB varies the aperture; nominal f-labels (f/13 = f/12.7) must not break it."""
    r = Roll()
    b = r.aperture_bracket(1 / 500, base_f, offsets)
    g = detect(r.shots)
    assert names(g, "exposure") == [b]
    expected = [round(o, 2) for o in offsets]
    assert g[0].ev_offsets == pytest.approx(expected, abs=0.02)


def test_iso_bracket_in_manual_with_auto_iso():
    r = Roll()
    b = [r.shot(1 / 250, iso=iso, bias=float(o)) for iso, o in ((400, 0), (200, -1), (800, 1))]
    assert names(detect(r.shots), "exposure") == [b]


def test_nominal_labels_snap_to_exact_stops():
    from bracketeering.model import log2_aperture_stops, log2_iso, log2_shutter
    assert log2_aperture_stops(13) == pytest.approx(22 / 3)  # f/13 is really f/12.7
    assert log2_aperture_stops(11) - log2_aperture_stops(13) == pytest.approx(-1 / 3)
    assert log2_shutter(1 / 125) == pytest.approx(-7)
    assert log2_shutter(1 / 13) == pytest.approx(-11 / 3)
    assert log2_iso(640) == pytest.approx(8 / 3)
    assert log2_aperture_stops(4.8) == pytest.approx(2 * math.log2(4.8))  # half-stop label: kept as recorded


def test_back_to_back_brackets_no_pause():
    r = Roll()
    b1 = r.bracket(1 / 250, [0, -1, 1])
    b2 = r.bracket(1 / 250, [0, -1, 1])
    b3 = r.bracket(1 / 250, [0, -1, 1])
    assert names(detect(r.shots), "exposure") == [b1, b2, b3]


def test_back_to_back_without_metadata_uses_firing_order():
    r = Roll()
    b1 = r.bracket(1 / 250, [-1, 0, 1], mode=0)
    b2 = r.bracket(1 / 250, [-1, 0, 1], mode=0)
    assert names(detect(r.shots), "exposure") == [b1, b2]


def test_large_gap_splits_runs():
    r = Roll()
    b1 = r.bracket(1 / 250, [0, -1, 1])
    r.pause(30)
    b2 = r.bracket(1 / 250, [0, -1, 1])
    assert names(detect(r.shots), "exposure") == [b1, b2]


def test_long_exposure_gap_accounts_for_shutter_time():
    r = Roll()
    b = r.bracket(4.0, [0, -2, 2])  # 4s, 1s, 16s exposures
    assert names(detect(r.shots), "exposure") == [b]


def test_ae_burst_is_a_burst_not_a_bracket():
    r = Roll()
    burst = [r.shot(s, gap=0.1, mode=0) for s in (1 / 200, 1 / 250, 1 / 200, 1 / 160, 1 / 200)]  # metering jitter
    r.pause(10)
    single = r.shot(1 / 100, mode=0)
    g = detect(r.shots)
    assert names(g, "exposure") == []
    assert names(g, "burst") == [burst]
    assert names(g, "single") == [[single]]


def test_bracket_then_burst_are_separated():
    r = Roll()
    b = r.bracket(1 / 250, [0, -1, 1], gap=0.1)
    burst = [r.shot(1 / 250, gap=0.1, mode=0) for _ in range(6)]
    g = detect(r.shots)
    assert names(g, "exposure") == [b]
    assert names(g, "burst") == [burst]


def test_slow_singles_are_not_a_burst():
    r = Roll()
    for _ in range(4):
        r.shot(1 / 250, gap=3.0, mode=0)
    assert all(x.kind == "single" for x in detect(r.shots))


def test_bursts_can_be_disabled():
    r = Roll()
    for _ in range(5):
        r.shot(1 / 500, gap=0.05, mode=0)
    assert all(x.kind == "single" for x in detect(r.shots, EngineConfig(detect_bursts=False)))


def test_representative_is_the_middle_exposure():
    r = Roll()
    b = r.bracket(1 / 125, [-2, -1, 0, 1, 2])
    g = detect(r.shots)[0]
    assert g.shots[g.representative].name == b[2]
    r = Roll()
    b = r.bracket(1 / 125, [0, -1, 1])
    g = detect(r.shots)[0]
    assert g.shots[g.representative].name == b[0]


def test_min_confidence_demotes_weak_brackets():
    r = Roll()
    r.bracket(1 / 125, [-1, 0, 1], mode=0)  # no metadata: decent but not certain
    weak = detect(r.shots)[0]
    assert weak.kind == "exposure"
    demoted = detect(r.shots, EngineConfig(min_confidence=min(0.999, weak.confidence + 0.01), detect_bursts=False))
    assert all(x.kind == "single" for x in demoted)


def test_same_exposure_burst_is_not_a_bracket():
    r = Roll()
    for _ in range(6):
        r.shot(1 / 500, gap=0.05, mode=0)
    g = detect(r.shots)
    assert names(g, "exposure") == []
    assert [x.kind for x in g] == ["burst"]


def test_deleted_middle_frame_still_grouped_with_warning():
    r = Roll()
    full = r.bracket(1 / 125, [0, -1, 1, -2, 2])
    r.shots.pop(1)  # delete the -1 frame
    g = detect(r.shots)
    assert names(g, "exposure") == [[full[0], *full[2:]]]
    assert any("missing" in w for w in g[0].warnings)


def test_deleted_end_frame_still_grouped():
    r = Roll()
    full = r.bracket(1 / 125, [0, -1, 1, -2, 2])
    r.shots.pop()  # delete the +2 frame
    g = detect(r.shots)
    assert names(g, "exposure") == [full[:-1]]


def test_focal_length_change_breaks_group():
    r = Roll()
    a = r.shot(1 / 125, focal=24.0)
    b = r.shot(1 / 250, focal=70.0)
    c = r.shot(1 / 60, focal=70.0)
    assert [a] not in names(detect(r.shots), "exposure")
    assert all(a not in grp for grp in names(detect(r.shots), "exposure"))


def test_cameras_are_kept_apart():
    r1, r2 = Roll(make="Canon"), Roll(make="SONY", model="ILCE-7M4", start=500)
    b1 = r1.bracket(1 / 125, [0, -1, 1])
    b2 = r2.bracket(1 / 125, [0, -1, 1])
    g = detect(r1.shots + r2.shots)
    assert sorted(names(g, "exposure")) == sorted([b1, b2])


def test_no_subsecond_timestamps():
    r = Roll(subsec=False)
    b1 = r.bracket(1 / 500, [0, -1, 1], gap=0.1)
    r.pause(5)
    b2 = r.bracket(1 / 500, [0, -1, 1], gap=0.1)
    assert names(detect(r.shots), "exposure") == [b1, b2]


def test_maker_note_positions_split_identical_runs():
    r = Roll()
    out = []
    for _ in range(2):
        out.append([r.shot(1 / 125 * 2 ** o, mode=0, bracket_kind="exposure", bracket_position=i + 1)
                    for i, o in enumerate([0, -1, 1, -2, 2])])
    assert names(detect(r.shots), "exposure") == out


def test_focus_stack_from_maker_note():
    r = Roll()
    stack = [r.shot(1 / 100, mode=0, bracket_kind="focus", gap=0.3) for _ in range(12)]
    g = detect(r.shots)
    assert names(g, "focus") == [stack]


def test_focus_stack_inferred_from_distance():
    r = Roll()
    stack = [r.shot(1 / 100, mode=0, focus_distance=0.3 * 1.1 ** i, gap=0.3) for i in range(10)]
    assert names(detect(r.shots), "focus") == [stack]


def test_manual_mode_bracket_with_slow_operator():
    r = Roll()
    b = [r.shot(1 / 125 * 2 ** o, gap=3.0, mode=1) for o in (-2, 0, 2)]
    assert names(detect(r.shots), "exposure") == [b]


def test_sidecars_are_one_shot(tmp_path):
    from bracketeering.model import FileMeta
    from bracketeering.scan import pair_sidecars
    from pathlib import Path
    metas = [
        FileMeta(Path("/x/IMG_0001.CR3"), capture_time=10.0, exposure_time=0.01),
        FileMeta(Path("/x/IMG_0001.JPG"), capture_time=10.0, exposure_time=0.01),
        FileMeta(Path("/x/IMG_0002.JPG"), capture_time=11.0, exposure_time=0.01),
    ]
    shots = pair_sidecars(metas)
    assert sorted(len(s.files) for s in shots) == [1, 2]
    assert next(s for s in shots if len(s.files) == 2).meta.path.suffix == ".CR3"


def test_known_orders():
    assert known_orders(3)[(1, 0, 2)].startswith("0,-,+")
    assert known_orders(5)[(2, 1, 3, 0, 4)].startswith("0,-,+")
    assert (0, 1, 2, 3, 4, 5, 6) in known_orders(7)


def test_config_max_size():
    r = Roll()
    r.bracket(1 / 125, [0, -1, 1, -2, 2])
    g = detect(r.shots, EngineConfig(max_size=3))
    assert all(x.size <= 3 for x in g)


@pytest.mark.parametrize("profile,min_f1", [
    ("canon-aeb", 0.99), ("sony-no-subsec", 0.99), ("nikon-mixed-settings", 0.95),
    ("no-bracket-metadata", 0.93), ("manual-brackets", 0.93), ("worst-case", 0.90),
])
def test_synthetic_accuracy(profile, min_f1):
    hits = truth = pred = 0
    for seed in range(3):
        shots, t = generate(PROFILES[profile], 200, seed + 1000)
        m = evaluate(detect(shots), t)
        hits, truth, pred = hits + m.exact_hits, truth + m.truth_groups, pred + m.predicted_groups
    p, r = hits / pred, hits / truth
    assert 2 * p * r / (p + r) >= min_f1


@pytest.mark.parametrize("profile", ["canon-aeb", "no-bracket-metadata", "worst-case"])
def test_bursts_and_brackets_stay_apart(profile):
    from bracketeering.synth import evaluate_bursts, generate_labeled
    bf = bfb = kf = kfb = 0
    for seed in range(3):
        shots, brackets, bursts = generate_labeled(PROFILES[profile], 200, seed + 2000)
        b = evaluate_bursts(detect(shots), bursts, brackets)
        bf, bfb, kf, kfb = bf + b.burst_frames, bfb + b.burst_frames_in_brackets, kf + b.bracket_frames, kfb + b.bracket_frames_in_bursts
    assert bfb / bf < 0.02
    assert kfb / kf < 0.03
