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


@pytest.mark.parametrize("offsets", [[0, -1, 1, -2, 2], [-2, -1, 0, 1, 2], [0, -2, 2], [-3, -1.5, 0, 1.5, 3, 4.5, -4.5]])
def test_bracket_shapes(offsets):
    r = Roll()
    b = r.bracket(1 / 60, offsets)
    assert names(detect(r.shots), "exposure") == [b]


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


def test_singles_and_burst_are_not_grouped():
    r = Roll()
    for s in (1 / 200, 1 / 250, 1 / 200, 1 / 160, 1 / 200):  # AE burst with metering jitter
        r.shot(s, gap=0.1, mode=0)
    r.pause(10)
    r.shot(1 / 100, mode=0)
    g = detect(r.shots)
    assert names(g, "exposure") == []
    assert all(x.kind == "single" for x in g)


def test_same_exposure_burst_is_not_a_bracket():
    r = Roll()
    for _ in range(6):
        r.shot(1 / 500, gap=0.05, mode=0)
    assert names(detect(r.shots), "exposure") == []


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
