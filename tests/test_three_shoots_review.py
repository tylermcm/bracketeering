"""Regression tests for the review of WhiteSands '26 (Z 7_2), Japan 4-24 (D810) and Alaska Aug '21
(D800/D5/D850), and the 0.3.1 China re-test. Each test replays one finding."""

from pathlib import Path

import pytest

from bracketeering.engine import detect
from bracketeering.model import FileMeta, Shot
from bracketeering.pixels import PixelStats
from bracketeering.scan import pair_sidecars

T0 = 1_780_000_000.0


class Body:
    def __init__(self, make="NIKON CORPORATION", model="NIKON D800", serial="6001234", count=52900):
        self.cam = dict(make=make, model=model, serial=serial, lens="24-70mm f/2.8", focal_length=50.0)
        self.t, self.count, self.shots = T0, count, []

    def frame(self, name, after, *, slot=None, iso=400.0, shutter=1 / 250, drive="continuous", folder="",
              kind=None, interval=None, file_number=None, **kw):
        self.t += after
        self.count += 1
        m = FileMeta(Path(folder) / name, capture_time=self.t, time_precise=True, exposure_time=shutter, f_number=8.0,
                     iso=iso, exposure_mode=0, bracket_kind=kind or ("exposure" if slot is not None else "none"),
                     bracket_value=slot, drive_mode=drive, shutter_count=self.count, interval_shooting=interval,
                     file_number=file_number, **self.cam, **kw)
        self.shots.append(Shot([m]))
        return name


def kinds(groups, kind):
    return [[s.name for s in g.shots] for g in groups if g.kind == kind]


def test_long_exposure_single_frame_cycles_stay_brackets():
    """Aurora: 5F at 1/3 EV, 13-30 s exposures on a tripod, presses 22-66 s apart, single-frame drive."""
    b = Body(model="NIKON D850")
    expected = []
    for c in range(3):
        cyc = []
        for i, (v, exp) in enumerate(zip((0, -2 / 3, -1 / 3, 1 / 3, 2 / 3), (15, 8, 13, 20, 30))):
            cyc.append(b.frame(f"DSC_{1722 + 5 * c + i}.NEF", 240 if i == 0 else 45 + 7 * i, slot=v, shutter=exp, drive="single"))
        expected.append(cyc)
    assert kinds(detect(b.shots), "exposure") == expected


def test_tripod_waterfall_cycle_stays_whole():
    b = Body()
    cyc = [b.frame(f"DSC_{8774 + i}.NEF", 0 if i == 0 else gap, slot=v, shutter=1.0, drive="single")
           for i, (v, gap) in enumerate(zip((0, -2 / 3, -1 / 3, 1 / 3, 2 / 3), (0, 3.6, 7.6, 4.1, 5.0)))]
    assert kinds(detect(b.shots), "exposure") == [cyc]


def test_wildlife_cycle_at_drive_speed_stays_whole_and_is_flagged():
    """D800 7F cycles panning with a bear: frames don't register, but it's one press."""
    b = Body()
    cyc = [b.frame(f"DSC_{9850 + i}.NEF", 0 if i == 0 else 0.2, slot=v)
           for i, v in enumerate((0, -1, -2 / 3, -1 / 3, 1 / 3, 2 / 3, 1))]
    for i, s in enumerate(b.shots):  # every frame in a different place: the subject is moving
        s.pixels = PixelStats(0, 0, 0, 0.4, "ok", _pattern(i))
    g = detect(b.shots)
    assert kinds(g, "exposure") == [cyc]
    assert any("moving" in w for w in g[0].warnings)


def test_edited_exports_merge_into_their_source_shot():
    b = Body()
    src = b.frame("DSC_9568.JPG", 0, folder="105ND800")
    b.frame("DSC_9569.JPG", 0.15, folder="105ND800")
    source = b.shots[0].meta
    export = FileMeta(Path("Bears/DSC_9568.jpg"), capture_time=source.capture_time, time_precise=True,
                      exposure_time=source.exposure_time, f_number=8.0, iso=source.iso, **b.cam)  # no maker notes
    shots = pair_sidecars([f for s in b.shots for f in s.files] + [export])
    assert len(shots) == 2
    merged = next(s for s in shots if len(s.files) == 2)
    assert merged.meta.path.name == src
    assert kinds(detect(shots), "burst") == [["DSC_9568.JPG", "DSC_9569.JPG"]]


def test_identical_capture_times_never_form_a_burst():
    b = Body()
    b.frame("DSC_9857.JPG", 0)
    b.shots.append(Shot([FileMeta(Path("Re-edits/DSC_9857-Edit.jpg"), capture_time=b.t, time_precise=True,
                                  exposure_time=1 / 250, f_number=8.0, iso=800.0, **b.cam)]))
    assert kinds(detect(b.shots), "burst") == []


def test_zero_filled_file_is_unreadable(tmp_path):
    from bracketeering.metadata.native import read_native
    p = tmp_path / "DSC_8771.NEF"
    p.write_bytes(b"\x00" * 4096)
    assert "zeros" in read_native(p).error


def test_focus_shift_stack_is_a_focus_stack_not_a_timelapse():
    b = Body(model="NIKON Z 7_2")
    stack = [b.frame(f"_DSC{i:04d}.NEF", 0 if i == 1 else 1.24, kind="focus", interval=False, file_number=i)
             for i in range(1, 11)]
    g = detect(b.shots)
    assert kinds(g, "focus") == [stack]
    assert kinds(g, "timelapse") == []


def test_regular_hand_presses_are_not_a_timelapse_when_the_timer_was_off():
    """_DSC0504-0508: the dog at dinner, five presses ~1.12 s apart, IntervalShooting = Off."""
    b = Body(model="NIKON Z 7_2")
    for i, after in enumerate((0, 1.12, 1.13, 1.11, 1.12)):
        b.frame(f"_DSC{504 + i:04d}.NEF", after, interval=False, drive="single")
    assert kinds(detect(b.shots), "timelapse") == []


def test_burst_continues_across_a_file_number_series_switch():
    """_DSC8486, _DSC0059, _DSC0060, _DSC8487: consecutive shutter counts, 0.13 s apart."""
    b = Body(model="NIKON Z 7_2")
    names = [b.frame(n, 0 if i == 0 else 0.13, file_number=int(n[4:8])) for i, n in
             enumerate(("_DSC8486.NEF", "_DSC0059.NEF", "_DSC0060.NEF", "_DSC8487.NEF"))]
    assert kinds(detect(b.shots), "burst") == [names]


def test_featureless_frames_dont_break_a_burst_or_a_scene():
    """Moon in an empty sky: two holds 11 s apart; and black misfires 0.16 s apart."""
    from bracketeering.pixels import assign_scenes
    b = Body(model="NIKON Z 7_2")
    hold1 = [b.frame(f"_DSC{389 + i:04d}.NEF", 0 if i == 0 else 0.15) for i in range(5)]
    hold2 = [b.frame(f"_DSC{394 + i:04d}.NEF", 11 if i == 0 else 0.15) for i in range(5)]
    flat = bytes([128]) * (64 * 44)
    for s in b.shots:
        s.pixels = PixelStats(0, 0, 0, 0.1, "under", flat, bytes([30]) * (28 * 20))
    g = detect(b.shots)
    assert kinds(g, "burst") == [hold1, hold2]
    assign_scenes(g)
    assert g[0].scene == g[1].scene


def test_complete_cycles_after_a_different_step_are_not_partial():
    """Alaska: 3F at 2/3 EV right after 3F at 1/3 EV were reported as '3 of 5'."""
    b = Body()
    for c, step in enumerate((1 / 3, 1 / 3, 2 / 3, 2 / 3)):
        for i, v in enumerate((0, -step, step)):
            b.frame(f"DSC_{8549 + 3 * c + i}.NEF", 20 if i == 0 else 0.2, slot=v)
    g = detect(b.shots)
    assert len(kinds(g, "exposure")) == 4
    assert not any(w.startswith("partial") for x in g for w in x.warnings)


def test_recompose_split_fragment_is_flagged_partial():
    """_DSC1519 | _DSC1520-1523: the fragment is 4 of a 5F cycle."""
    b = Body(model="NIKON Z 7_2")
    b.frame("_DSC1519.NEF", 0, slot=0)
    for i, v in enumerate((-4 / 3, -2 / 3, 2 / 3, 4 / 3)):
        b.frame(f"_DSC{1520 + i}.NEF", 19 if i == 0 else 0.13, slot=v)
    g = detect(b.shots)
    frag = next(x for x in g if x.kind == "exposure")
    assert [s.name for s in frag.shots] == [f"_DSC{1520 + i}.NEF" for i in range(4)]
    assert "partial bracket: 4 of 5 slots" in frag.warnings


def _pattern(i: int) -> bytes:
    """A distinct, detailed band-passed-looking thumbnail per i (structures that don't register)."""
    import random
    rng = random.Random(i)
    return bytes(rng.choice((90, 128, 170)) for _ in range(64 * 44))


def test_slow_presses_with_refocus_are_not_a_burst():
    """_DSC1073-1075: 0.82 / 0.77 s apart, focus 199.5 -> 100 -> 66.8 m (camera in continuous all day)."""
    b = Body(model="NIKON Z 7_2")
    for i, (after, dist) in enumerate(zip((0, 0.82, 0.77), (199.5, 100.0, 66.8))):
        b.frame(f"_DSC{1073 + i}.NEF", after, focus_distance=dist)
    assert kinds(detect(b.shots), "burst") == []


def test_slow_burst_on_a_steady_subject_survives():
    """_DSC0705-0707: 0.4 s exposures every 0.53 s, constant focus: a real slow burst."""
    b = Body(model="NIKON Z 7_2")
    names = [b.frame(f"_DSC{705 + i}.NEF", 0 if i == 0 else 0.53, shutter=0.4, focus_distance=3.2) for i in range(3)]
    assert kinds(detect(b.shots), "burst") == [names]
