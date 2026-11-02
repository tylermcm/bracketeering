"""Regression tests replaying the hand-checked real shoot in docs/reviews/2026-11-02-china-8-26.md.

Frames, bracket slots (Nikon ExposureBracketValue), timings and Auto ISO behaviour follow the review;
each test pins one of its findings.
"""

from pathlib import Path

import pytest

from bracketeering.engine import detect
from bracketeering.model import FileMeta, Shot
from bracketeering.scan import pair_sidecars

T0 = 1_787_000_000.0
NIKON = dict(make="NIKON CORPORATION", model="NIKON Z 7_2", serial="3041025", lens="NIKKOR Z 50mm f/1.8 S",
             focal_length=50.0, utc_offset=-7 * 3600)


class Shoot:
    def __init__(self):
        self.t = T0
        self.shots: list[Shot] = []

    def frame(self, n: int, after: float, *, slot=None, iso=500.0, shutter=1 / 60, mode=0, folder="", drive="continuous"):
        self.t += after
        m = FileMeta(Path(folder) / f"_DSC{n}.NEF", capture_time=self.t, time_precise=True, exposure_time=shutter,
                     f_number=1.8, iso=iso, exposure_mode=mode, exposure_program=1 if mode == 1 else 3,
                     bracket_kind="exposure" if slot is not None else "none", bracket_value=slot,
                     drive_mode=drive, shutter_count=20482 + n - 1703, **NIKON)
        self.shots.append(Shot([m]))
        return f"_DSC{n}.NEF"


def groups_by_kind(groups, kind):
    return [[s.name for s in g.shots] for g in groups if g.kind == kind]


@pytest.fixture
def shoot():
    s = Shoot()
    # _DSC1690-1694 (5F program, 0 / -4/3 / -2/3 / +2/3 / +4/3): recomposed between presses, 5.8 s
    s.frame(1690, 0, slot=0, iso=800)
    s.frame(1691, 0.13, slot=-4 / 3, iso=320)
    s.frame(1692, 5.8, slot=-2 / 3, iso=450)
    s.frame(1693, 0.13, slot=2 / 3, iso=1000)
    s.frame(1694, 0.13, slot=4 / 3, iso=1600)
    # 1695-1697 deleted; 1698/1699 are the tail of that cycle, different subjects
    s.frame(1698, 40, slot=2 / 3, iso=900)
    s.frame(1699, 4.0, slot=4 / 3, iso=1400)
    s.frame(1700, 20, slot=0, iso=640)  # same pose, 0.18 s
    s.frame(1701, 0.18, slot=-4 / 3, iso=250)
    # _DSC1703-1705: slots 1-3; re-pressed after 1.11 s; Auto ISO re-metered between presses
    s.frame(1703, 60, slot=0, iso=1000)
    s.frame(1704, 1.11, slot=-4 / 3, iso=360)
    s.frame(1705, 0.13, slot=-2 / 3, iso=560)
    # _DSC1706-1708: 0 -> 2.07 s -> -4/3 -> 0.13 s -> -2/3, one composition
    s.frame(1706, 30, slot=0, iso=800)
    s.frame(1707, 2.07, slot=-4 / 3, iso=280)
    s.frame(1708, 0.13, slot=-2 / 3, iso=450)
    # _DSC1709-1711: M mode, ISO fixed 500: shutter-driven AEB (1/15, 1/40, 1/25)
    s.frame(1709, 300, slot=0, shutter=1 / 15, mode=1)
    s.frame(1710, 1.13, slot=-4 / 3, shutter=1 / 40, mode=1)
    s.frame(1711, 0.97, slot=-2 / 3, shutter=1 / 25, mode=1)
    # bracketing off from 1712: a continuous burst with Auto ISO wandering in 1/3 stops
    for i, iso in enumerate((500, 640, 500, 400, 500, 640)):
        s.frame(1712 + i, 30 if i == 0 else 0.18, iso=float(iso))
    # _DSC1746-1750 / _DSC1751-1755: two presses at 5.6 fps, 16.71 s apart
    for i in range(5):
        s.frame(1746 + i, 60 if i == 0 else 0.18)
    for i in range(5):
        s.frame(1751 + i, 16.71 if i == 0 else 0.18)
    return s


def test_brackets_follow_the_camera_cycles(shoot):
    g = detect(shoot.shots)
    assert groups_by_kind(g, "exposure") == [
        ["_DSC1690.NEF", "_DSC1691.NEF"],
        ["_DSC1692.NEF", "_DSC1693.NEF", "_DSC1694.NEF"],
        ["_DSC1700.NEF", "_DSC1701.NEF"],
        ["_DSC1703.NEF", "_DSC1704.NEF", "_DSC1705.NEF"],
        ["_DSC1706.NEF", "_DSC1707.NEF", "_DSC1708.NEF"],
        ["_DSC1709.NEF", "_DSC1710.NEF", "_DSC1711.NEF"],
    ]
    singles = {n for grp in groups_by_kind(g, "single") for n in grp}
    assert {"_DSC1698.NEF", "_DSC1699.NEF"} <= singles


def test_offsets_come_from_the_camera_not_auto_iso(shoot):
    by_first = {g.shots[0].name: g for g in detect(shoot.shots) if g.kind == "exposure"}
    for first in ("_DSC1703.NEF", "_DSC1706.NEF", "_DSC1709.NEF"):
        grp = by_first[first]
        assert grp.ev_offsets == pytest.approx([0, -4 / 3, -2 / 3], abs=0.01)
        assert grp.shots[grp.representative].name == first  # the metered frame
        assert any("partial bracket: 3 of 5" in w for w in grp.warnings)
        assert grp.step_ev == pytest.approx(0.67, abs=0.01)
    assert not any("by hand" in e for e in by_first["_DSC1709.NEF"].evidence)


def test_frames_with_bracketing_off_are_never_brackets(shoot):
    g = detect(shoot.shots)
    burst_frames = {f"_DSC{n}.NEF" for n in range(1712, 1718)}
    in_brackets = {s.name for grp in g if grp.kind == "exposure" for s in grp.shots}
    assert not burst_frames & in_brackets
    assert sorted(burst_frames) in groups_by_kind(g, "burst")


def test_two_presses_are_two_bursts(shoot):
    bursts = groups_by_kind(detect(shoot.shots), "burst")
    assert [f"_DSC{n}.NEF" for n in range(1746, 1751)] in bursts
    assert [f"_DSC{n}.NEF" for n in range(1751, 1756)] in bursts


def test_copies_in_a_picks_folder_are_not_a_burst(shoot):
    # _winners/ holds a copy of _DSC1747 (same body, same shutter count)
    original = next(s for s in shoot.shots if s.name == "_DSC1747.NEF").meta
    copy = FileMeta(Path("_winners/_DSC1747.NEF"), **{k: getattr(original, k) for k in (
        "make", "model", "serial", "lens", "capture_time", "time_precise", "exposure_time", "f_number", "iso",
        "exposure_mode", "focal_length", "bracket_kind", "drive_mode", "shutter_count", "utc_offset")})
    metas = [f for s in shoot.shots for f in s.files] + [copy]
    shots = pair_sidecars(metas)
    assert len(shots) == len(shoot.shots)
    merged = next(s for s in shots if len(s.files) == 2)
    assert merged.meta.path == Path("_DSC1747.NEF")  # the original stays primary
    bursts = groups_by_kind(detect(shots), "burst")
    assert [f"_DSC{n}.NEF" for n in range(1746, 1751)] in bursts


def _thumb(kind: int, shift: int = 0) -> bytes:
    """A 28x20 grayscale thumbnail of one of two synthetic compositions, optionally drifted."""
    from bracketeering.pixels import THUMB_H, THUMB_W
    px = bytearray()
    for y in range(THUMB_H):
        for x in range(THUMB_W):
            xs = x + shift
            v = (xs * 9 + (y * 37 if kind else 0) + (80 if (xs // 5 + y // 4) % 2 == kind else 0)) % 256
            px.append(v)
    return bytes(px)


def _with_thumb(shot: Shot, thumb: bytes) -> Shot:
    from bracketeering.pixels import PixelStats
    shot.pixels = PixelStats(0, 0.0, 0.0, 0.4, "ok", thumb)
    return shot


def test_recomposed_first_frame_splits_from_its_cycle():
    """DSC_1254-1256: slot 0 shot on one framing, slots 2-3 on another (36 % shift), quickly."""
    s = Shoot()
    s.frame(1254, 0, slot=0)
    s.frame(1255, 0.8, slot=-2 / 3)
    s.frame(1256, 0.2, slot=2 / 3)
    _with_thumb(s.shots[0], _thumb(1))
    _with_thumb(s.shots[1], _thumb(0))
    _with_thumb(s.shots[2], _thumb(0, shift=1))
    g = detect(s.shots)
    assert groups_by_kind(g, "exposure") == [["_DSC1255.NEF", "_DSC1256.NEF"]]
    assert groups_by_kind(g, "single") == [["_DSC1254.NEF"]]


def test_scene_survives_drift_and_one_odd_frame():
    """_DSC0878-0894: one composition with 5-8 % handheld drift; dHash cut it into 11 scenes."""
    from bracketeering.pixels import assign_scenes
    s = Shoot()
    for i, after in enumerate((0, 2.3, 1.6, 3.1, 1.9, 2.6, 1.4, 2.8)):  # separate presses, irregular
        s.frame(878 + i, after, drive="single")
    groups = detect(s.shots)
    assert [g.kind for g in groups] == ["single"] * 8
    for i, grp in enumerate(groups):
        _with_thumb(grp.shots[0], _thumb(1) if i == 4 else _thumb(0, shift=i % 3))  # frame 4: someone walked in
    assign_scenes(groups)
    scenes = [grp.scene for grp in groups]
    assert len({sc for i, sc in enumerate(scenes) if i != 4}) == 1
    assert scenes[4] != scenes[0]
