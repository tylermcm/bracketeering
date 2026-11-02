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


# ----------------------------------------------------------------- second review (0.3.0 re-test)

def _partial(groups, first):
    g = next(g for g in groups if g.kind == "exposure" and g.shots[0].name == first)
    return [w for w in g.warnings if w.startswith("partial")]


def test_complete_cycles_are_not_called_partial():
    """3F cycles next to a 5F session, and 1-EV cycles next to a 2/3-EV session, are complete."""
    s = Shoot()
    # in the camera's firing order: metered slot, then darkest to brightest (0, -4/3, -2/3, +2/3, +4/3)
    for k, slots in enumerate(((0, -2 / 3, 2 / 3), (0, -4 / 3, -2 / 3, 2 / 3, 4 / 3), (0, -2, -1, 1, 2), (0, -2 / 3, 2 / 3))):
        for i, v in enumerate(slots):
            s.frame(1224 + 10 * k + i, 30 if i == 0 else 0.15, slot=v)
    g = detect(s.shots)
    assert len([x for x in g if x.kind == "exposure"]) == 4
    for k in range(4):
        assert _partial(g, f"_DSC{1224 + 10 * k}.NEF") == []


def test_partial_cycles_are_flagged_with_the_right_size(shoot):
    g = detect(shoot.shots)
    assert _partial(g, "_DSC1700.NEF") == ["partial bracket: 2 of 5 slots"]  # 0, -4/3 in a 2/3-EV 5F session
    assert _partial(g, "_DSC1703.NEF") == ["partial bracket: 3 of 5 slots"]


def test_camera_program_setting_wins():
    s = Shoot()
    for i, v in enumerate((0, -1, 1)):
        s.frame(1300 + i, 0 if i == 0 else 0.15, slot=v)
        s.shots[-1].meta.bracket_program = 5  # exiftool BracketProgram "5F"
    assert _partial(detect(s.shots), "_DSC1300.NEF") == ["partial bracket: 3 of 5 slots"]


def test_slow_separate_presses_in_continuous_drive_are_not_a_burst():
    """Body left in continuous drive all day: _DSC0600-0604 (~2 s apart), _DSC1073-1075 (~0.8 s)."""
    s = Shoot()
    for i, (after, iso) in enumerate(zip((0, 2.1, 1.9, 2.0, 2.2), (2000, 3600, 2200, 2500, 2500))):
        s.frame(600 + i, after, iso=float(iso))
    s.frame(1073, 60)
    s.frame(1074, 0.82)
    s.frame(1075, 0.77)
    # _DSC0705-0707: 0.4 s exposures fired every 0.53 s, constant settings: a real slow burst
    s.frame(705, 60, shutter=0.4)
    s.frame(706, 0.53, shutter=0.4)
    s.frame(707, 0.53, shutter=0.4)
    g = detect(s.shots)
    assert groups_by_kind(g, "burst") == [["_DSC705.NEF", "_DSC706.NEF", "_DSC707.NEF"]]


PIL = pytest.importorskip("PIL")


def _world(seed: int, w: int = 1600, h: int = 700):
    """A wide synthetic place: sky, structures with windows, foliage texture."""
    import random
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    im = Image.new("L", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=int(170 - 60 * y / h) if y < h * 0.55 else 70)
    for _ in range(25):
        x, bw, top = rng.randrange(w), rng.randrange(30, 160), rng.randrange(int(h * 0.15), int(h * 0.5))
        d.rectangle([x, top, x + bw, int(h * 0.6)], fill=rng.randrange(30, 200))
        for _ in range(rng.randrange(0, 6)):
            wx, wy = x + rng.randrange(max(1, bw - 10)), top + rng.randrange(max(1, int(h * 0.6) - top - 10))
            d.rectangle([wx, wy, wx + 8, wy + 12], fill=rng.randrange(0, 255))
    for _ in range(3000):
        x, y, r = rng.randrange(w), rng.randrange(int(h * 0.55), h), rng.randrange(2, 6)
        d.ellipse([x - r, y - r, x + r, y + r], fill=rng.randrange(20, 140))
    return im


def _view(world, x0: int, exposure: float = 1.0) -> bytes:
    from bracketeering.pixels import band_pass
    fw = 720
    im = world.crop((x0, 110, x0 + fw, 110 + 480)).resize((600, 400))
    if exposure != 1.0:
        im = im.point(lambda p: max(0, min(255, int(p * exposure))))
    return band_pass(im)


def test_similarity_separates_drift_from_recompose():
    from bracketeering.pixels import DIFFERENT, SAME_SCENE, thumb_similarity
    w = _world(1)
    base = _view(w, 300)
    for drifted in (_view(w, 322), _view(w, 343), _view(w, 357, exposure=0.4), _view(w, 300, exposure=2.0)):
        assert thumb_similarity(base, drifted) >= SAME_SCENE  # 3-8 % handheld drift, bracket exposures
    for other in (_view(w, 559), _view(w, 444), _view(_world(2), 300)):
        assert thumb_similarity(base, other) < DIFFERENT  # 36 % / 20 % recompose, another place


def test_quick_recompose_splits_a_bracket_cycle():
    """DSC_1254 -> 1255: 36 % recompose only 1.6 s apart (0.3.0 scored it +0.657 and kept the cycle)."""
    w = _world(3)
    s = Shoot()
    s.frame(1254, 0, slot=0)
    s.frame(1255, 1.6, slot=-2 / 3)
    s.frame(1256, 0.2, slot=2 / 3)
    for shot, thumb in zip(s.shots, (_view(w, 300), _view(w, 559, 0.6), _view(w, 565, 1.6))):
        _with_thumb(shot, thumb)
    g = detect(s.shots)
    assert groups_by_kind(g, "exposure") == [["_DSC1255.NEF", "_DSC1256.NEF"]]
    assert groups_by_kind(g, "single") == [["_DSC1254.NEF"]]


def test_scene_does_not_chain_across_subjects():
    """Scene 195: a slow walk around a statue chained 41 groups / 4 subjects via the last-4 lookback."""
    from bracketeering.pixels import assign_scenes
    w = _world(4, w=3200)
    s = Shoot()
    xs = [100 + 70 * i for i in range(24)]  # each view drifts ~10 % from the last; 0 and 23 share nothing
    for i in range(len(xs)):
        s.frame(1198 + i, 0 if i == 0 else 6.0 + (i % 3), drive="single")
    groups = detect(s.shots)
    for grp, x in zip(groups, xs):
        _with_thumb(grp.shots[0], _view(w, x))
    assign_scenes(groups)
    first, last = groups[0].scene, groups[-1].scene
    assert first != last
