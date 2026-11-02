import struct

import pytest

pytest.importorskip("PIL")

from bracketeering import analyze  # noqa: E402
from bracketeering.preview import preview_source  # noqa: E402
from bracketeering.pixels import analyse_file  # noqa: E402
from exifbuilder import LONG, SHORT, _box, build_tiff, standard_tags  # noqa: E402
from imagegen import jpeg_bytes, scene  # noqa: E402


def _tiff_raw_with_preview(jpeg: bytes, decoy: bytes | None = None) -> bytes:
    """TIFF-based RAW: IFD0 points at a JPEG preview (and optionally a lossless-JPEG decoy strip)."""
    ifd0, exif = standard_tags()
    def build(prev_off, decoy_off):
        extra = [(0x0201, LONG, prev_off), (0x0202, LONG, len(jpeg))]
        if decoy is not None:
            extra += [(0x0103, SHORT, 6), (0x0111, LONG, decoy_off), (0x0117, LONG, len(decoy))]
        return build_tiff(ifd0, exif, extra_ifd0=extra)
    head = build(0, 0)
    prev_off = len(head) + (len(head) % 2)
    decoy_off = prev_off + len(jpeg)
    head = build(prev_off, decoy_off)
    return head.ljust(prev_off, b"\x00") + jpeg + (decoy or b"")


def _lossless_jpeg_decoy(size: int) -> bytes:
    # SOI + SOF3 (lossless) header: raw sensor data, must not be chosen as a preview
    return b"\xff\xd8\xff\xc3" + struct.pack(">H", 11) + b"\x0c\x00\x10\x00\x10\x01\x01\x11\x00" + b"\x00" * size


def test_preview_from_tiff_raw_skips_lossless_raw_data(tmp_path):
    jpeg = jpeg_bytes(scene())
    p = tmp_path / "DSC_0001.NEF"
    p.write_bytes(_tiff_raw_with_preview(jpeg, decoy=_lossless_jpeg_decoy(len(jpeg) * 3)))
    assert preview_source(p) == jpeg


def test_preview_from_raf(tmp_path):
    jpeg = jpeg_bytes(scene())
    header = b"FUJIFILMCCD-RAW 0201FF383501".ljust(84, b"\x00") + struct.pack(">II", 160, len(jpeg))
    p = tmp_path / "DSCF0001.RAF"
    p.write_bytes(header.ljust(160, b"\x00") + jpeg + b"\x00" * 100)
    assert preview_source(p) == jpeg


def test_preview_from_cr3_prvw(tmp_path):
    jpeg = jpeg_bytes(scene())
    prvw_uuid = bytes.fromhex("eaf42b5e1c984b88b9fbb7dc406e4d16")
    prvw = _box(b"PRVW", b"\x00" * 8 + struct.pack(">HH", 320, 214) + b"\x00" * 4 + jpeg)
    data = _box(b"ftyp", b"crx \x00\x00\x00\x01crx isom") + _box(b"uuid", prvw_uuid + b"\x00" * 8 + prvw)
    p = tmp_path / "IMG_0001.CR3"
    p.write_bytes(data)
    assert preview_source(p) == jpeg


def test_exposure_flags(tmp_path):
    cases = {"ok": 1.0, "over": 3.0, "under": 0.12}
    for expected, gain in cases.items():
        p = tmp_path / f"{expected}.jpg"
        p.write_bytes(jpeg_bytes(scene(gain=gain)))
        st = analyse_file(p)
        assert st is not None and st.flag == expected, (expected, st)


def _shoot(d):
    # 5-frame bracket of scene 0 (0,-1,+1,-2,+2), a single of the same scene, then a different scene.
    for i, (ev, gain) in enumerate([(0, 0.4), (-1, 0.2), (1, 0.8), (-2, 0.1), (2, 1.6)]):
        den = int(250 / 2 ** ev)
        (d / f"IMG_{i + 1:04d}.JPG").write_bytes(jpeg_bytes(scene(0, gain), exposure=(1, den), bias=(ev, 1), second=0, subsec=f"{i}0"))
    (d / "IMG_0006.JPG").write_bytes(jpeg_bytes(scene(0, 0.45), exposure=(1, 200), mode=0, second=20))
    (d / "IMG_0007.JPG").write_bytes(jpeg_bytes(scene(1), exposure=(1, 200), mode=0, second=50))


def test_best_exposed_and_scenes(tmp_path):
    _shoot(tmp_path)
    groups, errors = analyze([tmp_path], backend="native", pixels=True)
    assert not errors
    kinds = [g.kind for g in groups]
    assert kinds == ["exposure", "single", "single"]
    bracket = groups[0]
    assert bracket.shots[bracket.representative].name == "IMG_0001.JPG"
    # +1 EV keeps its highlights in this scene; +2 EV clips -> +1 is the exposed-to-the-right pick
    assert bracket.shots[bracket.best_exposed].name == "IMG_0003.JPG"
    assert bracket.shots[4].pixels.flag == "over"
    assert bracket.shots[3].pixels.flag == "under"
    assert groups[0].scene == groups[1].scene  # same scene re-shot 20 s later
    assert groups[2].scene != groups[1].scene  # different subject
