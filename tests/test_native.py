import io
from datetime import datetime, timezone

import pytest

from bracketeering.metadata.native import read_native
from exifbuilder import (
    build_tiff, cr3_with_exif, jpeg_with_exif, raf_with_exif, rw2_with_embedded_jpeg, standard_tags,
)

EXPECTED_TS = datetime(2024, 5, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() + 0.123


def _check(m):
    assert m.error is None, m.error
    assert m.make == "Canon"
    assert m.model == "Canon EOS R5"
    assert m.exposure_time == pytest.approx(1 / 250)
    assert m.f_number == 8
    assert m.iso == 100
    assert m.exposure_bias == 0
    assert m.exposure_mode == 2
    assert m.focal_length == 24
    assert m.serial == "012345678901"
    assert m.lens == "RF24-70mm F2.8 L IS USM"
    assert m.capture_time == pytest.approx(EXPECTED_TS)
    assert m.time_precise
    assert m.flash_fired is False


@pytest.mark.parametrize("big_endian", [False, True])
def test_jpeg(tmp_path, big_endian):
    ifd0, exif = standard_tags()
    p = tmp_path / "IMG_0001.JPG"
    p.write_bytes(jpeg_with_exif(build_tiff(ifd0, exif, big_endian=big_endian)))
    m = read_native(p)
    _check(m)
    assert m.file_number == 1


@pytest.mark.parametrize("suffix,magic", [(".NEF", 42), (".DNG", 42), (".ORF", 0x4F52), (".CR2", 42)])
def test_tiff_based_raw(tmp_path, suffix, magic):
    ifd0, exif = standard_tags()
    p = tmp_path / f"DSC_0042{suffix}"
    p.write_bytes(build_tiff(ifd0, exif, magic=magic) + b"\x00" * 1024)
    _check(read_native(p))


def test_raf(tmp_path):
    ifd0, exif = standard_tags()
    p = tmp_path / "DSCF0001.RAF"
    p.write_bytes(raf_with_exif(build_tiff(ifd0, exif)))
    _check(read_native(p))


def test_cr3(tmp_path):
    ifd0, exif = standard_tags()
    p = tmp_path / "IMG_0001.CR3"
    p.write_bytes(cr3_with_exif(ifd0, exif))
    _check(read_native(p))


def test_rw2_embedded_jpeg(tmp_path):
    ifd0, exif = standard_tags()
    p = tmp_path / "P1000001.RW2"
    p.write_bytes(rw2_with_embedded_jpeg("Canon", build_tiff(ifd0, exif)))
    _check(read_native(p))


def test_negative_bias_and_no_offset(tmp_path):
    ifd0, exif = standard_tags(bias=(-4, 3), offset="", subsec="")
    exif = [t for t in exif if t[0] not in (0x9011, 0x9291)]
    p = tmp_path / "a.jpg"
    p.write_bytes(jpeg_with_exif(build_tiff(ifd0, exif)))
    m = read_native(p)
    assert m.exposure_bias == pytest.approx(-4 / 3)
    assert not m.time_precise
    assert m.capture_time == datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def test_garbage_is_reported_not_raised(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe1\x00")
    assert read_native(p).error
    q = tmp_path / "x.nef"
    q.write_bytes(b"II*\x00\xff\xff\xff\x7f")
    assert read_native(q).error


def test_matches_pillow(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (8, 8))
    exif = Image.Exif()
    exif[0x010F] = "NIKON CORPORATION"
    exif[0x0110] = "NIKON Z 8"
    ifd = exif.get_ifd(0x8769)
    ifd[0x829A] = 1 / 640
    ifd[0x829D] = 5.6
    ifd[0x8827] = 400
    ifd[0x9003] = "2023:01:02 03:04:05"
    ifd[0x9291] = "50"
    ifd[0xA402] = 2
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    p = tmp_path / "DSC_1234.JPG"
    p.write_bytes(buf.getvalue())
    m = read_native(p)
    assert m.error is None
    assert m.make == "NIKON CORPORATION"
    assert m.exposure_time == pytest.approx(1 / 640)
    assert m.f_number == pytest.approx(5.6)
    assert m.iso == 400
    assert m.exposure_mode == 2
    assert m.time_precise
    assert m.file_number == 1234


@pytest.mark.parametrize("mode,slot,kind,drive", [
    (0b110001, (-4, 3), "exposure", "continuous"),  # Continuous, Exposure Bracketing, Auto ISO
    (0b100001, None, "none", "continuous"),  # Continuous, Auto ISO (bracketing off)
    (0b000000, None, "none", "single"),  # Single-Frame
])
def test_nikon_makernote(tmp_path, mode, slot, kind, drive):
    from exifbuilder import UNDEFINED, nikon_makernote
    ifd0, exif = standard_tags()
    ifd0 = [(t, ty, ("NIKON CORPORATION" if t == 0x010F else "NIKON Z 7_2" if t == 0x0110 else v)) for t, ty, v in ifd0]
    exif = exif + [(0x927C, UNDEFINED, nikon_makernote(mode, slot, 20483))]
    p = tmp_path / "_DSC1704.NEF"
    p.write_bytes(build_tiff(ifd0, exif) + b"\x00" * 512)
    m = read_native(p)
    assert m.error is None, m.error
    assert (m.bracket_kind, m.drive_mode, m.shutter_count) == (kind, drive, 20483)
    assert m.bracket_value == (pytest.approx(slot[0] / slot[1]) if slot else None)
    assert m.utc_offset == 2 * 3600
