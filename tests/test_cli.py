import json

from bracketeering.cli import main
from exifbuilder import RATIONAL, SRATIONAL, build_tiff, jpeg_with_exif, standard_tags


def _write(path, exposure, bias, second, subsec):
    ifd0, exif = standard_tags(exposure=exposure, bias=bias, dto=f"2024:05:01 12:00:{second:02d}", subsec=subsec)
    path.write_bytes(jpeg_with_exif(build_tiff(ifd0, exif)))


def _shoot(d):
    # 0,-1,+1 bracket, then an unrelated single 30s later
    _write(d / "IMG_0001.JPG", (1, 250), (0, 1), 0, "10")
    _write(d / "IMG_0002.JPG", (1, 500), (-1, 1), 0, "30")
    _write(d / "IMG_0003.JPG", (1, 125), (1, 1), 0, "50")
    ifd0, exif = standard_tags(exposure=(1, 60), mode=0, dto="2024:05:01 12:00:40")
    (d / "IMG_0004.JPG").write_bytes(jpeg_with_exif(build_tiff(ifd0, exif)))


def test_scan_json(tmp_path, capsys):
    src = tmp_path / "card"
    src.mkdir()
    _shoot(src)
    assert main(["scan", str(src), "--format", "json", "--backend", "native"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["exposure_brackets"] == 1
    assert data["summary"]["singles"] == 1
    b = data["brackets"][0]
    assert [f["files"][0].rsplit("/", 1)[1] for f in b["frames"]] == ["IMG_0001.JPG", "IMG_0002.JPG", "IMG_0003.JPG"]
    assert b["ev_offsets"] == [0.0, -1.0, 1.0]


def test_organize_copy_and_dry_run(tmp_path, capsys):
    src, dst = tmp_path / "card", tmp_path / "out"
    src.mkdir()
    _shoot(src)
    assert main(["organize", str(src), "--dest", str(dst), "--dry-run", "--backend", "native"]) == 0
    assert not dst.exists()
    assert main(["organize", str(src), "--dest", str(dst), "--mode", "copy", "--backend", "native"]) == 0
    folders = [p for p in dst.iterdir() if p.is_dir()]
    assert len(folders) == 1 and "HDR3x1EV" in folders[0].name
    assert sorted(p.name for p in folders[0].iterdir()) == ["IMG_0001.JPG", "IMG_0002.JPG", "IMG_0003.JPG"]
    assert len(list(src.iterdir())) == 4  # copy leaves sources alone
