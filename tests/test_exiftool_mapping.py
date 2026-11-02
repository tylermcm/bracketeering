import pytest

from bracketeering.metadata.exiftool import classify_bracket, meta_from_record


@pytest.mark.parametrize("record,expected", [
    ({"AEBBracketValue": "+1", "BracketShotNumber": 2}, ("exposure", 2)),
    ({"AEBBracketValue": "0"}, ("none", None)),
    ({"BracketMode": "AEB", "BracketShotNumber": 3}, ("exposure", 3)),
    ({"BracketMode": "FEB"}, ("flash", None)),
    ({"BracketMode": "Off"}, ("none", None)),
    ({"ReleaseMode2": "Continuous - Exposure Bracketing", "SequenceNumber": 2}, ("exposure", 2)),
    ({"ReleaseMode2": "Normal"}, (None, None)),
    ({"ShootingMode": "Continuous, Exposure Bracketing"}, ("exposure", None)),
    ({"ShootingMode": "Single-Frame, White Balance Bracketing"}, ("wb", None)),
    ({"AutoBracketing": "On", "SequenceNumber": 1}, ("exposure", 1)),
    ({"AutoBracketing": "0.3 EV, 3 frames"}, ("exposure", None)),
    ({"AutoBracketing": "Off"}, ("none", None)),
    ({"DriveMode": "Continuous Shooting, Shot 4; AE Bracketing"}, ("exposure", 4)),
    ({"FocusBracketing": "Enable"}, ("focus", None)),
    ({"FocusBracketing": "Disable"}, ("none", None)),
    ({"BracketSettings": "No Bracket"}, ("none", None)),
    ({"BracketSettings": "3 Images, Sequence 0/-/+"}, ("exposure", None)),
    ({"AutoExposureBracketing": "On (shot 2)"}, ("exposure", 2)),
    ({}, (None, None)),
])
def test_classify(record, expected):
    assert classify_bracket(record) == expected


def test_meta_from_record():
    m = meta_from_record({
        "SourceFile": "/a/IMG_0007.CR3", "Make": "Canon", "Model": "Canon EOS R5", "SerialNumber": 123,
        "DateTimeOriginal": "2024:05:01 12:00:00", "SubSecTimeOriginal": 45, "OffsetTimeOriginal": "+00:00",
        "ExposureTime": 0.004, "FNumber": 8, "ISO": 100, "ExposureCompensation": -1,
        "ExposureMode": 2, "FocalLength": 24, "BracketMode": "AEB", "BracketShotNumber": 2,
    })
    assert m.serial == "123"
    assert m.time_precise and m.capture_time % 1 == pytest.approx(0.45)
    assert (m.bracket_kind, m.bracket_position) == ("exposure", 2)
    assert m.exposure_mode == 2 and m.file_number == 7
