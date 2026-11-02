import pytest

from bracketeering.metadata.exiftool import classify_bracket, meta_from_record


@pytest.mark.parametrize("record,expected", [
    ({"AEBBracketValue": "+1", "BracketShotNumber": 2}, ("exposure", 2)),
    ({"AEBBracketValue": "0"}, ("none", None)),
    ({"BracketMode": "AEB", "BracketShotNumber": 3}, ("exposure", 3)),
    ({"BracketMode": "FEB"}, ("flash", None)),
    ({"BracketMode": "Off"}, ("none", None)),
    ({"ReleaseMode2": "Continuous - Exposure Bracketing", "SequenceNumber": 2}, ("exposure", 2)),
    ({"ReleaseMode2": "Normal"}, ("none", None)),
    # Nikon writes bracketing *configuration* on every frame; ShootingMode says whether it was on
    ({"ExposureBracketValue": 0, "BracketSet": "AE", "BracketProgram": "Disabled",
      "AutoBracketModeM": "Flash/Speed", "ShootingMode": "Continuous"}, ("none", None)),
    ({"ExposureBracketValue": -1.333, "BracketSet": "AE", "BracketProgram": "5F",
      "AutoBracketModeM": "Flash/Speed", "ShootingMode": "Continuous, Exposure Bracketing, Auto ISO"}, ("exposure", None)),
    ({"BracketProgram": "Disabled"}, ("none", None)),
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


@pytest.mark.parametrize("record,expected", [
    ({"ContinuousDrive": "Continuous, High"}, ("continuous", None)),
    ({"ContinuousDrive": "Single"}, ("single", None)),
    ({"ReleaseMode": "Normal"}, ("single", None)),
    ({"ReleaseMode2": "Continuous - Hi", "SequenceNumber": 4}, ("continuous", 4)),
    ({"ShootingMode": "Single-Frame"}, ("single", None)),
    ({"ShootingMode": "Continuous, Exposure Bracketing"}, ("continuous", None)),
    ({"DriveMode": "Continuous Shooting, Shot 3"}, ("continuous", 3)),
    ({"DriveMode": "Continuous Low", "SequenceNumber": 0}, ("continuous", None)),
    ({"BurstMode": "Off"}, ("single", None)),
    ({"BurstMode": "On", "SequenceNumber": 2}, ("continuous", 2)),
    ({"DriveMode": "Self-timer (12 sec)"}, ("single", None)),
    ({}, (None, None)),
])
def test_classify_drive(record, expected):
    from bracketeering.metadata.exiftool import classify_drive
    assert classify_drive(record) == expected


def test_nikon_slot_and_shutter_count():
    from bracketeering.metadata.exiftool import bracket_slot
    m = meta_from_record({
        "SourceFile": "/a/_DSC1704.NEF", "Make": "NIKON CORPORATION", "Model": "NIKON Z 7_2",
        "DateTimeOriginal": "2026:08:26 04:47:23", "SubSecTimeOriginal": "10", "OffsetTimeOriginal": "-07:00",
        "ExposureTime": 1 / 60, "FNumber": 1.8, "ISO": 360, "ExposureBracketValue": -1.3333333,
        "ShootingMode": "Continuous, Exposure Bracketing, Auto ISO", "BracketSet": "AE", "ShutterCount": 20483,
    })
    assert m.bracket_kind == "exposure"
    assert m.bracket_value == pytest.approx(-4 / 3, abs=1e-3)
    assert m.shutter_count == 20483
    assert m.utc_offset == -7 * 3600
    assert bracket_slot({"ExposureBracketValue": "+2/3"}) == pytest.approx(2 / 3)
