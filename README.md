# bracketeering

Fast, accurate detection and grouping of camera **exposure brackets** (AEB / HDR sets), **bursts** and **focus stacks** from a folder of photos. Optionally, it also finds **near-duplicate scenes** and per-frame **exposure problems**.

It's built as the grouping step of a photo-culling pipeline. Every file gets a group, a kind and a role, so the next stage knows which frames are alternatives of one another and which single frame stands for each group. The JSON contract is documented in [SCHEMA.md](SCHEMA.md).

Example output:

```
$ bracketeer scan /Volumes/CARD
B00001  bracket  5 frames  step 1 EV  [+0 -1 +1 -2 +2]  conf 0.99
         *IMG_0412.CR3, IMG_0413.CR3, IMG_0414.CR3, IMG_0415.CR3, IMG_0416.CR3
B00007  bracket  3 frames  step 2 EV  [+0 -2 +2]  conf 0.98
         *IMG_0430.CR3, IMG_0431.CR3, IMG_0432.CR3
U00008  burst    9 frames  conf 0.90
         IMG_0433.CR3, ..., *IMG_0437.CR3, ...
F00009  focus    24 frames  conf 0.95
         IMG_0440.CR3, ...

1873 shots (3746 files): 214 exposure brackets, 41 bursts, 3 focus stacks, 611 singles   (* = representative frame)
```

- **No required dependencies.** Pure Python 3.10+. A built-in EXIF reader handles JPEG, TIFF, DNG, CR2, CR3, NEF, NRW, ARW, SR2, ORF, RW2, PEF, RAF and other TIFF-based RAWs. If [exiftool](https://exiftool.org) is installed, it covers everything else and adds maker-note bracket tags.
- **Efficient.** The reader seeks straight to the IFD entries it needs and reads a few KB per file, even from a 60 MB RAW. Files are read in parallel. In one test, 3,000 RAW-sized files took about 1 s end to end. Grouping runs at 10–20k shots/s.
- **Pixel analysis is optional.** `--pixels` (needs Pillow) decodes each RAW's embedded JPEG preview, not the RAW itself, at about 5 ms per file.
- **Explainable.** Every group lists its evidence, a confidence score, and warnings such as missing frames or partial brackets.

## Install

```bash
pip install -e .            # provides the `bracketeer` command
pip install -e '.[pixels]'  # + Pillow, for --pixels
pip install -e '.[dev]'     # + pytest, pillow for the test-suite
```

## Usage

```bash
bracketeer scan PATH...                       # human-readable report
bracketeer scan PATH... -v                    # include singles and evidence
bracketeer scan PATH... --format json -o report.json
bracketeer scan PATH... --format csv          # one row per file
bracketeer scan PATH... --pixels --format json # + exposure stats, best-exposed frame, near-duplicate scenes

bracketeer organize PATH... --dest ./sorted --dry-run
bracketeer organize PATH... --dest ./sorted --mode copy|move|symlink|hardlink
```

Useful options:
- `--backend auto|native|exiftool`
- `--max-gap`: seconds allowed between bracket frames (default 2)
- `--manual-gap`: for hand-adjusted brackets in M mode (default 6)
- `--burst-gap`: seconds between burst frames (default 1)
- `--min-confidence`: weaker bracket candidates become singles (default 0.5)
- `--scene-window`, `--hash-threshold`: near-duplicate tuning
- `--max-size`, `--min-size`, `--no-focus`, `--no-bursts`

`organize` creates one folder per group, for example `20240501-120000_IMG_0412_HDR5x1EV/` or `…_BURST9/`. It keeps RAW+JPEG sidecars together and never overwrites existing files.

### Python API

```python
from bracketeering import analyze, EngineConfig

groups, unreadable = analyze(["/photos/2024-05-01"], EngineConfig(min_confidence=0.8), pixels=True)
for g in groups:
    rep = g.shots[g.representative]          # the frame that stands for the group
    if g.kind == "exposure":
        best = g.shots[g.best_exposed] if g.best_exposed is not None else rep
        print("bracket", g.size, g.ev_offsets, "best exposed:", best.meta.path, "scene", g.scene)
    elif g.kind == "burst":
        print("burst", [s.meta.path for s in g.shots])

from bracketeering.report import to_dict    # the SCHEMA.md structure, ready for json.dumps
```

`bracketeering.detect(shots)` runs only the grouping engine. Use it if you already have metadata from your own source, such as a catalog database.

## How it works

1. **Read metadata.** The engine reads capture time with sub-seconds and UTC offset, shutter, aperture, ISO, exposure compensation, EXIF `ExposureMode` (2 = auto bracket), focal length, lens, body serial, white balance, flash and subject distance. With `--backend exiftool` it also reads maker-note bracket tags (Canon `BracketMode`/`BracketShotNumber`/`AEBBracketValue`, Sony `ReleaseMode*`, Nikon `ShootingMode`, Fujifilm/Pentax `AutoBracketing`, Olympus `DriveMode`, Panasonic `BracketSettings`, …) and normalises them to a bracket kind and shot position.
2. **Pair sidecars.** `IMG_1.CR3` and `IMG_1.JPG` in the same folder become one shot when their timestamps agree. The RAW is the authoritative source.
3. **Split into runs** per camera body. A run is a chain of consecutive frames with the same lens and focal length and comparable exposure data. Each frame must start within `max_gap` of the *end* of the previous exposure, so long-exposure brackets work. Large frame-counter jumps also end a run.
4. **Carve out focus stacks.** These are constant-exposure blocks flagged by maker notes, or with monotonic focus distance.
5. **Segment each run by dynamic programming.** Each candidate group gets an additive log-odds-style score, and singles score zero. The engine finds the partition with the highest total score:
   - **Exposure structure.** Brightness is measured from all three settings: log2(shutter) + log2(ISO/100) − 2·log2(f-number). That makes shutter-driven brackets (A/Av and M modes), aperture-driven brackets (S/Tv mode) and auto-ISO brackets all register. Cameras write rounded labels (f/13 is really f/12.7, 1/125 s is really 1/128 s), so standard ⅓-stop labels are snapped back to their exact values first. Exposures must be distinct and evenly spaced. The step is fitted on the 1/3- and 1/2-stop grid and tolerates nominal shutter-speed rounding. An interior gap of exactly *k* steps means *k*−1 frames were deleted. The group is still accepted, with a warning.
   - **Firing order.** Groups that match the orders cameras actually use score higher: `0,-,+`, `-,0,+`, `+,0,-`, `0,+,-`, `0,-,--,+,++`, … The engine also matches partial brackets with a missing frame at either end.
   - **Metadata.** Evidence includes `ExposureMode = auto bracket`, maker-note bracket mode, shot positions `1..n` (a reset to 1 inside a group is a hard veto), and exposure compensation matching the measured deltas. A frame flagged as bracketed is penalised if it's left ungrouped. If a body is seen writing `ExposureMode=2` elsewhere in the shoot, its frames with other modes count *against* bracketing.
   - **Context.** Groups that change two exposure variables at once, change white balance or fire a flash are penalised. A large time gap favours a boundary there.
   - **Burst veto.** Auto-exposure bursts wander in 1/3-stop increments and can look like small brackets. A group without metadata support is dissolved if ungrouped frames, or groups with a different signature, sit next to it at the same cadence and exposure level.
6. **Second pass with a prior.** Photographers rarely change bracket settings mid-shoot. The camera's dominant signature (size, firing order, step) becomes a prior, and the DP runs again. This settles the hard cases, such as back-to-back brackets fired in continuous drive with no pause. Group evaluations are cached, so the second pass costs almost nothing.
7. **Bursts.** After brackets are settled, chains of leftover frames shot at burst cadence (≤ `burst_max_gap` apart, same lens and focal length) become burst groups. Because bracket detection runs first, a burst is "fast, but not a bracket". That's how metering-jitter bursts and repeated exposures are kept apart from real brackets.
8. **Representatives.** Each group gets a representative frame. For brackets it's the middle exposure (≈ 0 EV); for bursts and stacks it's the middle frame.
9. **Pixels (optional).** Embedded previews are found in each container (JPEG `APP1`, TIFF/NEF/ARW/DNG/CR2 preview IFDs and SubIFDs, RW2 `JpgFromRaw`, RAF, CR3 `PRVW`). Lossless-JPEG raw data is skipped. Each preview is decoded at reduced size, and the engine computes:
   - **Exposure stats** per frame: highlight clipping (any channel ≥ 250), crushed shadows, mean level, and an `ok`/`over`/`under` flag.
   - **`best_exposed`** per bracket: the brightest frame whose highlights hold (≤ 0.5% clipped), i.e. expose to the right. Of the usable frames it has the cleanest shadows.
   - **Near-duplicate scenes:** consecutive groups from one body, at most `scene_window` apart, whose representatives have a similar perceptual hash (dHash). Gradient-based hashing tolerates exposure differences.

## Accuracy

`scripts/benchmark.py` simulates realistic shoots with known ground truth. The simulator includes AEB in several firing orders and both drive modes, back-to-back brackets, deleted frames, hand-adjusted manual brackets, auto-exposure bursts with metering jitter, focus stacks, single shots, and quick handheld pairs with an exposure tweak. Shutter speeds are quantised to real nominal values. Results over 20 seeds (~4,000 brackets and ~900 bursts per profile):

| profile | what's hard about it | bracket precision | bracket recall | burst recall | burst→bracket | bracket→burst |
|---|---|---|---|---|---|---|
| canon-aeb | baseline: full metadata | 0.999 | 0.999 | 1.000 | 0.00% | 0.01% |
| sony-no-subsec | timestamps only to the second | 0.999 | 0.999 | 0.952 | 0.00% | 0.02% |
| nikon-mixed-settings | size 2–9 and step ⅓–3 EV change every bracket | 0.991 | 0.982 | 1.000 | 0.00% | 0.10% |
| manual-brackets | user-driven brackets, 1–4 s apart, no bracket flags | 0.996 | 0.979 | 0.994 | 0.27% | 0.35% |
| no-bracket-metadata | no ExposureMode, no exposure compensation, AE bursts | 0.979 | 0.968 | 0.997 | 0.15% | 0.76% |
| worst-case | all of the above, no sub-seconds, 50% back-to-back | 0.980 | 0.964 | 0.941 | 0.71% | 1.52% |

Precision and recall are for exact groups: the predicted group contains exactly the right frames. The last two columns measure confusion between the two kinds: the share of burst frames wrongly placed in a bracket, and of bracket frames wrongly placed in a burst. Without sub-second timestamps, a burst sometimes absorbs a neighbouring single. That's why burst recall is lower in those profiles.

These numbers come from a simulator written alongside the engine, so they reflect its assumptions about camera behaviour. Real shoots are the true test. Most remaining errors are inherently ambiguous without metadata, for example a 1/3-EV bracket next to an auto-exposure burst. Use `--backend exiftool` when maker notes are available.

```bash
python scripts/benchmark.py --seeds 20
python -m pytest
```

## Layout

```
src/bracketeering/
  metadata/native.py    dependency-free EXIF reader (JPEG/TIFF/RAW/RAF/CR3)
  metadata/exiftool.py  batched exiftool backend + maker-note normalisation
  scan.py               file discovery, RAW+JPEG pairing
  engine.py             run splitting, scoring, DP segmentation, priors, bursts
  preview.py            embedded JPEG preview extraction
  pixels.py             exposure stats, best-exposed pick, near-duplicate scenes
  report.py             text / JSON (SCHEMA.md) / CSV output
  organize.py           copy / move / link groups into folders
  synth.py              shoot simulator + evaluation metrics
  cli.py                `bracketeer` command
```

## Notes and limitations

- Times in reports are UTC when the file records an offset (`OffsetTimeOriginal`), and camera-local otherwise.
- The native reader doesn't decode maker notes. Bracket detection still works from EXIF and exposure analysis, but maker-note shot positions need `--backend exiftool`.
- HEIC/HEIF and less common RAW containers fall back to exiftool when it's installed. Without exiftool, they're reported as unreadable.
- White-balance and flash brackets are not grouped as exposure brackets. ISO brackets (auto-ISO AEB) are.
- ⅓-stop brackets need metadata support (EXIF auto-bracket flag, maker notes or matching exposure compensation). Without it, a ⅓-stop series looks the same as metering drift in a burst, so it isn't grouped.
- Depth-of-field series (several apertures at the *same* exposure, with the shutter compensating) aren't exposure brackets. They come out as a burst or singles.
- Pixel exposure stats come from the camera's JPEG preview, which clips highlights earlier than the RAW does, so `over` is conservative. Treat the stats as a triage signal, not a RAW-accurate measurement.
- Near-duplicate detection only links *consecutive* groups from the same body within `scene_window`. Returning to a subject after shooting something else starts a new scene.
- HEIC files have no extractable preview without extra libraries, so `--pixels` skips them.
