# bracketeering

Fast, accurate detection and grouping of camera **exposure brackets** (AEB / HDR sets) and **focus stacks** from a folder of photos.

Example output:

```
$ bracketeer scan /Volumes/CARD
B00001  bracket  5 frames  step 1 EV  [+0 -1 +1 -2 +2]  conf 0.99
         IMG_0412.CR3, IMG_0413.CR3, IMG_0414.CR3, IMG_0415.CR3, IMG_0416.CR3
B00007  bracket  3 frames  step 2 EV  [+0 -2 +2]  conf 0.98
         IMG_0430.CR3, IMG_0431.CR3, IMG_0432.CR3
F00009  focus    24 frames  conf 0.95
         IMG_0440.CR3, ...

1873 shots (3746 files): 214 exposure brackets, 3 focus stacks, 611 singles
```

- **No required dependencies.** Pure Python 3.10+. A built-in EXIF reader handles JPEG, TIFF, DNG, CR2, CR3, NEF, NRW, ARW, SR2, ORF, RW2, PEF, RAF and other TIFF-based RAWs. If [exiftool](https://exiftool.org) is installed, it covers everything else and adds maker-note bracket tags.
- **Efficient.** The reader seeks straight to the IFD entries it needs and reads a few KB per file, even from a 60 MB RAW. Files are read in parallel. In one test, 3,000 RAW-sized files took about 1 s end to end. Grouping runs at 10–20k shots/s.
- **Explainable.** Every group lists its evidence, a confidence score, and warnings such as missing frames or partial brackets.

## Install

```bash
pip install -e .            # provides the `bracketeer` command
pip install -e '.[dev]'     # + pytest, pillow for the test-suite
```

## Usage

```bash
bracketeer scan PATH...                       # human-readable report
bracketeer scan PATH... -v                    # include singles and evidence
bracketeer scan PATH... --format json -o report.json
bracketeer scan PATH... --format csv

bracketeer organize PATH... --dest ./sorted --dry-run
bracketeer organize PATH... --dest ./sorted --mode copy|move|symlink|hardlink
```

Useful options: `--backend auto|native|exiftool`, `--max-gap` (seconds allowed between frames, default 2), `--manual-gap` (for hand-adjusted brackets in M mode, default 6), `--max-size`, `--min-size`, `--no-focus`.

`organize` creates one folder per group, for example `20240501-120000_IMG_0412_HDR5x1EV/`. It keeps RAW+JPEG sidecars together and never overwrites existing files.

### Python API

```python
from bracketeering import analyze, EngineConfig

groups, unreadable = analyze(["/photos/2024-05-01"], EngineConfig(max_gap=3))
for g in groups:
    if g.kind == "exposure":
        print(g.size, g.step_ev, g.ev_offsets, g.confidence, [str(p) for s in g.shots for p in s.paths])
```

`bracketeering.detect(shots)` runs only the grouping engine. Use it if you already have metadata from your own source, such as a catalog database.

## How it works

1. **Read metadata.** The engine reads capture time with sub-seconds and UTC offset, shutter, aperture, ISO, exposure compensation, EXIF `ExposureMode` (2 = auto bracket), focal length, lens, body serial, white balance, flash and subject distance. With `--backend exiftool` it also reads maker-note bracket tags (Canon `BracketMode`/`BracketShotNumber`/`AEBBracketValue`, Sony `ReleaseMode*`, Nikon `ShootingMode`, Fujifilm/Pentax `AutoBracketing`, Olympus `DriveMode`, Panasonic `BracketSettings`, …) and normalises them to a bracket kind and shot position.
2. **Pair sidecars.** `IMG_1.CR3` and `IMG_1.JPG` in the same folder become one shot when their timestamps agree. The RAW is the authoritative source.
3. **Split into runs** per camera body. A run is a chain of consecutive frames with the same lens and focal length and comparable exposure data. Each frame must start within `max_gap` of the *end* of the previous exposure, so long-exposure brackets work. Large frame-counter jumps also end a run.
4. **Carve out focus stacks.** These are constant-exposure blocks flagged by maker notes, or with monotonic focus distance.
5. **Segment each run by dynamic programming.** Each candidate group gets an additive log-odds-style score, and singles score zero. The engine finds the partition with the highest total score:
   - **Exposure structure.** Exposures must be distinct and evenly spaced. The step is fitted on the 1/3- and 1/2-stop grid and tolerates nominal shutter-speed rounding. An interior gap of exactly *k* steps means *k*−1 frames were deleted. The group is still accepted, with a warning.
   - **Firing order.** Groups that match the orders cameras actually use score higher: `0,-,+`, `-,0,+`, `+,0,-`, `0,+,-`, `0,-,--,+,++`, … The engine also matches partial brackets with a missing frame at either end.
   - **Metadata.** Evidence includes `ExposureMode = auto bracket`, maker-note bracket mode, shot positions `1..n` (a reset to 1 inside a group is a hard veto), and exposure compensation matching the measured deltas. A frame flagged as bracketed is penalised if it's left ungrouped. If a body is seen writing `ExposureMode=2` elsewhere in the shoot, its frames with other modes count *against* bracketing.
   - **Context.** Groups that change two exposure variables at once, change white balance or fire a flash are penalised. A large time gap favours a boundary there.
   - **Burst veto.** Auto-exposure bursts wander in 1/3-stop increments and can look like small brackets. A group without metadata support is dissolved if ungrouped frames, or groups with a different signature, sit next to it at the same cadence and exposure level.
6. **Second pass with a prior.** Photographers rarely change bracket settings mid-shoot. The camera's dominant signature (size, firing order, step) becomes a prior, and the DP runs again. This settles the hard cases, such as back-to-back brackets fired in continuous drive with no pause. Group evaluations are cached, so the second pass costs almost nothing.

## Accuracy

`scripts/benchmark.py` simulates realistic shoots with known ground truth. The simulator includes AEB in several firing orders and both drive modes, back-to-back brackets, deleted frames, hand-adjusted manual brackets, auto-exposure bursts with metering jitter, focus stacks, single shots, and quick handheld pairs with an exposure tweak. Shutter speeds are quantised to real nominal values. Results over 20 seeds (~4,000 groups per profile):

| profile | what's hard about it | exact-group precision | exact-group recall | pairwise F1 |
|---|---|---|---|---|
| canon-aeb | baseline: full metadata | 0.999 | 0.999 | 1.000 |
| sony-no-subsec | timestamps only to the second | 0.999 | 0.999 | 0.999 |
| nikon-mixed-settings | size 2–9 and step ⅓–3 EV change every bracket | 0.990 | 0.978 | 0.999 |
| manual-brackets | user-driven brackets, 1–4 s apart, no bracket flags | 0.988 | 0.970 | 0.995 |
| no-bracket-metadata | no ExposureMode, no exposure compensation, AE bursts | 0.979 | 0.968 | 0.997 |
| worst-case | all of the above, no sub-seconds, 50% back-to-back | 0.977 | 0.959 | 0.993 |

"Exact" means the predicted group contains exactly the right frames. Most remaining errors are inherently ambiguous without metadata, for example a 1/3-EV bracket next to an auto-exposure burst. Use `--backend exiftool` when maker notes are available.

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
  engine.py             run splitting, scoring, DP segmentation, priors
  report.py             text / JSON / CSV output
  organize.py           copy / move / link groups into folders
  synth.py              shoot simulator + evaluation metrics
  cli.py                `bracketeer` command
```

## Notes and limitations

- Times in reports are UTC when the file records an offset (`OffsetTimeOriginal`), and camera-local otherwise.
- The native reader doesn't decode maker notes. Bracket detection still works from EXIF and exposure analysis, but maker-note shot positions need `--backend exiftool`.
- HEIC/HEIF and less common RAW containers fall back to exiftool when it's installed. Without exiftool, they're reported as unreadable.
- White-balance, flash and ISO-only brackets are not grouped as exposure brackets.
