# bracketeering

Fast, accurate detection and grouping of camera **exposure brackets** (AEB / HDR sets), **bursts** (continuous-drive sequences), **timelapses** (interval-timer sequences) and **focus stacks** from a folder of photos. Optionally, it also finds **near-duplicate scenes** and per-frame **exposure problems**.

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
- **Pipeline handoff:** see [docs/HANDOFF.md](docs/HANDOFF.md) for what it does, how to call it and where it belongs in a culling pipeline.
- **Explainable.** Every group lists its evidence, a confidence score, and warnings such as missing frames or partial brackets.

## Install

```bash
pip install -e .            # provides the `bracketeer` command
pip install -e '.[pixels]'  # + Pillow, for --pixels
pip install -e '.[dev]'     # + pytest, pillow for the test-suite
```

If pip says the build backend is **missing the `build_editable` hook**, the build ended up using setuptools older than 64. Common causes:
- The system pip on Ubuntu 22.04 / Debian (pip 22.0.x, often with "Defaulting to user installation"). A bug there hides the freshly downloaded setuptools from the build.
- `--no-build-isolation`, or conda.

The reliable fix is a virtual environment with an up-to-date pip:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Debian/Ubuntu: sudo apt install python3-venv
python -m pip install --upgrade pip
python -m pip install -e '.[pixels]'
```

You can also skip installing entirely, since there are no required dependencies: `PYTHONPATH=src python -m bracketeering scan PATH`.

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
- `--exclude PATTERN`: skip subfolders by name, e.g. `--exclude _winners` (repeatable)
- `--scene-window`, `--scene-similarity`: near-duplicate tuning
- `--max-size`, `--min-size`, `--no-focus`, `--no-bursts`, `--no-timelapse`

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

1. **Read metadata.** The engine reads capture time with sub-seconds and UTC offset, shutter, aperture, ISO, exposure compensation, EXIF `ExposureMode` (2 = auto bracket), focal length, lens, body serial, white balance, flash and subject distance. With `--backend exiftool` it also reads maker-note bracket tags (Canon `BracketMode`/`BracketShotNumber`/`AEBBracketValue`, Sony `ReleaseMode*`, Nikon `ShootingMode`, Fujifilm/Pentax `AutoBracketing`, Olympus `DriveMode`, Panasonic `BracketSettings`, …) and normalises them to a bracket kind and shot position. The native reader also decodes **Nikon maker notes** directly: `ShootingMode` (bracketing on/off, continuous drive), `ExposureBracketValue` (this frame's slot in the bracket program) and `ShutterCount`. Configuration tags that Nikon writes on every frame (`BracketSet`, `AutoBracketModeM`) are ignored, because they don't say whether *this* frame was bracketed.
2. **Pair sidecars and merge copies.** `IMG_1.CR3` and `IMG_1.JPG` in the same folder become one shot when their timestamps agree. The RAW is the authoritative source. Copies and edited exports of the same capture in other folders or formats (same body and shutter count, *or* same capture time, frame number and exposure), such as a `_picks/` or `Edits/` folder, are merged into that shot too. Frames with identical capture times never form a burst. Otherwise a copy and its original look like a 0-second burst.
3. **Camera bracket cycles** (`cycles.py`). When the camera recorded each frame's bracket slot, brackets are rebuilt from those slots rather than inferred from measured exposure. With Auto ISO the camera re-meters on every press, so a 0 / −4/3 / −2/3 cycle can *measure* +0.85 / −0.64 / 0.
   - **New cycle:** slots follow the program order. Metered-first (0, −, +) or under-first (−, 0, +) is learned per bracketing session from adjacent frames (a 0 → − step only happens metered-first), which culling can't distort; the default is metered-first. A repeated or earlier slot starts a new cycle: a re-press restart, or the tail of a cycle whose first frames were deleted.
   - **Recompose split:** a cycle splits where the shooter recomposed. Within 3 s it stays together unless the previews clearly differ across a non-drive gap; at drive speed a mismatch means a moving subject (wildlife), so the cycle stays whole and gets flagged "frames don't register". Across a longer pause it stays together only on positive evidence: with `--pixels`, the previews register strongly, or decently at almost the same position (a tripod). Without `--pixels`, the pause must fit the cycle's own rhythm (30 s aurora exposures in single-frame drive have long gaps everywhere). A featureless frame, such as a black misfire, is never evidence for joining.
   - **Offsets:** they're the camera's own (`[0, -1.33, -0.67]`), the representative is the metered frame, and partial cycles are flagged ("partial bracket: 3 of 5 slots"). The program size comes from the camera's `BracketProgram` setting when exiftool provides it. Otherwise it's inferred from the slots: programs are evenly spaced, so a cycle reaching ±k steps has 2k+1 frames, and neighbouring cycles tell symmetric programs from one-sided ones.
   - **Bracketing off:** frames the camera recorded with bracketing *off* can never be grouped as a bracket.
4. **Split into runs** per camera body, for frames without bracket slots. Frames are ordered and chained by the body's **shutter count** when known, so a file-number restart or series switch (`_DSC8xxx` → `_DSC0xxx`, focus-shift folders) doesn't break a burst. A run is a chain of consecutive frames with the same lens and focal length and comparable exposure data. Each frame must start within `max_gap` of the *end* of the previous exposure, so long-exposure brackets work. Large frame-counter jumps also end a run.
5. **Carve out focus stacks.** These are constant-exposure blocks flagged by maker notes (including Nikon focus-shift shooting, `FocusMode = Fshift`), or with monotonic focus distance.
6. **Segment each run by dynamic programming.** Each candidate group gets an additive log-odds-style score, and singles score zero. The engine finds the partition with the highest total score:
   - **Exposure structure.** Brightness is measured from all three settings: log2(shutter) + log2(ISO/100) − 2·log2(f-number). That makes shutter-driven brackets (A/Av and M modes), aperture-driven brackets (S/Tv mode) and auto-ISO brackets all register. Cameras write rounded labels (f/13 is really f/12.7, 1/125 s is really 1/128 s), so standard ⅓-stop labels are snapped back to their exact values first. Exposures must be distinct and evenly spaced. The step is fitted on the 1/3- and 1/2-stop grid and tolerates nominal shutter-speed rounding. An interior gap of exactly *k* steps means *k*−1 frames were deleted. The group is still accepted, with a warning.
   - **Firing order.** Groups that match the orders cameras actually use score higher: `0,-,+`, `-,0,+`, `+,0,-`, `0,+,-`, `0,-,--,+,++`, … The engine also matches partial brackets with a missing frame at either end.
   - **Metadata.** Evidence includes `ExposureMode = auto bracket`, maker-note bracket mode, shot positions `1..n` (a reset to 1 inside a group is a hard veto), and exposure compensation matching the measured deltas. A frame flagged as bracketed is penalised if it's left ungrouped. If a body is seen writing `ExposureMode=2` elsewhere in the shoot, its frames with other modes count *against* bracketing.
   - **Context.** Groups that change two exposure variables at once, change white balance or fire a flash are penalised. A large time gap favours a boundary there.
   - **Burst veto.** Auto-exposure bursts wander in 1/3-stop increments and can look like small brackets. A group without metadata support is dissolved if ungrouped frames, or groups with a different signature, sit next to it at the same cadence and exposure level.
7. **Second pass with a prior.** Photographers rarely change bracket settings mid-shoot. The camera's dominant signature (size, firing order, step) becomes a prior, and the DP runs again. This settles the hard cases, such as back-to-back brackets fired in continuous drive with no pause. Group evaluations are cached, so the second pass costs almost nothing.
8. **Timelapses.** Leftover frames fired at a fixed interval of ≥ 1 s (within 0.4 % + 20 ms; ≥ 2 s for whole-second timestamps), at least 5 in a row, become a timelapse, unless the camera recorded its interval timer as off or the frames are focus-shift frames. Continuous drive is never that slow, and timers are far more precise than any drive or human. A slowing burst can't be mistaken for one, because a timelapse doesn't start right on the heels of a drive-speed frame.
9. **Bursts** (`bursts.py`). A burst is **one press of the shutter in continuous drive**. It's detected with a probabilistic sequence model that gets the same care as brackets:
   - Leftover frames are chained: same body and lens, consecutive frame numbers, ≤ 2.5 s apart. Every interval in a chain is explained as either **drive** or **break**.
     - **Drive:** the camera fired the next frame. The interval sits near the burst's local cadence, with a few % jitter and occasional autofocus/buffer slowdowns, and the cadence fits a drive rate prior (~1–60 fps).
     - **Break:** a separate press. The pause follows a log-normal (~0.9 s typical), and pauses under ~0.2 s are rare because humans can't re-press that fast.
   - Both sides are densities over the same intervals, so dynamic programming finds the most likely labelling: which frames are bursts, and exactly where each burst starts and ends.
   - A burst may **change pace** without splitting (a 20 fps burst that drops to 5 fps when the buffer fills, or zooming). An interval at the *new* pace is a drive interval; a pause that fits neither pace is a re-press.
   - With slow shutters, the drive fires "exposure + fixed overhead" apart, so regularity is judged on the dead time between frames.
   - **Metadata** (Nikon natively, other makes with `--backend exiftool`): continuous vs single drive, and the burst **sequence number**, where a reset to 1 forces a new burst. That separates back-to-back presses even without sub-second times. Zoom changes, exposure jumps and frame-counter gaps count against "same press". Drive mode counts only when the body was used in both modes; many people leave continuous drive on all day, and then it says nothing. A pace slower than ~1 fps is never a burst, and a human re-press is timed from the *end* of the exposure.
   - **Pixels** (with `--pixels`): a subject change between consecutive previews splits a burst, even at full drive speed.
   - **Coarse timestamps:** without sub-seconds, the model uses the probability of landing in the same or next second, a decaying probability for several separate presses within one second, and an edge prior (a separate press right next to a running burst is unlikely).
   - **Confidence** is the log-odds of "one press" vs "separate presses". On simulated shoots, bursts at ≥ 0.9 were exactly right 95% of the time.
10. **Representatives.** Each group gets a representative frame. For brackets it's the middle exposure (≈ 0 EV); for bursts, timelapses and stacks it's the middle frame.
11. **Pixels (optional).** Embedded previews are found in each container (JPEG `APP1`, TIFF/NEF/ARW/DNG/CR2 preview IFDs and SubIFDs, RW2 `JpgFromRaw`, RAF, CR3 `PRVW`). Lossless-JPEG raw data is skipped. Each preview is decoded at reduced size, and the engine computes:
   - **Exposure stats** per frame: highlight clipping (any channel ≥ 250), crushed shadows, mean level, and an `ok`/`over`/`under` flag.
   - **`best_exposed`** per bracket: the brightest frame whose highlights hold (≤ 0.5% clipped), i.e. expose to the right. Of the usable frames it has the cleanest shadows.
   - **Preview similarity** answers "do these two frames register?", like phase correlation. A 64×44 thumbnail is band-passed, keeping structure at the scale of objects and dropping overall layout, brightness and pixel-level texture. The best normalised cross-correlation is then found over shifts up to ~12 % of the frame (coarse search, then refined). Same composition with handheld drift or a different exposure scores ~0.95; a 20–60 % recompose or another place scores ~0.15–0.25.
     - Plain low-resolution correlation compares layout (sky above, ground below), which a recompose keeps. Version 0.3.0 used it and scored a 36 % recompose at +0.66.
     - dHash flips bits on fine texture (foliage, ivy, fog).
   - **Featureless previews** (a moon in an empty sky, black misfires) have nothing to register, so their similarity is *unknown*, not "different". Scenes then fall back to timing and camera settings.
   - **Near-duplicate scenes:** each group's representative is compared with the last 4 groups from the same body within `scene_window`, not just the previous one, so one odd frame doesn't cut a scene in two. Groups share a scene if they register at or above `--scene-similarity` (0.5). They also share one if they register weakly (≥ 0.15) while their overall tonal layout matches strongly (≥ 0.85) within 20 s, which covers handheld drift on fine texture such as ivy. A scene never takes in a group that clearly differs from its *first* group in both structure and layout, so a slow walk around a subject can't chain into one long scene.
   - **Bursts and bracket cycles** use the similarity cautiously. In a burst, a mismatch only lowers confidence: one hold of the shutter stays one burst even if the subject moves through the frame. In a bracket cycle, it splits only across a real pause (see step 3).

## Accuracy

`scripts/benchmark.py` runs two simulators with known ground truth. These numbers come from simulators written alongside the engine, so they reflect its assumptions about camera behaviour. Real shoots are the true test.

### Bracket shoots

These include AEB in several firing orders and both drive modes, back-to-back brackets, deleted frames, hand-adjusted manual brackets, continuous-drive bursts with auto-exposure, focus stacks, singles, and quick handheld pairs. Shutter speeds are quantised to real nominal values. Results over 20 seeds (~4,000 brackets and ~900 bursts per profile):

| profile | what's hard about it | bracket precision | bracket recall | burst recall | burst→bracket | bracket→burst |
|---|---|---|---|---|---|---|
| canon-aeb | baseline: full metadata | 0.999 | 0.999 | 0.962 | 0.00% | 0.00% |
| sony-no-subsec | timestamps only to the second | 1.000 | 1.000 | 0.914 | 0.00% | 0.00% |
| nikon-mixed-settings | size 2–9 and step ⅓–3 EV change every bracket | 0.993 | 0.985 | 0.971 | 0.00% | 0.02% |
| manual-brackets | user-driven brackets, 1–4 s apart, no bracket flags | 0.998 | 0.982 | 0.962 | 0.04% | 0.13% |
| no-bracket-metadata | no ExposureMode, no exposure compensation | 0.983 | 0.973 | 0.969 | 0.00% | 0.49% |
| worst-case | all of the above, no sub-seconds, 50% back-to-back | 0.975 | 0.951 | 0.910 | 0.26% | 0.94% |

### Action shoots (burst-focused)

These include:
- continuous drive at 1.5–60 fps with 3% jitter;
- autofocus hiccups;
- buffer slowdowns (2–4× slower after 15–80 frames);
- bursts of 2–120 frames;
- re-presses after 0.25–1.5 s pauses (which must be separate bursts);
- quick human taps and interval-timer sequences (which must not be bursts);
- zoom changes and pans to a new subject mid-burst;
- AEB sets mixed in.

Results over 20 seeds (~3,900 bursts per profile):

| profile | burst precision | burst recall | merged | split | taps/timer → burst | bracket recall | timelapse recall |
|---|---|---|---|---|---|---|---|
| action (sub-second times, no maker notes) | 0.946 | 0.907 | 3.1% | 3.3% | 0.96% | 1.000 | 1.000 |
| action-drive-meta (`--backend exiftool`) | 0.978 | 0.988 | 0.0% | 1.2% | 0.00% | 1.000 | 1.000 |
| action-pixels (`--pixels`, with pans) | 0.946 | 0.901 | 3.3% | 3.4% | 0.93% | 1.000 | 1.000 |
| action-no-subsec-meta | 0.996 | 0.980 | 0.0% | 2.0% | 0.00% | 1.000 | 0.778 |
| action-no-subsec (no maker notes) | 0.648 | 0.362 | 32.7% | 17.2% | 0.00% | 1.000 | 0.778 |

Precision and recall are for exact groups: a predicted burst must contain exactly the right frames. "Merged" counts predicted bursts that join two real presses; "split" counts real bursts broken apart.

**What limits burst accuracy:**
- **With sub-second timestamps:** most remaining errors are genuinely ambiguous. Examples: a 2-frame burst at 3 fps vs two quick taps, a re-press pause barely longer than an autofocus hiccup, or a slow burst with a hiccup on its first frame.
- **Cameras that record only whole seconds and no maker notes:** this is the hard limit. A re-press less than a second after the previous burst is invisible in the timestamps; it accounts for most of the errors in that row. The camera's burst **sequence numbers** solve it completely (0.996 / 0.980 with them), so **use `--backend exiftool` for such cameras**. In the bracket simulator's no-subsec profiles, a burst sometimes absorbs a neighbouring single for the same reason.
- **Without metadata, ⅓-stop exposure series are never called brackets**, because they're indistinguishable from auto-exposure metering drift inside a burst. Brackets with a ⅔-stop step or larger, or with any bracket metadata, are unaffected.

```bash
python scripts/benchmark.py --seeds 20
python -m pytest
```

## Layout

```
src/bracketeering/
  metadata/native.py    dependency-free EXIF reader (JPEG/TIFF/RAW/RAF/CR3)
  metadata/exiftool.py  batched exiftool backend + maker-note normalisation
  scan.py               file discovery, RAW+JPEG pairing, merging copies across folders
  cycles.py             brackets rebuilt from the camera's own bracket slots (maker notes)
  engine.py             bracket detection: run splitting, scoring, DP segmentation, priors
  bursts.py             burst model (drive vs break, pace changes) + timelapse detection
  preview.py            embedded JPEG preview extraction
  pixels.py             exposure stats, best-exposed pick, near-duplicate scenes
  report.py             text / JSON (SCHEMA.md) / CSV output
  organize.py           copy / move / link groups into folders
  synth.py              shoot simulator + evaluation metrics
  cli.py                `bracketeer` command
```

## Notes and limitations

- Times in reports are camera-local, with the UTC offset when the file records one (`OffsetTimeOriginal`).
- **Most accurate: `--backend exiftool`.** The native reader can't see everything. Nikon stores focus distance in encrypted lens data, and the native reader can't read it, so the "focus jumped between slow presses" evidence and the camera's `BracketProgram` setting need exiftool.
- The native reader decodes Nikon maker notes only. For other makes, maker-note bracket and drive tags need `--backend exiftool`; without them, detection works from EXIF and exposure analysis.
- The similarity thresholds (0.5 same scene, 0.35 different) are set from synthetic textured scenes and not yet calibrated on real photos. `scripts/similarity_pairs.py` prints the similarity of frame pairs, for calibrating on real shoots.
- HEIC/HEIF and less common RAW containers fall back to exiftool when it's installed. Without exiftool, they're reported as unreadable.
- White-balance and flash brackets are not grouped as exposure brackets. ISO brackets (auto-ISO AEB) are.
- ⅓-stop brackets need metadata support (EXIF auto-bracket flag, maker notes or matching exposure compensation). Without it, a ⅓-stop series looks the same as metering drift in a burst, so it is never grouped as a bracket.
- A timelapse needs at least 5 frames. Shorter interval-timer runs fall back to singles.
- Depth-of-field series (several apertures at the *same* exposure, with the shutter compensating) aren't exposure brackets. They come out as a burst or singles.
- Pixel exposure stats come from the camera's JPEG preview, which clips highlights earlier than the RAW does, so `over` is conservative. Treat the stats as a triage signal, not a RAW-accurate measurement.
- Near-duplicate detection only links groups from the same body within `scene_window`, looking back 4 groups. Returning to a subject after shooting several other things starts a new scene.
- HEIC files have no extractable preview without extra libraries, so `--pixels` skips them.
