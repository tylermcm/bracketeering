# bracketeering — handoff for the culling pipeline

**What it is:** the grouping step of the culling pipeline. Before any frame is scored, it decides *what
competes with what*. Every photo comes out labelled as part of an exposure bracket, a burst, a focus stack,
a timelapse, or as a single, with a representative frame per group and a near-duplicate "scene" id.

**Status:** 0.3.4, branch `ccr-fd1f1f35-hi49ff`. 176 tests pass. Validated by hand on four real Nikon shoots
(about 5,000 shots). See "How good is it" for what is and isn't verified.

---

## 1. What it does

| Group kind | Meaning | How it's detected |
|---|---|---|
| `exposure` | One camera auto-bracket (or hand-adjusted) set | Camera maker notes when present (Nikon: `ShootingMode` + `ExposureBracketValue` slot per frame); otherwise exposure analysis (evenly spaced steps on the ⅓/½-stop grid, known firing orders, EXIF ExposureMode). |
| `burst` | One press-and-hold of the shutter in continuous drive | A sequence model: each gap between frames is either "the drive fired it" (steady pace, buffer/AF slowdowns allowed) or "a separate press" (human pause). Re-presses split bursts; hand taps, timers and pauses don't form them. |
| `focus` | A focus stack (e.g. Nikon focus-shift) | Maker notes (`FocusMode = Fshift`) or monotonic focus distance at constant exposure. |
| `timelapse` | Interval-timer sequence | ≥ 5 frames at a fixed interval ≥ 1 s, timer precision; blocked if the camera recorded its interval timer as off. |
| `single` | Everything else | — |

Every group also gets:
- **`representative`:** the frame that stands for the group. For a bracket it's the metered (0 EV) frame; for bursts, stacks and timelapses, the middle frame.
- **`confidence`** (0–1). For bursts, ≥ 0.9 was exactly right 95% of the time in testing.
- **`evidence`** and **`warnings`** in plain language. Examples: `partial bracket: 3 of 5 slots`, `frames don't register (subject or camera moving): an HDR merge may ghost`.
- **`scene`** (with `--pixels`): groups of the same subject shot close together share a scene id. These are near-duplicates; compare them against each other, never against unrelated photos.

With `--pixels`, every frame also gets exposure stats from its embedded JPEG preview: % highlights clipped, % shadows crushed, mean level, and an `ok` / `over` / `under` flag. Each bracket also gets **`best_exposed`**: the brightest frame whose highlights still hold.

**Also handled:**
- **One shot, many files:** RAW+JPEG sidecars, copies in other folders (`_winners/`) and edited exports (`Bears/`, `Re-edits/`) all merge into one shot. The original RAW is primary.
- **Unreadable files:** zero-filled or corrupt files are reported as errors.
- **Ordering:** frames are ordered by the body's shutter count, so file-number restarts don't break groups.
- **Times:** shown in camera-local time.

## 2. How to call it

```bash
pip install -e '.[pixels]'          # in a venv; see README if pip complains about build_editable
bracketeer scan SHOOT_DIR --backend exiftool --pixels --format json -o groups.json
```

**Recommended flags for the pipeline:**

| Flag | Why |
|---|---|
| `--backend exiftool` | Most accurate. Needs `sudo apt install libimage-exiftool-perl`. The native reader is faster and reads Nikon maker notes itself, but it can't see focus distance (encrypted in Nikon lens data) or `BracketProgram`. |
| `--pixels` | Exposure stats, `best_exposed`, scenes, recompose detection. About 5 ms per file; reads embedded previews, never decodes RAW. |
| `--exclude NAME` | Skip export or picks folders if you don't want them in the index at all. Copies are merged either way. |

**From Python** (if the pipeline imports it):

```python
from bracketeering import analyze
from bracketeering.report import to_dict

groups, unreadable = analyze([shoot_dir], backend="exiftool", pixels=True)
data = to_dict(groups, unreadable)        # identical to the JSON output
```

**Output contract:** `SCHEMA.md` (schema `bracketeering/2`). The two parts the pipeline needs:
- `files{path}` gives every input file exactly once, with `group`, `kind`, `role` (`representative` / `member` / `single`), `position`, `primary` (false for sidecars, copies and exports), `scene`, `best_exposed` and `error`.
- `groups[]` gives each group's `kind`, `confidence`, `representative`, `best_exposed`, `ev_offsets` (the camera's own slots when known), `warnings`, and per-frame `exposure` stats.

## 3. Recommended culling order

Grouping comes **first**, before any image scoring, because it decides what competes with what.

```
1. Ingest + group     bracketeer --backend exiftool --pixels → groups.json
2. Route by kind      apply the user's targeting choices per group kind
3. Hard gates         only clear failures: misfire/black, severe blur, unusable clipping
4. Technical scores   sharpness, motion, eyes/faces, noise. Scored, not dropped; compared within the group
5. Aesthetic rank     TOPIQ only within a comparison set (burst, scene), only on survivors
6. Decide + explain   keep/reject per group with reasons; low-confidence groups → review
```

**Step 2, rules per kind:**

| Kind | User wants it culled | User doesn't |
|---|---|---|
| `exposure` (bracket) | Judge the bracket **once**, on `best_exposed` (or `representative`), against the other groups in its scene. Keep or reject the whole set (HDR merge), or keep only `best_exposed` in "pick one" mode. | Keep the whole set untouched. |
| `burst` | A set of alternatives: score frames against each other, keep the best 1–2. | Keep all. |
| `focus`, `timelapse` | Usually pass through: every frame belongs to the sequence. | Keep all. |
| `single` | Compete within its `scene` (near-duplicates); keep the top k. | — |

**Rules that matter:**
1. **Never run the over/under-exposure filter on bracket frames.** They're dark and bright on purpose; use `best_exposed` instead.
2. **Score, don't drop, in steps 3–4, then decide once.** Two passes that each drop frames can lose a whole scene to one strict stage. If every frame in a group fails, keep the least-bad and flag it.
3. **Make sharpness relative.** Raw sharpness scores mostly measure scene texture, so rank frames against their siblings in the same burst or scene. Shutter speed vs focal length (metadata) predicts handheld blur for free.
4. **Only let TOPIQ compare real alternatives.** It is strongly biased by brightness: never rank the frames of one bracket with it. Run it within bursts and scenes, and only on survivors (also faster).
5. **Use confidence.** Groups below ~0.7, or carrying warnings, go to a review bucket rather than being culled automatically.
6. **Write results back.** XMP sidecars (ratings, labels) so Lightroom shows the outcome. Keep the reasons ("rejected: blur vs 3 sharper siblings") for tuning.
7. **Measure the whole pipeline against the user's past culls.** Lightroom ratings on old shoots are free ground truth.

## 4. How good is it

**Real shoots (hand-checked against Nikon maker notes and previews), all Nikon:**

| Shoot | Body | Result (last full check: 0.3.3) |
|---|---|---|
| China 8-26 | Z 7_2 | 54/58 expected groupings; the 4 misses were a 0.3.3 regression, fixed in 0.3.4 (not yet re-verified) |
| Japan 4-24 | D810 | Clean: all 108 bracket cycles, bursts, scenes |
| WhiteSands '26 | Z 7_2 | All bracket cycles; focus-shift stacks; bursts across file-number switches |
| Alaska Aug '21 | D800/D5/D850 | Copies and exports merged; wildlife cycles whole; aurora tripod cycles recovered |

Across all four: **no false brackets, no bursts mixing bracket frames.**

**Simulated shoots** (`python scripts/benchmark.py`): brackets 0.98–1.00 precision. Bursts 0.95 precision / 0.91 recall with sub-second times, 0.98–0.99 with camera drive metadata.

**Known limits:**
- **Only tested on Nikon.** Canon, Sony and Fuji maker-note handling is written from exiftool's documentation and unit-tested on hand-built records, but never run on real files. The [ExifTool sample repository](https://exiftool.org/sample_images.html) can verify the reader per brand.
- **Native reader blind spots:** without exiftool, `_DSC1073–1075`-type cases (slow separate presses distinguished by refocus) need focus distance the native reader can't decode.
- **Whole-second timestamps** (no sub-seconds): back-to-back re-presses under 1 s apart are invisible without camera sequence numbers.
- **Scene similarity is the weakest part.** Thresholds are set from synthetic images plus a few real cases. Hazy, low-contrast horizons can over-merge, and borderline pairs (`DSC_1267/1268`) can split.
- **⅓-stop brackets without camera metadata** are never grouped, because they're indistinguishable from metering drift.

## 5. What we did (history)

| Version | Change |
|---|---|
| 0.1 | Bracket engine: dependency-free EXIF reader (JPEG/TIFF RAWs/RAF/CR3), exiftool backend, evidence-scored segmentation with a per-camera prior, deleted-frame tolerance, burst veto. Synthetic benchmark. |
| 0.2 | Pipeline contract (per-file index, representatives, schema), pixel analysis (exposure stats, best-exposed, scenes), first burst grouping. Exact ⅓-stop snapping of nominal values (fixed ⅓-stop aperture brackets). |
| 0.2+ | Bursts rebuilt as a probabilistic drive-vs-press sequence model (pace changes, buffer slowdowns, dead time for slow shutters, coarse timestamps); timelapse kind; action-shoot simulator. |
| 0.3.0 | First real review (China): Nikon maker notes natively; brackets rebuilt from camera slots; exiftool false-bracket bug; copies merged; local times. |
| 0.3.1 | Partial-bracket warnings from the camera program; drive mode counts only when informative; registration-style image similarity; scene anchor guard. |
| 0.3.2 | Three more shoots: long-exposure cycles, edited exports, zero-filled files, focus-shift, shutter-count ordering, featureless frames. |
| 0.3.3 | Slot order learned per session from adjacent frames (culled sets); slow-press loophole; haze; JPEGs without EXIF. |
| 0.3.4 | Bracket cycles across a pause need positive registration (fixed the 0.3.3 regression). |

Full findings: `docs/reviews/`.

## 6. Repo map and working on it

```
src/bracketeering/
  metadata/native.py     EXIF + Nikon maker-note reader     metadata/exiftool.py  exiftool backend + tag mapping
  scan.py                discovery, sidecars, copy/export merging
  cycles.py              brackets from camera slots          engine.py   bracket inference, orchestration
  bursts.py              burst model + timelapses            pixels.py   previews: exposure, similarity, scenes
  report.py              JSON/CSV/text (SCHEMA.md)           cli.py      `bracketeer`
tests/                   176 tests; test_china_review.py / test_three_shoots_review.py replay real findings
scripts/                 benchmark.py, check_makernotes.py (native vs exiftool), similarity_pairs.py
```

**Rules for changing it:**
- Every real-shoot finding becomes a regression test before it is fixed.
- After any change, run `python -m pytest` and `python scripts/benchmark.py`.
- Re-run the hand-checked shoots: the reviewer session on the photo server holds about 135 automated checks.

## 7. Recommended next steps

1. **Re-verify 0.3.4** on the four shoots (one run of the existing checks).
2. **Export those checks into the repo:** exiftool JSON metadata with GPS stripped, no images, plus the expected groups, under `tests/fixtures/`. Every change is then tested against real shoots without the server.
3. **Wire it into the pipeline** using the order above. Measure end-to-end agreement with past Lightroom culls.
4. **Test other brands:** run `scripts/check_makernotes.py` on the ExifTool sample repository; ideally get one real Canon or Sony shoot.
5. **Stop tuning edge cases.** Remaining issues are scene-threshold calls and data limits; spend effort on the pipeline and new data instead.
