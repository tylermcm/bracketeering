# Culling engine spec: rework of CLI-Culler

**Goal:** turn `tylermcm/cli-culler` (package `aiculler`) into the headless culling engine for the
image-triage app. Given a shoot, it returns **keep / reject / review for every file, with reasons**.
It uses `bracketeering` to decide *what competes with what*, and the app's five models to decide *which
frame wins*.

**Principles:**
1. **The app only calls it.** No culling logic in the app: in = folder + settings, out = `decisions.json`
   (+ optional XMP). Keep CLI-Culler's no-GUI, callback-driven design.
2. **Compare only real alternatives.** Scores are compared within a comparison set (a burst, a scene),
   never across the whole shoot, and never across the frames of one exposure bracket.
3. **Score, don't drop; decide once.** Gates reject only clear failures. Everything else is a score;
   one decision step at the end applies the user's settings.
4. **Cheap → expensive.** Each stage runs on fewer frames than the last. Q-ReAlign (0.8B) only on
   survivors of sets that still need ranking.
5. **Measured against the user's own culls.** An evaluation harness reports agreement with past
   Lightroom ratings; every change is judged by it.

---

## 1. Interface

### Input

```python
from aiculler import cull, CullSettings
result = cull(folder, settings=CullSettings(...), models=ModelRegistry.from_app_manifest(...),
              cache_db="culler.sqlite", on_progress=callback)
```

CLI equivalent: `ai-culler cull FOLDER --settings settings.json --out decisions.json [--xmp]`.

`CullSettings` (JSON-serialisable; defaults in bold):

| Setting | Values | Meaning |
|---|---|---|
| `brackets` | **`keep_set`** / `pick_one` / `ignore` | Judge a bracket once and keep/reject the whole set; or keep only its best-exposed frame; or leave brackets out of culling (all kept) |
| `bursts_keep` | int, **2** | Best N frames kept per burst (`0` = leave bursts out of culling, all kept) |
| `scene_keep` | int, **1** | Best N groups kept per near-duplicate scene |
| `focus_stacks`, `timelapses` | **`keep_all`** / `cull` | Sequences where every frame matters are passed through by default |
| `review_below` | float, **0.7** | Groups with grouping confidence below this, or with warnings, → `review` instead of auto-decisions |
| `require_eyes_open` | **true** | Blinks count against a frame; in group shots, all detected faces |
| `keep_people_coverage` | **true** | Each recognised person keeps at least one acceptable frame in the shoot |
| `gates` | thresholds | Only for clear failures (see §3) |

### Output: `decisions.json` (schema `aiculler-decisions/1`)

```jsonc
{
  "schema": "aiculler-decisions/1",
  "engine_version": "…", "bracketeering_version": "…", "models": {"maniqa": "<rev>", "...": "..."},
  "settings": { "...": "..." },
  "summary": {"files": 2304, "keep": 410, "reject": 1700, "review": 194},
  "files": {
    "/abs/path/_DSC1703.NEF": {
      "decision": "keep",                  // keep | reject | review
      "group": "B00581", "kind": "exposure", "role": "representative", "scene": 461,
      "rank_in_set": 1, "set_size": 3,     // rank within the comparison set it competed in
      "score": 0.83,                       // final combined score (comparable only within the set)
      "scores": {"maniqa": 0.71, "q_realign": 3.9, "face_sharpness": 0.8, "eyes_open": 1.0, "preference": 0.4},
      "reasons": ["bracket kept as a set (keep_set)", "beat 2 other groups in scene 461 on appeal"],
      "primary": true                      // false for sidecars/copies/exports: they follow their shot
    }
  }
}
```

Every input file appears exactly once (bracketeering already guarantees one entry per file). Non-primary files (JPEG sidecars, copies, exports) always get their shot's decision.

**XMP (optional, `--xmp`):** write `xmp:Rating` and `xmp:Label` (keep = 3★, review = yellow label, reject = 1★ + red). Never overwrite existing user ratings unless `--overwrite-ratings` is given.

---

## 2. Models (use the app's, pinned)

| Key | Model | Revision | Used for |
|---|---|---|---|
| `tinyclip` | TinyCLIP ViT-8M-16-Text-3M-YFCC15M (ONNX) | `9463a9c508a344c837ffefe9d724f3827bf2dc79` | Image embeddings: scene refinement, preference learning, text search |
| `q_realign` | Q-ReAlign Mini 0.8B | `fe1f45a7574c9e9d908875af9f7e90cb946aa19f` | Appeal ranking (last stage) |
| `maniqa` | MANIQA, KonIQ-10k checkpoint | `18bbbb8a87d06d7066de3fb0412139520fd6ff91` | Technical quality |
| `face_landmarker` | MediaPipe Face Landmarker float16 `.task` (mediapipe 0.10.35) | `e8098a34013a130b8d8c88de0772f162b450dfad` | Faces, landmarks, blinks; face crops |
| `adaface` | AdaFace IR-50 (UniFace ONNX export) | `bfcc7d48…` (**confirm full hash in the app's `ai_manifest.py`**) | Identity, for people coverage |

**Rules:**
- **Don't download or re-pin models in this repo.** `ModelRegistry.from_app_manifest(path)` resolves model files from the app's manifest (`ai_manifest.py` / `DEFAULT_CAPABILITY_ORDER`). Tests use a fake registry.
- **Reuse the app's inference code** for Q-ReAlign, MANIQA, MediaPipe and AdaFace (pre/post-processing, score scales) rather than re-deriving it. Port it behind the scorer interface (§4).
- **Remove** CLI-Culler's CLIP-L and TOPIQ defaults (`features.HeadlessFeatureExtractor`), or keep them only as optional legacy scorers.

---

## 3. Pipeline

```
0. Ingest        bracketeering.analyze(folder, backend="exiftool" if available, pixels=True)
                 → groups, scenes, roles, per-frame exposure stats, previews; store in SQLite
1. Embed         TinyCLIP embedding per primary shot (cached)
2. Scene refine  merge/split bracketeering scenes using TinyCLIP + bracketeering's registration
3. Route         per group kind + settings → which frames compete, in which comparison sets
4. Gates         clear failures only → reject with reason (never on bracket frames)
5. Technical     MediaPipe faces/eyes + face-crop sharpness, MANIQA, relative within the set
6. Appeal        Q-ReAlign only on survivors of sets with > 1 candidate left
7. Preference    optional learned score from the user's past culls (TinyCLIP embeddings)
8. Decide        per-set ranking → keep/reject/review per settings; people coverage; reasons
9. Write         decisions.json (+ XMP)
```

### 0. Ingest
- **Dependency:** `bracketeering @ git+https://github.com/tylermcm/bracketeering@ccr-fd1f1f35-hi49ff` (≥ 0.3.4). Read its `docs/HANDOFF.md` and `SCHEMA.md` first.
- **Previews:** replace CLI-Culler's `PreviewExtractor` (scans for JPEG start markers) with `bracketeering.preview.preview_source`, which parses containers properly and skips lossless raw data. Decode previews once at ~1024 px and cache them.
- **Identity:** key the store by **capture** rather than path. Use bracketeering's shot (body + shutter count, or capture time + frame number); files are members of a shot. Copies, exports and sidecars must never be scored twice.
- **Errors:** bracketeering's `errors` → `decision: "review"`, reason "unreadable: …".

### 2. Scene refinement (TinyCLIP)
Bracketeering's weakest output is near-duplicate scenes: hazy horizons over-merge, and some same-view pairs split. Refine its scenes per camera body in capture order:
- **Split** a scene where consecutive groups' TinyCLIP cosine similarity < `split_cos`.
- **Merge** adjacent scenes within 120 s where cosine ≥ `merge_cos` and bracketeering's layout/registration doesn't contradict.
- **Thresholds:** `split_cos` and `merge_cos` are tuned with the evaluation harness; start ~0.80 / ~0.92.
- Record each change as a reason ("scene merged by embedding similarity 0.94").

### 3. Routing: comparison sets

| Group kind | Setting | What competes |
|---|---|---|
| `exposure` (bracket) | `keep_set` | The bracket competes **as one unit** in its scene, represented by `best_exposed` (else `representative`). The decision applies to all its frames. |
| | `pick_one` | Same, then keep only `best_exposed`; reject the other frames ("not best exposed"). |
| | `ignore` | All frames kept; not ranked. |
| `burst` | `bursts_keep = N` | Its frames compete with each other; keep top N. The burst's best frame then represents it in the scene. |
| `focus`, `timelapse` | `keep_all` | Kept, not ranked (unless set to `cull`). |
| `single` | — | Competes in its scene. |

A scene is then a comparison set of *units* (single frames, bracket units, burst winners); keep the top `scene_keep`.

### 4. Gates: clear failures only
- Black or misfired frame: bracketeering `exposure.mean` < 0.02 or the frame is featureless.
- Unusable clipping: `highlights_clipped` > 0.25 or `shadows_crushed` > 0.6.
- Severe blur: MANIQA in the bottom 2 % of the shoot **and** face/whole-frame sharpness far below its set's median.
- **Never gate bracket frames on exposure;** brackets are judged by their best-exposed frame.
- If a gate would reject every candidate in a set, keep the least-bad and mark it `review`.

### 5. Technical scores (relative)
- **MediaPipe Face Landmarker:**
  - Faces, eye openness (blink), face size and position.
  - **Face-crop sharpness:** variance of the Laplacian on the largest or most central faces. For portraits this beats whole-frame sharpness.
- **MANIQA:** quality score. KonIQ-trained models react to brightness and noise, so only compare it within a set, never across a bracket's frames.
- **Metadata blur prior:** shutter time vs focal length and drive mode (handheld 1/8 s at 200 mm is suspect).
- **Normalise within the comparison set** (rank or z-score), not globally. Raw sharpness mostly measures scene texture.

### 6. Appeal (Q-ReAlign Mini 0.8B)
- Run **only** on candidates still competing in sets with > 1 survivor, batched, cached by capture identity + model revision.
- Never use it to choose between a bracket's own frames.
- Log how many frames reached this stage. It is the runtime driver.

### 7. Preference learning
Keep CLI-Culler's `PreferenceLearningScorer` (PCA + logistic regression) and its leave-one-out diagnostics, but:
- Train it on **TinyCLIP** embeddings.
- Take labels from the user's past culls (§6).
- Use it as one more score, weighted low until the harness shows it helps.

### 8. Decide
- **Combined score per candidate:** weighted sum of set-normalised scores. Start with appeal 0.4, technical 0.4 (MANIQA 0.25, face 0.15), preference 0.2, minus penalties (blink, gate-adjacent issues). Weights live in config; the harness tunes them.
- **Apply keep counts** per routing table. A group with grouping confidence < `review_below` or with bracketeering warnings goes to **review**, not auto-reject.
- **People coverage (AdaFace):** cluster face embeddings into identities. If a person's only acceptable frames were all rejected, promote their best one to `keep` with the reason "only good frame of person #3".
- **Reasons:** human-readable, specific ("rejected: eyes closed; 2 sharper siblings in burst U00366").

### 9. Performance
- Cache everything per capture identity and model revision in SQLite. Re-running with new *settings* must not re-run any model.
- Use ONNX Runtime provider selection (keep CLI-Culler's CUDA → CoreML → CPU logic).
- Report a per-stage timing summary.
- **Target:** a 2,000-shot shoot on a laptop GPU in minutes, dominated by Q-ReAlign on ≲ 25 % of frames.

---

## 4. Code structure

```
src/aiculler/
  api.py            cull(), CullSettings, result types (the app's entry point)
  ingest.py         bracketeering integration, capture identity, preview cache
  models/registry.py ModelRegistry.from_app_manifest(); fake registry for tests
  scorers/          one module per model, common interface:
                    class Scorer: name, revision, def score(batch: list[Candidate]) -> list[dict]
                    tinyclip.py  maniqa.py  q_realign.py  faces.py (MediaPipe)  adaface.py  metadata.py
  scenes.py         scene refinement
  routing.py        comparison sets from groups + settings
  gates.py          clear-failure gates
  decide.py         normalisation, weights, keep counts, people coverage, reasons
  outputs.py        decisions.json, XMP
  evaluate.py       harness (§6)
  storage.py        (existing, extended: shots, files, scores(capture, scorer, revision, json))
  preference_learning.py, session_review.py, run_logging.py   (existing, adapted)
  cli.py            (existing; add `cull`, `evaluate`; keep review-session / compare-rankings)
```

**Retire or park:** `ranking.ActiveQuicksortCuller` / `GlobalRanker` (global ranking), `composite_ranking` (replaced by `decide.py`), `adapter.RankingAwareAdapter`, and the CLIP-L/TOPIQ extractor. Delete them once `cull` covers their use, or move them to `legacy/`. Text/profile prompt scoring may stay as an optional scorer for semantic search.

---

## 5. Milestones (each shippable, each measured)

1. **Skeleton + harness.**
   - `cull()` with bracketeering ingest, routing, the decision layer using *only* bracketeering's exposure stats and simple whole-frame sharpness, `decisions.json`, and `evaluate`.
   - **Done when** it runs on one real shoot end to end and `evaluate` reports agreement with that shoot's Lightroom ratings.
2. **Faces + MANIQA.** MediaPipe and MANIQA scorers, set-relative normalisation, gates. Measure the change in agreement.
3. **Q-ReAlign appeal.** Only on survivors. Measure agreement and runtime.
4. **TinyCLIP.** Scene refinement plus preference learning. Measure the scene changes on the hand-checked shoots and the agreement delta.
5. **People coverage (AdaFace)** and XMP output.
6. **App adapter.** A thin call from the image-triage app; the app shows decisions and reasons.

---

## 6. Evaluation harness (the accuracy number)

- **Ground truth: the user's past culls.**
  - Read Lightroom ratings/labels from XMP sidecars (`xmp:Rating`, `xmp:Label`).
  - Or read picks and ratings from a **read-only copy** of the `.lrcat` (SQLite: `Adobe_images` rating/pick joined to `AgLibraryFile` / `AgLibraryFolder` for paths).
  - **Kept** = picked or ≥ 3★ (configurable); **rejected** = rejected flag or unrated.
- **Metrics:**
  - Precision/recall of `keep` against kept.
  - **Per-set agreement:** did the engine keep a frame the user kept, in each burst and scene? This matters more than global precision.
  - `review` rate, and runtime per stage.
- `ai-culler evaluate SHOOT --truth lightroom.lrcat|xmp --out report.md` writes a report with the worst disagreements (file, decision, user's choice, reasons) for inspection.
- **Fixtures:** export small, anonymised fixtures (bracketeering JSON + cached scores + user labels, no images, no GPS) under `tests/fixtures/` so changes are checked in CI.

---

## 7. Non-goals / later

- No GUI in this repo. No model training except the lightweight preference model.
- No editing or file moving (CLI-Culler's `stage` may stay as a convenience).
- Later: active learning from the app's keep/reject clicks (`ThreadSafeLearningEngine` hook), and per-genre weight profiles (portrait, wildlife, landscape).

## 8. Open questions for the user

- Where does the app's model inference code live (repo + module) for Q-ReAlign, MANIQA, MediaPipe and AdaFace?
- Which shoots have reliable Lightroom culls to use as ground truth, and is it XMP or catalog?
- Default keep counts per genre: is 2 per burst and 1 per scene right for your shooting?
