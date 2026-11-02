# Output schema `bracketeering/2`

`bracketeer scan --format json` (or `report.to_dict()` in Python) returns one JSON object. The `schema` field changes only when the format changes incompatibly. Fields may be *added* within a schema version, so consumers should ignore keys they don't know.

## Top level

| key | type | meaning |
|---|---|---|
| `schema` | `"bracketeering/2"` | format version |
| `tool_version` | string | package version that produced the file |
| `summary` | object | counts: `shots`, `files`, `exposure_brackets`, `focus_stacks`, `bursts`, `timelapses`, `singles`, `near_duplicate_scenes`, `unreadable` |
| `groups` | array | every shot belongs to exactly one group, singles included, in capture order |
| `files` | object | per-file index keyed by path; **every input file appears exactly once** |
| `errors` | array | `{file, error}` for files whose metadata could not be read (they still appear in `files`, as singles) |

## `groups[]`

| key | type | meaning |
|---|---|---|
| `id` | string | `B…` exposure bracket, `U…` burst, `T…` timelapse, `F…` focus stack, `S…` single |
| `kind` | `"exposure"` \| `"burst"` \| `"timelapse"` \| `"focus"` \| `"single"` | group type |
| `size` | int | number of shots (a RAW+JPEG pair is one shot) |
| `confidence` | float 0–1 | detection confidence (1.0 for singles). For bursts it is the log-odds of "one press of the shutter" vs "separate presses", squashed to 0–1. On simulated shoots, bursts at ≥ 0.9 were exactly right 95% of the time and those at 0.5–0.7 about 77% |
| `camera` | string \| null | make + model |
| `start` | ISO time \| null | capture time of the first frame, camera-local, with its UTC offset when the file records one (e.g. `2026-08-26T04:47:21.990-07:00`) |
| `representative` | path | the frame that stands for the group. Exposure brackets: the middle exposure (≈ 0 EV). Bursts, timelapses and focus stacks: the middle frame. |
| `best_exposed` | path \| null | exposure brackets with `--pixels`: brightest frame whose highlights hold (expose to the right) |
| `scene` | int \| null | with `--pixels`: near-duplicate scene id. Groups sharing an id are the same subject shot repeatedly |
| `ev_offsets` | float[] \| null | exposure brackets: each frame's offset in stops, in capture order. When the camera recorded its bracket program (maker notes), these are the camera's own slots relative to the metered frame (e.g. `[0, -1.33, -0.67]`). Otherwise they're measured, relative to the middle exposure |
| `step_ev` | float \| null | exposure brackets: bracket step in stops |
| `evidence` | string[] | why the group was formed (human-readable) |
| `warnings` | string[] | e.g. missing frames, incomplete bracket, weak candidate demoted |
| `frames` | array | per shot: `files` (paths, primary first), `role`, `time`, `shutter`, `aperture`, `iso`, `exposure_bias`, `ev`; with `--pixels` also `best_exposed` (bool) and `exposure` |

`frames[].exposure` (with `--pixels`): `highlights_clipped` (fraction of pixels with any channel clipped), `shadows_crushed` (fraction with luma ≤ 5/255), `mean` (luma 0–1), `flag` (`"ok"`, `"over"` or `"under"`). These are measured on the camera's embedded JPEG preview. Highlights clip there slightly earlier than in the RAW, so `over` is conservative.

## `files{path}`

| key | type | meaning |
|---|---|---|
| `group` | string | id of the group the file belongs to |
| `kind` | string | that group's kind |
| `role` | `"representative"` \| `"member"` \| `"single"` | the file's role in its group |
| `position` | int | 1-based capture position within the group |
| `primary` | bool | false for a sidecar (e.g. the JPEG half of a RAW+JPEG pair) and for copies of the same capture in other folders (e.g. a `_picks/` folder); they share their shot's role |
| `scene` | int \| null | near-duplicate scene id (with `--pixels`) |
| `best_exposed` | bool \| null | exposure brackets with `--pixels` |
| `error` | string \| null | metadata read error, if any |

## Using it in a culling pipeline

- **The user doesn't target brackets:** for every file whose `kind == "exposure"`, keep the whole group as one unit (e.g. for HDR merging).
- **Brackets are targeted, one frame wanted:** use `best_exposed` when present, otherwise `representative`.
- **Bursts:** one press of the shutter in continuous drive. The group is a set of alternatives: rank its frames (sharpness, then aesthetics) and keep the best. Two presses separated by a short pause are two bursts. Groups sharing a `scene` id are the same subject.
- **Timelapses:** interval-timer sequences, where every frame is part of the sequence. These are usually excluded from culling rather than reduced to one frame.
- **Near-duplicates:** groups sharing a `scene` id are alternatives of the same subject. Compare their representatives against each other, not against unrelated photos.
- **Confidence:** pass `--min-confidence` to have weak bracket candidates reported as singles instead. A wrong merge costs more in culling than a missed bracket.
