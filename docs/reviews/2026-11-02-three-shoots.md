# 0.3.1 on three more shoots + China re-test (summary)

Hand-checked by a Claude Code session on the photographer's machine (Nikon maker notes for bracket
cycles, phase correlation + preview viewing for framing). Drove 0.3.2.

| Shoot | Bodies | Shots | Result with 0.3.1 |
|---|---|---|---|
| WhiteSands '26 | Z 7_2 | 917 | 91/91 bracket cycles right; focus-shift stacks called timelapse/bursts; a false timelapse (hand presses 1.12 s apart, timer off); 3 bursts broken at `_DSC8xxx`/`_DSC0xxx` file-number switches; moon bursts fragmented (featureless frames); hazy-horizon scenes drift |
| Japan 4-24 | D810 | 531 | clean: 108/108 cycles, bursts, scenes |
| Alaska Aug '21 | D800, D5, D850 | 2727 | 968/968 copies merged; long-exposure single-frame aurora cycles (13–30 s, presses 22–66 s apart) all broken into singles by the 3 s pause rule; 13 edited exports formed fake 500 fps bursts; 6 zero-filled NEFs not reported; 28 wildlife 7F cycles split at < 0.5 s on similarity; partial warnings: 14 false (3F 2/3 EV after 3F 1/3 EV), ≈140 missed (fragments) |
| China 8-26 (re-test) | Z 7_2 | 1137 | 58/58 expected groupings; but the stricter similarity broke 3 bursts (one 20-frame hold, black misfires, blurred frames) and split the ivy scene; `_DSC1073–1075` (0.8 s presses, AF re-acquired) still a burst; native backend missed 9 partial warnings |

Fixes in 0.3.2 (regression tests in `tests/test_three_shoots_review.py`):
1. Cycle recompose split relative to the cycle's own rhythm; similarity splits only across a real pause; "frames don't register" flag for moving subjects.
2. Copies *and* edited exports merged on any shared identity (shutter count, or capture time + frame number + exposure); identical capture times never form a burst.
3. Zero-filled / unreadable files reported as errors (native check + exiftool's `Error`).
4. Nikon focus-shift (`FocusMode = Fshift`) → focus stack; `IntervalShooting = Off` blocks timelapse; tighter timer precision.
5. Ordering and chaining by shutter count.
6. Featureless previews → similarity unknown; bursts never split on similarity; scenes use structure + layout.
7. Partial warnings: program inferred from the whole cycle and its contiguous bracketing session.
8. Slow pace + focus jump between frames counts against a burst.
