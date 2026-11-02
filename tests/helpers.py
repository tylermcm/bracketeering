import math
from pathlib import Path

from bracketeering.model import FileMeta, Shot
from bracketeering.synth import nominal_shutter

_F = [1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 9.0, 10, 11, 13, 14, 16, 18, 20, 22]


def nominal_fnumber(stops: float) -> float:
    """Nominal f-number label for an aperture `stops` = 2*log2(N)."""
    return min(_F, key=lambda f: abs(2 * math.log2(f) - stops))

T0 = 1_700_000_000.0


class Roll:
    """Fluent builder for a sequence of shots from one camera."""

    def __init__(self, make="Canon", model="EOS R5", subsec=True, start=1):
        self.make, self.model, self.subsec = make, model, subsec
        self.t = T0
        self.n = start
        self.shots: list[Shot] = []

    def shot(self, shutter, *, gap=0.2, mode=2, bias=None, fnum=8.0, iso=100, focal=24.0, ext="CR3", **kw):
        name = f"IMG_{self.n:04d}.{ext}"
        self.n += 1
        m = FileMeta(Path(name), make=self.make, model=self.model, capture_time=self.t if self.subsec else float(int(self.t)),
                     time_precise=self.subsec, exposure_time=shutter, f_number=fnum, iso=iso, exposure_bias=bias,
                     exposure_mode=mode, focal_length=focal, **kw)
        self.shots.append(Shot([m]))
        self.t += shutter + gap
        return name

    def bracket(self, base, offsets, **kw):
        """Shutter-driven bracket using the nominal shutter labels cameras write."""
        return [self.shot(nominal_shutter(math.log2(base) + o), bias=float(o), **kw) for o in offsets]

    def aperture_bracket(self, shutter, base_f, offsets, **kw):
        """Shutter-priority bracket: the aperture moves (wider = brighter), shutter stays fixed."""
        return [self.shot(shutter, fnum=nominal_fnumber(2 * math.log2(base_f) - o), bias=float(o), **kw) for o in offsets]

    def pause(self, seconds):
        self.t += seconds


def names(groups, kind=None):
    return [[s.name for s in g.shots] for g in groups if kind is None or g.kind == kind]
