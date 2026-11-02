from pathlib import Path

from bracketeering.model import FileMeta, Shot

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
        return [self.shot(base * 2 ** o, bias=float(o), **kw) for o in offsets]

    def pause(self, seconds):
        self.t += seconds


def names(groups, kind=None):
    return [[s.name for s in g.shots] for g in groups if kind is None or g.kind == kind]
