# src/numclass/progress.py
from __future__ import annotations

import sys
import time


class Progress:
    def __init__(self, total: int, *, enabled: bool = True):
        self.total = max(1, int(total))
        self.enabled = enabled
        self.start = time.perf_counter()
        self.last_draw = 0.0
        self.spin = "|/-\\"
        self.i = 0

    def update(self, done: int, label: str = ""):
        THROTTLE = 0.05
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self.last_draw < THROTTLE:  # throttle to avoid flicker
            return
        self.last_draw = now
        self.i = (self.i + 1) % len(self.spin)
        frac = min(max(done / self.total, 0.0), 1.0)
        pct = int(frac * 100)
        bar_len = 24
        fill = int(frac * bar_len)
        bar = "#" * fill + "-" * (bar_len - fill)
        msg = f"\r[{self.spin[self.i]}] [{bar}] {pct:3d}%  {label[:50]}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def done(self):
        if not self.enabled:
            return
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
