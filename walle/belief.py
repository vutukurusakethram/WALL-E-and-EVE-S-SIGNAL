"""Bayesian belief grid over EVE's location."""
import numpy as np


def erf(x):
    """Vectorised erf (Abramowitz-Stegun 7.1.26, error < 1.5e-7)."""
    s, x = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def ndtr(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


class Belief:
    def __init__(self, free, coords, sigma, uniform_mix=0.10):
        self.free = free
        self.H, self.W = free.shape
        self.ys, self.xs = np.mgrid[0:self.H, 0:self.W]
        cx, cy = coords
        g = np.exp(-((self.xs - cx) ** 2 + (self.ys - cy) ** 2) / (2 * sigma ** 2)) * free
        g /= g.sum()
        u = free / free.sum()
        # a little uniform mass keeps the agent honest if the first signal was wrong / spoofed
        self.p = (1 - uniform_mix) * g + uniform_mix * u
        self.p *= free
        self._norm()

    def _norm(self):
        s = self.p.sum()
        if s <= 0 or not np.isfinite(s):
            self.p = self.free / self.free.sum()
        else:
            self.p /= s

    def _dist(self, pos):
        return np.hypot(self.xs - pos[0], self.ys - pos[1])

    # ---------------------------------------------------------------- updates
    def update_signal(self, pos, reading, length_scale, noise, spoof_prob):
        """P(reading | EVE at cell) = (1-eps) N(reading; exp(-d/L), noise) + eps * Uniform(0,1)."""
        mu = np.exp(-self._dist(pos) / length_scale)
        gauss = np.exp(-((reading - mu) ** 2) / (2 * noise ** 2)) / (noise * np.sqrt(2 * np.pi))
        self.p *= (1 - spoof_prob) * gauss + spoof_prob * 1.0
        self._norm()

    def update_clue(self, estimate, sigma, decoy_prior):
        """A clue points at `estimate`. It is either reliable (gaussian around EVE) or a decoy (uniform)."""
        d2 = (self.xs - estimate[0]) ** 2 + (self.ys - estimate[1]) ** 2
        reliable = np.exp(-d2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        self.p *= (1 - decoy_prior) * reliable + decoy_prior / self.free.sum()
        self._norm()

    def update_not_detected(self, pos, detect_radius):
        """Scanned, EVE not seen -> EVE is (almost surely) not within the detection radius."""
        self.p[self._dist(pos) <= detect_radius] *= 1e-3
        self._norm()

    def collapse(self, pos):
        self.p[:] = 0
        self.p[int(pos[1]), int(pos[0])] = 1.0

    # ------------------------------------------------------------------ stats
    def peak(self):
        i = np.unravel_index(np.argmax(self.p), self.p.shape)
        return (int(i[1]), int(i[0]))

    def entropy(self):
        q = self.p[self.p > 0]
        return float(-(q * np.log(q)).sum())

    def mass_near(self, pos, radius):
        return float(self.p[self._dist(pos) <= radius].sum())

    def region_size(self, mass=0.9):
        q = np.sort(self.p.ravel())[::-1]
        return int(np.searchsorted(np.cumsum(q), mass) + 1)

    def summary(self, detect_radius):
        pk = self.peak()
        return {"peak": list(pk), "peak_p": round(float(self.p[pk[1], pk[0]]), 4),
                "entropy": round(self.entropy(), 3),
                "confidence": round(self.mass_near(pk, detect_radius), 4),
                "cells_90pct": self.region_size(0.9)}
