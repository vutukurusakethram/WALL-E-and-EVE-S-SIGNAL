"""Simulated environment: loads a mission config and returns observations for each action.

The agent only ever sees what `step()` returns and what `public_info()` exposes.
EVE's position and the clue layout are hidden and revealed only through interaction.
"""
from collections import deque
import numpy as np

from .config import resolve_signal_coords, signal_sigma

DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
CLUE_KINDS = ["tire tracks", "scorch marks", "weak beacon ping", "dropped plant sample", "energy residue"]


def bfs_dist(free, src):
    H, W = free.shape
    dist = np.full((H, W), np.inf)
    sx, sy = int(src[0]), int(src[1])
    dist[sy, sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and free[ny, nx] and dist[ny, nx] == np.inf:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))
    return dist


def bfs_path(free, src, dst):
    """Shortest 4-connected path as a list of (x, y) cells including src and dst."""
    H, W = free.shape
    src, dst = (int(src[0]), int(src[1])), (int(dst[0]), int(dst[1]))
    prev = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for dx, dy in DIRS4:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < W and 0 <= ny < H and free[ny, nx] and (nx, ny) not in prev:
                prev[(nx, ny)] = cur
                q.append((nx, ny))
    if dst not in prev:
        return None
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


class Environment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])          # builds the world
        self.srng = np.random.default_rng(cfg["seed"] + 1)     # sensor noise stream
        m = cfg["map"]
        self.W, self.H = m["width"], m["height"]
        self.start = tuple(cfg["start"])

        free = np.ones((self.H, self.W), dtype=bool)
        for x, y in m["obstacles"]:
            free[y, x] = False
        if m.get("obstacle_density", 0) > 0:
            free &= ~(self.rng.random((self.H, self.W)) < m["obstacle_density"])
        free[self.start[1], self.start[0]] = True
        self.free = free
        self.home_dist = bfs_dist(free, self.start)
        reachable = free & np.isfinite(self.home_dist)

        # --- initial signal (what the agent is told) ---
        sig = cfg["signal"]
        self.signal = {
            "coords": resolve_signal_coords(cfg),
            "strength": sig["strength"],
            "timestamp": sig["timestamp"],
            "sigma": signal_sigma(sig, now=0.0),
        }

        # --- hidden truth: EVE ---
        ev = cfg["eve"]
        if ev["position"] is not None:
            self.eve = tuple(ev["position"])
            if not reachable[self.eve[1], self.eve[0]]:
                raise ValueError("eve.position must be a reachable free cell")
        else:
            ys, xs = np.nonzero(reachable)
            keep = ~((xs == self.start[0]) & (ys == self.start[1]))
            xs, ys = xs[keep], ys[keep]
            cx, cy = self.signal["coords"]
            w = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * self.signal["sigma"] ** 2))
            i = self.rng.choice(len(xs), p=w / w.sum())
            self.eve = (int(xs[i]), int(ys[i]))
        self.eve_status = ev.get("status", "active")
        self.reachable = reachable
        self.clues = self._make_clues()
        self.discovered = set()

        # --- mutable state ---
        b = cfg["budget"]
        self.pos = self.start
        self.battery = b["battery"]
        self.time_left = b["time"]
        self.recharges_used = 0
        self.max_recharges = b["max_recharges"]
        self.done = False
        self.end_reason = None

    # ------------------------------------------------------------------ world
    def _make_clues(self):
        cc, rng = self.cfg["clues"], self.rng
        clues = []

        def build(i, pos, kind, reliable):
            if reliable:
                est = np.array(self.eve, float) + rng.normal(0, cc["sigma"], 2)
                est = np.clip(est, [0, 0], [self.W - 1, self.H - 1])
            else:  # decoy: points somewhere random
                ys, xs = np.nonzero(self.reachable)
                j = rng.integers(len(xs))
                est = np.array([xs[j], ys[j]], float)
            return {"id": i, "pos": tuple(int(v) for v in pos), "kind": kind,
                    "reliable": bool(reliable), "estimate": [round(float(est[0]), 2), round(float(est[1]), 2)]}

        if cc.get("list"):
            for i, c in enumerate(cc["list"]):
                clues.append(build(i, c["pos"], c.get("kind", CLUE_KINDS[i % len(CLUE_KINDS)]), c.get("reliable", True)))
            return clues

        ys, xs = np.nonzero(self.reachable)
        cells = [(int(x), int(y)) for x, y in zip(xs, ys) if (x, y) != self.start and (x, y) != self.eve]
        near = [c for c in cells if np.hypot(c[0] - self.eve[0], c[1] - self.eve[1]) <= 7]
        used = set()
        for i in range(cc["count"]):
            pool = near if (rng.random() < 0.6 and near) else cells
            pool = [c for c in pool if c not in used] or cells
            pos = pool[rng.integers(len(pool))]
            used.add(pos)
            clues.append(build(i, pos, CLUE_KINDS[i % len(CLUE_KINDS)], rng.random() >= cc["decoy_prob"]))
        return clues

    # -------------------------------------------------------- what agent may see
    def public_info(self):
        c, s, b = self.cfg["costs"], self.cfg["sensor"], self.cfg["budget"]
        return {
            "width": self.W, "height": self.H, "free": self.free.copy(), "start": self.start,
            "budget": dict(b), "costs": dict(c), "sensor": dict(s),
            "signal": dict(self.signal), "clue_sigma": self.cfg["clues"]["sigma"],
        }

    def truth(self):
        """Ground truth, used ONLY for evaluation after the mission."""
        return {"eve_pos": self.eve, "eve_status": self.eve_status, "clues": self.clues}

    def _state(self, ok, msg, extra=None):
        out = {"ok": ok, "msg": msg, "pos": list(self.pos), "battery": self.battery,
               "time_left": self.time_left, "scan": None}
        if extra:
            out.update(extra)
        return out

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.done:
            return self._state(False, "mission already ended")
        c, s = self.cfg["costs"], self.cfg["sensor"]
        t = action.get("type")
        obs = None

        if t == "move":
            tx, ty = action["to"]
            adj = abs(tx - self.pos[0]) + abs(ty - self.pos[1]) == 1
            if not adj:
                obs = self._state(False, "move target is not adjacent")
            elif not (0 <= tx < self.W and 0 <= ty < self.H) or not self.free[ty, tx]:
                obs = self._state(False, "blocked by obstacle / edge")
            elif self.battery < c["move"] or self.time_left < c["move_time"]:
                obs = self._state(False, "not enough battery or time to move")
            else:
                self.pos = (int(tx), int(ty))
                self.battery -= c["move"]
                self.time_left -= c["move_time"]
                obs = self._state(True, "moved")

        elif t == "scan":
            if self.battery < c["scan"] or self.time_left < c["scan_time"]:
                obs = self._state(False, "not enough battery or time to scan")
            else:
                self.battery -= c["scan"]
                self.time_left -= c["scan_time"]
                d = float(np.hypot(self.pos[0] - self.eve[0], self.pos[1] - self.eve[1]))
                reading = float(np.exp(-d / s["length_scale"]) + self.srng.normal(0, s["noise"]))
                if self.srng.random() < s["spoof_prob"]:          # spoofed / decoy signal
                    reading = float(self.srng.random())
                found = d <= s["detect_radius"]
                new_clues = []
                for cl in self.clues:
                    if cl["id"] in self.discovered:
                        continue
                    if np.hypot(self.pos[0] - cl["pos"][0], self.pos[1] - cl["pos"][1]) <= s["scan_radius"]:
                        self.discovered.add(cl["id"])
                        new_clues.append({"id": cl["id"], "kind": cl["kind"], "pos": list(cl["pos"]),
                                          "estimate": cl["estimate"]})   # 'reliable' is NOT exposed
                obs = self._state(True, "scanned", {"scan": {
                    "reading": round(reading, 4), "clues": new_clues, "eve_detected": bool(found),
                    "eve_pos": list(self.eve) if found else None,
                    "eve_status": self.eve_status if found else None}})

        elif t == "recharge":
            if self.pos != self.start:
                obs = self._state(False, "can only recharge at the charger")
            elif self.recharges_used >= self.max_recharges:
                obs = self._state(False, "no recharges left")
            elif self.time_left < c["recharge_time"]:
                obs = self._state(False, "not enough time to recharge")
            else:
                self.battery = self.cfg["budget"]["battery"]
                self.time_left -= c["recharge_time"]
                self.recharges_used += 1
                obs = self._state(True, "recharged")

        elif t == "declare":
            self.done, self.end_reason = True, "declared"
            obs = self._state(True, "mission declared complete")
        else:
            obs = self._state(False, f"unknown action {t!r}")

        obs["battery"], obs["time_left"], obs["pos"] = self.battery, self.time_left, list(self.pos)
        if not self.done:
            if self.time_left <= 0:
                self.done, self.end_reason = True, "timeout"
            elif self.battery <= 0 and self.pos != self.start:
                self.done, self.end_reason = True, "stranded"
        obs["done"], obs["end_reason"] = self.done, self.end_reason
        return obs
