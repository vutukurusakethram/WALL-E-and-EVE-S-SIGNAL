"""Autonomous agent: decides where to move, what to scan, when to recharge or turn back.

Decision rule ("eig" policy)
    utility(cell) = (expected information gain of a scan there
                     + P(EVE is within detection radius) * current entropy) / battery cost
    cost = move cost to get there + scan cost
Only cells from which WALL-E can still get back to the charger (with a safety margin) are allowed.
"""
import numpy as np

from .belief import Belief, ndtr
from .env import bfs_dist, bfs_path

N_BINS = 10   # reading discretisation used for the information-gain calculation


class Agent:
    def __init__(self, info, acfg, policy=None):
        self.info, self.cfg = info, acfg
        self.policy = policy or acfg["policy"]
        self.free = info["free"]
        self.start = tuple(info["start"])
        self.pos = self.start
        b, c, s = info["budget"], info["costs"], info["sensor"]
        self.full = b["battery"]
        self.battery = b["battery"]
        self.time_left = b["time"]
        self.recharges_left = b["max_recharges"]
        self.mv, self.sc, self.mvt, self.sct, self.rt = c["move"], c["scan"], c["move_time"], c["scan_time"], c["recharge_time"]
        self.L, self.noise, self.spoof = s["length_scale"], s["noise"], s["spoof_prob"]
        self.detect_r = s["detect_radius"]
        self.margin = acfg["safety_margin"]
        self.clue_sigma = info["clue_sigma"]

        sig = info["signal"]
        self.belief = Belief(self.free, sig["coords"], sig["sigma"])
        self.home = bfs_dist(self.free, self.start)

        ys, xs = np.nonzero(self.free & np.isfinite(self.home))
        self.cx, self.cy = xs, ys                      # candidate scan cells == possible EVE cells
        D = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
        self.det_mask = (D <= self.detect_r).astype(float)
        # P(reading bin | EVE at cell j, scanned from cell i) -- static, so precompute once
        mu = np.exp(-D / self.L)
        edges = np.linspace(0, 1, N_BINS + 1)
        cdf = ndtr((edges[None, None, :] - mu[:, :, None]) / self.noise)
        cdf[:, :, 0], cdf[:, :, -1] = 0.0, 1.0
        bp = np.diff(cdf, axis=2)
        self.bp = np.maximum((1 - self.spoof) * bp + self.spoof / N_BINS, 1e-12)

        self.plan, self.goal, self.target = [], None, None
        self.pending_reason, self.leg_reason = "", ""
        self.confirmed, self.eve_status = None, None
        self.clues, self.readings, self.scan_cells = [], [], []
        self.failures = 0

    # ------------------------------------------------------------ belief helpers
    def confidence(self):
        return self.belief.mass_near(self.belief.peak(), self.detect_r)

    def _flat(self):
        return self.belief.p[self.cy, self.cx]

    def _utilities(self, d_cur):
        pf, H = self._flat(), self.belief.entropy()
        cost = np.where(np.isfinite(d_cur), d_cur * self.mv + self.sc, np.inf)
        pdet = self.det_mask @ pf
        if self.policy == "greedy":                       # baseline: go to the most likely spot
            util = pdet - 1e-4 * np.nan_to_num(d_cur, posinf=1e6)
            return util, np.zeros_like(pdet), pdet
        py = np.einsum("n,ink->ik", pf, self.bp)
        mi = np.einsum("n,ink->i", pf, self.bp * (np.log(self.bp) - np.log(np.maximum(py, 1e-300))[:, None, :]))
        util = (mi + pdet * H) / cost
        return util, mi, pdet

    # ------------------------------------------------------------------ acting
    def act(self):
        """Returns (action, decision_tag, reasoning_text)."""
        b = self.belief
        if self.failures >= 3:
            return {"type": "declare"}, "declare", "Several actions failed in a row; stopping safely."
        if self.confirmed:
            return {"type": "declare"}, "declare", (
                f"EVE visually confirmed at {self.confirmed} (status: {self.eve_status}). Mission goal reached.")
        if self.scan_cells and self.confidence() >= self.cfg["stop_confidence"]:
            return {"type": "declare"}, "declare", (
                f"Belief converged: {self.confidence():.0%} of probability sits around {b.peak()}.")
        if not self.plan and self.goal is None:
            immediate = self._replan()
            if immediate:
                return immediate
        return self._advance()

    def _advance(self):
        if self.plan:
            nxt = self.plan.pop(0)
            tag = "replan" if self.pending_reason else "move"
            reason = self.pending_reason or f"{self.leg_reason} ({len(self.plan)} moves left)"
            self.pending_reason = ""
            return {"type": "move", "to": list(nxt)}, tag, reason
        if self.goal == "scan":
            self.goal = None
            lead = self.pending_reason + " " if self.pending_reason else ""
            self.pending_reason = ""
            return {"type": "scan"}, "scan", (
                f"{lead}Scanning at {self.pos} to update the belief about EVE's location.")
        self.goal = None                                   # arrived home after a turn-back
        return self.act()

    def _replan(self):
        b = self.belief
        d_cur = bfs_dist(self.free, self.pos)[self.cy, self.cx]
        d_home = self.home[self.cy, self.cx]

        def feasible(batt, tl, with_time=True):
            need = d_cur * self.mv + self.sc + d_home * self.mv + self.margin
            ok = np.isfinite(d_cur) & (need <= batt)
            if with_time:
                ok &= (d_cur * self.mvt + self.sct <= tl)
            return ok

        util, mi, pdet = self._utilities(d_cur)
        feas = feasible(self.battery, self.time_left)
        at_home = self.pos == self.start
        can_recharge = (at_home and self.recharges_left > 0 and self.battery < self.full
                        and self.time_left > self.rt + self.sct)

        if can_recharge:
            feas_full = feasible(self.full, self.time_left - self.rt)
            best_full = util[feas_full].max() if feas_full.any() else 0.0
            best_now = util[feas].max() if feas.any() else 0.0
            if best_full > 1.3 * best_now:
                why = ("no useful scan is reachable with the current battery" if not feas.any()
                       else f"a full battery allows a {best_full / max(best_now, 1e-9):.1f}x better scan")
                return {"type": "recharge"}, "recharge", (
                    f"At the charger with battery {self.battery}/{self.full}: {why}. Recharging "
                    f"({self.recharges_left} recharge(s) left).")

        if not feas.any():
            if at_home:
                return {"type": "declare"}, "declare", "No affordable scan remains and no recharge is possible. Reporting."
            if feasible(self.battery, self.time_left, with_time=False).any():
                return {"type": "declare"}, "declare", "Time budget exhausted. Reporting best estimate."
            path = bfs_path(self.free, self.pos, self.start)
            self.plan, self.goal = path[1:], "home"
            self.leg_reason = "Turning back to the charger"
            self.pending_reason = (f"TURN BACK: battery {self.battery} only covers the return trip "
                                   f"({int(self.home[self.pos[1], self.pos[0]])} moves) plus margin {self.margin}.")
            return None

        i = int(np.argmax(np.where(feas, util, -np.inf)))
        self.target = (int(self.cx[i]), int(self.cy[i]))
        path = bfs_path(self.free, self.pos, self.target)
        self.plan, self.goal = path[1:], "scan"
        pk = b.peak()
        self.leg_reason = f"Heading to scan at {self.target}"
        self.pending_reason = (
            f"REPLAN: best scan spot {self.target} (info gain {mi[i]:.2f} nats, P(EVE within {self.detect_r} cells)="
            f"{pdet[i]:.2f}, cost {d_cur[i] * self.mv + self.sc:.0f} battery). Belief peak {pk} "
            f"p={b.p[pk[1], pk[0]]:.3f}, entropy {b.entropy():.2f} nats; {int(feas.sum())} of {len(feas)} cells affordable "
            f"with return-to-charger reserve.")
        return None

    # ---------------------------------------------------------------- observing
    def observe(self, action, obs):
        self.battery, self.time_left, self.pos = obs["battery"], obs["time_left"], tuple(obs["pos"])
        if not obs["ok"]:
            self.failures += 1
            self.plan, self.goal = [], None
            return
        self.failures = 0
        t = action["type"]
        if t == "recharge":
            self.recharges_left -= 1
        elif t == "scan":
            sc = obs["scan"]
            self.scan_cells.append(self.pos)
            self.readings.append({"pos": list(self.pos), "reading": sc["reading"]})
            self.belief.update_signal(self.pos, sc["reading"], self.L, self.noise, self.spoof)
            for cl in sc["clues"]:
                self.clues.append(cl)
                self.belief.update_clue(cl["estimate"], self.clue_sigma, self.cfg["decoy_prior"])
            if sc["eve_detected"]:
                self.belief.collapse(sc["eve_pos"])
                self.confirmed, self.eve_status = tuple(sc["eve_pos"]), sc["eve_status"]
            else:
                self.belief.update_not_detected(self.pos, self.detect_r)
