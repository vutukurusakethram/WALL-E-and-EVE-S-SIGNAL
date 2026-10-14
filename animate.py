#!/usr/bin/env python3
"""Replay a mission as an animated GIF.
Usage: python animate.py configs/mission_01.yaml [--out outputs/mission_01/animation.gif] [--fps 8] [--policy eig|greedy]"""
import argparse, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from walle.config import load_config
from walle.runner import run_mission

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--out", default=None)
ap.add_argument("--fps", type=int, default=8)
ap.add_argument("--policy", choices=["eig", "greedy"], default=None)
a = ap.parse_args()

cfg = load_config(a.config)
res = run_mission(cfg, None, policy=a.policy)
env, trace, snaps, rep = res["env"], res["trace"], res["snaps"], res["report"]
free, start, full = env.free, env.start, cfg["budget"]["battery"]
H, W = free.shape

# build one frame per step (+ hold frames on scans / the final result)
frames = [(0, 1)]
for r in trace:
    hold = 6 if r["action"]["type"] in ("scan", "recharge", "declare") or r["decision"] in ("replan",) else 1
    frames.append((r["step"], hold))
frames = [i for i, h in frames for _ in range(h)] + [trace[-1]["step"]] * 16

def belief_at(step):
    best = snaps[0][1]
    for st, p in snaps:
        if st <= step: best = p
    return best

fig = plt.figure(figsize=(9, 6.4))
ax = fig.add_axes([0.05, 0.22, 0.62, 0.70])
axb = fig.add_axes([0.72, 0.52, 0.25, 0.40])
axt = fig.add_axes([0.05, 0.02, 0.92, 0.16]); axt.axis("off")
axi = fig.add_axes([0.72, 0.22, 0.25, 0.22]); axi.axis("off")
cmap = plt.get_cmap("magma").copy()

def draw(i):
    step = frames[i]
    ax.clear(); axb.clear()
    p = belief_at(step)
    ax.imshow(np.where(free, p / max(p.max(), 1e-12), np.nan), cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.imshow(np.where(free, np.nan, 1.0), cmap="Greys", origin="upper", vmin=0, vmax=1.6)
    done = [r for r in trace if r["step"] <= step]
    pts = [start] + [tuple(r["pos_after"]) for r in done if r["action"]["type"] == "move"]
    ax.plot(*zip(*pts), "-", color="white", lw=1.5, alpha=.8)
    k = 0
    for r in done:
        if r["action"]["type"] == "scan":
            k += 1
            ax.plot(*r["pos_after"], "o", color="orange", ms=9, mec="k")
            ax.annotate(str(k), r["pos_after"], fontsize=6.5, ha="center", va="center", weight="bold")
            for cl in r["observation"]["scan"]["clues"]:
                ax.plot(*cl["pos"], "D", color="#2ecc71", ms=8, mec="k")
        if r["decision"] == "recharge":
            ax.plot(*start, "s", color="yellow", ms=13, mec="k", zorder=1)
    ax.plot(*start, "c^", ms=11, mec="k")
    cur = tuple(done[-1]["pos_after"]) if done else start
    ax.plot(*cur, "o", color="deepskyblue", ms=13, mec="white", mew=2, zorder=5)
    ax.annotate("W", cur, color="k", fontsize=8, ha="center", va="center", weight="bold", zorder=6)
    found = done and done[-1]["decision"] == "declare" and rep["status"] == "CONFIRMED"
    if found:
        ax.plot(*rep["inferred_location"], "*", color="lime", ms=22, mec="k", zorder=7)
    ax.set_xticks(range(0, W, 2)); ax.set_yticks(range(0, H, 2)); ax.tick_params(labelsize=7)
    ax.set_title(f"{cfg['name']}   (step {step})", fontsize=10)

    batt = done[-1]["battery"] if done else full
    axb.bar([0], [batt], color="#2ca02c" if batt > full * .4 else "#ff7f0e" if batt > full * .2 else "#d62728")
    axb.set_ylim(0, full); axb.set_xticks([]); axb.set_title(f"Battery {batt}/{full}", fontsize=9)

    axt.clear(); axt.axis("off")
    axi.clear(); axi.axis("off")
    if done:
        r = done[-1]; b = r["belief"]
        axt.text(0, 1, "\n".join(textwrap.wrap(f"[{r['decision'].upper()}] {r['reasoning']}", 135)[:4]),
                 va="top", fontsize=8, family="monospace")
        axi.text(0, 1, f"confidence {b['confidence']:.0%}\nentropy {b['entropy']:.2f}\nregion(90%) {b['cells_90pct']} cells\n"
                       f"time left {r['time_left']}", va="top", fontsize=8.5)
    if i >= len(frames) - 10:
        axi.text(0, .4, f"RESULT: {rep['status']}\nat {tuple(rep['inferred_location'])}", va="top", fontsize=10, weight="bold", color="#1f4e79")

out = a.out or str(Path("outputs") / cfg["name"].replace(" ", "_") / "animation.gif")
Path(out).parent.mkdir(parents=True, exist_ok=True)
anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / a.fps)
anim.save(out, writer=PillowWriter(fps=a.fps), dpi=80)
print("saved", out, f"({len(frames)} frames)")
