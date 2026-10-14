"""Heartbeat-sensor style replay of a mission as an animated GIF."""
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle
from .runner import run_mission


def make_animation(cfg, out, fps=12, policy=None):
    res = run_mission(cfg, None, policy=policy)
    env, trace, snaps, rep = res["env"], res["trace"], res["snaps"], res["report"]
    free, start, full = env.free, env.start, cfg["budget"]["battery"]
    H, W = free.shape
    RMAX = 2.2 * cfg["sensor"]["length_scale"]        # how far the ping ring travels
    P = 16                                             # frames per ping

    G, DG, AMB, RED = "#39ff14", "#0b3d0b", "#ffb000", "#ff3030"
    cmap = LinearSegmentedColormap.from_list("hb", ["#000000", "#06330a", "#1fbf2a", G])

    # frame plan: (step, pulse_frame or None)
    frames = [(0, None)] * 6
    for r in trace:
        t, d = r["action"]["type"], r["decision"]
        if t == "scan":
            frames += [(r["step"], k) for k in range(P)] + [(r["step"], None)] * 4
        elif t == "recharge":
            frames += [(r["step"], None)] * 10
        elif t == "declare":
            frames += [(r["step"], None)] * 36
        else:
            frames += [(r["step"], None)] * (4 if d in ("replan", "turn_back") or r["reasoning"].startswith("TURN") else 1)

    def belief_at(step):
        best = snaps[0][1]
        for st, p in snaps:
            if st <= step:
                best = p
        return best

    plt.rcParams["text.color"] = G
    fig = plt.figure(figsize=(9.4, 6.4), facecolor="black")
    ax = fig.add_axes([0.04, 0.22, 0.60, 0.72], facecolor="black")
    axb = fig.add_axes([0.70, 0.56, 0.06, 0.36], facecolor="black")
    axi = fig.add_axes([0.79, 0.56, 0.20, 0.36]); axi.axis("off")
    axs = fig.add_axes([0.70, 0.26, 0.28, 0.20], facecolor="black")
    axt = fig.add_axes([0.04, 0.02, 0.92, 0.15]); axt.axis("off")
    wall_cmap = ListedColormap(["#145214"])

    def style(axx):
        for s in axx.spines.values():
            s.set_color(DG)
        axx.tick_params(colors=MG if False else "#1f8f1f", labelsize=6)

    def draw(i):
        step, k = frames[i]
        reveal = k is None or k >= P // 2
        recs = [r for r in trace if r["step"] < step or (r["step"] == step and reveal)]
        prev = [r for r in trace if r["step"] <= step]
        cur_rec = prev[-1] if prev else None
        ax.clear()
        ax.set_facecolor("black")
        p = belief_at(step if reveal else step - 1)
        ax.imshow(np.where(free, (p / max(p.max(), 1e-12)) ** 0.6, np.nan), cmap=cmap, origin="upper",
                  vmin=0, vmax=1, interpolation="bilinear")
        ax.imshow(np.where(free, np.nan, 1.0), cmap=wall_cmap, origin="upper", vmin=0, vmax=1)
        ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color=DG, lw=.4)
        ax.set_xticks(range(0, W, 5)); ax.set_yticks(range(0, H, 5)); style(ax)

        pts = [start] + [tuple(r["pos_after"]) for r in recs if r["action"]["type"] == "move"]
        ax.plot(*zip(*pts), "-", color="#1f8f1f", lw=1.2, alpha=.8)
        n = 0
        for r in recs:
            if r["action"]["type"] == "scan":
                n += 1
                ax.add_patch(Circle(r["pos_after"], .32, fill=False, ec=G, lw=1.2, alpha=.7))
                for cl in r["observation"]["scan"]["clues"]:
                    ax.plot(*cl["pos"], "D", mfc="none", mec=AMB, ms=7, mew=1.5)
        ax.plot(*start, "^", color="#7dffb0", ms=9)
        pos = tuple(cur_rec["pos_after"]) if cur_rec else start

        # --- the ping: expanding rings, brighter when the signal return is strong ---
        if k is not None:
            reading = float(np.clip(trace[step - 1]["observation"]["scan"]["reading"], 0, 1))
            base = 0.45 + 0.55 * reading
            for lag in (0.0, 0.12, 0.24):
                f = k / (P - 1) - lag
                if f > 0:
                    ax.add_patch(Circle(pos, f * RMAX, fill=False, ec=G, lw=3.0 - 2 * f,
                                        alpha=max(0.0, (1 - f)) * base * (1 - lag * 2)))
        ax.plot(*pos, "o", color="white", ms=8, mec=G, mew=2, zorder=6)

        # --- heartbeat blip at the current best estimate ---
        conf = cur_rec["belief"]["confidence"] if cur_rec else 0
        pk = np.unravel_index(np.argmax(p), p.shape)
        phase = (i % 14) / 14.0                      # double thump: lub-dub
        thump = max(0, np.sin(phase * 2 * np.pi)) ** 3 + 0.6 * max(0, np.sin((phase - .25) * 2 * np.pi)) ** 3
        locked = rep["status"] == "CONFIRMED" and cur_rec is not None and cur_rec["decision"] == "declare"
        if locked:
            ex, ey = rep["inferred_location"]
            for j, rr in enumerate((0.6, 1.1, 1.7)):
                ax.add_patch(Circle((ex, ey), rr + thump * (0.5 + .3 * j), fill=False, ec=RED, lw=2, alpha=.9 - .25 * j))
            ax.plot(ex, ey, "o", color=RED, ms=10, mec="white")
            ax.annotate("EVE LOCKED", (ex, ey), xytext=(0, 18), textcoords="offset points", ha="center", color=RED, fontsize=9, weight="bold")
        elif cur_rec is not None and n > 0:
            col = G if conf >= .25 else "#1fbf2a"
            ax.add_patch(Circle((pk[1], pk[0]), .45 + thump * (0.4 + conf), fill=False, ec=col, lw=1.8, alpha=.95))
            ax.plot(pk[1], pk[0], "o", color=col, ms=4 + 8 * conf)
        ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)
        ax.set_title(f"HEARTBEAT SENSOR  //  {cfg['name']}  //  step {step}", fontsize=9, color=G, loc="left")

        # --- battery ---
        batt = cur_rec["battery"] if cur_rec else full
        axb.clear(); axb.set_facecolor("black")
        axb.bar([0], [batt], color=G if batt > full * .4 else AMB if batt > full * .2 else RED, width=.8)
        axb.set_ylim(0, full); axb.set_xticks([]); style(axb)
        axb.set_title("BATT", fontsize=7, color=G)

        # --- stats ---
        axi.clear(); axi.axis("off")
        if cur_rec:
            b = cur_rec["belief"]
            axi.text(0, 1, f"{batt}/{full}\nCONF {b['confidence']:.0%}\nENTROPY {max(0, b["entropy"]):.2f}\nZONE {b['cells_90pct']} cells\nTIME {cur_rec['time_left']}",
                     va="top", fontsize=8.5, family="monospace", color=G, linespacing=1.5)

        # --- signal-return strip (one point per ping) ---
        axs.clear(); axs.set_facecolor("black"); style(axs)
        rd = [r["observation"]["scan"]["reading"] for r in recs if r["action"]["type"] == "scan"]
        if rd:
            xs = np.arange(1, len(rd) + 1)
            axs.plot(xs, rd, "-o", color=G, lw=1.6, ms=3)
            axs.fill_between(xs, 0, rd, color=G, alpha=.15)
        axs.set_ylim(-.15, 1.1); axs.set_xlim(.5, max(6, len(rd)) + .5)
        axs.set_title("SIGNAL RETURN PER PING", fontsize=7, color=G); axs.grid(color=DG, lw=.4)

        # --- caption ---
        axt.clear(); axt.axis("off")
        if k is not None:
            txt = f"[PING #{n + (0 if reveal else 1)}] signal return {np.clip(trace[step - 1]['observation']['scan']['reading'], -9, 9):+.2f}"
        elif cur_rec:
            txt = f"[{cur_rec['decision'].upper()}] {cur_rec['reasoning']}"
        else:
            txt = "[STANDBY] waiting for first ping"
        axt.text(0, 1, "\n".join(textwrap.wrap(txt, 130)[:4]), va="top", fontsize=8, family="monospace", color=G)
        if i >= len(frames) - 30:
            axi.text(0, .1, f"RESULT\n{rep['status']}\n{tuple(rep['inferred_location'])}", va="bottom", fontsize=10,
                     weight="bold", color=RED if rep["status"] == "CONFIRMED" else AMB, family="monospace")


    Path(out).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps).save(
        str(out), writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    return str(out)
