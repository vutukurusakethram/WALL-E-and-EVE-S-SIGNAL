"""Visualisation: ASCII map and a 4-panel matplotlib figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ascii_map(free, belief, path, scans, clues, agent_pos, start, eve_true=None):
    H, W = free.shape
    bmax = belief.max() if belief.max() > 0 else 1.0
    grid = [["." for _ in range(W)] for _ in range(H)]
    for y in range(H):
        for x in range(W):
            if not free[y, x]:
                grid[y][x] = "#"
            else:
                lvl = int(9 * belief[y, x] / bmax)
                grid[y][x] = str(lvl) if lvl >= 1 else "."
    for x, y in path:
        grid[y][x] = "*"
    for x, y in scans:
        grid[y][x] = "S"
    for cl in clues:
        grid[cl["pos"][1]][cl["pos"][0]] = "?"
    if eve_true is not None:
        grid[eve_true[1]][eve_true[0]] = "E"
    grid[start[1]][start[0]] = "C"
    grid[agent_pos[1]][agent_pos[0]] = "W"
    lines = ["    " + " ".join(f"{x % 10}" for x in range(W))]
    for y in range(H):
        lines.append(f"{y:3d} " + " ".join(grid[y]))
    legend = ("Legend: # wall | C charger | W WALL-E (final) | * path | S scan spot | ? clue found | "
              "E EVE true position (evaluation only) | 1-9 belief level (9 = most likely), . ~0")
    return "\n".join(lines) + "\n" + legend


def make_figure(outfile, free, snapshots, trace, clues_found, start, eve_true, title, detect_r=1.5, truth_clues=None):
    H, W = free.shape
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    ax_prior, ax_final, ax_path, ax_batt = axes.ravel()
    cmap = plt.get_cmap("magma").copy()

    def heat(ax, p, ttl):
        q = np.where(free, p, np.nan)
        ax.imshow(q, cmap=cmap, origin="upper", interpolation="nearest")
        ax.imshow(np.where(free, np.nan, 1.0), cmap="Greys", origin="upper", vmin=0, vmax=1.5)
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks(range(0, W, 2)); ax.set_yticks(range(0, H, 2))
        ax.tick_params(labelsize=7)

    # 1. prior
    heat(ax_prior, snapshots[0][1], "1. Prior belief (from the initial signal)")
    ax_prior.plot(*start, "c^", ms=11, mec="k", label="charger / start")
    ax_prior.legend(loc="upper right", fontsize=8)

    # 2. belief just before the last scan (the final scan often collapses it to a single cell)
    pre = snapshots[-3][1] if len(snapshots) >= 3 else snapshots[-1][1]
    heat(ax_final, pre, "2. Belief just before the final scan")
    ax_final.plot(*eve_true, "*", color="lime", ms=16, mec="k", label="EVE (true, evaluation only)")
    ax_final.legend(loc="upper right", fontsize=8)

    # 3. path + strategy
    heat(ax_path, pre * 0.5, "3. Path, scans, clues and strategy changes")
    pts = [tuple(start)] + [tuple(r["pos_after"]) for r in trace if r["action"]["type"] == "move"]
    xs, ys = zip(*pts)
    ax_path.plot(xs, ys, "-", color="white", lw=1.6, alpha=.9)
    ax_path.plot(*start, "c^", ms=11, mec="k", label="charger")
    k = 0
    for r in trace:
        a = r["action"]["type"]
        px, py = r["pos_after"]
        if a == "scan":
            k += 1
            ax_path.plot(px, py, "o", color="orange", ms=9, mec="k")
            ax_path.annotate(str(k), (px, py), color="k", fontsize=6.5, ha="center", va="center", weight="bold")
        elif r["decision"] == "turn_back" or (a == "move" and r["reasoning"].startswith("TURN BACK")):
            ax_path.plot(px, py, "v", color="red", ms=10, mec="k")
        elif a == "recharge":
            ax_path.plot(px, py, "s", color="yellow", ms=10, mec="k")
    for cl in clues_found:
        ok = cl.get("consistent", True)
        ax_path.plot(*cl["pos"], "D", color="#2ecc71" if ok else "#e74c3c", ms=8, mec="k")
    ax_path.plot(*eve_true, "*", color="lime", ms=16, mec="k")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color="w", mfc="orange", mec="k", ms=9, label="scan (numbered)"),
               Line2D([], [], marker="v", color="w", mfc="red", mec="k", ms=9, label="turn back"),
               Line2D([], [], marker="s", color="w", mfc="yellow", mec="k", ms=9, label="recharge"),
               Line2D([], [], marker="D", color="w", mfc="#2ecc71", mec="k", ms=8, label="clue (consistent)"),
               Line2D([], [], marker="D", color="w", mfc="#e74c3c", mec="k", ms=8, label="clue (conflicting)"),
               Line2D([], [], marker="*", color="w", mfc="lime", mec="k", ms=13, label="EVE (true)")]
    ax_path.legend(handles=handles, loc="lower right", fontsize=7, framealpha=.85)

    # 4. battery trace
    steps = [r["step"] for r in trace]
    batt = [r["battery"] for r in trace]
    ax_batt.plot(steps, batt, color="#1f77b4", lw=2)
    ax_batt.set_title("4. Battery over time (dips = moving/scanning, jumps = recharge)", fontsize=11)
    ax_batt.set_xlabel("step"); ax_batt.set_ylabel("battery"); ax_batt.grid(alpha=.3)
    ax2 = ax_batt.twinx()
    ax2.plot(steps, [r["belief"]["entropy"] for r in trace], color="#d62728", lw=1.5, ls="--")
    ax2.set_ylabel("belief entropy (nats)  [red dashed]", color="#d62728")
    for r in trace:
        if r["action"]["type"] == "scan":
            ax_batt.axvline(r["step"], color="orange", alpha=.25)
        if r["action"]["type"] == "recharge":
            ax_batt.axvline(r["step"], color="gold", lw=2.5)
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
