"""Runs one mission end-to-end and writes trace / report / visualisations."""
import json
from pathlib import Path
import numpy as np

from .agent import Agent
from .env import Environment
from .report import build_report, to_markdown
from . import viz


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def run_mission(cfg, outdir=None, policy=None, max_steps=1500, verbose=False):
    env = Environment(cfg)
    info = env.public_info()
    agent = Agent(info, cfg["agent"], policy)
    trace, snaps = [], [(0, agent.belief.p.copy())]
    dr = info["sensor"]["detect_radius"]
    step = 0
    while not env.done and step < max_steps:
        action, tag, reason = agent.act()
        obs = env.step(action)
        agent.observe(action, obs)
        step += 1
        if action["type"] == "scan" and obs["ok"]:
            snaps.append((step, agent.belief.p.copy()))
        rec = {"step": step, "time_left": obs["time_left"], "battery": obs["battery"],
               "pos_after": obs["pos"], "action": action, "decision": tag, "reasoning": reason,
               "observation": {"ok": obs["ok"], "msg": obs["msg"], "scan": obs["scan"]},
               "belief": agent.belief.summary(dr)}
        trace.append(_clean(rec))
        if verbose:
            print(f"[{step:3d}] {action['type']:<8} batt={obs['battery']:<4} pos={tuple(obs['pos'])}  {tag}")
    if not env.done:
        env.done, env.end_reason = True, "step_limit"
    snaps.append((step, agent.belief.p.copy()))

    report = build_report(info, agent, trace, env.truth(), env.end_reason, cfg)
    result = {"report": report, "trace": trace, "agent": agent, "env": env}
    if outdir:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config_used.json").write_text(json.dumps(_clean(cfg), indent=2))
        with open(out / "trace.jsonl", "w") as f:
            for r in trace:
                f.write(json.dumps(r) + "\n")
        (out / "report.json").write_text(json.dumps(_clean(report), indent=2))
        (out / "report.md").write_text(to_markdown(report))
        path_cells = [tuple(r["pos_after"]) for r in trace if r["action"]["type"] == "move"]
        txt = viz.ascii_map(info["free"], agent.belief.p, path_cells, agent.scan_cells, agent.clues,
                            agent.pos, info["start"], eve_true=env.eve)
        (out / "ascii_final.txt").write_text(txt + "\n")
        clues_f = report["evidence"]["clues"]
        viz.make_figure(out / "mission_overview.png", info["free"], snaps, trace, clues_f, info["start"],
                        env.eve, f"{cfg['name']}  -  {report['status']} at {tuple(report['inferred_location'])}, "
                        f"confidence {report['confidence']:.0%}  (policy: {agent.policy})", dr)
    return result
