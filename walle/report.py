"""Final mission report: evidence, inferred location/status, confidence, recommended next step."""
import numpy as np
from .env import bfs_dist


def build_report(info, agent, trace, truth, end_reason, cfg):
    b, s = agent.belief, info["sensor"]
    mv = info["costs"]["move"]
    if agent.confirmed:
        loc, conf, status = agent.confirmed, 1.0, "CONFIRMED"
    else:
        loc = b.peak()
        conf = b.mass_near(loc, s["detect_radius"])
        status = "PROBABLE" if conf >= 0.6 else "POSSIBLE" if conf >= 0.25 else "UNKNOWN"
    eve_status = agent.eve_status or "not observed (unknown until EVE is reached)"

    # evidence: clues and readings judged against the final belief
    clues = []
    for c in agent.clues:
        d = float(np.hypot(c["estimate"][0] - loc[0], c["estimate"][1] - loc[1]))
        c = dict(c, dist_to_inferred=round(d, 2), consistent=bool(d <= 2.5 * info["clue_sigma"]))
        clues.append(c)
    outliers = []
    for r in agent.readings:
        d = float(np.hypot(r["pos"][0] - loc[0], r["pos"][1] - loc[1]))
        expected = float(np.exp(-d / s["length_scale"]))
        if abs(r["reading"] - expected) > 4 * s["noise"]:
            outliers.append(dict(r, expected=round(expected, 3)))
    best = sorted(agent.readings, key=lambda r: -r["reading"])[:3]

    # recommended next step
    d_from_here = bfs_dist(agent.free, agent.pos)[loc[1], loc[0]]
    d_home_loc = agent.home[loc[1], loc[0]]
    round_trip = (d_from_here + d_home_loc) * mv
    at_home = agent.pos == agent.start
    if status == "CONFIRMED":
        if agent.battery >= round_trip + agent.margin:
            rec = (f"Proceed to EVE at {loc}: {int(d_from_here)} moves ({d_from_here * mv:.0f} battery). Battery "
                   f"{agent.battery} covers the trip there and back to the charger with margin.")
        elif agent.recharges_left > 0 or at_home:
            rec = (f"Return to the charger, recharge ({agent.recharges_left} left), then go to EVE at {loc}: the "
                   f"one-way trip from the charger is {int(d_home_loc)} moves, a round trip costs {2 * d_home_loc * mv:.0f} battery.")
        else:
            rec = (f"EVE is at {loc} but the remaining battery ({agent.battery}) cannot cover the round trip "
                   f"({round_trip:.0f}) and no recharges are left. Request a relay / extra battery pack.")
    elif status in ("PROBABLE", "POSSIBLE"):
        rec = (f"Verify with a close-range scan at {loc} ({int(d_from_here)} moves away). "
               + ("Battery is sufficient." if agent.battery >= round_trip + agent.sc + agent.margin
                  else "Recharge first: battery would not cover the round trip."))
    else:
        top = np.argsort(agent._flat())[::-1][:3]
        cells = [(int(agent.cx[i]), int(agent.cy[i])) for i in top]
        rec = f"Evidence is inconclusive. Search the highest-probability unexplored cells next: {cells}."

    err = float(np.hypot(loc[0] - truth["eve_pos"][0], loc[1] - truth["eve_pos"][1]))
    return {
        "mission": cfg["name"], "end_reason": end_reason, "status": status,
        "inferred_location": list(loc), "confidence": round(conf, 4), "eve_status": eve_status,
        "region_cells_90pct": b.region_size(0.9),
        "evidence": {
            "initial_signal": info["signal"], "scans": len(agent.readings),
            "strongest_readings": best, "clues": clues, "outlier_readings_possible_spoofs": outliers},
        "resources": {
            "battery_left": agent.battery, "battery_full": agent.full, "time_left": agent.time_left,
            "steps": len(trace), "scans": len(agent.readings),
            "recharges_used": info["budget"]["max_recharges"] - agent.recharges_left},
        "recommended_next_step": rec,
        "evaluation_only": {"true_eve_position": list(truth["eve_pos"]), "true_eve_status": truth["eve_status"],
                            "error_cells": round(err, 2), "correct": bool(err <= s["detect_radius"]),
                            "decoy_clues_actually_present": sum(1 for c in truth["clues"] if not c["reliable"])},
    }


def to_markdown(rep):
    ev, rs, ge = rep["evidence"], rep["resources"], rep["evaluation_only"]
    L = [f"# Mission report: {rep['mission']}", "",
         f"**Outcome:** {rep['status']}  |  **Ended by:** {rep['end_reason']}", "",
         "| Item | Value |", "|---|---|",
         f"| Inferred EVE location | {tuple(rep['inferred_location'])} |",
         f"| Confidence | {rep['confidence']:.1%} (probability within detection radius of that cell) |",
         f"| 90% credible region | {rep['region_cells_90pct']} cells |",
         f"| EVE status | {rep['eve_status']} |",
         f"| Steps / scans / recharges | {rs['steps']} / {rs['scans']} / {rs['recharges_used']} |",
         f"| Battery left | {rs['battery_left']} / {rs['battery_full']} |",
         f"| Time left | {rs['time_left']} |", "",
         "## Evidence found", "",
         f"- Initial signal: coords {ev['initial_signal']['coords']}, strength {ev['initial_signal']['strength']}",
         "- Strongest scan readings: " + ", ".join(f"{r['reading']:.2f} at {tuple(r['pos'])}" for r in ev["strongest_readings"])]
    if ev["clues"]:
        L += ["", "| Clue | Found at | Points to | Distance to inferred EVE | Verdict |", "|---|---|---|---|---|"]
        for c in ev["clues"]:
            L.append(f"| {c['kind']} | {tuple(c['pos'])} | {tuple(c['estimate'])} | {c['dist_to_inferred']} | "
                     f"{'consistent' if c['consistent'] else 'conflicting (likely decoy)'} |")
    else:
        L.append("- No clues were found.")
    if ev["outlier_readings_possible_spoofs"]:
        L.append("- Possible spoofed readings: " + ", ".join(
            f"{r['reading']:.2f} at {tuple(r['pos'])} (expected about {r['expected']})" for r in ev["outlier_readings_possible_spoofs"]))
    L += ["", "## Recommended next step", "", rep["recommended_next_step"], "",
          "## Evaluation only (ground truth, never shown to the agent)", "",
          f"True EVE position {tuple(ge['true_eve_position'])} ({ge['true_eve_status']}); error {ge['error_cells']} cells; "
          f"correct: **{ge['correct']}**; decoy clues in this world: {ge['decoy_clues_actually_present']}."]
    return "\n".join(L) + "\n"
