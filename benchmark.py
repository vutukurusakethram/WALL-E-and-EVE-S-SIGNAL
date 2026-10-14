#!/usr/bin/env python3
"""Generalisation benchmark over random missions: EIG planner vs greedy baseline.
Usage: python benchmark.py [--n 40] [--out outputs/benchmark]"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from walle.config import deep_merge, DEFAULTS
from walle.runner import run_mission


def random_cfg(seed):
    r = np.random.default_rng(10_000 + seed)
    return deep_merge(DEFAULTS, {
        "name": f"random_{seed}", "seed": seed,
        "map": {"width": 20, "height": 20, "obstacle_density": float(r.choice([0.0, 0.08, 0.15]))},
        "signal": {"coords": [int(r.integers(6, 19)), int(r.integers(4, 19))],
                   "strength": float(r.uniform(0.3, 0.9)), "timestamp": -int(r.integers(0, 15))},
        "budget": {"battery": int(r.choice([50, 70, 100])), "time": 300, "max_recharges": 3},
        "sensor": {"noise": float(r.choice([0.05, 0.08, 0.12])), "spoof_prob": float(r.choice([0.0, 0.05, 0.1]))},
        "clues": {"count": 5, "decoy_prob": float(r.choice([0.2, 0.4]))},
    })


def run_benchmark(n=40, out="outputs/benchmark", progress=None):
    Path(out).mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(n):
        if progress:
            progress(seed, n)
        for pol in ("eig", "greedy"):
            try:
                res = run_mission(random_cfg(seed), None, policy=pol)
            except ValueError:
                continue
            rp = res["report"]
            rows.append({"seed": seed, "policy": pol, "status": rp["status"], "correct": rp["evaluation_only"]["correct"],
                         "confirmed": rp["status"] == "CONFIRMED", "error": rp["evaluation_only"]["error_cells"],
                         "steps": rp["resources"]["steps"], "scans": rp["resources"]["scans"],
                         "recharges": rp["resources"]["recharges_used"], "end": rp["end_reason"]})
    with open(Path(out) / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for pol in ("eig", "greedy"):
        R = [r for r in rows if r["policy"] == pol]
        summ[pol] = {"missions": len(R), "confirmed_rate": float(np.mean([r["confirmed"] for r in R])),
                     "correct_rate": float(np.mean([r["correct"] for r in R])),
                     "mean_error_cells": float(np.mean([r["error"] for r in R])),
                     "mean_steps": float(np.mean([r["steps"] for r in R])),
                     "mean_scans": float(np.mean([r["scans"] for r in R])),
                     "mean_recharges": float(np.mean([r["recharges"] for r in R]))}
    (Path(out) / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="outputs/benchmark")
    a = ap.parse_args()
    print(json.dumps(run_benchmark(a.n, a.out), indent=2))
