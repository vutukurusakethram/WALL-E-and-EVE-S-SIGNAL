#!/usr/bin/env python3
"""WALL-E searches for EVE.

* Press Run on this file (or `python run.py`) -> opens the web control panel in your browser.
* Command line still works:  python run.py configs/mission_01.yaml [--out DIR] [--policy eig|greedy] [-v]
"""
import sys

if len(sys.argv) == 1:
    from walle.webui import serve
    serve()
    sys.exit(0)

import argparse
from pathlib import Path
from walle.config import load_config
from walle.runner import run_mission

ap = argparse.ArgumentParser(description="WALL-E searches for EVE")
ap.add_argument("config")
ap.add_argument("--out", default=None, help="output folder (default outputs/<config name>)")
ap.add_argument("--policy", choices=["eig", "greedy"], default=None)
ap.add_argument("-v", "--verbose", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
out = a.out or str(Path("outputs") / Path(a.config).stem)
res = run_mission(cfg, out, policy=a.policy, verbose=a.verbose)
r = res["report"]
print(f"\n{r['mission']}: {r['status']} at {tuple(r['inferred_location'])} "
      f"(confidence {r['confidence']:.0%}, ended by {r['end_reason']})")
print("Next step:", r["recommended_next_step"])
print("Files written to", out)
