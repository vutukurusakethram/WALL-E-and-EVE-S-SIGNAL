#!/usr/bin/env python3
"""Heartbeat-sensor style replay of a mission.
Usage: python animate.py configs/mission_01.yaml [--out file.gif] [--fps 12] [--policy eig|greedy]
(The plain version is still available as animate_classic.py)"""
import argparse
from pathlib import Path
from walle.config import load_config
from walle.anim import make_animation

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--out", default=None)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--policy", choices=["eig", "greedy"], default=None)
a = ap.parse_args()
cfg = load_config(a.config)
out = a.out or str(Path("outputs") / Path(a.config).stem / "animation.gif")
print("saved", make_animation(cfg, out, fps=a.fps, policy=a.policy))
