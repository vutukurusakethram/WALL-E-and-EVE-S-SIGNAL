"""Mission config loading (JSON or YAML), defaults and validation."""
import copy
import json
import math
from pathlib import Path

DEFAULTS = {
    "name": "mission",
    "seed": 0,
    "map": {"width": 20, "height": 20, "obstacles": [], "obstacle_density": 0.0},
    "start": [0, 0],                      # also the charger location
    "eve": {"position": None, "status": "active"},   # HIDDEN truth. None -> sampled from seed
    "signal": {                           # the only thing the agent knows about EVE at t=0
        "coords": None,                   # [x, y] estimate ...
        "bearing_deg": None,              # ... or bearing (0=east, 90=south) + range
        "range": None,
        "strength": 0.6,                  # 0..1, higher = more trustworthy / precise
        "timestamp": 0,                   # when it was received (older = less precise)
    },
    "budget": {"battery": 100, "time": 300, "max_recharges": 3},
    "costs": {"move": 1, "scan": 3, "move_time": 1, "scan_time": 2, "recharge_time": 10},
    "sensor": {
        "length_scale": 6.0,              # signal = exp(-distance / length_scale)
        "noise": 0.08,                    # gaussian noise on each reading
        "spoof_prob": 0.0,                # chance a reading is a spoofed random value
        "scan_radius": 3.0,               # clues within this radius are revealed by a scan
        "detect_radius": 1.5,             # EVE is visually confirmed within this radius
    },
    "clues": {"count": 5, "decoy_prob": 0.2, "sigma": 1.5, "list": None},
    "agent": {"policy": "eig", "safety_margin": 6, "stop_confidence": 0.95, "decoy_prior": 0.25},
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    cfg = deep_merge(DEFAULTS, raw)
    if "name" not in raw:
        cfg["name"] = path.stem
    validate(cfg)
    return cfg


def validate(cfg):
    W, H = cfg["map"]["width"], cfg["map"]["height"]
    sx, sy = cfg["start"]
    if not (0 <= sx < W and 0 <= sy < H):
        raise ValueError("start is outside the map")
    for x, y in cfg["map"]["obstacles"]:
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"obstacle {(x, y)} is outside the map")
    if cfg["budget"]["battery"] <= 0 or cfg["budget"]["time"] <= 0:
        raise ValueError("battery and time budgets must be positive")
    if not 0 <= cfg["signal"]["strength"] <= 1:
        raise ValueError("signal.strength must be in [0, 1]")
    sig = cfg["signal"]
    if sig["coords"] is None and (sig["bearing_deg"] is None or sig["range"] is None):
        raise ValueError("signal needs coords OR bearing_deg + range")


def resolve_signal_coords(cfg):
    sig = cfg["signal"]
    W, H = cfg["map"]["width"], cfg["map"]["height"]
    if sig["coords"] is not None:
        x, y = sig["coords"]
    else:
        a = math.radians(sig["bearing_deg"])
        x = cfg["start"][0] + sig["range"] * math.cos(a)
        y = cfg["start"][1] + sig["range"] * math.sin(a)
    return [min(max(float(x), 0), W - 1), min(max(float(y), 0), H - 1)]


def signal_sigma(sig, now=0.0):
    """Prior uncertainty (in cells) of the initial signal: weaker/older = wider."""
    s = min(max(sig["strength"], 0.0), 1.0)
    age = max(0.0, now - sig.get("timestamp", 0))
    return (2.0 + 8.0 * (1.0 - s)) * (1.0 + 0.03 * age)
