import numpy as np
from walle.config import load_config
from walle.env import Environment
from walle.belief import Belief
from walle.runner import run_mission


def test_belief_normalised_and_signal_update_moves_peak():
    free = np.ones((10, 10), bool)
    b = Belief(free, [5, 5], 3.0)
    assert abs(b.p.sum() - 1) < 1e-9
    # strong reading (~1.0) at (2,2) -> EVE very close to (2,2)
    b.update_signal((2, 2), 0.95, 6.0, 0.08, 0.0)
    pk = b.peak()
    assert abs(pk[0] - 2) <= 1 and abs(pk[1] - 2) <= 1


def test_env_hides_truth_and_charges_costs():
    cfg = load_config("configs/mission_01.yaml")
    env = Environment(cfg)
    info = env.public_info()
    assert "eve" not in info and "clues" not in info
    o = env.step({"type": "move", "to": [1, 0]})
    assert o["ok"] and o["battery"] == cfg["budget"]["battery"] - cfg["costs"]["move"]
    o = env.step({"type": "move", "to": [5, 5]})
    assert not o["ok"]


def test_agent_never_strands_and_report_is_complete():
    for path in ["configs/mission_01.yaml", "configs/mission_02_walls_noise_decoys.yaml", "configs/mission_03_bearing.json"]:
        res = run_mission(load_config(path))
        assert res["report"]["end_reason"] != "stranded"
        assert res["report"]["recommended_next_step"]
        assert len(res["trace"]) > 0 and "reasoning" in res["trace"][0]


if __name__ == "__main__":
    test_belief_normalised_and_signal_update_moves_peak()
    test_env_hides_truth_and_charges_costs()
    test_agent_never_strands_and_report_is_complete()
    print("all tests passed")
