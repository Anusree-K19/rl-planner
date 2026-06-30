import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.blocksworld_sim import BlocksWorld, state
from rl.qlearning import QLearningAgent, _ser_trajectory

BLOCKS = ["a", "b", "c", "d"]
GOAL = {("on", "a", "b")}


def configs():
    c1 = state(("ontable", "a"), ("ontable", "b"), ("ontable", "c"), ("ontable", "d"),
               ("clear", "a"), ("clear", "b"), ("clear", "c"), ("clear", "d"),
               ("handempty",))
    c2 = state(("ontable", "a"), ("on", "b", "c"), ("ontable", "c"), ("ontable", "d"),
               ("clear", "a"), ("clear", "b"), ("clear", "d"), ("handempty",))
    c3 = state(("ontable", "a"), ("on", "b", "c"), ("on", "c", "d"), ("ontable", "d"),
               ("clear", "a"), ("clear", "b"), ("handempty",))
    return [c1, c2, c3]


if __name__ == "__main__":
    world = BlocksWorld(BLOCKS)
    starts = configs()
    trajectories = []
    print("Training the bridge from multiple starting configurations:\n")
    for i, start in enumerate(starts, 1):
        agent = QLearningAgent(world, start, GOAL, seed=i)
        agent.train(episodes=800)
        traj, ok = agent.greedy_trajectory()
        plan = [f"({a[0]} {' '.join(a[1:])})" for (_, a, _) in traj]
        b_pos = "b on table" if ("ontable", "b") in start else "b on another block"
        print(f"  config {i} ({b_pos}): solved={ok}  bridge={' -> '.join(plan)}")
        if ok:
            trajectories.append(traj)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(here), "results")
    os.makedirs(out_dir, exist_ok=True)
    payload = {"greedy": _ser_trajectory(trajectories[0]),
               "successful": [_ser_trajectory(t) for t in trajectories[1:]]}
    with open(os.path.join(out_dir, "trajectories.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(trajectories)} trajectories to: "
          f"{os.path.join(out_dir, 'trajectories.json')}")
