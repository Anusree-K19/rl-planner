import os
import sys
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.blocksworld_sim import BlocksWorld, state
from rl.qlearning import QLearningAgent

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

EPISODES = 500
MAX_STEPS = 50


class RandomAgent:
    def __init__(self, world, start, goal, max_steps=MAX_STEPS, seed=0):
        self.world = world
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.rng = random.Random(seed)

    def run_episode(self):
        s = self.start
        for t in range(self.max_steps):
            actions = self.world.applicable_actions(s)
            if not actions:
                return False, t
            a = self.rng.choice(actions)
            s = self.world.step(s, a)
            if self.world.goal_reached(s, self.goal):
                return True, t + 1
        return False, self.max_steps

    def run(self, episodes):
        successes, lengths = [], []
        for _ in range(episodes):
            ok, steps = self.run_episode()
            successes.append(1 if ok else 0)
            if ok:
                lengths.append(steps)
        return successes, lengths


def make_flat_start(blocks):
    preds = [("handempty",)]
    for b in blocks:
        preds.append(("ontable", b))
        preds.append(("clear", b))
    return state(*preds)


def rolling(xs, w):
    return [100 * sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1])
            for i in range(len(xs))]


def run_size(blocks):
    world = BlocksWorld(blocks)
    start = make_flat_start(blocks)
    goal = {("on", blocks[0], blocks[1])}

    rl = QLearningAgent(world, start, goal, max_steps=MAX_STEPS, seed=0)
    rl_stats = rl.train(episodes=EPISODES)
    rl_succ = [1 if s else 0 for s in rl_stats["successes"]]
    traj, ok = rl.greedy_trajectory()
    rl_len = len(traj) if ok else None

    rnd = RandomAgent(world, start, goal, seed=0)
    rnd_succ, rnd_lengths = rnd.run(EPISODES)
    rnd_avg = sum(rnd_lengths) / len(rnd_lengths) if rnd_lengths else None

    return {"blocks": len(blocks),
            "rl_final_success": 100 * sum(rl_succ[-50:]) / 50,
            "rl_solution_length": rl_len,
            "random_success": 100 * sum(rnd_succ) / len(rnd_succ),
            "random_avg_length": round(rnd_avg, 1) if rnd_avg else None,
            "_rl_succ": rl_succ, "_rnd_succ": rnd_succ}


def plot_learning_comparison(row):
    eps = list(range(1, EPISODES + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, rolling(row["_rl_succ"], 20), color="#238b45", linewidth=2.2,
            label="Q-learning (RL)")
    ax.plot(eps, rolling(row["_rnd_succ"], 20), color="#cb181d", linewidth=2.2,
            label="random exploration")
    ax.set_xlabel("episode")
    ax.set_ylabel("success rate (%)  [rolling, 20]")
    ax.set_ylim(-5, 105)
    ax.set_title(f"RL vs random: reaching the bridge ({row['blocks']} blocks)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "baseline_comparison.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    sizes = [["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"]]
    rows = [run_size(b) for b in sizes]

    print(f"{'blocks':>6}  {'RL success':>11}  {'RL steps':>9}  "
          f"{'random success':>15}  {'random steps':>13}")
    print("-" * 64)
    for r in rows:
        print(f"{r['blocks']:>6}  {r['rl_final_success']:>10.0f}%  "
              f"{str(r['rl_solution_length']):>9}  {r['random_success']:>14.0f}%  "
              f"{str(r['random_avg_length']):>13}")

    mid = next(r for r in rows if r["blocks"] == 3)
    fig_path = plot_learning_comparison(mid)

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(os.path.join(RESULTS, "baselines.json"), "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved figure to: {fig_path}")
    print(f"Saved data to:   {os.path.join(RESULTS, 'baselines.json')}")
