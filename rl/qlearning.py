import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.blocksworld_sim import BlocksWorld, state

GOAL_REWARD = 100
STEP_PENALTY = -1
TIMEOUT_PENALTY = -10


class QLearningAgent:
    def __init__(self, world, start, goal, alpha=0.5, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99,
                 max_steps=20, seed=0):
        self.world = world
        self.start = start
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.Q = {}

    def _q(self, s, a):
        return self.Q.get((s, a), 0.0)

    def _best_value(self, s):
        actions = self.world.applicable_actions(s)
        if not actions:
            return 0.0
        return max(self._q(s, a) for a in actions)

    def _choose(self, s, actions, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        best = max(self._q(s, a) for a in actions)
        best_actions = [a for a in actions if self._q(s, a) == best]
        return self.rng.choice(best_actions)

    def train(self, episodes=500):
        epsilon = self.epsilon_start
        rewards, lengths, successes, trajectories = [], [], [], []
        for _ in range(episodes):
            s = self.start
            total = 0.0
            steps = 0
            traj = []
            reached = False
            for t in range(self.max_steps):
                actions = self.world.applicable_actions(s)
                if not actions:
                    break
                a = self._choose(s, actions, epsilon)
                s2 = self.world.step(s, a)
                steps += 1
                reached = self.world.goal_reached(s2, self.goal)
                timed_out = (t == self.max_steps - 1) and not reached
                if reached:
                    r = GOAL_REWARD
                elif timed_out:
                    r = STEP_PENALTY + TIMEOUT_PENALTY
                else:
                    r = STEP_PENALTY
                if reached or timed_out:
                    target = r
                else:
                    target = r + self.gamma * self._best_value(s2)
                self.Q[(s, a)] = self._q(s, a) + self.alpha * (target - self._q(s, a))
                traj.append((s, a, s2))
                total += r
                s = s2
                if reached or timed_out:
                    break
            rewards.append(total)
            lengths.append(steps)
            successes.append(reached)
            if reached:
                trajectories.append(traj)
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
        return {"rewards": rewards, "lengths": lengths,
                "successes": successes, "trajectories": trajectories}

    def greedy_trajectory(self):
        s = self.start
        traj = []
        for _ in range(self.max_steps):
            actions = self.world.applicable_actions(s)
            if not actions:
                break
            best = max(self._q(s, a) for a in actions)
            a = next(act for act in actions if self._q(s, act) == best)
            s2 = self.world.step(s, a)
            traj.append((s, a, s2))
            s = s2
            if self.world.goal_reached(s, self.goal):
                return traj, True
        return traj, False


def _ser_state(s):
    return sorted(list(p) for p in s)


def _ser_action(a):
    return list(a)


def _ser_trajectory(traj):
    return [{"state": _ser_state(s), "action": _ser_action(a),
             "next_state": _ser_state(s2)} for (s, a, s2) in traj]


def save_results(stats, greedy_traj, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    training = {"rewards": stats["rewards"], "lengths": stats["lengths"],
                "successes": [bool(x) for x in stats["successes"]]}
    with open(os.path.join(out_dir, "q_training_stats.json"), "w") as f:
        json.dump(training, f, indent=2)
    successful = [_ser_trajectory(t) for t in stats["trajectories"][-20:]]
    with open(os.path.join(out_dir, "trajectories.json"), "w") as f:
        json.dump({"greedy": _ser_trajectory(greedy_traj),
                   "successful": successful}, f, indent=2)


if __name__ == "__main__":
    blocks = ["a", "b"]
    world = BlocksWorld(blocks)
    start = state(("ontable", "a"), ("ontable", "b"),
                  ("clear", "a"), ("clear", "b"), ("handempty",))
    goal = {("on", "a", "b")}

    agent = QLearningAgent(world, start, goal, seed=0)
    episodes = 500
    stats = agent.train(episodes=episodes)

    block = 50
    print(f"Trained {episodes} episodes. Success rate per {block}-episode block:")
    succ = stats["successes"]
    for i in range(0, episodes, block):
        chunk = succ[i:i + block]
        rate = 100.0 * sum(chunk) / len(chunk)
        print(f"  episodes {i:4d}-{i + len(chunk) - 1:4d}: {rate:5.1f}%")

    traj, ok = agent.greedy_trajectory()
    print(f"\nGreedy policy solves the bridge: {ok}")
    if ok:
        plan = [f"({a[0]} {' '.join(a[1:])})" for (_, a, _) in traj]
        print("Learned bridge:", " -> ".join(plan))

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(here), "results")
    save_results(stats, traj, out_dir)
    print(f"\nSaved training stats and trajectories to: {out_dir}")
