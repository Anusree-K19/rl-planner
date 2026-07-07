import os
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PROBLEMS = os.path.join(ROOT, "problems")


def _load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _rolling(xs, w):
    out = []
    for i in range(len(xs)):
        window = xs[max(0, i - w + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def _block_count(name):
    path = os.path.join(PROBLEMS, name + ".pddl")
    if not os.path.exists(path):
        return None
    m = re.search(r"\(:objects(.*?)\)", open(path).read(), re.S)
    return len(m.group(1).split()) if m else None


def plot_learning_curve(stats):
    rewards = stats["rewards"]
    successes = [1 if s else 0 for s in stats["successes"]]
    eps = list(range(1, len(rewards) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(eps, rewards, color="#9ecae1", linewidth=0.8, label="per-episode reward")
    ax1.plot(eps, _rolling(rewards, 20), color="#08519c", linewidth=2.2,
             label="rolling mean (20)")
    ax1.set_ylabel("reward (return)")
    ax1.set_title("RL learning curve: reward per episode")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax2.plot(eps, [100 * x for x in _rolling(successes, 20)],
             color="#238b45", linewidth=2.2)
    ax2.set_ylabel("success rate (%)")
    ax2.set_xlabel("episode")
    ax2.set_title("Rolling success rate (20-episode window)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "learning_curve.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_success(before, after):
    total = len(before)
    nb = sum(1 for r in before if r["status"] == "solved")
    na = sum(1 for r in after if r["status"] == "solved")
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["before repair", "after repair"],
                  [100 * nb / total, 100 * na / total],
                  color=["#fb6a4a", "#41ab5d"], width=0.5)
    for bar, n in zip(bars, [nb, na]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{n}/{total}", ha="center", fontweight="bold")
    ax.set_ylabel("tasks solved (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Solvability before vs after domain repair")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "success_comparison.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_before_after_metrics(before, after):
    names = [r["problem"] for r in before]
    bmap = {r["problem"]: r for r in before}
    amap = {r["problem"]: r for r in after}
    blen = [bmap[n]["length"] or 0 for n in names]
    alen = [amap[n]["length"] or 0 for n in names]
    brt = [bmap[n]["runtime"] for n in names]
    art = [amap[n]["runtime"] for n in names]
    x = range(len(names))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar([i - w / 2 for i in x], blen, w, label="before", color="#fb6a4a")
    ax1.bar([i + w / 2 for i in x], alen, w, label="after", color="#41ab5d")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("plan length (actions)")
    ax1.set_title("Plan length (0 = unsolved)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.bar([i - w / 2 for i in x], brt, w, label="before", color="#fb6a4a")
    ax2.bar([i + w / 2 for i in x], art, w, label="after", color="#41ab5d")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("planning runtime (s)")
    ax2.set_title("Planning runtime")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "before_after_metrics.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_scaling(after):
    pts = []
    for r in after:
        if r["status"] == "solved":
            n = _block_count(r["problem"])
            if n is not None:
                pts.append((n, r["length"], r["runtime"], r["problem"]))
    pts.sort()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#08519c")
    for p in pts:
        ax1.annotate(p[3], (p[0], p[1]), textcoords="offset points", xytext=(5, 5),
                     fontsize=8)
    ax1.set_xlabel("number of blocks")
    ax1.set_ylabel("plan length (actions)")
    ax1.set_title("Plan length vs problem size (after repair)")
    ax1.grid(True, alpha=0.3)
    ax2.plot([p[0] for p in pts], [p[2] for p in pts], "o-", color="#238b45")
    ax2.set_xlabel("number of blocks")
    ax2.set_ylabel("planning runtime (s)")
    ax2.set_title("Runtime vs problem size (after repair)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "scaling.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def print_summary(before, after):
    total = len(before)
    nb = sum(1 for r in before if r["status"] == "solved")
    na = sum(1 for r in after if r["status"] == "solved")
    op = os.path.join(RESULTS, "learned_operator.pddl")
    macros = open(op).read().count("(:action") if os.path.exists(op) else 0
    amap = {r["problem"]: r for r in after}

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Success rate before repair: {nb}/{total} ({100 * nb / total:.0f}%)")
    print(f"Success rate after repair:  {na}/{total} ({100 * na / total:.0f}%)")
    print(f"Macro-actions added:        {macros}")
    print()
    print(f"{'problem'.ljust(14)} {'blocks':>6}  {'after':>11}  "
          f"{'length':>6}  {'runtime(s)':>10}")
    print("-" * 56)
    for r in before:
        n = r["problem"]
        a = amap[n]
        length = a["length"] if a["status"] == "solved" else "-"
        print(f"{n.ljust(14)} {str(_block_count(n)):>6}  "
              f"{a['status']:>11}  {str(length):>6}  {a['runtime']:>10.3f}")
if __name__ == "__main__":
    evaluation = _load("evaluation.json")
    stats = _load("q_training_stats.json")
    if evaluation is None:
        raise SystemExit("results/evaluation.json not found - run evaluate.py first")
    before, after = evaluation["before"], evaluation["after"]
    made = []
    if stats is not None:
        made.append(plot_learning_curve(stats))
    else:
        print("(skipping learning curve: q_training_stats.json not found - "
              "run rl/qlearning.py)")
    made.append(plot_success(before, after))
    made.append(plot_before_after_metrics(before, after))
    made.append(plot_scaling(after))
    print_summary(before, after)
    print("\nFigures written:")
    for m in made:
        print("  " + m)
