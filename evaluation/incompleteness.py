import os
import sys
import json
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.planner import run_planner

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PROBLEMS = os.path.join(ROOT, "problems")
COMPLETE = os.path.join(ROOT, "domains", "blocksworld_complete.pddl")

LEVELS = [
    ("complete", []),
    ("slight", ["stack"]),
    ("moderate", ["stack", "unstack"]),
    ("severe", ["stack", "unstack", "pick-up"]),
]


def remove_action(domain_text, action_name):
    marker = f"(:action {action_name}"
    start = domain_text.find(marker)
    if start == -1:
        return domain_text
    depth = 0
    for i in range(start, len(domain_text)):
        if domain_text[i] == "(":
            depth += 1
        elif domain_text[i] == ")":
            depth -= 1
            if depth == 0:
                return domain_text[:start] + domain_text[i + 1:]
    return domain_text


def degraded_domain(removed):
    text = open(COMPLETE).read()
    for name in removed:
        text = remove_action(text, name)
    return text


def evaluate_level(removed, problems):
    text = degraded_domain(removed)
    fd, path = tempfile.mkstemp(suffix=".pddl")
    os.close(fd)
    with open(path, "w") as f:
        f.write(text)
    solved, runtimes = 0, []
    try:
        for p in problems:
            r = run_planner(path, p)
            runtimes.append(r.runtime)
            if r.success:
                solved += 1
    finally:
        os.remove(path)
    return solved, sum(runtimes) / len(runtimes)


def plot(rows, total):
    labels = [f"{r['level']}\n(-{r['removed']} actions)" for r in rows]
    pct = [100 * r["solved"] / total for r in rows]
    rts = [r["avg_runtime"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars = ax1.bar(labels, pct, color="#6baed6", width=0.6)
    for bar, r in zip(bars, rows):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{r['solved']}/{total}", ha="center", fontweight="bold")
    ax1.set_ylabel("tasks solvable (%)")
    ax1.set_ylim(0, 112)
    ax1.set_title("Baseline solvability vs degree of incompleteness")
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.plot([r["level"] for r in rows], rts, "o-", color="#08519c")
    ax2.set_ylabel("avg planning runtime (s)")
    ax2.set_title("Planning runtime vs degree of incompleteness")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "incompleteness.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    problems = sorted(os.path.join(PROBLEMS, f)
                      for f in os.listdir(PROBLEMS) if f.endswith(".pddl"))
    total = len(problems)
    rows = []
    print(f"{'level':<12} {'removed':<28} {'solvable':>9} {'avg runtime(s)':>15}")
    print("-" * 66)
    for name, removed in LEVELS:
        solved, avg_rt = evaluate_level(removed, problems)
        rows.append({"level": name, "removed": len(removed),
                     "removed_actions": removed, "solved": solved,
                     "avg_runtime": round(avg_rt, 4)})
        rm = ", ".join(removed) if removed else "(none)"
        print(f"{name:<12} {rm:<28} {f'{solved}/{total}':>9} {avg_rt:>15.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "incompleteness.json"), "w") as f:
        json.dump({"total": total, "levels": rows}, f, indent=2)
    out = plot(rows, total)
    print(f"\nSaved figure to: {out}")
    print(f"Saved data to:   {os.path.join(RESULTS, 'incompleteness.json')}")
