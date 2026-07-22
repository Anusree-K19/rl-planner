import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from planner.planner import run_planner

DOMAINS = os.path.join(ROOT, "domains")
REPAIRED = os.path.join(DOMAINS, "blocksworld_repaired.pddl")
PROBLEMS_SCALING = os.path.join(ROOT, "problems_scaling")
RESULTS = os.path.join(ROOT, "results")
IMAGES = os.path.join(ROOT, "images")

BRIDGE = "learned_bridge"

FAMILY_LABEL = {"stacking": "stacking goals",
                "disassembly": "disassembly goals"}


def n_blocks(problem_path):
    text = open(problem_path).read()
    m = re.search(r"\(:objects(.*?)\)", text, re.S)
    if not m:
        return None
    return len(m.group(1).split("-")[0].split())


def family_of(problem_path):
    text = open(problem_path).read()
    gi = text.find("(:goal")
    goal = text[gi:] if gi != -1 else text
    return "stacking" if re.search(r"\(on\s+\w+\s+\w+\)", goal) else "disassembly"


def problem_paths():
    return sorted(os.path.join(PROBLEMS_SCALING, f)
                  for f in os.listdir(PROBLEMS_SCALING) if f.endswith(".pddl"))


def measure(domain, paths):
    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        result = run_planner(domain, path)

        found = result.success
        plan = result.plan or []
        runtime = result.runtime

        rows.append({
            "problem": name,
            "family": family_of(path),
            "blocks": n_blocks(path),
            "solved": bool(found),
            "plan_length": len(plan) if found else None,
            "runtime": runtime,
            "uses_bridge": any(BRIDGE in a for a in plan) if found else False,
        })
        print(f"  {name:<20} {result.status:<12} "
              f"len={rows[-1]['plan_length']} "
              f"bridge={rows[-1]['uses_bridge']}")
    return rows


def plot(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["solved"]:
            groups[r["family"]].append(
                (r["blocks"], r["plan_length"], r["runtime"], r["problem"])
            )

    os.makedirs(IMAGES, exist_ok=True)
    written = []

    for family, data in groups.items():
        data.sort()
        blocks = [d[0] for d in data]
        labels = [d[3] for d in data]
        panels = [("plan length (actions)", [d[1] for d in data]),
                  ("planning runtime (s)", [d[2] for d in data])]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (ylabel, ys) in zip(axes, panels):
            ax.plot(blocks, ys, "o-" if len(blocks) > 1 else "o",
                    linewidth=2, markersize=7)
            for x, y, lab in zip(blocks, ys, labels):
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(7, 7), fontsize=8)
            ax.set_xlabel("number of blocks")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel.split(' (')[0].capitalize()} vs problem "
                         f"size - {FAMILY_LABEL[family]}", fontsize=11)
            ax.set_xticks(sorted(set(blocks)))
            ax.margins(x=0.25, y=0.25)
            if "runtime" in ylabel:
                ax.set_ylim(0, max(ys) * 1.4)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(IMAGES, f"scaling_{family}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    return written


def main():
    if not os.path.isdir(PROBLEMS_SCALING):
        sys.exit(f"missing directory: {PROBLEMS_SCALING}")
    if not os.path.isfile(REPAIRED):
        sys.exit(f"repaired domain not found: {REPAIRED}\n"
                 f"run main.py first so the repaired domain exists.")

    paths = problem_paths()
    print(f"\nProblem-size stress test: {len(paths)} instances "
          f"against the repaired domain\n")
    rows = measure(REPAIRED, paths)

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, "scaling.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    written = plot(rows)

    leaked = [r["problem"] for r in rows
              if r["family"] == "disassembly" and r["uses_bridge"]]
    print(f"\n  wrote {os.path.relpath(csv_path, ROOT)}")
    for p in written:
        print(f"  wrote {os.path.relpath(p, ROOT)}")
    if leaked:
        print(f"\n  WARNING: {BRIDGE} appears in disassembly plans: {leaked}")
    else:
        print(f"\n  regression check passed: {BRIDGE} absent from all "
              f"disassembly plans\n")


if __name__ == "__main__":
    main()
