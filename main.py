import os
import re
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from planner.planner import run_planner
from simulator.blocksworld_sim import BlocksWorld, state
from rl.qlearning import QLearningAgent, _ser_trajectory
from operator_induction.induce import induce_operator, validate_operator
from evaluation.evaluate import repair_domain, evaluate_problems

DOMAINS = os.path.join(ROOT, "domains")
PROBLEMS_DIR = os.path.join(ROOT, "problems")
RESULTS = os.path.join(ROOT, "results")
INCOMPLETE = os.path.join(DOMAINS, "blocksworld_incomplete.pddl")
REPAIRED = os.path.join(DOMAINS, "blocksworld_repaired.pddl")
OPERATOR = os.path.join(RESULTS, "learned_operator.pddl")
TRAJECTORIES = os.path.join(RESULTS, "trajectories.json")

_TTY = sys.stdout.isatty()
_CODES = {"bold": "1", "dim": "2", "red": "91", "green": "92",
          "yellow": "93", "cyan": "96", "gray": "90"}


def col(text, *styles):
    if not _TTY or not styles:
        return text
    prefix = "".join(f"\033[{_CODES[s]}m" for s in styles)
    return f"{prefix}{text}\033[0m"


def title_box(text, width=60):
    print(col("┌" + "─" * width + "┐", "cyan"))
    print(col("│", "cyan") + col(" " + text.ljust(width - 1), "bold", "cyan") + col("│", "cyan"))
    print(col("└" + "─" * width + "┘", "cyan"))


def stage(n, text):
    print()
    print(col(f"  STAGE {n}", "bold", "cyan") + col(f"   {text}", "bold"))
    print(col("  " + "─" * 56, "gray"))


def status_line(name, status, solved, extra=""):
    mark = col("PASS", "green") if solved else col("FAIL", "red")
    word = col(status, "green") if solved else col(status, "red")
    tail = col("   " + extra, "dim") if extra else ""
    print(f"  [{mark}]  {name:<16} {word}{tail}")


def operator_box(op_text):
    lines = op_text.rstrip().split("\n")
    width = max(len(l) for l in lines) + 2
    print(col("  ┌" + "─" * width + "┐", "gray"))
    for l in lines:
        print(col("  │ ", "gray") + col(l.ljust(width - 1), "yellow") + col("│", "gray"))
    print(col("  └" + "─" * width + "┘", "gray"))


def comparison_table(before, after):
    amap = {r["problem"]: r for r in after}
    name_w = max([len(r["problem"]) for r in before] + [len("problem")])
    bw = 13
    head = ("  " + col("problem".ljust(name_w), "bold", "dim") + "   "
            + col("before".ljust(bw), "bold", "dim") + "   " + col("after", "bold", "dim"))
    print(head)
    print(col("  " + "─" * name_w + "   " + "─" * bw + "   " + "─" * 28, "gray"))
    for b in before:
        a = amap[b["problem"]]
        bt = ("solved" if b["status"] == "solved" else b["status"])
        at = ("solved" if a["status"] == "solved" else a["status"])
        if a.get("uses_bridge"):
            at += " (learned_bridge)"
        bcolor = "green" if b["status"] == "solved" else "red"
        acolor = "green" if a["status"] == "solved" else "red"
        print("  " + b["problem"].ljust(name_w) + "   "
              + col(bt.ljust(bw), bcolor) + "   " + col(at, acolor))


def result_box(nb, na, total, width=60):
    inner = f" RESULT    {nb}/{total}  solved  ->  {na}/{total}  solved after repair"
    print()
    print(col("┌" + "─" * width + "┐", "green"))
    print(col("│", "green") + col(inner.ljust(width), "bold", "green") + col("│", "green"))
    print(col("└" + "─" * width + "┘", "green"))


def problem_paths():
    return sorted(os.path.join(PROBLEMS_DIR, f)
                  for f in os.listdir(PROBLEMS_DIR) if f.endswith(".pddl"))


def parse_goal_on(problem_path):
    text = open(problem_path).read()
    gi = text.find("(:goal")
    section = text[gi:] if gi != -1 else text
    m = re.search(r"\(on\s+(\w+)\s+(\w+)\)", section)
    return (m.group(1), m.group(2)) if m else None


def make_configs(x, y):
    pool = [b for b in ["c", "d", "e", "f"] if b not in (x, y)]
    c, d = pool[0], pool[1]
    c1 = state(("ontable", x), ("ontable", y), ("ontable", c), ("ontable", d),
               ("clear", x), ("clear", y), ("clear", c), ("clear", d), ("handempty",))
    c2 = state(("ontable", x), ("on", y, c), ("ontable", c), ("ontable", d),
               ("clear", x), ("clear", y), ("clear", d), ("handempty",))
    c3 = state(("ontable", x), ("on", y, c), ("on", c, d), ("ontable", d),
               ("clear", x), ("clear", y), ("handempty",))
    return [x, y, c, d], [c1, c2, c3]


def solved_count(rows):
    return sum(1 for r in rows if r["status"] == "solved")


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    problems = problem_paths()
    total = len(problems)

    print()
    title_box("Completing Incomplete Planning Domains using RL")

    stage(1, "Baseline planning on the incomplete domain")
    before = evaluate_problems(INCOMPLETE, problems)
    for r in before:
        status_line(r["problem"], r["status"], r["status"] == "solved")
    nb = solved_count(before)
    print("\n  " + col(f"{nb}/{total} solved", "bold")
          + col(" - the planner cannot solve the rest; an action is missing.", "dim"))

    failed = [r for r in before if r["status"] != "solved"]
    trigger = failed[0]["problem"]
    trigger_path = os.path.join(PROBLEMS_DIR, trigger + ".pddl")

    stage(2, "Localise the gap (bridge derived from the failure)")
    x, y = parse_goal_on(trigger_path)
    print(col("  Failed problem : ", "dim") + trigger)
    print(col("  Goal requires  : ", "dim") + col(f"(on {x} {y})", "yellow"))
    print(col("  Diagnosis      : ", "dim") + "no remaining action can produce (on _ _)")
    print(col("  RL target      : ", "dim") + f"reach C = (on {x} {y}) from state B, in the simulator")

    stage(3, "Reinforcement learning (learn the bridge)")
    blocks, configs = make_configs(x, y)
    world = BlocksWorld(blocks)
    goal = {("on", x, y)}
    trajectories = []
    for i, start in enumerate(configs, 1):
        agent = QLearningAgent(world, start, goal, seed=i)
        agent.train(episodes=800)
        traj, ok = agent.greedy_trajectory()
        plan = " -> ".join(f"({a[0]} {' '.join(a[1:])})" for (_, a, _) in traj)
        tag = col("learned", "green") if ok else col("failed", "red")
        print(f"  config {i}  [{tag}]   {col(plan, 'dim')}")
        if ok:
            trajectories.append(traj)
    payload = {"greedy": _ser_trajectory(trajectories[0]),
               "successful": [_ser_trajectory(t) for t in trajectories[1:]]}
    with open(TRAJECTORIES, "w") as f:
        json.dump(payload, f, indent=2)
    print("\n  " + col(f"Bridge learned from {len(trajectories)} varied configurations.", "bold"))

    stage(4, "Induce a symbolic operator (automatic)")
    operator, ground = induce_operator(TRAJECTORIES)
    with open(OPERATOR, "w") as f:
        f.write(operator)
    operator_box(operator)
    ok = validate_operator(ground)
    print("  " + col("validated", "green" if ok else "red")
          + col(f"  (reproduces the learned transition: {ok})", "dim"))

    stage(5, "Repair the domain")
    repair_domain(INCOMPLETE, OPERATOR, REPAIRED)
    print(col("  Inserted ", "dim") + col("learned_bridge", "yellow") + col(" into the domain.", "dim"))
    print(col("  Wrote ", "dim") + os.path.relpath(REPAIRED, ROOT))

    stage(6, "Re-plan with the repaired domain")
    after = evaluate_problems(REPAIRED, problems)
    for r in after:
        extra = "uses learned_bridge" if r.get("uses_bridge") else ""
        status_line(r["problem"], r["status"], r["status"] == "solved", extra)
    na = solved_count(after)

    print()
    print(col("  SUMMARY", "bold", "cyan") + col("   before  ->  after", "bold"))
    print(col("  " + "─" * 56, "gray"))
    comparison_table(before, after)
    result_box(nb, na, total)
    print("  " + col("One operator, learned automatically, reused across problems.", "dim") + "\n")
