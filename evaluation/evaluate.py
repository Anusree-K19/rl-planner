import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.planner import run_planner

LEARNED_NAME = "learned_bridge"


def build_repaired_domain(incomplete_path, operator_path, out_path):
    operator = open(operator_path).read().strip()
    domain = open(incomplete_path).read()
    idx = domain.rstrip().rfind(")")
    repaired = domain[:idx].rstrip() + "\n\n  " + operator + "\n)\n"
    with open(out_path, "w") as f:
        f.write(repaired)
    return out_path


def _uses_learned(plan):
    return any(LEARNED_NAME in a for a in plan)


def evaluate(problems, incomplete_domain, repaired_domain, planner_fn=run_planner):
    rows = []
    before_solved = after_solved = 0
    for prob in problems:
        before = planner_fn(incomplete_domain, prob)
        after = planner_fn(repaired_domain, prob)
        before_solved += int(before.success)
        after_solved += int(after.success)
        rows.append({
            "problem": os.path.basename(prob),
            "before_status": before.status,
            "before_length": before.plan_length,
            "before_runtime": round(before.runtime, 3),
            "after_status": after.status,
            "after_length": after.plan_length,
            "after_runtime": round(after.runtime, 3),
            "uses_learned": _uses_learned(after.plan),
        })
    return rows, before_solved, after_solved


def print_table(rows, before_solved, after_solved, total):
    name_w = max(len(r["problem"]) for r in rows)
    print(f"{'problem'.ljust(name_w)}   before           after")
    print("-" * (name_w + 38))
    for r in rows:
        before = r["before_status"]
        if r["before_status"] == "solved":
            before = f"solved ({r['before_length']})"
        after = r["after_status"]
        if r["after_status"] == "solved":
            tag = ", learned_bridge" if r["uses_learned"] else ""
            after = f"solved ({r['after_length']}{tag})"
        print(f"{r['problem'].ljust(name_w)}   {before.ljust(16)} {after}")
    print()
    print(f"Before repair: {before_solved}/{total} solved")
    print(f"After repair:  {after_solved}/{total} solved")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    domains = os.path.join(root, "domains")
    incomplete = os.path.join(domains, "blocksworld_incomplete.pddl")
    operator = os.path.join(root, "results", "learned_operator.pddl")
    repaired = os.path.join(domains, "blocksworld_repaired.pddl")

    build_repaired_domain(incomplete, operator, repaired)
    print(f"Repaired domain written to: {repaired}\n")

    prob_dir = os.path.join(root, "problems")
    problems = sorted(os.path.join(prob_dir, f)
                      for f in os.listdir(prob_dir) if f.endswith(".pddl"))

    rows, before_solved, after_solved = evaluate(problems, incomplete, repaired)
    print_table(rows, before_solved, after_solved, len(problems))

    out = os.path.join(root, "results", "evaluation.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "before_solved": before_solved,
                   "after_solved": after_solved, "total": len(problems)},
                  f, indent=2)
    print(f"\nSaved evaluation to: {out}")
