import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.planner import run_planner

LEARNED_ACTION = "learned_bridge"


def repair_domain(incomplete_path, operator_path, out_path):
    domain = open(incomplete_path).read()
    operator = open(operator_path).read().strip()
    idx = domain.rstrip().rfind(")")
    repaired = domain[:idx] + "\n  " + operator + "\n)\n"
    with open(out_path, "w") as f:
        f.write(repaired)
    return out_path


def uses_learned_action(plan):
    return any(LEARNED_ACTION in a for a in plan)


def evaluate_problems(domain, problems):
    rows = []
    for path in problems:
        name = os.path.splitext(os.path.basename(path))[0]
        r = run_planner(domain, path)
        rows.append({"problem": name, "status": r.status,
                     "length": r.plan_length if r.success else None,
                     "runtime": round(r.runtime, 3),
                     "uses_bridge": uses_learned_action(r.plan)})
    return rows


def _cell(row):
    if row["status"] != "solved":
        return row["status"]
    tag = ", learned_bridge" if row["uses_bridge"] else ""
    return f"solved ({row['length']} steps{tag})"


def print_comparison(before, after):
    before_cells = {r["problem"]: _cell(r) for r in before}
    after_cells = {r["problem"]: _cell(r) for r in after}
    name_w = max([len(r["problem"]) for r in before] + [len("problem")])
    before_w = max([len(v) for v in before_cells.values()] + [len("before")])
    head = f"{'problem'.ljust(name_w)}   {'before'.ljust(before_w)}   after"
    print(head)
    print("-" * len(head))
    for b in before:
        n = b["problem"]
        print(f"{n.ljust(name_w)}   {before_cells[n].ljust(before_w)}   {after_cells[n]}")
    nb = sum(1 for r in before if r["status"] == "solved")
    na = sum(1 for r in after if r["status"] == "solved")
    print()
    print(f"Solved before repair: {nb}/{len(before)}")
    print(f"Solved after repair:  {na}/{len(after)}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    domains = os.path.join(root, "domains")
    incomplete = os.path.join(domains, "blocksworld_incomplete.pddl")
    operator = os.path.join(root, "results", "learned_operator.pddl")
    repaired = os.path.join(domains, "blocksworld_repaired.pddl")
    problems_dir = os.path.join(root, "problems")
    problems = sorted(os.path.join(problems_dir, f)
                      for f in os.listdir(problems_dir) if f.endswith(".pddl"))
    repair_domain(incomplete, operator, repaired)
    print(f"Repaired domain written to: {repaired}\n")
    before = evaluate_problems(incomplete, problems)
    after = evaluate_problems(repaired, problems)
    print_comparison(before, after)
    out_path = os.path.join(root, "results", "evaluation.json")
    with open(out_path, "w") as f:
        json.dump({"before": before, "after": after}, f, indent=2)
    print(f"\nSaved evaluation to: {out_path}")
