import os
import glob
import time
import tempfile
import subprocess
from dataclasses import dataclass

FAST_DOWNWARD = os.environ.get(
    "FAST_DOWNWARD", os.path.expanduser("~/downward/fast-downward.py")
)

DEFAULT_ALIAS = "lama-first"
DEFAULT_TIME_LIMIT = 120


@dataclass
class PlanResult:
    success: bool
    plan: list
    runtime: float
    status: str
    exit_code: int | None = None

    @property
    def plan_length(self):
        return len(self.plan)


def _parse_plan_file(path):
    actions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            actions.append(line)
    return actions


def run_planner(domain, problem, alias=DEFAULT_ALIAS,
                time_limit=DEFAULT_TIME_LIMIT, fast_downward=FAST_DOWNWARD):
    fast_downward = os.path.expanduser(fast_downward)
    if not os.path.isfile(fast_downward):
        raise FileNotFoundError(
            f"Fast Downward not found at '{fast_downward}'. "
            f"Set the FAST_DOWNWARD environment variable to its location."
        )

    domain = os.path.abspath(domain)
    problem = os.path.abspath(problem)
    for p in (domain, problem):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"PDDL file not found: {p}")

    cmd = ["python3", fast_downward, "--alias", alias, domain, problem]

    with tempfile.TemporaryDirectory(prefix="fd_run_") as tmpdir:
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True,
                                  text=True, timeout=time_limit)
            runtime = time.perf_counter() - start
            exit_code = proc.returncode
            stdout = proc.stdout or ""
        except subprocess.TimeoutExpired:
            runtime = time.perf_counter() - start
            return PlanResult(False, [], runtime, "timeout", None)

        plan_files = sorted(glob.glob(os.path.join(tmpdir, "sas_plan*")))
        if plan_files:
            plan = _parse_plan_file(plan_files[-1])
            return PlanResult(True, plan, runtime, "solved", exit_code)

        lowered = stdout.lower()
        if (exit_code == 12 or "no solution" in lowered
                or "unsolvable" in lowered
                or "without finding a solution" in lowered):
            return PlanResult(False, [], runtime, "unsolvable", exit_code)

        return PlanResult(False, [], runtime, "error", exit_code)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    complete = os.path.join(root, "domains", "blocksworld_complete.pddl")
    incomplete = os.path.join(root, "domains", "blocksworld_incomplete.pddl")
    problem = os.path.join(root, "problems", "p1_stack2.pddl")

    print(f"Fast Downward: {FAST_DOWNWARD}\n")

    for label, dom in [("COMPLETE   (expect: solved)", complete),
                       ("INCOMPLETE (expect: unsolvable)", incomplete)]:
        r = run_planner(dom, problem)
        print(label)
        print(f"  success : {r.success}")
        print(f"  status  : {r.status}")
        print(f"  exit    : {r.exit_code}")
        print(f"  runtime : {r.runtime:.3f}s")
        print(f"  length  : {r.plan_length}")
        if r.plan:
            print(f"  plan    : {' -> '.join(r.plan)}")
        print()
