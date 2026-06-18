import os
import sys
import json

ACTION_NAME = "learned_bridge"
VAR_NAMES = ["?x", "?y", "?z", "?w", "?v", "?u"]


def _deser_state(lst):
    return frozenset(tuple(p) for p in lst)


def load_starts_and_ends(path):
    with open(path) as f:
        data = json.load(f)
    trajs = []
    if data.get("greedy"):
        trajs.append(data["greedy"])
    trajs.extend(data.get("successful") or [])
    pairs = []
    for traj in trajs:
        if not traj:
            continue
        start = _deser_state(traj[0]["state"])
        end = _deser_state(traj[-1]["next_state"])
        pairs.append((start, end))
    return pairs


def induce_preconditions(pairs):
    starts = [s for (s, _) in pairs]
    common = set(starts[0])
    for s in starts[1:]:
        common &= set(s)
    return common


def induce_effects(pairs):
    add = delete = None
    for (s, e) in pairs:
        a = set(e) - set(s)
        d = set(s) - set(e)
        add = a if add is None else (add & a)
        delete = d if delete is None else (delete & d)
    return (add or set()), (delete or set())


def _constants_in(preds):
    consts = []
    for p in sorted(preds):
        for arg in p[1:]:
            if arg not in consts:
                consts.append(arg)
    return consts


def build_lifting(add, delete):
    ordered = _constants_in(add) + _constants_in(delete)
    mapping = {}
    for c in ordered:
        if c not in mapping:
            mapping[c] = VAR_NAMES[len(mapping)]
    return mapping


def _lift_pred(pred, mapping):
    if len(pred) == 1:
        return pred
    args = pred[1:]
    if any(a not in mapping for a in args):
        return None
    return (pred[0],) + tuple(mapping[a] for a in args)


def _lift_set(preds, mapping):
    out = []
    for p in preds:
        lifted = _lift_pred(p, mapping)
        if lifted is not None:
            out.append(lifted)
    return sorted(out)


def _fmt(pred):
    return "(" + " ".join(pred) + ")"


def render_operator(params, preconditions, add, delete, name=ACTION_NAME):
    pre = " ".join(_fmt(p) for p in preconditions)
    eff = " ".join([_fmt(p) for p in add] + [f"(not {_fmt(p)})" for p in delete])
    return (f"(:action {name}\n"
            f" :parameters ({' '.join(params)})\n"
            f" :precondition (and {pre})\n"
            f" :effect (and {eff}))\n")


def induce_operator(traj_path):
    pairs = load_starts_and_ends(traj_path)
    if not pairs:
        raise ValueError("no successful trajectories found")
    add, delete = induce_effects(pairs)
    pre_ground = induce_preconditions(pairs)
    mapping = build_lifting(add, delete)
    operator = render_operator(
        list(mapping.values()),
        _lift_set(pre_ground, mapping),
        _lift_set(add, mapping),
        _lift_set(delete, mapping),
    )
    ground = {"start": pairs[0][0], "end": pairs[0][1],
              "precondition": pre_ground, "add": add, "delete": delete}
    return operator, ground


def validate_operator(ground):
    s, e = ground["start"], ground["end"]
    if not set(ground["precondition"]).issubset(s):
        return False
    result = (set(s) - set(ground["delete"])) | set(ground["add"])
    return frozenset(result) == e


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    traj_path = os.path.join(root, "results", "trajectories.json")

    operator, ground = induce_operator(traj_path)

    print("Induced operator:\n")
    print(operator)

    ok = validate_operator(ground)
    print(f"Validation (operator reproduces the learned B -> C transition): {ok}")

    out_path = os.path.join(root, "results", "learned_operator.pddl")
    with open(out_path, "w") as f:
        f.write(operator)
    print(f"\nSaved operator to: {out_path}")
