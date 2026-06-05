from itertools import permutations

class BlocksWorld:
    def __init__(self, blocks):
        self.blocks = list(blocks)

    def _schema(self, action):
        name = action[0]
        if name == "pick-up":
            (x,) = action[1:]
            pre = {("clear", x), ("ontable", x), ("handempty",)}
            add = {("holding", x)}
            dele = {("ontable", x), ("clear", x), ("handempty",)}
        elif name == "put-down":
            (x,) = action[1:]
            pre = {("holding", x)}
            add = {("clear", x), ("handempty",), ("ontable", x)}
            dele = {("holding", x)}
        elif name == "stack":
            (x, y) = action[1:]
            pre = {("holding", x), ("clear", y)}
            add = {("clear", x), ("handempty",), ("on", x, y)}
            dele = {("holding", x), ("clear", y)}
        elif name == "unstack":
            (x, y) = action[1:]
            pre = {("on", x, y), ("clear", x), ("handempty",)}
            add = {("holding", x), ("clear", y)}
            dele = {("clear", x), ("handempty",), ("on", x, y)}
        else:
            raise ValueError(f"unknown action: {action}")
        return pre, add, dele

    def all_actions(self):
        actions = []
        for b in self.blocks:
            actions.append(("pick-up", b))
            actions.append(("put-down", b))
        for x, y in permutations(self.blocks, 2):
            actions.append(("stack", x, y))
            actions.append(("unstack", x, y))
        return actions

    def is_applicable(self, state, action):
        pre, _, _ = self._schema(action)
        return pre.issubset(state)

    def applicable_actions(self, state):
        return [a for a in self.all_actions() if self.is_applicable(state, a)]

    def step(self, state, action):
        pre, add, dele = self._schema(action)
        if not pre.issubset(state):
            raise ValueError(f"action {action} not applicable in state {state}")
        return frozenset((state - dele) | add)

    @staticmethod
    def goal_reached(state, goal):
        return set(goal).issubset(state)


def state(*predicates):
    return frozenset(predicates)


if __name__ == "__main__":
    world = BlocksWorld(["a", "b"])
    s0 = state(("ontable", "a"), ("ontable", "b"),
               ("clear", "a"), ("clear", "b"), ("handempty",))
    goal = {("on", "a", "b")}
    print("initial state:", sorted(s0))
    print("applicable:", world.applicable_actions(s0))
    plan = [("pick-up", "a"), ("stack", "a", "b")]
    s = s0
    for act in plan:
        s = world.step(s, act)
        print(f"after {act}: {sorted(s)}")
    print("goal reached?", world.goal_reached(s, goal))
