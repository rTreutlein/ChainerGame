"""Independent rule interpreter and finite conditional-probability enumeration.

The interpreter does not call World.probabilities. Enumeration is evaluator-only;
learners never receive rules or the resulting conditional distributions.
"""

from itertools import product
from math import prod
from .world import World


def numeric_contexts(world, target):
    for position in range(world.config.numeric_width + 1):
        for velocity in (-3, -2, -1, 1, 2, 3):
            yield [
                {
                    (target + "_Pos", (str(position),)): 1,
                    (target + "_Vel", (str(velocity),)): 1,
                }
            ] * world.config.deep_lags


def numeric_matching_contexts(world, target, rules):
    """Reachable histories, integrating only variables inspected by learned rules.

    Each current position/velocity state has total weight one, distributed
    uniformly over its reachable pasts and independent auxiliary observations.
    Unmentioned distractors integrate out without exponential enumeration.
    """
    from collections import Counter

    core_heads = (target + "_Pos", target + "_Vel")
    literals = [literal for rule in rules for literal in rule.body]
    depth = max(
        [1] + [literal.lag for literal in literals if literal.head in core_heads]
    )
    paths = [[history[-1]] for history in numeric_contexts(world, target)]
    width = world.config.numeric_width
    for _ in range(depth - 1):
        if len(paths) * 6 > 1048576:
            raise ValueError("numeric history expansion exceeds exact evaluator limit")
        expanded = []
        for history in paths:
            row = history[-1]
            x, v = [
                next(
                    int(args[0])
                    for (head, args), truth in row.items()
                    if truth and head == wanted
                )
                for wanted in core_heads
            ]
            z = x + v
            position = -z if z < 0 else 2 * width - z if z > width else z
            for velocity in (-3, -2, -1, 1, 2, 3):
                expanded.append(
                    history
                    + [
                        {
                            (core_heads[0], (str(position),)): 1,
                            (core_heads[1], (str(velocity),)): 1,
                        }
                    ]
                )
        paths = expanded
    factors = {}
    for literal in literals:
        if literal.head in core_heads:
            continue
        key = (literal.head, literal.lag)
        arguments = [args for head, args in world.atoms if head == literal.head]
        if arguments == [()]:
            factors[key] = [{(literal.head, ()): value} for value in (0, 1)]
        elif arguments:
            factors[key] = [{(literal.head, args): 1} for args in arguments]
        else:
            factors[key] = [{}]
    if len(paths) * prod(len(values) for values in factors.values()) > 1048576:
        raise ValueError("numeric behavioural matching exceeds exact evaluator limit")
    alternatives = list(product(*factors.values()))

    def state_key(row):
        return tuple(sorted(row))

    counts = Counter(state_key(history[-1]) for history in paths)
    length = max(
        [world.config.deep_lags, depth] + [literal.lag for literal in literals]
    )
    for path in paths:
        weight = 1 / (counts[state_key(path[-1])] * len(alternatives))
        for cells in alternatives:
            history = [{} for _ in range(length - len(path))] + [
                dict(row) for row in path
            ]
            for (_, lag), cell in zip(factors, cells):
                history[-lag].update(cell)
            yield history, weight


def numeric_rule_value(rule, history, width, phase=0):
    row = history[-1]
    position = next(
        int(a[0]) for (h, a), v in row.items() if v and h == rule.target + "_Pos"
    )
    velocity = next(
        int(a[0]) for (h, a), v in row.items() if v and h == rule.target + "_Vel"
    )
    if not World.holds(rule, history):
        return None
    if rule.formula:
        from .scoring import formula_coefficients

        values = {"1": 1, "x1": position, "v1": velocity}

        def evaluate(formula):
            return sum(
                float(coefficient) * values[name]
                for name, coefficient in formula_coefficients(formula).items()
            )

        region, expression = rule.condition.split(":", 1)
        z = evaluate(expression)
        actual_region = "below" if z < 0 else "above" if z > width else "inside"
        if actual_region != region:
            return None
        value = evaluate(rule.formula)
        return int(value) if value == int(value) else value
    return int(rule.value) if World.holds(rule, history) else None


def interpret(world, history, phase):
    result = {}
    noise = world.config.observation_noise
    for target, domain in world.domains.items():
        rules = [r for r in world.rules[phase] if r.target == target]
        if world.types[target] == "numeric":
            size = len(domain)
            values = [numeric_rule_value(r, history, size - 1, phase) for r in rules]
            chosen = next(v for v in values if v is not None)
            reliability = world.config.strength * (1 - noise)
            result[target] = [
                (1 - reliability) / size + reliability * (i == chosen)
                for i in range(size)
            ]
        else:
            active = [r for r in rules if World.holds(r, history)]
            probability = 1 - prod(1 - r.strength for r in active)
            p = probability * (1 - noise) + (1 - probability) * noise
            result[target] = [1 - p, p]
    return result


def conditional(world, rule, phase=0):
    """Exact P(value | body) for IID Boolean inputs and categorical entities.

    Numeric calibration uses uniform reachable (position,velocity) contexts,
    explicitly distinct from the trajectory's visitation distribution.
    """
    target = rule.target
    if world.types[target] == "numeric":
        histories = numeric_contexts(world, target)
    else:
        literals = list(rule.body)
        literals += [
            literal
            for r in world.rules[phase]
            if r.target == target
            for literal in r.body
        ]
        groups = {}
        for literal in literals:
            head, lag = literal.head, literal.lag
            if head.endswith(("_At", "_Did")):
                key = (head, lag)
                groups[key] = [
                    {(head, (f"e{i}",)): int(i == chosen) for i in range(4)}
                    for chosen in range(4)
                ]
            elif head.endswith(("_Pos", "_Vel")):
                values = (
                    range(world.config.numeric_width + 1)
                    if head.endswith("_Pos")
                    else (-3, -2, -1, 1, 2, 3)
                )
                groups[(head, lag)] = [{(head, (str(v),)): 1} for v in values]
            else:
                groups[(head, lag)] = [{(head, literal.args): v} for v in (0, 1)]
        if prod(len(g) for g in groups.values()) > 1048576:
            raise ValueError("conditional enumeration exceeds exact evaluator limit")

        def enumerate_histories():
            for cells in product(*groups.values()):
                history = [
                    {}
                    for _ in range(
                        max(
                            world.config.deep_lags,
                            max(literal.lag for literal in literals),
                        )
                    )
                ]
                for (_, lag), cell in zip(groups, cells):
                    history[-lag].update(cell)
                yield history

        histories = enumerate_histories()
    total = count = 0
    # Restrict to this target: other independent targets need not be enumerated.
    from copy import copy

    subworld = copy(world)
    subworld.types = {target: world.types[target]}
    subworld.domains = {target: world.domains[target]}
    for history in histories:
        if World.holds(rule, history):
            total += interpret(subworld, history, phase)[target][
                world.domains[target].index(rule.value)
            ]
            count += 1
    return total / count if count else None
