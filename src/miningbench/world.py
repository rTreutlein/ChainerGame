"""Seeded, fully observed input streams with stochastic hidden mechanisms."""

from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import random

KINDS = (
    "single",
    "and2",
    "and3",
    "and4",
    "negation",
    "lag",
    "xor",
    "or",
    "numeric",
    "binding",
)


@dataclass(frozen=True, order=True)
class Literal:
    head: str
    args: tuple[str, ...] = ()
    lag: int = 1
    negative: bool = False


@dataclass(frozen=True)
class Rule:
    target: str
    body: tuple[Literal, ...]
    value: str = "1"
    formula: str = ""
    strength: float = 1.0
    condition: str = ""


@dataclass(frozen=True)
class Config:
    seed: int = 1
    samples: int = 256
    kinds: tuple[str, ...] = KINDS
    targets: int = 1  # copies of each requested type
    distractors: int = 0
    strength: float = 1.0
    observation_noise: float = 0.0
    drift: bool = False
    deep_lags: int = 3
    entity_holdout: bool = False
    numeric_width: int = 32

    def __post_init__(self):
        if self.samples < 8 or self.targets < 1 or self.distractors < 0:
            raise ValueError("samples >= 8, targets >= 1, distractors >= 0 required")
        if not self.kinds or any(k not in KINDS for k in self.kinds):
            raise ValueError("unknown or empty rule kinds")
        if not 0 < self.strength <= 1 or not 0 <= self.observation_noise <= 0.5:
            raise ValueError("strength in (0,1], observation_noise in [0,.5] required")


def rng_for(seed, name):
    # Independent channels preserve matched streams when a knob adds distractors/targets.
    return random.Random(
        int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
    )


class World:
    def __init__(self, config):
        self.config = config
        self.types = {
            f"T{i}_{k}": k for i in range(config.targets) for k in config.kinds
        }
        self.domains = {
            t: tuple(map(str, range(config.numeric_width + 1)))
            if k == "numeric"
            else ("0", "1")
            for t, k in self.types.items()
        }
        self.atoms = []
        for t, k in self.types.items():
            if k == "numeric":
                self.atoms.extend(
                    (t + "_Pos", (str(i),)) for i in range(config.numeric_width + 1)
                )
                self.atoms.extend(
                    (t + "_Vel", (str(i),)) for i in (-3, -2, -1, 1, 2, 3)
                )
            elif k == "binding":
                self.atoms.extend(
                    (t + "_" + h, (f"e{i}",))
                    for h in ("At", "Did")
                    for i in range(8 if config.entity_holdout else 4)
                )
            else:
                self.atoms.extend((t + f"_F{i}", ()) for i in range(4))
        self.atoms.extend((f"D{i}", ()) for i in range(config.distractors))
        self.features = tuple(
            Literal(h, a, lag)
            for lag in range(1, config.deep_lags + 1)
            for h, a in self.atoms
            if lag <= 3 or h.startswith("D")
        )
        self.rules = [self.plant(phase) for phase in (0, 1)]

    def plant(self, phase):
        result = []
        for t, k in self.types.items():

            def lit(i, lag=1, negative=False):
                return Literal(
                    t + f"_F{i}", (), lag, negative ^ (bool(phase) and i == 0)
                )

            if k == "single":
                bodies = [(lit(0),)]
            elif k in ("and2", "and3", "and4"):
                bodies = [tuple(lit(i) for i in range(int(k[-1])))]
            elif k == "negation":
                bodies = [(lit(0), lit(1, negative=True))]
            elif k == "lag":
                bodies = [(lit(0, 1), lit(1, 2), lit(2, 3))]
            elif k == "xor":
                bodies = [
                    (lit(0), lit(1, negative=True)),
                    (lit(0, negative=True), lit(1)),
                ]
            elif k == "or":
                bodies = [(lit(0), lit(1)), (lit(2), lit(3))]
            elif k == "binding":
                # Phase changes equality into one concrete exception plus equality elsewhere.
                bodies = [(Literal(t + "_At", ("$x",)), Literal(t + "_Did", ("$x",)))]
                if phase:
                    bodies = [
                        (
                            Literal(t + "_At", (f"e{i}",)),
                            Literal(t + "_Did", (f"e{i}",)),
                        )
                        for i in range(1, 4)
                    ]
            else:
                body = (Literal(t + "_Pos", ("$x1",)), Literal(t + "_Vel", ("$v1",)))
                width = self.config.numeric_width
                motion = "x1-v1" if phase else "x1+v1"
                for region, formula in [
                    ("inside", motion),
                    ("below", f"-({motion})"),
                    ("above", f"{2 * width}-({motion})"),
                ]:
                    result.append(
                        Rule(
                            t,
                            body,
                            "*",
                            formula,
                            self.config.strength,
                            region + ":" + motion,
                        )
                    )
                continue
            result.extend(Rule(t, b, strength=self.config.strength) for b in bodies)
        return result

    @staticmethod
    def holds(rule, history):
        variables = sorted(
            {a for literal in rule.body for a in literal.args if a.startswith("$")}
        )
        domains = {}
        for variable in variables:
            candidates = None
            for literal in rule.body:
                if literal.negative:
                    continue
                for position, argument in enumerate(literal.args):
                    if argument != variable:
                        continue
                    values = {
                        args[position]
                        for (head, args), truth in history[-literal.lag].items()
                        if truth
                        and head == literal.head
                        and len(args) == len(literal.args)
                    }
                    candidates = values if candidates is None else candidates & values
            if candidates is None:
                candidates = {
                    argument
                    for row in history
                    for (_, args) in row
                    for argument in args
                }
            domains[variable] = sorted(candidates)
        for values in product(*(domains[variable] for variable in variables)):
            binding = dict(zip(variables, values))
            if all(
                bool(
                    history[-literal.lag].get(
                        (literal.head, tuple(binding.get(a, a) for a in literal.args)),
                        0,
                    )
                )
                != literal.negative
                for literal in rule.body
            ):
                return True
        return False

    def probabilities(self, history, phase):
        """Exact conditional distribution; independent rule firings combine by noisy OR.

        Observation noise affects the outcome channel, never hidden input state.
        Numeric failures select a uniform outcome; Boolean failures give false.
        """
        q = {}
        for t, k in self.types.items():
            rules = [r for r in self.rules[phase] if r.target == t]
            n = self.config.observation_noise
            if k == "numeric":
                x, velocity = [
                    next(
                        int(a[0])
                        for (h, a), v in history[-1].items()
                        if h == t + "_" + head and v
                    )
                    for head in ("Pos", "Vel")
                ]
                width = self.config.numeric_width
                z = x - velocity if phase else x + velocity
                value = -z if z < 0 else 2 * width - z if z > width else z
                size = width + 1
                q[t] = [
                    (1 - self.config.strength) / size
                    + self.config.strength * (i == value)
                    for i in range(size)
                ]
                q[t] = [(1 - n) * p + n / size for p in q[t]]
            else:
                miss = 1.0
                for r in rules:
                    if self.holds(r, history):
                        miss *= 1 - r.strength
                p = n + (1 - 2 * n) * (1 - miss)
                q[t] = [1 - p, p]
        return q

    def stream(self):
        streams = {t: rng_for(self.config.seed, t) for t in self.types}
        ds = {
            h: rng_for(self.config.seed, h) for h, _ in self.atoms if h.startswith("D")
        }
        labels = {t: rng_for(self.config.seed, "label:" + t) for t in self.types}
        positions = {
            t: streams[t].randrange(self.config.numeric_width + 1)
            for t, k in self.types.items()
            if k == "numeric"
        }
        velocities = {t: streams[t].choice((-3, -2, -1, 1, 2, 3)) for t in positions}
        warmup = self.config.deep_lags
        history = []
        for step in range(self.config.samples + warmup):
            facts = {}
            for t, k in self.types.items():
                r = streams[t]
                if k == "numeric":
                    width = self.config.numeric_width
                    x, velocity = positions[t], velocities[t]
                    # Random velocity interventions prevent a short periodic orbit.
                    if r.random() < 0.2:
                        velocity = r.choice((-3, -2, -1, 1, 2, 3))
                    facts.update(
                        {(t + "_Pos", (str(i),)): int(i == x) for i in range(width + 1)}
                    )
                    facts.update(
                        {
                            (t + "_Vel", (str(i),)): int(i == velocity)
                            for i in (-3, -2, -1, 1, 2, 3)
                        }
                    )
                    z = x + velocity
                    positions[t] = -z if z < 0 else 2 * width - z if z > width else z
                    velocities[t] = -velocity if z < 0 or z > width else velocity
                elif k == "binding":
                    offset = (
                        4
                        if self.config.entity_holdout
                        and step >= warmup + self.config.samples // 2
                        else 0
                    )
                    a, b = offset + r.randrange(4), offset + r.randrange(4)
                    facts.update(
                        {
                            (t + "_" + h, (f"e{i}",)): int(i == v)
                            for h, v in [("At", a), ("Did", b)]
                            for i in range(8 if self.config.entity_holdout else 4)
                        }
                    )
                else:
                    facts.update({(t + f"_F{i}", ()): r.randrange(2) for i in range(4)})
            facts.update({(h, ()): r.randrange(2) for h, r in ds.items()})
            if step >= warmup:
                phase = int(
                    self.config.drift and step - warmup >= self.config.samples // 2
                )
                q = self.probabilities(history, phase)
                y = {}
                for t, probs in q.items():
                    u = labels[t].random()
                    acc = 0.0
                    for i, p in enumerate(probs):
                        acc += p
                        if u < acc:
                            y[t] = i
                            break
                yield {
                    "step": step - warmup,
                    "time": step,
                    "phase": phase,
                    "history": tuple(history),
                    "q": q,
                    "y": y,
                    "current": facts,
                    "holdout": self.config.entity_holdout
                    and step - warmup >= self.config.samples // 2,
                }
            history = (history + [facts])[-warmup:]

    def metadata(self):
        return {
            "config": asdict(self.config),
            "types": self.types,
            "domains": self.domains,
            "rules": [[asdict(r) for r in rs] for rs in self.rules],
        }


def serial_sample(sample):
    out = {k: v for k, v in sample.items() if k not in ("history", "current")}
    out["history"] = [
        [[h, list(a), v] for (h, a), v in row.items()] for row in sample["history"]
    ]
    return out
