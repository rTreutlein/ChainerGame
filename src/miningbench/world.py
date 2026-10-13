"""Seeded, fully observed input streams with stochastic hidden mechanisms."""
from dataclasses import asdict, dataclass, replace
from itertools import product
import hashlib
import json
import random

KINDS = ('single', 'and2', 'and3', 'and4', 'negation', 'lag', 'xor', 'or', 'numeric', 'binding')


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
    value: str = '1'
    formula: str = ''
    strength: float = 1.0


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

    def __post_init__(self):
        if self.samples < 8 or self.targets < 1 or self.distractors < 0:
            raise ValueError('samples >= 8, targets >= 1, distractors >= 0 required')
        if not self.kinds or any(k not in KINDS for k in self.kinds):
            raise ValueError('unknown or empty rule kinds')
        if not 0 < self.strength <= 1 or not 0 <= self.observation_noise <= .5:
            raise ValueError('strength in (0,1], observation_noise in [0,.5] required')


def rng_for(seed, name):
    # Independent channels preserve matched streams when a knob adds distractors/targets.
    return random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big'))


class World:
    def __init__(self, config):
        self.config = config
        self.types = {f'T{i}_{k}': k for i in range(config.targets) for k in config.kinds}
        self.domains = {t: tuple(map(str, range(5))) if k == 'numeric' else ('0', '1')
                        for t, k in self.types.items()}
        self.atoms = []
        for t, k in self.types.items():
            if k == 'numeric':
                self.atoms.extend((t + '_Pos', (str(i),)) for i in range(5))
            elif k == 'binding':
                self.atoms.extend((t + '_' + h, (f'e{i}',)) for h in ('At', 'Did') for i in range(4))
            else:
                self.atoms.extend((t + f'_F{i}', ()) for i in range(4))
        self.atoms.extend((f'D{i}', ()) for i in range(config.distractors))
        self.features = tuple(Literal(h, a, lag) for lag in (1, 2, 3) for h, a in self.atoms)
        self.rules = [self.plant(phase) for phase in (0, 1)]

    def plant(self, phase):
        result = []
        for t, k in self.types.items():
            def lit(i, lag=1, negative=False):
                return Literal(t + f'_F{i}', (), lag, negative ^ (bool(phase) and i == 0))
            if k == 'single':
                bodies = [(lit(0),)]
            elif k in ('and2', 'and3', 'and4'):
                bodies = [tuple(lit(i) for i in range(int(k[-1])))]
            elif k == 'negation':
                bodies = [(lit(0), lit(1, negative=True))]
            elif k == 'lag':
                bodies = [(lit(0, 1), lit(1, 2), lit(2, 3))]
            elif k == 'xor':
                bodies = [(lit(0), lit(1, negative=True)), (lit(0, negative=True), lit(1))]
            elif k == 'or':
                bodies = [(lit(0), lit(1)), (lit(2), lit(3))]
            elif k == 'binding':
                # Phase changes equality into one concrete exception plus equality elsewhere.
                bodies = [(Literal(t+'_At', ('$x',)), Literal(t+'_Did', ('$x',)))]
                if phase:
                    bodies = [(Literal(t+'_At', (f'e{i}',)), Literal(t+'_Did', (f'e{i}',))) for i in range(1, 4)]
            else:
                for region, formula in [('interior', '2*x1-x2' if not phase else 'x2'),
                                        ('wall0', 'x2'), ('wall4', 'x2')]:
                    result.append(Rule(t, (Literal(t+'_Region', (region,)),), '*', formula, self.config.strength))
                continue
            result.extend(Rule(t, b, strength=self.config.strength) for b in bodies)
        return result

    @staticmethod
    def holds(rule, history):
        variables = sorted({a for l in rule.body for a in l.args if a.startswith('$')})
        for values in product(('e0', 'e1', 'e2', 'e3'), repeat=len(variables)):
            binding = dict(zip(variables, values))
            if all(bool(history[-l.lag].get((l.head, tuple(binding.get(a, a) for a in l.args)), 0)) != l.negative
                   for l in rule.body):
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
            if k == 'numeric':
                x1, x2 = [next(int(a[0]) for (h, a), v in history[-lag].items() if h == t+'_Pos' and v)
                          for lag in (1, 2)]
                value = x2 if x1 in (0, 4) or phase else 2*x1-x2
                # The input stream is a noiseless reflecting unit-velocity trajectory.
                q[t] = [(1-self.config.strength)/5 + self.config.strength*(i == value) for i in range(5)]
                q[t] = [(1-n)*p+n/5 for p in q[t]]
            else:
                miss = 1.0
                for r in rules:
                    if self.holds(r, history):
                        miss *= 1-r.strength
                p = n + (1-2*n)*(1-miss)
                q[t] = [1-p, p]
        return q

    def stream(self):
        streams = {t: rng_for(self.config.seed, t) for t in self.types}
        ds = {h: rng_for(self.config.seed, h) for h, _ in self.atoms if h.startswith('D')}
        labels = {t: rng_for(self.config.seed, 'label:'+t) for t in self.types}
        positions = {t: streams[t].randrange(8) for t, k in self.types.items() if k == 'numeric'}
        history = []
        for step in range(self.config.samples+3):
            facts = {}
            for t, k in self.types.items():
                r = streams[t]
                if k == 'numeric':
                    z = (positions[t]+step) % 8
                    x = z if z <= 4 else 8-z
                    facts.update({(t+'_Pos', (str(i),)): int(i == x) for i in range(5)})
                elif k == 'binding':
                    a, b = r.randrange(4), r.randrange(4)
                    facts.update({(t+'_'+h, (f'e{i}',)): int(i == v)
                                  for h, v in [('At', a), ('Did', b)] for i in range(4)})
                else:
                    facts.update({(t+f'_F{i}', ()): r.randrange(2) for i in range(4)})
            facts.update({(h, ()): r.randrange(2) for h, r in ds.items()})
            if step >= 3:
                phase = int(self.config.drift and step-3 >= self.config.samples//2)
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
                yield {'step': step-3, 'time': step, 'phase': phase,
                       'history': tuple(history), 'q': q, 'y': y, 'current': facts}
            history = (history+[facts])[-3:]

    def metadata(self):
        return {'config': asdict(self.config), 'types': self.types, 'domains': self.domains,
                'rules': [[asdict(r) for r in rs] for rs in self.rules]}


def serial_sample(sample):
    out = {k: v for k, v in sample.items() if k not in ('history', 'current')}
    out['history'] = [[[h, list(a), v] for (h, a), v in row.items()] for row in sample['history']]
    return out
