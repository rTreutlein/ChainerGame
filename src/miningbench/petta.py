"""Frozen PeTTaChainer adapter. Run only inside the mandatory capped wrapper."""
from dataclasses import replace
import re
from .world import Literal, Rule


def parse(text):
    tokens = iter(re.findall(r'\(|\)|[^\s()]+', str(text)))
    def term(token):
        if token != '(':
            return token
        out = []
        for token in tokens:
            if token == ')':
                return out
            out.append(term(token))
        raise ValueError('unclosed term')
    return term(next(tokens))


def decode(implication, domains):
    """Translate period offsets back to event lags, retaining entity equalities."""
    if implication[0] != 'Implication':
        raise ValueError('not an implication')
    body, consequent = implication[1:]
    target = consequent[0]
    if target not in domains:
        raise ValueError('non-target conclusion')
    value = '1' if len(consequent) == 2 else consequent[1]
    if value.startswith('$'):
        raise ValueError('variable outcome is not a categorical prediction')
    clauses = body[1:] if body[0] == 'And' else [body]
    times = {consequent[-1]: 1}
    links = [c for c in clauses if c[0] == 'NextPeriod']
    for _ in range(len(links)+1):
        for _, earlier, later in links:
            if later in times:
                times[earlier] = times[later]+1
    literals = []
    for clause in clauses:
        if clause[0] == 'NextPeriod':
            continue
        neg = clause[0] == 'Not'
        c = clause[1] if neg else clause
        if c[-1] not in times or not c[-1].startswith('$'):
            raise ValueError('unbound or ground time')
        literals.append(Literal(c[0], tuple(c[1:-1]), times[c[-1]], neg))
    return Rule(target, tuple(literals), value)


class Petta:
    def __init__(self, features, domains, budget=80, max_new=4, max_live=8):
        from pettachainer import PeTTaChainer
        self.h = PeTTaChainer()
        self.domains, self.budget = domains, budget
        self.label_counts = {t: [0]*len(d) for t,d in domains.items()}
        self.h.set_evidence_confidence_k(1)
        self.h.set_hypothesis_mining(max_new, max_live)
        self.h.set_positive_only_predicate('NextPeriod')
        self.h.set_time_link('NextPeriod')
        self.input_heads = {l.head: len(l.args)+1 for l in features}
        for head, pos in self.input_heads.items():
            self.h.set_predicate_time(head, pos, 0)
        for target, domain in domains.items():
            self.h.set_predicate_time(target, 1 if len(domain) == 2 else 2, 1)
            self.h.set_observed_predicate(target)
        self.live = {}
        self.negative_rates = {}
        self.unparsed = {}
        self.raw = []
        self.period = 0
        self.work = None
        self.coverage = 0
        self.anticipated = 0

    def add_inputs(self, row, time):
        self.h.add_atoms_no_check([f'(: in-{time}-{i} ({h} {" ".join(a)} p{time}) (STV {v} 1))'
                                   for i, ((h, a), v) in enumerate(row.items())])

    def statement(self, t, value, period):
        return f'({t} p{period})' if len(self.domains[t]) == 2 else f'({t} {value} p{period})'

    def predict(self, history):
        # An event y_t is the end-of-period outcome of context x_(t-1).
        # Context and outcome use layers 0/1 of that decision period. No lagged
        # feature predicates, negated proxy predicates or planted rules are supplied.
        if self.period == 0:
            for i, row in enumerate(history):
                self.add_inputs(row, i)
                if i:
                    self.h.add_atoms_no_check([f'(: link{i} (NextPeriod p{i-1} p{i}) (STV 1 1))'])
            self.period = 2
        else:
            self.period += 1
            self.add_inputs(history[-1], self.period)
        statements, indices = [], []
        for t, domain in self.domains.items():
            for v in (('1',) if len(domain) == 2 else domain):
                statements.append(f'(: $p {self.statement(t, v, self.period)} $tv)')
                indices.append((t, v))
        rows = self.h.query_many_refs(statements, steps=self.budget)
        predicted = {t: [0.]*len(d) for t, d in self.domains.items()}
        for (t, v), answers in zip(indices, rows):
            stvs = [a.tv for a in answers if a.tv[0] == 'STV']
            tv = max(stvs, key=lambda tv: float(tv[2])) if stvs else ['STV', 1/len(self.domains[t]), 0]
            self.coverage += bool(stvs)
            p = float(tv[1])
            predicted[t][self.domains[t].index(v)] = p
            if len(self.domains[t]) == 2:
                predicted[t][0] = 1-p
        for t, p in predicted.items():
            s = sum(p)
            predicted[t] = [x/s for x in p] if s else [1/len(p)]*len(p)
        return predicted

    def learn(self, history, labels):
        positives, negatives = [], []
        for t, y in labels.items():
            self.label_counts[t][y] += 1
            domain = self.domains[t]
            for v in (('1',) if len(domain) == 2 else domain):
                stmt = self.statement(t, v, self.period)
                truth = int(domain[y] == v)
                atom = f'(: label-{self.period}-{t}-{v} {stmt} (STV {truth} 1))'
                (positives if truth else negatives).append((t, v, atom))
        self.h.add_atoms_no_check([a for _, _, a in positives])
        # A ground-headed live conclusion is exactly the documented anticipation
        # domain. Supply explicit negatives outside it, so never-successful goals
        # can exercise failure mining too. No missing outcome becomes unknown.
        covered = {(r.target, r.value) for r in self.live.values()}
        manual = [a for t, v, a in negatives if (t, v) not in covered]
        self.h.add_atoms_no_check(manual)
        self.anticipated += sum((t, v) in covered for t, v, _ in negatives)
        self.h.add_atoms_no_check([f'(: close{self.period} (NextPeriod p{self.period-1} p{self.period}) (STV 1 1))'])
        verdicts = self.h.review_hypotheses(steps=self.budget)
        self.raw = [str(v) for v in verdicts]
        events = []
        for raw in self.raw:
            row = parse(raw)
            if not isinstance(row, list) or len(row) != 5 or row[0] != 'hypothesis':
                raise ValueError(f'unrecognized hypothesis verdict: {raw}')
            _, name, implication, tv, verdict = row
            try:
                rule = decode(implication, self.domains)
                positive_tv = tv[1] if tv[0] == 'CTV' else tv
                rule = replace(rule, strength=float(positive_tv[1]))
            except (ValueError, IndexError, TypeError) as error:
                self.unparsed[name] = {'raw': raw, 'reason': str(error)}
                continue
            old = name in self.live
            if verdict == 'dropped':
                self.live.pop(name, None)
                self.negative_rates.pop(name, None)
            else:
                self.live[name] = rule
                if tv[0] == 'CTV':
                    self.negative_rates[name] = float(tv[2][1])
            events.append({'event': 'observed_retirement' if verdict == 'dropped' else ('review' if old else 'candidate'),
                           'candidate': name, 'target': rule.target, 'strength': rule.strength,
                           'raw': raw})
        return events

    def rules(self):
        # A probabilistic planted rule can have strength below 0.5. Recovery
        # asks which condition raises its target rate, not which label wins.
        rules = []
        for name, r in self.live.items():
            counts = self.label_counts[r.target]
            baseline = counts[self.domains[r.target].index(r.value)]/max(1,sum(counts))
            other = self.negative_rates.get(name, baseline)
            if r.strength > other:
                rules.append(r)
            if name in self.negative_rates and other > r.strength:
                # CTV's negative branch conditions on NOT(the whole body).
                # De Morgan gives one disjunct per negated literal.
                rules.extend(replace(r, body=(replace(l, negative=not l.negative),), strength=other)
                             for l in r.body)
        return rules
