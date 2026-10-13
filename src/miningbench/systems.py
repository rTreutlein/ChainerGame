"""Learners receive public schema, history and labels, never planted rules."""

from dataclasses import replace
from .world import Rule


class Marginal:
    """Best empirical squared-error decision stump; bounded sufficient statistics."""

    def __init__(self, features, domains):
        self.features, self.domains = features, domains
        self.counts = {
            t: [[[0] * len(d) for _ in range(2)] for _ in features]
            for t, d in domains.items()
        }
        self.total = {t: [0] * len(d) for t, d in domains.items()}
        self.selected = {}
        self.work = 0

    def vector(self, history):
        return [
            int(history[-literal.lag].get((literal.head, literal.args), 0))
            for literal in self.features
        ]

    def predict(self, history):
        self.work = 0
        self.x = self.vector(history)
        result = {}
        for t, domain in self.domains.items():
            best, chosen = -1.0, None
            for j, cells in enumerate(self.counts[t]):
                gain = sum(sum(v * v for v in c) / sum(c) for c in cells if sum(c))
                if gain > best:
                    best, chosen = gain, j
            self.selected[t] = chosen
            c = self.counts[t][chosen][self.x[chosen]]
            result[t] = [(v + 0.5) / (sum(c) + len(domain) * 0.5) for v in c]
        return result

    def learn(self, history, labels):
        for t, y in labels.items():
            self.total[t][y] += 1
            for j, value in enumerate(self.x):
                self.counts[t][j][value][y] += 1
        self.work = len(self.features) * len(labels)
        return []

    def rules(self):
        rules = []
        for t, j in self.selected.items():
            cells = self.counts[t][j]
            total = [
                sum(cell[value] for cell in cells)
                for value in range(len(self.domains[t]))
            ]
            n = sum(total)
            if n < 8:
                continue
            gain = (
                sum(sum(v * v for v in c) / sum(c) for c in cells if sum(c))
                - sum(v * v for v in total) / n
            ) / n
            if gain <= 0.005:
                continue
            for side, c in enumerate(cells):
                if sum(c) < 4:
                    continue
                values = [1] if len(total) == 2 else range(len(total))
                for value in values:
                    p = (c[value] + 0.5) / (sum(c) + len(total) * 0.5)
                    if p > total[value] / n:
                        rules.append(
                            Rule(
                                t,
                                (replace(self.features[j], negative=not bool(side)),),
                                self.domains[t][value],
                                strength=p,
                            )
                        )
        return rules


class BudgetedMarginal(Marginal):
    """Reference fixed-count search: compare lag/relevance priority with flat search.

    Each unit examines and updates one feature/target contingency table. This is
    an externally checkable algorithmic budget, not a claim about PeTTa's internals.
    """

    def __init__(self, features, domains, budget=80, flat=False):
        super().__init__(features, domains)
        self.budget = budget
        self.flat = flat
        self.step = 0
        self.weights = None
        self.schedule = []
        self.tiers = []
        self.set_targets({t: 1.0 for t in domains})

    def set_targets(self, weights):
        if self.weights == weights:
            return
        self.weights = dict(weights)
        targets = (
            list(self.domains)
            if self.flat
            else sorted(self.domains, key=lambda t: -weights[t])
        )
        self.schedule = [(t, j) for t in targets for j in range(len(self.features))]
        if not self.flat:
            from itertools import groupby

            def priority(item):
                return -weights[item[0]], self.features[item[1]].lag

            self.schedule.sort(key=priority)
            self.tiers = [list(items) for _, items in groupby(self.schedule, priority)]

    def predict(self, history):
        self.active = []
        # Reserve at most one input read per target for its learned incumbent.
        # The remaining allowance schedules candidate updates. Prediction never
        # forgets a learned rule merely because another table is being sampled.
        allowance = max(0, self.budget - len(self.domains))
        tiers = [self.schedule] if self.flat else self.tiers
        for tier in tiers:
            count = min(allowance - len(self.active), len(tier))
            offset = (self.step * count) % len(tier)
            self.active.extend(tier[(offset + i) % len(tier)] for i in range(count))
            if len(self.active) == allowance:
                break
        self.x = {}

        def read(index):
            if index not in self.x:
                literal = self.features[index]
                self.x[index] = int(
                    history[-literal.lag].get((literal.head, literal.args), 0)
                )
            return self.x[index]

        for _, index in self.active:
            read(index)
        result = {}
        for target, domain in self.domains.items():
            index = self.selected.get(target)
            counts = (
                self.total[target]
                if index is None
                else self.counts[target][index][read(index)]
            )
            result[target] = [
                (value + 0.5) / (sum(counts) + len(domain) * 0.5) for value in counts
            ]
        self.work = {
            "table_updates": 0,
            "feature_reads": len(self.x),
            "candidate_evaluations": 0,
            "base_rate_updates": 0,
            "by_lag": {},
            "budget": self.budget,
        }
        self.step += 1
        return result

    def quality(self, target, index):
        """Empirical Gini gain on the observations actually sampled for this table."""
        cells = self.counts[target][index]
        total = [
            sum(cell[value] for cell in cells)
            for value in range(len(self.domains[target]))
        ]
        count = sum(total)
        if not count:
            return -1.0, 0, -index
        conditional = (
            sum(
                sum(value * value for value in cell) / sum(cell)
                for cell in cells
                if sum(cell)
            )
            / count
        )
        marginal = sum(value * value for value in total) / count**2
        return conditional - marginal, count, -index

    def learn(self, history, labels):
        for target, value in labels.items():
            self.total[target][value] += 1
        candidates = {target: {index} for target, index in self.selected.items()}
        lag_counts = {}
        for target, index in self.active:
            self.counts[target][index][self.x[index]][labels[target]] += 1
            candidates.setdefault(target, set()).add(index)
            lag = str(self.features[index].lag)
            lag_counts[lag] = lag_counts.get(lag, 0) + 1
        for target, indices in candidates.items():
            ranked = [(self.quality(target, index), index) for index in indices]
            quality, index = max(ranked)
            if quality[0] > 0:
                self.selected[target] = index
            else:
                self.selected.pop(target, None)
        self.work.update(
            table_updates=len(self.active),
            candidate_evaluations=sum(len(indices) for indices in candidates.values()),
            base_rate_updates=len(labels),
            by_lag=lag_counts,
        )
        return []
