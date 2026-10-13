"""Learners receive public schema, history and labels, never planted rules."""
from dataclasses import replace
from .world import Rule


class Marginal:
    """Best empirical squared-error decision stump; bounded sufficient statistics."""
    def __init__(self, features, domains):
        self.features, self.domains = features, domains
        self.counts = {t: [[[0]*len(d) for _ in range(2)] for _ in features] for t, d in domains.items()}
        self.total = {t: [0]*len(d) for t, d in domains.items()}
        self.selected = {}
        self.work = 0

    def vector(self, history):
        return [int(history[-l.lag].get((l.head, l.args), 0)) for l in self.features]

    def predict(self, history):
        self.x = self.vector(history)
        result = {}
        for t, domain in self.domains.items():
            best, chosen = -1., None
            for j, cells in enumerate(self.counts[t]):
                gain = sum(sum(v*v for v in c)/sum(c) for c in cells if sum(c))
                if gain > best:
                    best, chosen = gain, j
            self.selected[t] = chosen
            c = self.counts[t][chosen][self.x[chosen]]
            result[t] = [(v+0.5)/(sum(c)+len(domain)*.5) for v in c]
        return result

    def learn(self, history, labels):
        for t, y in labels.items():
            self.total[t][y] += 1
            for j, value in enumerate(self.x):
                self.counts[t][j][value][y] += 1
        self.work = len(self.features)*len(labels)
        return []

    def rules(self):
        rules = []
        for t, j in self.selected.items():
            total = self.total[t]
            n = sum(total)
            if n < 8:
                continue
            cells = self.counts[t][j]
            gain = (sum(sum(v*v for v in c)/sum(c) for c in cells if sum(c))
                    -sum(v*v for v in total)/n)/n
            if gain <= .005:
                continue
            for side, c in enumerate(cells):
                if sum(c) < 4:
                    continue
                values = [1] if len(total) == 2 else range(len(total))
                for value in values:
                    p = (c[value]+.5)/(sum(c)+len(total)*.5)
                    if p > total[value]/n:
                        rules.append(Rule(t, (replace(self.features[j], negative=not bool(side)),),
                                          self.domains[t][value], strength=p))
        return rules


class BudgetedMarginal(Marginal):
    """Reference fixed-count search: compare lag/relevance priority with flat search.

    Each unit examines and updates one feature/target contingency table. This is
    an externally checkable algorithmic budget, not a claim about PeTTa's internals.
    """
    def __init__(self, features, domains, budget=80, flat=False):
        super().__init__(features,domains)
        self.budget = budget
        self.flat = flat
        self.step = 0
        self.weights = {t:1. for t in domains}

    def set_targets(self, weights):
        self.weights = weights

    def predict(self, history):
        self.x = self.vector(history)
        targets = sorted(self.domains, key=lambda t:-self.weights[t])
        candidates = [(t,j) for t in targets for j in range(len(self.features))]
        if not self.flat:
            candidates.sort(key=lambda item:(-self.weights[item[0]],self.features[item[1]].lag))
        # Flat arm rotates through all tables; prioritized arm allocates by lag
        # and relevance. Both have exactly the same maximum update allowance.
        offset = (self.step*self.budget) % len(candidates) if self.flat else 0
        self.active = (candidates+ candidates)[offset:offset+min(self.budget,len(candidates))]
        result = {}
        for t,domain in self.domains.items():
            eligible = [j for target,j in self.active if target == t]
            if not eligible:
                c = self.total[t]
            else:
                def quality(j):
                    cells = self.counts[t][j]
                    return sum(sum(v*v for v in c)/sum(c) for c in cells if sum(c))
                j = max(eligible,key=quality)
                self.selected[t] = j
                c = self.counts[t][j][self.x[j]]
            result[t] = [(v+.5)/(sum(c)+len(domain)*.5) for v in c]
        return result

    def learn(self, history, labels):
        for t,y in labels.items():
            self.total[t][y] += 1
        lag_counts = {}
        for t,j in self.active:
            self.counts[t][j][self.x[j]][labels[t]] += 1
            lag = str(self.features[j].lag)
            lag_counts[lag] = lag_counts.get(lag,0)+1
        self.work = {'table_updates':len(self.active), 'by_lag':lag_counts, 'budget':self.budget}
        self.step += 1
        return []
