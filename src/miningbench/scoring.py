"""Exact DNF matching, proper probability scores and cost summaries."""
import ast
from fractions import Fraction
from dataclasses import replace
from itertools import permutations
import math
import statistics


def formula_key(source):
    """Exact affine algebra for the numeric template language; no evaluation."""
    if not source:
        return ''
    def walk(node):
        if isinstance(node, ast.Name) and node.id in ('x1','x2','x3'):
            return {node.id: Fraction(1)}
        if isinstance(node, ast.Constant) and isinstance(node.value, (int,float)):
            return {'1': Fraction(str(node.value))}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return {k:-v for k,v in walk(node.operand).items()}
        if isinstance(node, ast.BinOp):
            a,b = walk(node.left), walk(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                sign = -1 if isinstance(node.op, ast.Sub) else 1
                return {k:a.get(k,0)+sign*b.get(k,0) for k in a.keys()|b.keys()}
            if isinstance(node.op, ast.Mult):
                if set(a) <= {'1'}:
                    return {k:a.get('1',0)*v for k,v in b.items()}
                if set(b) <= {'1'}:
                    return {k:b.get('1',0)*v for k,v in a.items()}
        raise ValueError('formula outside affine template language: '+source)
    coefficients = walk(ast.parse(source,mode='eval').body)
    return repr(tuple(sorted((k,str(v)) for k,v in coefficients.items() if v)))


def canonical(rule):
    variables = sorted({a for l in rule.body for a in l.args if a.startswith('$')})
    if len(variables) > 6:
        raise ValueError('alpha normalization supports at most six variables')
    forms = []
    for order in permutations(variables):
        names = {v: f'$v{i}' for i, v in enumerate(order)}
        body = frozenset(replace(l, args=tuple(names.get(a, a) for a in l.args)) for l in rule.body)
        forms.append(tuple(sorted(body)))
    return (rule.target, rule.value, formula_key(rule.formula), min(forms))


def normal_form(rules):
    """Blake canonical cover (all prime implicants), grouped by conclusion.

    Consensus plus absorption gives Boolean equivalence, not just syntactic
    equality. Relational literals preserve shared variables under alpha renaming;
    numeric formulas use the documented template vocabulary, not sampled equality.
    """
    groups = {}
    for rule in rules:
        t, v, f, body = canonical(rule)
        cube = frozenset(body)
        if any(replace(l, negative=not l.negative) in cube for l in cube):
            continue
        groups.setdefault((t, v, f), set()).add(cube)
    output = set()
    for key, cubes in groups.items():
        # Existential variable scopes are local to a rule. Never apply a
        # propositional consensus across independently quantified conjunctions.
        relational = {c for c in cubes if any(a.startswith('$') for l in c for a in l.args)}
        output.update((*key,tuple(sorted(c))) for c in relational)
        cubes = cubes-relational
        while True:
            grown = set(cubes)
            for a in cubes:
                for b in cubes:
                    opposite = [x for x in a if replace(x, negative=not x.negative) in b]
                    if len(opposite) == 1:
                        x = opposite[0]
                        grown.add((a | b)-{x, replace(x, negative=not x.negative)})
            reduced = {a for a in grown if not any(b < a for b in grown)}
            if len(reduced) > 4096:
                raise ValueError('DNF cover exceeds 4096 cubes; cannot claim exact matching')
            if reduced == cubes:
                break
            cubes = reduced
        output.update((*key, tuple(sorted(c))) for c in cubes)
    return output


def recovery(learned, planted, types):
    a, b = normal_form(learned), normal_form(planted)
    def score(x, y):
        n = len(x & y)
        return {'precision': n/len(x) if x else 0.0, 'recall': n/len(y) if y else 1.0,
                'matched': n, 'learned': len(x), 'planted': len(y), 'equivalent': x == y}
    return {'all': score(a, b), 'by_type': {k: score({x for x in a if types.get(x[0]) == k},
                                                    {x for x in b if types.get(x[0]) == k})
                                          for k in sorted(set(types.values()))}}


def losses(predicted, exact):
    if len(predicted) != len(exact) or any(not math.isfinite(p) or p < 0 for p in predicted):
        raise ValueError('invalid predictive distribution')
    if abs(sum(predicted)-1) > 1e-8:
        raise ValueError('predictive distribution must sum to one')
    p = [max(1e-12, v) for v in predicted]
    ce = -sum(q*math.log(v) for q, v in zip(exact, p) if q)
    entropy = -sum(q*math.log(q) for q in exact if q)
    # Mean over categories: binary case equals the familiar (p-q)^2.
    brier = sum((v-q)**2 for v, q in zip(predicted, exact))/len(exact)
    return {'log_loss': ce, 'excess_log_loss': max(0., ce-entropy), 'brier': brier,
            'expected_brier': brier+sum(q*(1-q) for q in exact)/len(exact)}


def average(rows):
    return {k: statistics.fmean(r[k] for r in rows) for k in rows[0]} if rows else {}


class RunningAverage:
    def __init__(self):
        self.count = 0
        self.sums = {}

    def add(self, row):
        self.count += 1
        for key, value in row.items():
            self.sums[key] = self.sums.get(key, 0.)+value

    def result(self):
        return {k:v/self.count for k,v in self.sums.items()} if self.count else {}


class CostMeter:
    def __init__(self, samples):
        self.samples = samples
        self.count = 0
        self.total = self.early = self.late = self.xy = 0.

    def add(self, value):
        quarter = max(1, self.samples//4)
        if self.count < quarter:
            self.early += value
        if self.count >= self.samples-quarter:
            self.late += value
        self.xy += self.count*value
        self.total += value
        self.count += 1

    def result(self):
        n = self.count
        quarter = max(1,self.samples//4)
        early, late = self.early/quarter, self.late/quarter
        slope = (self.xy-(n-1)/2*self.total)/(n*(n*n-1)/12) if n > 1 else 0
        return {'mean_seconds':self.total/n, 'early_seconds':early, 'late_seconds':late,
                'late_over_early':late/early if early else None, 'seconds_per_step_slope':slope,
                'constant_cost_screen':n >= 64 and late <= 2*early, 'screen_is_proof':False}
