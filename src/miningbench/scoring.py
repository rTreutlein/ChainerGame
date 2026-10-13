"""Exact DNF matching, proper probability scores and cost summaries."""
import ast
from fractions import Fraction
from dataclasses import replace
from itertools import permutations
import math


def formula_coefficients(source):
    """Exact affine algebra for the numeric template language; no evaluation."""
    if not source:
        return {}
    def walk(node):
        if isinstance(node, ast.Name) and node.id in ('x1','x2','x3','v1'):
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
    return {k:v for k,v in coefficients.items() if v}


def formula_key(source):
    return repr(tuple(sorted((k,str(v)) for k,v in formula_coefficients(source).items()))) if source else ''


def canonical(rule):
    variables = sorted({a for l in rule.body for a in l.args if a.startswith('$')})
    if len(variables) > 6:
        raise ValueError('alpha normalization supports at most six variables')
    forms = []
    for order in permutations(variables):
        names = {v: f'$v{i}' for i, v in enumerate(order)}
        body = frozenset(replace(l, args=tuple(names.get(a, a) for a in l.args)) for l in rule.body)
        forms.append(tuple(sorted(body)))
    return (rule.target, rule.value, (formula_key(rule.formula), rule.condition), min(forms))


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


SUFFICIENCY_MARGIN = .1


def sufficient_rules(learned, planted, margin=SUFFICIENCY_MARGIN):
    thresholds = {}
    for rule in planted:
        thresholds[rule.target] = min(thresholds.get(rule.target, 1.), rule.strength)
    return [r for r in learned if r.strength >= thresholds.get(r.target, 1.)-margin]


def recovery(learned, planted, types, world=None, phase=0):
    sufficient = sufficient_rules(learned, planted)
    a, b = normal_form(sufficient), normal_form(planted)
    if world is not None:
        from .truth import numeric_contexts, numeric_rule_value
        for target, kind in types.items():
            if kind != 'numeric':
                continue
            # A rule denotes a set of reachable (context, prediction) pairs.
            # This admits grounded wall transitions and formula refinements.
            for rules, normalized in ((sufficient,a),(planted,b)):
                normalized.difference_update({item for item in normalized if item[0] == target})
                for index, history in enumerate(numeric_contexts(world,target)):
                    for rule in rules:
                        if rule.target != target:
                            continue
                        value = numeric_rule_value(rule,history,world.config.numeric_width,phase)
                        if value is not None:
                            normalized.add((target,str(value),'reachable',index))
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
                'relative_growth': slope*n/(self.total/n) if self.total else 0.,
                'constant_cost_screen':n >= 1000 and slope*n <= .5*(self.total/n), 'screen_is_proof':False}


def association_scores(learned, planted, world, phase):
    from .truth import conditional
    sufficient = sufficient_rules(learned, planted)
    rows = []
    for rule in learned:
        if rule in sufficient or rule.formula:
            continue
        exact = conditional(world,rule,phase)
        rows.append({'rule': repr(rule), 'strength':rule.strength, 'exact_conditional':exact,
                     'squared_error':(rule.strength-exact)**2 if exact is not None else None})
    errors = [r['squared_error'] for r in rows if r['squared_error'] is not None]
    return {'rules':rows, 'count':len(rows), 'mean_squared_error':sum(errors)/len(errors) if errors else None,
            'sufficiency_margin':SUFFICIENCY_MARGIN}
