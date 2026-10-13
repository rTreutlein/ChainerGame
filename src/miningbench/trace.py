"""Directed-miner trace v1: costs and decisions, not inferred internal behavior.

Custom adapters receive set_targets(requests) before predict(), and may return
these event dictionaries from learn(). The benchmark checks reported decisions;
raw events remain available to verify claims against cases in samples.jsonl.
"""
from collections import Counter, defaultdict
import math

DECISIONS = {
    'target': ('signal', 'priority', 'action_relevance'),
    'split': ('parent', 'candidate', 'literal', 'residual_cases', 'gain'),
    'queue': ('candidate', 'expected_gain', 'estimated_cost', 'lag', 'prior', 'selected', 'best_available_gain_per_cost'),
    'template': ('candidate', 'template', 'role', 'bound_variables', 'used_variables', 'learned_constant', 'constant_from_data'),
    'pair': ('candidate', 'cells', 'singles_stalled', 'interaction_score'),
    'overlap': ('parents', 'candidate', 'joint_cases', 'joint_rate'),
    'soft_sample': ('candidate', 'strength', 'positive_mass', 'negative_mass'),
    'generalize': ('parents', 'candidate', 'skeletons', 'variables', 'variable_cap', 'pooled_cases'),
    'specialize': ('parent', 'candidate', 'exception_cases'),
    'promote': ('candidate', 'support', 'parent_log_loss', 'child_log_loss', 'complexity_cost', 'confident'),
    'retire': ('candidate', 'reason', 'opportunities'),
    'compile': ('candidate', 'indexed', 'backfilled_cases', 'duplicate_samples'),
    'budget': ('active_targets', 'beam', 'candidate_updates', 'pair_updates', 'new_candidates', 'new_cap'),
    'invent': ('condition', 'uses', 'name'),
}


class TraceAudit:
    def __init__(self):
        self.counts = Counter()
        self.sums = defaultdict(float)
        self.metrics = Counter()
        self.errors = []
        self.error_count = 0

    def add(self, e):
        kind = e['event']
        if kind not in DECISIONS:
            return
        def error(message):
            self.error_count += 1
            if len(self.errors) < 32:
                self.errors.append({'event': kind, 'step': e.get('step'), 'error': message})
        missing = [f for f in DECISIONS[kind] if f not in e]
        if missing:
            error('missing fields: '+', '.join(missing))
            return
        self.counts[kind] += 1
        def metric(name, value):
            self.sums[name] += value
            self.metrics[name] += 1
        if kind == 'target':
            if e['signal'] not in ('surprise','low_confidence','gap'):
                error('unknown target signal')
            metric('irrelevant_target_fraction', e['action_relevance'] == 0)
        elif kind == 'split':
            metric('split_gain', e['gain'])
            metric('split_residual_cases', len(e['residual_cases']))
        elif kind == 'queue' and e['selected']:
            if e['estimated_cost'] <= 0:
                error('nonpositive estimated cost')
            else:
                metric('queue_gain_per_cost_regret', e['best_available_gain_per_cost']-e['expected_gain']/e['estimated_cost'])
                metric('selected_lag', e['lag'])
                metric('selected_lag_prior', e['prior'])
        elif kind == 'template':
            if e['template'] not in ('order','offset','linear','sum','difference','declared','expression'):
                error('unknown numeric template')
            if not set(e['used_variables']) <= set(e['bound_variables']):
                error('template uses unbound variables')
            if e['learned_constant'] is not None and not e['constant_from_data']:
                error('constant enumerated instead of learned')
            if e['role'] not in ('condition','formula'):
                error('unknown template role')
            metric('formula_fraction', e['role'] == 'formula')
        elif kind == 'pair':
            # Each cell is [outcome-zero mass, outcome-one mass].
            cells = e['cells']
            if len(cells) != 4 or any(len(c) != 2 or min(c) < 0 for c in cells):
                error('expected a nonnegative 2x2x2 outcome table')
            else:
                rates = [c[1]/sum(c) if sum(c) else 0 for c in cells]
                metric('pair_interaction_contrast', abs(rates[0]-rates[1]-rates[2]+rates[3]))
            metric('pairs_before_singles_stall', not e['singles_stalled'])
        elif kind == 'overlap':
            metric('joint_support', len(e['joint_cases']))
        elif kind == 'soft_sample':
            if abs(e['positive_mass']-e['strength']) > 1e-8 or abs(e['negative_mass']-(1-e['strength'])) > 1e-8:
                error('sample does not use strength-weighted masses')
        elif kind == 'generalize':
            if len(set(e['skeletons'])) != 1 or len(e['variables']) > e['variable_cap']:
                error('incompatible skeletons or variable cap exceeded')
            metric('pooled_cases', len(set(e['pooled_cases'])))
            if len(e['pooled_cases']) != len(set(e['pooled_cases'])):
                error('pooled evidence duplicated')
        elif kind == 'specialize':
            metric('exception_support', len(e['exception_cases']))
        elif kind == 'promote':
            gain = e['parent_log_loss']-e['child_log_loss']-e['complexity_cost']
            metric('penalized_promotion_gain', gain)
            if not e['confident'] or gain <= 0:
                error('promotion lacks confidently positive penalized gain')
        elif kind == 'retire':
            metric('retirement_opportunities', e['opportunities'])
        elif kind == 'compile':
            metric('backfilled_cases', len(e['backfilled_cases']))
            if not e['indexed'] or e['duplicate_samples']:
                error('nonindexed promotion or duplicated evidence')
        elif kind == 'budget':
            cap = e['active_targets']*e['beam']
            if e['candidate_updates'] > cap or e['pair_updates'] > cap or e['new_candidates'] > e['new_cap']:
                error('work exceeds configured beam or proposal budget')
            metric('candidate_updates', e['candidate_updates'])
            metric('pair_updates', e['pair_updates'])
        elif kind == 'invent':
            metric('invented_condition_reuse', e['uses'])

    def result(self):
        return {'version': 1, 'decisions': {k: {'events': self.counts[k],
                'status': 'reported' if self.counts[k] else 'not_reported'} for k in DECISIONS},
                'means': {k: v/self.metrics[k] for k,v in self.sums.items()},
                'violation_count': self.error_count, 'violations': self.errors}


def audit(events):
    a = TraceAudit()
    for event in events:
        a.add(event)
    return a.result()


def target_requests(predictions, labels, domains, learned_targets):
    """Observable signals for the *next* step; never uses planted truth/probability."""
    requests = []
    for i, (target, domain) in enumerate(domains.items()):
        p = predictions.get(target, [1/len(domain)]*len(domain))
        entropy = -sum(v*math.log(v) for v in p if v)
        surprise = -math.log(max(1e-12,p[labels[target]])) if target in labels else 0
        # Public decision utility: every third competing target is diagnostic only.
        relevance = float(i % 3 != 2)
        for signal, weight in [('gap', float(target not in learned_targets)),
                               ('low_confidence', entropy), ('surprise', surprise)]:
            requests.append({'target': target, 'signal': signal, 'action_relevance': relevance,
                             'priority': weight*relevance, 'weight': weight})
    return requests
