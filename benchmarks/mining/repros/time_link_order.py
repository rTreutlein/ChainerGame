"""Link-last ingestion misses a temporal candidate on the frozen miner.

Run via benchmarks/mining/repros/run.sh time-link-order.
Both arms receive exactly the same explicit facts and declared time places.
Only link insertion order differs. No planted rule is supplied to the miner.
"""
import json
from pettachainer import PeTTaChainer


def probe(link_first):
    h = PeTTaChainer()
    h.set_evidence_confidence_k(1)
    h.set_hypothesis_mining(4, 8)
    h.set_time_link('NextPeriod')
    h.set_positive_only_predicate('NextPeriod')
    # Equal layers prevent contemporaneous F -> Y from masking the lagged rule.
    h.set_predicate_time('F', 2, 0)
    h.set_predicate_time('Y', 2, 0)
    h.add_atoms_no_check(['(: f0 (F entity t0) (STV 1 1))'])
    for i in range(1, 17):
        link = f'(: link{i} (NextPeriod t{i-1} t{i}) (STV 1 1))'
        label = f'(: y{i} (Y entity t{i}) (STV {int(i % 2 == 1)} 1))'
        if link_first:
            h.add_atoms_no_check([link, label])
        else:
            h.add_atoms_no_check([label, link])
        h.add_atoms_no_check([f'(: f{i} (F entity t{i}) (STV {int(i % 2 == 0)} 1))'])
    return [str(v) for v in h.review_hypotheses('(Y entity $t)', steps=40)]


if __name__ == '__main__':
    print(json.dumps({'link_first': probe(True), 'link_last': probe(False)}, indent=2))
