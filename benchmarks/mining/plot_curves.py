"""Export base-case learning curves from completed runs (optional matplotlib)."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
results = [json.loads(p.read_text()) for p in root.glob('base-*/result.json')]
kinds = sorted({k for r in results for k in r['by_type']})
fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
colors = {'oracle':'#338855', 'marginal':'#4477aa', 'petta':'#cc6677'}
for ax, kind in zip(axes.flat, kinds):
    for r in results:
        if kind not in r['by_type']:
            continue
        system = r['metadata']['system']
        points = r['learning_curve']
        ax.plot([p['samples'] for p in points], [p['window'][kind]['brier'] for p in points],
                label=system, color=colors[system], marker='.', linewidth=1.5)
    ax.set_title(kind)
    ax.set_ylabel('Window probability Brier')
    ax.set_xlabel('Samples observed')
    ax.grid(alpha=.2)
axes.flat[0].legend()
fig.suptitle('Planted mining benchmark: prequential error by sample window (seed 1)')
fig.tight_layout()
fig.savefig(root/'learning_curves.svg')
plt.close(fig)
