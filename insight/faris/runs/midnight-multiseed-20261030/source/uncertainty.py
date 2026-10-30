from pathlib import Path
import time,json
R=Path(__file__).resolve().parents[1]
import numpy as np
from scipy.stats import ks_2samp
def interval(y, selected, reference, folds):
    rng = np.random.default_rng(20261031)
    strata = [np.flatnonzero((folds == f) & (y == k)) for f in range(5) for k in (0, 1)]
    orders, ends = [], []
    for p in (selected, reference):
        order = np.argsort(p, kind='stable')
        orders.append(order)
        ends.append(np.flatnonzero(np.r_[p[order][1:] != p[order][:-1], True]))
    deltas = []
    for repeat in range(2000):
        assert time.time() < json.loads((R/'CLOCK.json').read_text())['deadline_epoch'] - 20
        sample = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        count = np.bincount(sample, minlength=70000)
        value = count * (y/y.sum() - (1-y)/(len(y)-y.sum()))
        a, b = [np.max(abs(np.cumsum(value[order])[end])) for order, end in zip(orders, ends)]
        deltas.append(float(a-b))
    return dict(replicates=2000, seed=20261031, percentile95=np.quantile(deltas,[.025,.975]).tolist(),
                excludes_adaptive_selection_and_refitting=True, independent_confirmation=False)
