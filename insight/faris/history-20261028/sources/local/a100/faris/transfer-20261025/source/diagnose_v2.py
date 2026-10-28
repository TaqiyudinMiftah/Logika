"""Exploratory error anatomy on previously exposed OOF; never trains a predictor."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logit
import baseline as b
from robustness import R, A, INPUT, MAIN, PERIODIC, inputs

reference = np.load(PERIODIC / 'OOF.npz')
y, folds = reference['target'], reference['fold']
features = {}
raw = np.full(len(y), np.nan)
corrected = np.full(len(y), np.nan)
spline = np.full(len(y), np.nan)
for f in range(5):
    original, ix, q, context, yc, columns, p, frame, mature, models = inputs(f)
    vi = ix['validation']
    for name in ['aux_account_months', 'repayment_installment_count', 'bureau_score_mean', 'beh_spectral_window_supported', 'has_repayments']:
        if name not in features:
            features[name] = np.full(len(y), np.nan)
        if name in q:
            features[name][vi] = pd.to_numeric(q[name].iloc[6000:], errors='coerce').to_numpy()
        elif name in frame:
            features[name][vi] = pd.to_numeric(frame[name].iloc[vi], errors='coerce').to_numpy()
    h, sp = models['baseline']
    raw[vi] = p[6000:]
    corrected[vi] = b.corrected_probability(h, q.iloc[6000:], p[6000:])[0]
    spline[vi] = b.predict_spline(sp, frame.iloc[vi])

groups = {
    'all': np.ones(len(y), dtype=bool),
    'recent_account_months_0_to_1_or_absent': ~np.isfinite(features['aux_account_months']) | (features['aux_account_months'] <= 1),
    'recent_account_months_3': features['aux_account_months'] == 3,
    'no_repayment_installments': ~np.isfinite(features['repayment_installment_count']) | (features['repayment_installment_count'] == 0),
    'repayment_installments_1_to_11': (features['repayment_installment_count'] >= 1) & (features['repayment_installment_count'] < 12),
    'repayment_installments_at_least_12': features['repayment_installment_count'] >= 12,
    'bureau_score_missing': ~np.isfinite(features['bureau_score_mean']),
    'bureau_score_present': np.isfinite(features['bureau_score_mean']),
    'spectral_supported': features['beh_spectral_window_supported'] == 1,
    'spectral_unsupported': features['beh_spectral_window_supported'] == 0,
}
predictions = {'raw_Kumo': raw, 'corrected_Kumo': corrected, 'spline': spline,
               'baseline': reference['baseline'], 'maturity': reference['maturity'], 'periodicity': reference['true']}
output = {}
for name, mask in groups.items():
    row = dict(rows=int(mask.sum()), positives=int(y[mask].sum()))
    if mask.sum() and min(int(y[mask].sum()), int(mask.sum()-y[mask].sum())) >= 20:
        row['observed_default_rate'] = float(y[mask].mean())
        row['models'] = {k: dict(**b.scores(y[mask], p[mask]), mean_probability=float(p[mask].mean()), mean_residual=float((y[mask]-p[mask]).mean())) for k,p in predictions.items()}
    output[name] = row
b.dump(dict(status='complete', predefined_slices=output,
            interpretation='Exploratory on previously reused labels. Slice scores are not independent tests and cannot directly select a learned routing rule.',
            no_new_model_fit=True, source_sha256=b.sha(Path(__file__))), R/'ERROR_ANATOMY_v2.json')
np.savez_compressed(R/'STAGE_OOF_v2.npz', target=y, fold=folds, **predictions, **{f'feature__{k}':v for k,v in features.items()})
