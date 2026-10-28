"""Read-only data/OOF audit. No training or new submission is performed."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
V = ROOT/'v7'
OUT = V/'audit_noise/reports'


def metric(y, p):
    f, t, cut = roc_curve(y, p, drop_intermediate=False)
    j = int(np.argmax(t-f))
    return dict(KS=float(np.max(abs(t-f))), AUC=float(roc_auc_score(y, p)),
                signed_KS=float((t-f)[j]), threshold=float(cut[j]),
                peak_top_fraction=float(np.mean(p >= cut[j])),
                logloss=float(log_loss(y, p)), brier=float(brier_score_loss(y, p)))


def interval(k, n):
    if not n:
        return [None, None]
    p = k/n
    z = 1.959963984540054
    a = 1.+z*z/n
    h = z*np.sqrt(p*(1.-p)/n+z*z/(4.*n*n))
    return [float((p+z*z/(2.*n)-h)/a), float((p+z*z/(2.*n)+h)/a)]


def top(p, fraction):
    keep = np.zeros(len(p), bool)
    keep[np.argsort(-p, kind='stable')[:int(round(len(p)*fraction))]] = True
    return keep


def psi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(fa) or not len(fb):
        return None
    cuts = np.unique(np.quantile(fa, np.linspace(0., 1., 11)))
    cuts = np.r_[-np.inf, cuts[1:-1], np.inf]
    x = np.r_[np.histogram(fa, cuts)[0], np.sum(~np.isfinite(a))]+.5
    y = np.r_[np.histogram(fb, cuts)[0], np.sum(~np.isfinite(b))]+.5
    x, y = x/x.sum(), y/y.sum()
    return float(np.sum((y-x)*np.log(y/x)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(V/'audit_noise/data/application_metadata.parquet')
    train = pd.read_parquet(ROOT/'data/train_features.parquet')
    test = pd.read_parquet(ROOT/'data/test_features.parquet')
    train_meta = meta[meta.dataset == 'train'].set_index('application_id').reindex(train.application_id)
    test_meta = meta[meta.dataset == 'test'].set_index('application_id').reindex(test.application_id)
    assert not train_meta.isna().any().any()  # all raw training fields documented non-null
    assert np.array_equal(train.target.to_numpy(), train_meta.target.to_numpy())
    assert train.customer_id.tolist() == train_meta.customer_id.tolist()
    assert set(train.customer_id).isdisjoint(test.customer_id)
    y = train.target.to_numpy(int)
    dates = pd.to_datetime(train_meta.decision_datetime)
    test_dates = pd.to_datetime(test_meta.decision_datetime)
    # Raw dictionary explicitly says local, timezone-naive Asia/Jakarta timestamps.
    months, quarters = dates.dt.strftime('%Y-%m').to_numpy(), dates.dt.to_period('Q').astype(str).to_numpy()
    half = np.where(dates.dt.month.to_numpy() <= 6, 'H1', 'H2')
    results, predictions = {}, {}
    ids = ['V25', 'V167', 'V178', 'V185', 'V189', 'V193', 'V219', 'V221', 'V238', 'V239']
    folds = None
    for ident in ids:
        d = pd.read_csv(V/'runs'/ident/'OOF.csv', float_precision='round_trip')
        assert d.application_id.tolist() == train.application_id.tolist()
        assert np.array_equal(d.target.to_numpy(), y)
        if folds is None:
            folds = d.fold.to_numpy(int)
        assert np.array_equal(d.fold.to_numpy(int), folds)
        results[ident] = json.loads((V/'runs'/ident/'RESULT.json').read_text())
        predictions[ident] = d.prediction.to_numpy(float)
    b = predictions['V25']
    p = predictions['V221']
    control_metric = metric(y, b)
    rows, fold_rows, normalized = [], [], {}
    for ident in ids:
        score = predictions[ident]
        stats = metric(y, score)
        r = results[ident]
        assert abs(stats['KS']-r['OOF']['KS']) < 1e-12
        assert abs(stats['AUC']-r['OOF']['AUC']) < 1e-12
        norm = np.empty(len(score))
        deltas = []
        for fold in range(5):
            ix = folds == fold
            m, c = metric(y[ix], score[ix]), metric(y[ix], b[ix])
            norm[ix] = (rankdata(score[ix], method='average')-.5)/ix.sum()
            deltas.append(m['KS']-c['KS'])
            fold_rows.append(dict(model=ident, fold=fold, n=int(ix.sum()), events=int(y[ix].sum()),
                                  KS=m['KS'], AUC=m['AUC'], delta_KS=m['KS']-c['KS'],
                                  delta_AUC=m['AUC']-c['AUC'], mean_score=float(score[ix].mean()),
                                  peak_top_fraction=m['peak_top_fraction']))
        normalized[ident] = metric(y, norm)
        rows.append(dict(model=ident, **stats, mean_fold_KS=r['mean_fold_KS'], std_fold_KS=r['std_fold_KS'],
                         delta_pooled_KS=stats['KS']-control_metric['KS'],
                         delta_mean_fold_KS=float(np.mean(deltas)), fold_wins=int(np.sum(np.array(deltas)>0)),
                         rank_normalized_pooled_KS=normalized[ident]['KS'],
                         source_head_gap=float(np.mean([f['train_validation_head_KS_gap'] for f in r['folds']]))))
    pd.DataFrame(rows).to_csv(OUT/'MODEL_COMPARISON.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'FOLD_COMPARISON.csv', index=False)
    captures = []
    for fraction in [.1, .2, control_metric['peak_top_fraction'], .4, .5]:
        selected_b, selected_p = top(b, fraction), top(p, fraction)
        entrant, exit_ = selected_p & ~selected_b, selected_b & ~selected_p
        ks_at = lambda ix: float(y[ix].sum()/y.sum()-(ix.sum()-y[ix].sum())/(len(y)-y.sum()))
        captures.append(dict(population_fraction=fraction, baseline_bad_captured=int(y[selected_b].sum()),
                             candidate_bad_captured=int(y[selected_p].sum()),
                             net_bad_captured=int(y[selected_p].sum()-y[selected_b].sum()),
                             entrants=int(entrant.sum()), entrant_events=int(y[entrant].sum()), exit_events=int(y[exit_].sum()),
                             baseline_KS_at_fraction=ks_at(selected_b), candidate_KS_at_fraction=ks_at(selected_p)))
    cohorts = []
    for name, group in [('month', months), ('quarter', quarters)]:
        for value in sorted(set(group)):
            ix = group == value
            m, c = metric(y[ix], p[ix]), metric(y[ix], b[ix])
            cohorts.append(dict(group=name, value=value, n=int(ix.sum()), events=int(y[ix].sum()),
                                target_rate=float(y[ix].mean()), target_CI=interval(y[ix].sum(), ix.sum()),
                                KS_V25=c['KS'], KS_V221=m['KS'], delta_KS=m['KS']-c['KS'],
                                AUC_V25=c['AUC'], AUC_V221=m['AUC'], delta_AUC=m['AUC']-c['AUC'],
                                test_n=int(np.sum((test_dates.dt.strftime('%Y-%m').to_numpy() if name=='month' else test_dates.dt.to_period('Q').astype(str).to_numpy()) == value))))
    pd.DataFrame(cohorts).to_csv(OUT/'COHORT_COMPARISON.csv', index=False)
    spectral = pd.read_parquet(V/'light_fe/data/spectral_features.parquet').set_index('application_id')
    assert spectral.index.is_unique
    sx, st = spectral.reindex(train.application_id), spectral.reindex(test.application_id)
    drift = []
    for c in json.loads((ROOT/'BASELINE_LOCK.json').read_text())['Kumo']['features']+list(spectral.columns):
        if c in spectral.columns:
            a, e = sx[c].to_numpy(float), st[c].to_numpy(float)
        elif c in train and pd.api.types.is_numeric_dtype(train[c]):
            a, e = train[c].to_numpy(float), test[c].to_numpy(float)
        else:
            continue
        aa, ee = a[np.isfinite(a)], e[np.isfinite(e)]
        if not len(aa) or not len(ee):
            continue
        drift.append(dict(feature=c, train_missing=float(np.mean(~np.isfinite(a))),
                          test_missing=float(np.mean(~np.isfinite(e))), train_median=float(np.median(aa)),
                          test_median=float(np.median(ee)), train_test_marginal_KS=float(ks_2samp(aa, ee).statistic),
                          PSI=psi(a,e)))
    pd.DataFrame(drift).sort_values('PSI', ascending=False).to_csv(OUT/'FEATURE_DRIFT.csv', index=False)
    # Data-defined bins, shared across halves; every table below is diagnostic,
    # not a new fitted feature or an independent time-validation experiment.
    signals = {c:train[c].to_numpy(float) for c in [
        'declared_monthly_income', 'installment_observed_income_ratio', 'repayment_installment_count',
        'v2_ecom_30d_orders', 'v2_ecom_30d_spend', 'ecom_last_purchase_recency_days']}
    signals.update({c:sx[c].to_numpy(float) for c in spectral.columns if c != 'beh_spectral_window_supported'})
    pattern_rows = []
    rank_change = (rankdata(p)-rankdata(b))/len(y)
    delta = logit(np.clip(p, 1e-6, 1.-1e-6))-logit(np.clip(b, 1e-6, 1.-1e-6))
    for feature, values in signals.items():
        valid = np.isfinite(values)
        cuts = np.unique(np.quantile(values[valid], [.25,.5,.75]))
        groups = np.where(valid, np.searchsorted(cuts, values, side='right'), -1)
        for h in ['ALL', 'H1', 'H2']:
            for group in sorted(set(groups)):
                ix = (groups == group) & (np.ones(len(y), bool) if h=='ALL' else half==h)
                if ix.sum() < 50:
                    continue
                c, q = metric(y[ix], b[ix]), metric(y[ix], p[ix])
                pattern_rows.append(dict(feature=feature, half=h, bin=int(group), cuts=cuts.tolist(),
                                         n=int(ix.sum()), events=int(y[ix].sum()), target_rate=float(y[ix].mean()),
                                         target_CI=interval(y[ix].sum(), ix.sum()),
                                         mean_rank_change=float(rank_change[ix].mean()), mean_logit_change=float(delta[ix].mean()),
                                         delta_KS=q['KS']-c['KS'], delta_AUC=q['AUC']-c['AUC']))
    pd.DataFrame(pattern_rows).to_csv(OUT/'DATA_PATTERNS.csv', index=False)
    # Retrospective selection sensitivity among existing frozen predictions.
    # This is NOT nested CV and not an independent performance estimate:
    # historical candidate design already reused the full labeled dataset.
    candidate_ids = ['V25']
    vectors = [b]
    excluded = []
    for path in sorted((V/'runs').glob('V*/RESULT.json'), key=lambda q:int(q.parent.name[1:])):
        ident = path.parent.name
        if not 120 <= int(ident[1:]) <= 239:
            continue
        oof = path.parent/'OOF.csv'
        if not oof.exists():
            excluded.append(ident)
            continue
        d = pd.read_csv(oof, float_precision='round_trip')
        assert d.application_id.tolist() == train.application_id.tolist()
        assert np.array_equal(d.target.to_numpy(), y)
        candidate_ids.append(ident)
        vectors.append(d.prediction.to_numpy())
    matrix = np.array(vectors)
    order = np.argsort(matrix, axis=1, kind='stable')
    sorted_y = y[order]
    sorted_p = np.take_along_axis(matrix, order, axis=1)
    ends = np.column_stack([sorted_p[:,1:] != sorted_p[:,:-1], np.ones(len(candidate_ids), bool)])
    full_pos = np.cumsum(sorted_y, axis=1, dtype=np.int32)
    full_neg = np.arange(1, len(y)+1, dtype=np.int32)[None,:]-full_pos
    rng = np.random.default_rng(65027)
    trials = []
    strata = [np.flatnonzero((folds==f)&(y==c)) for f in range(5) for c in [0,1]]
    for repeat in range(40):
        select = np.zeros(len(y), bool)
        for ix in strata:
            select[rng.choice(ix, len(ix)//2, replace=False)] = True
        selected = select[order]
        pos = np.cumsum(selected & (sorted_y==1), axis=1, dtype=np.int32)
        neg = np.cumsum(selected & (sorted_y==0), axis=1, dtype=np.int32)
        np1, nn1 = int(np.sum(select & (y==1))), int(np.sum(select & (y==0)))
        s = np.max(np.where(ends, abs(pos/np1-neg/nn1), 0.), axis=1)
        e = np.max(np.where(ends, abs((full_pos-pos)/(int(y.sum())-np1)-(full_neg-neg)/(len(y)-int(y.sum())-nn1)), 0.), axis=1)
        winner = int(np.argmax(s))
        trials.append(dict(repeat=repeat, selected=candidate_ids[winner],
                           selection_delta_KS=float(s[winner]-s[0]), complement_delta_KS=float(e[winner]-e[0]),
                           V221_selection_delta=float(s[candidate_ids.index('V221')]-s[0]),
                           V221_complement_delta=float(e[candidate_ids.index('V221')]-e[0])))
        if repeat % 10 == 0:
            print('SELECTION_SENSITIVITY', repeat, 'models', len(candidate_ids), flush=True)
    pd.DataFrame(trials).to_csv(OUT/'RETROSPECTIVE_SELECTION_SENSITIVITY.csv', index=False)
    trial_frame = pd.DataFrame(trials)
    sensitivity = dict(candidate_count=len(candidate_ids), candidate_ids=candidate_ids, excluded_missing_OOF=excluded, repeats=40, seed=65027,
                       average_selection_delta=float(trial_frame.selection_delta_KS.mean()),
                       average_complement_delta=float(trial_frame.complement_delta_KS.mean()),
                       average_selection_minus_complement=float((trial_frame.selection_delta_KS-trial_frame.complement_delta_KS).mean()),
                       fraction_positive_complement=float((trial_frame.complement_delta_KS > 0.).mean()),
                       winner_counts=trial_frame.selected.value_counts().to_dict(),
                       limitation='Conditional fixed-prediction sensitivity; historical selection used all labels. This is not nested CV or an independent validation estimate.')
    receipt = json.loads((V/'reports/V221_KAGGLE_READONLY_RECEIPT.json').read_text())
    summary = dict(UTC=datetime.now(timezone.utc).isoformat(), raw_rows=len(meta), train_rows=len(y), test_rows=len(test),
                   events=int(y.sum()), unique_selected_customers=int(meta.customer_id.nunique()),
                   train_decision_range=[str(dates.min()),str(dates.max())],
                   test_decision_range=[str(test_dates.min()),str(test_dates.max())],
                   raw_source_manifest=json.loads((V/'audit_noise/data/RAW_SOURCE_MANIFEST.json').read_text()),
                   models=rows, cohorts=cohorts, risk_capture=captures, selection_sensitivity=sensitivity,
                   V221_vs_V25_CI=results['V221']['confidence_interval'], public_receipt=receipt,
                   maximum_feature_PSI=max(d['PSI'] for d in drift), top_feature_drift=sorted(drift,key=lambda d:d['PSI'],reverse=True)[:12],
                   input_sha256={str(q.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(q.read_bytes()).hexdigest() for q in [
                       V/'audit_noise/data/application_metadata.parquet', V/'light_fe/data/spectral_features.parquet', V/'runs/V221/OOF.csv', V/'runs/V25/OOF.csv']},
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   no_new_training=True, no_new_submission=True, no_hidden_test_labels=True)
    (OUT/'VALIDATION_NOISE_AUDIT.json').write_text(json.dumps(summary, indent=2, allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(models=rows, selection=sensitivity, cohorts=[x for x in cohorts if x['group']=='quarter'],
                          top_drift=summary['top_feature_drift'][:4], capture=captures)), flush=True)


if __name__ == '__main__':
    main()
