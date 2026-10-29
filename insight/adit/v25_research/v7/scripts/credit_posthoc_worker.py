"""V262: four small credit-conditioned corrections with inner-OOF teachers.

No Kumo training, no new raw-data extraction, no outer-validation tuning.
The fixed spline branch is never used as a teacher on its own training rows.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import argparse, json, sys, traceback
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import check_grad
ROOT = Path(__file__).resolve().parents[2]
V = ROOT / 'v7'
sys.path[:0] = [str(ROOT / 'scripts'), str(V / 'scripts')]
from baseline import dump, sha, design, corrected_probability
from adaptive_adapters import inner_predictions
from conservative_adapters import main as evaluate, residual_objective, solve
from stability_worker import postprocess as stability_postprocess

CONTINUOUS = ['re_late_30_posterior', 'installment_observed_income_ratio',
              'v2_existing_total_burden_observed', 're_outflow_income_geometric_ratio']
TERMS = ['late30_x_allocation_reliability', 'installment_x_conditional_deficit',
         'installment_x_outflow_pressure', 'total_burden_x_sparse_allocation']


def values(frame):
    raw = frame[CONTINUOUS].to_numpy(float, copy=True)
    raw[~np.isfinite(raw)] = np.nan
    return np.sign(raw) * np.log1p(np.abs(raw))


def counts(frame):
    cols = ['aux_allocation_count', 'risk_conditional_negative_months',
            'risk_conditional_nonnegative_months', 'aux_account_months']
    a = frame[cols].to_numpy(float, copy=True)
    a[~np.isfinite(a)] = np.nan
    assert np.all(np.isnan(a) | (a >= 0)), 'Count columns must be nonnegative'
    return a.T


def fit_reference(frame):
    a = values(frame)
    median = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in a.T])
    sd = np.array([max(float(np.nanstd(c)), 1e-8) if np.isfinite(c).any() else 1. for c in a.T])
    n, neg, pos, months = counts(frame)
    ok = np.isfinite(neg) & np.isfinite(pos) & ((neg + pos) > 0)
    mu = float((np.sum(neg[ok]) + .5) / (np.sum(neg[ok] + pos[ok]) + 1.))
    observed = np.isfinite(n)
    reliability = n[observed] / (n[observed] + 10.)
    center = float(reliability.mean()) if len(reliability) else 0.
    return dict(median=median, sd=sd, deficit_prior_mean=mu,
                reliability_mean=center, fit_rows=len(frame))


def interactions(frame, ref):
    a = values(frame)
    observed = np.isfinite(a)
    z = np.clip((np.where(observed, a, ref['median']) - ref['median']) / ref['sd'], -3., 3.)
    n, neg, pos, months = counts(frame)
    rep = np.isfinite(n)
    nr = np.where(rep, n, 0.)
    rel = nr / (nr + 10.)
    cash = np.isfinite(neg) & np.isfinite(pos) & ((neg + pos) > 0)
    exposure = np.where(cash, neg + pos, 0.)
    mu = ref['deficit_prior_mean']
    deficit = (np.where(cash, neg, 0.) + 2 * mu) / (exposure + 2.)
    account = np.isfinite(months) & (months > 0)
    m = np.where(account, months, 0.)
    mask = np.column_stack([observed[:, 0] & rep & (nr > 0),
                           observed[:, 1] & cash,
                           observed[:, 1] & observed[:, 3] & account,
                           observed[:, 2] & rep])
    x = np.column_stack([
        z[:, 0] * (rel - ref['reliability_mean']),
        z[:, 1] * (deficit - mu) * exposure / (exposure + 3.),
        z[:, 1] * z[:, 3] * m / (m + 3.),
        z[:, 2] * (ref['reliability_mean'] - rel),
    ])
    x = np.clip(x, -4., 4.) * mask
    assert np.isfinite(x).all()
    return x, mask


def fit_adapter(f, spec, columns):
    h = f['h']
    teacher, inner_heads, assignments, checks = inner_predictions(f, columns)
    # Only the 6k correction sample enters these references and projections.
    ref = fit_reference(h)
    raw, mask = interactions(h, ref)
    basis, _ = design(f['head'], h, f['kh'])
    projection = np.empty((basis.shape[1], len(TERMS)))
    for j in range(len(TERMS)):
        b = basis[mask[:, j]]
        projection[:, j] = np.linalg.solve(b.T @ b + 1e-3 * np.eye(b.shape[1]),
                                            b.T @ raw[mask[:, j], j])
    residual = np.clip(raw - basis @ projection, -4., 4.) * mask
    offset = logit(np.clip(teacher, 1e-6, 1 - 1e-6))
    beta, audit = solve(residual_objective, np.zeros(len(TERMS)),
                        (residual, offset, h.target.to_numpy(int),
                         spec['penalty'], spec['teacher_tether']))
    audit.update(inner_crossfit_used=True, inner_blocks=checks,
                 outer_validation_used_for_fitting=False,
                 spline_training_scores_used=False,
                 interaction_coefficients=dict(zip(TERMS, map(float, beta))),
                 supported_training_rows=dict(zip(TERMS, map(int, mask.sum(axis=0)))),
                 teacher='3-fold inner OOF corrected Kumo; 4000 fit / 2000 query',
                 allocation_count_is_DPD_denominator=False)
    model = dict(reference=ref, projection=projection, coefficients=beta,
                 strength=spec['strength'], cap=spec['logit_cap'], terms=TERMS,
                 inner_heads=inner_heads, inner_assignments=assignments,
                 inner_teacher_predictions=teacher)
    return model, audit, len(beta)


def probability(model, frame, kumo, head):
    p, _ = corrected_probability(head, frame, kumo)
    x, mask = interactions(frame, model['reference'])
    basis, _ = design(head, frame, kumo)
    residual = np.clip(x - basis @ model['projection'], -4., 4.) * mask
    delta = model['strength'] * np.clip(residual @ model['coefficients'], -model['cap'], model['cap'])
    out = expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + delta)
    out[delta == 0] = p[delta == 0]
    assert np.max(abs(delta)) <= model['strength'] * model['cap'] + 1e-12
    return out, delta


def postprocess(out, result, tr, y, folds, reference, pred, frames):
    result['overfitting_assessment'].update(
        inner_crossfit_used=True,
        diagnostic='New four-parameter head uses inner-OOF corrected-Kumo teacher. '
                   'No training predictions from spline are used. Outer labels only evaluate. '
                   '6k new-head training scores remain optimistic; development-fold reuse '
                   'and data-driven choice after prior residual audit remain selection risks.')
    result['engineered_interaction_features'] = TERMS
    result['features_changed_vs_V25'] = True
    result['raw_sources_changed_vs_V25'] = False
    result['Kumo_input_features_unchanged'] = True
    result['original_V25_head_unchanged'] = True
    result['new_target_fitted_parameters_per_fold'] = 4
    result['maximum_added_corrected_Kumo_logit'] = .075
    stability_postprocess(out, result, tr, y, folds, reference, pred, frames)
    supported = result['delta_KS'] > 0 and result['delta_AUC'] >= 0 and result['confidence_interval']['delta_KS'][0] > 0
    result['decision'] = 'EXPERIMENTAL_ONLY' if supported else 'KEEP_V25_NO_ESTABLISHED_GAIN'
    report = [f"# {result['experiment']}: credit-aware post-hoc on V25", '',
              result['modeling_formula'], '',
              f"Pooled OOF KS {result['OOF']['KS']:.10f}; AUC {result['OOF']['AUC']:.10f}.",
              f"Delta KS {result['delta_KS']:+.10f}; delta AUC {result['delta_AUC']:+.10f}; fold wins {result['fold_wins']}/5.",
              f"Paired bootstrap 95% delta-KS CI: {result['confidence_interval']['delta_KS']}.", '',
              'The extra head is trained against inner-OOF corrected Kumo, not an in-sample spline teacher. '
              'The original V25 head and 75:25 fusion remain fixed. Allocation-count reliability is only '
              'a coverage proxy, not the denominator of the late30 posterior. Deficit months are eligible '
              'repayment/account-matched months, not all account months.', '',
              'All five outer folds have been used in prior research. Bootstrap intervals are conditional '
              'on fitted predictions and do not correct experiment selection or training variability. '
              'Inherited DPD snapshot and global-reference provenance remains unverified. '
              'This is our operational credit post-hoc experiment, not a verified reproduction of the friend method.', '',
              f"Decision: {result['decision']}. V25 remains the public incumbent (user-reported 0.56822). '",
              'Candidate public score is unknown; no Kaggle submission was performed.']
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


def preflight():
    rng = np.random.default_rng(262031)
    frame = pd.DataFrame({c: rng.lognormal(size=100) for c in CONTINUOUS})
    for c in ['aux_allocation_count', 'risk_conditional_negative_months',
              'risk_conditional_nonnegative_months', 'aux_account_months']:
        frame[c] = rng.integers(0, 12, len(frame)).astype(float)
    frame.loc[0, :] = np.nan
    frame.loc[1, ['aux_allocation_count', 'risk_conditional_negative_months',
                  'risk_conditional_nonnegative_months', 'aux_account_months']] = 0.
    ref = fit_reference(frame)
    x, mask = interactions(frame, ref)
    assert np.all(x[0] == 0) and not mask[0].any()
    assert not mask[1, :3].any() and mask[1, 3]
    # Inference is batch-invariant and cannot fit statistics on query rows.
    ref_before = json.dumps(ref, default=lambda a: a.tolist(), sort_keys=True)
    for j in range(len(frame)):
        one, _ = interactions(frame.iloc[[j]], ref)
        assert np.allclose(one, x[j:j+1], atol=1e-13)
    assert json.dumps(ref, default=lambda a: a.tolist(), sort_keys=True) == ref_before
    z = rng.normal(-1., 1., len(frame)); y = rng.binomial(1, expit(z))
    beta = rng.normal(0., .05, 4); args = (x, z, y, .03, .2)
    err = check_grad(lambda b: residual_objective(b, *args)[0],
                     lambda b: residual_objective(b, *args)[1], beta)
    assert err < 1e-5
    fitted, audit = solve(residual_objective, np.zeros(4), args)
    assert residual_objective(fitted, *args)[0] <= residual_objective(np.zeros(4), *args)[0]
    dump(dict(status='PASS', synthetic_only=True, missing_support_gating=True,
              batch_invariance=True, gradient_error=float(err), optimizer=audit,
              source_sha256=sha(__file__)), V / 'credit_posthoc/reports/PREFLIGHT.json')
    print('CREDIT_POSTHOC_PREFLIGHT_PASS', flush=True)


def main(ident):
    pre = json.loads((V / 'credit_posthoc/reports/PREFLIGHT.json').read_text())
    assert pre['status'] == 'PASS' and pre['source_sha256'] == sha(__file__)
    spec = next(s for s in json.loads((V / 'credit_posthoc_plan.json').read_text()) if s['id'] == ident)
    evaluate(ident, plan_file='credit_posthoc_plan.json', adapter_fit=fit_adapter,
             adapter_probability=probability, method_formulas={spec['method']: spec['formula']},
             spectral_methods=[], worker_source='credit_posthoc_worker.py', postprocess=postprocess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run'); args = parser.parse_args()
    try:
        if args.preflight: preflight()
        else:
            assert args.run
            main(args.run)
    except Exception:
        if args.run: dump(dict(status='FAILED', traceback=traceback.format_exc()), V / 'runs' / args.run / 'STATUS.json')
        raise
