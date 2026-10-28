"""Five prespecified correction-head mechanisms; frozen Kumo and V25 inputs."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import argparse
import io
import json
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / 'v7'
sys.path[:0] = [str(ROOT / 'scripts'), str(V / 'scripts')]
from baseline import prepare, design, corrected_probability, dump, sha
from experiments import fit_head
from conservative_adapters import main as evaluate, solve
from stability_worker import postprocess


class NaturalCubicBasis(TransformerMixin, BaseEstimator):
    """Restricted cubic basis with four quantile knots and linear tails.

    Fit only on correction rows. Repeated knots become a linear-only feature;
    all columns retain a fixed width so missing/constant inputs remain safe.
    """
    def fit(self, x, y=None):
        a = np.asarray(x, float)
        self.n_features_in_ = a.shape[1]
        self.knots_ = np.quantile(a, [.05, .35, .65, .95], axis=0).T
        self.n_features_out_ = 3 * self.n_features_in_
        return self

    def transform(self, x):
        a = np.asarray(x, float)
        assert a.shape[1] == self.n_features_in_ and np.isfinite(a).all()
        blocks = []
        for j, knots in enumerate(self.knots_):
            span = knots[-1] - knots[0]
            if span <= 1e-10 or np.min(np.diff(knots)) <= 1e-10:
                blocks.append(np.column_stack([a[:, j], np.zeros((len(a), 2))]))
                continue
            t = (a[:, j] - knots[0]) / span
            k = (knots - knots[0]) / span
            den = k[-1] - k[-2]
            terms = [t]
            for q in k[:-2]:
                # Truncated-power restricted cubic spline. Cubic terms cancel
                # beyond the final knot, giving a linear rather than flat tail.
                terms.append(np.maximum(t-q, 0.)**3
                             - (k[-1]-q)/den * np.maximum(t-k[-2], 0.)**3
                             + (k[-2]-q)/den * np.maximum(t-k[-1], 0.)**3)
            blocks.append(np.column_stack(terms))
        return np.column_stack(blocks)


if __name__ == '__main__':
    sys.modules.setdefault('structured_heads', sys.modules[__name__])
    NaturalCubicBasis.__module__ = 'structured_heads'


def weighted_bce(beta, x, z, y, penalty, weights):
    eta = z + x @ beta
    w = weights / weights.sum()
    return (float(w @ (np.logaddexp(0., eta)-y*eta) + .5*np.dot(penalty, beta*beta)),
            x.T @ (w*(expit(eta)-y)) + penalty*beta)


def brier_objective(beta, x, z, y, penalty):
    p = expit(z + x @ beta)
    error = p-y
    return (float(np.mean(error*error) + .5*np.dot(penalty, beta*beta)),
            2.*x.T @ (error*p*(1.-p))/len(y) + penalty*beta)


def spline_groups(head, width):
    nums = list(head['encoder'].transformers_[0][2])
    assert head['encoder'].named_transformers_['smooth'].named_steps['basis'].n_features_out_ == width*len(nums)
    # Each numeric feature's spline coefficients AND missing indicator form
    # one group. Employment dummy block is one group; offset/intercept remain.
    groups = [np.r_[np.arange(2+j*width, 2+(j+1)*width), 2+len(nums)*width+j]
              for j in range(len(nums))]
    start = 2 + len(nums)*(width+1)
    if start < len(head['coefficients']):
        groups.append(np.arange(start, len(head['coefficients'])))
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(2, len(head['coefficients'])))
    return groups


def group_penalty(beta, groups, strength):
    return strength*sum(np.sqrt(len(ix))*np.linalg.norm(beta[ix]) for ix in groups)


def prox_group(value, step, groups, strength):
    out = value.copy()
    for ix in groups:
        norm = np.linalg.norm(value[ix])
        out[ix] *= max(0., 1.-step*strength*np.sqrt(len(ix))/max(norm, 1e-30))
    return out


def solve_group_lasso(x, z, y, penalty, groups, strength):
    """Convex ridge + group-lasso offset logistic fit, monotone proximal steps."""
    weights = np.ones(len(y))
    with threadpool_limits(limits=2):
        lipschitz = .25*np.linalg.eigvalsh(x.T@x/len(y)).max()+penalty.max()
        step = 1./max(lipschitz, 1e-10)
        beta = np.zeros(x.shape[1])
        for iteration in range(20000):
            value, gradient = weighted_bce(beta, x, z, y, penalty, weights)
            nxt = prox_group(beta-step*gradient, step, groups, strength)
            mapping = np.max(abs(nxt-beta))/step
            if mapping < 2e-7:
                beta = nxt
                break
            beta = nxt
        value, gradient = weighted_bce(beta, x, z, y, penalty, weights)
        mapping = float(np.max(abs(beta-prox_group(beta-step*gradient, step, groups, strength)))/step)
    assert mapping < 1e-6, ('group lasso proximal KKT residual', mapping)
    return beta, dict(objective=value+group_penalty(beta, groups, strength),
                      gradient_max_abs=mapping, proximal_gradient_max_abs=mapping,
                      raw_smooth_gradient_max_abs=float(np.max(abs(gradient))),
                      gradient_diagnostic='proximal KKT mapping, not smooth gradient',
                      iterations=iteration+1, active_groups=int(sum(np.linalg.norm(beta[ix]) > 1e-8 for ix in groups)),
                      total_groups=len(groups), group_lambda=strength)


def natural_head(h, k, columns):
    cats = h[columns].select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    nums = [c for c in columns if c not in cats]
    encoder = ColumnTransformer([
        ('smooth', Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('basis', NaturalCubicBasis()), ('scale', StandardScaler())]), nums),
        ('missing', MissingIndicator(features='all', sparse=False, error_on_new=False), nums),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])
    basis = encoder.fit_transform(prepare(h, columns, cats))
    z = logit(np.clip(k, 1e-6, 1.-1e-6))
    mean, sd = float(z.mean()), max(float(z.std()), 1e-8)
    x = np.column_stack([np.ones(len(h)), (z-mean)/sd, basis])
    penalty = np.full(x.shape[1], .003)
    penalty[0] = 0.
    beta, audit = solve(weighted_bce, np.zeros(x.shape[1]), (x, z, h.target.to_numpy(int), penalty, np.ones(len(h))))
    return dict(encoder=encoder, columns=columns, categorical=cats, offset_mean=mean,
                offset_sd=sd, coefficients=beta, spec=dict(correction_strength=.5, penalty=.003),
                positive_weight=1., **audit)


def fit_adapter(f, spec, columns):
    h, k = f['h'], f['kh']
    y = h.target.to_numpy(int)
    method = spec['method']
    head = natural_head(h, k, columns) if method == 'natural_cubic' else fit_head(h, k, columns, .003, .5)
    x, z = design(head, h, k)
    penalty = np.full(x.shape[1], .003)
    penalty[0] = 0.
    model = dict(kind=method, head=head)
    audit = dict(gradient_max_abs=head['gradient_max_abs'], iterations=head['iterations'])
    count = len(head['coefficients'])
    if method == 'group_sparse':
        width = head['encoder'].named_transformers_['smooth'].named_steps['basis'].n_features_out_ // len(head['encoder'].transformers_[0][2])
        groups = spline_groups(head, width)
        head['coefficients'], audit = solve_group_lasso(x, z, y, penalty, groups, spec['group_lambda'])
        model['groups'] = groups
    elif method == 'spline_pca':
        # Compress covariates only. Keep intercept and standardized Kumo logit
        # separate; PCA never sees outcomes, validation rows or test rows.
        pca = PCA(n_components=spec['retained_variance'], svd_solver='full', whiten=False)
        u = pca.fit_transform(x[:, 2:])
        xp = np.column_stack([x[:, :2], u])
        pen = np.full(xp.shape[1], .003)
        pen[0] = 0.
        beta, audit = solve(weighted_bce, np.zeros(xp.shape[1]), (xp, z, y, pen, np.ones(len(y))))
        model.update(pca=pca, coefficients=beta)
        count = len(beta)
        audit.update(retained_components=int(pca.n_components_), total_basis_columns=x.shape[1]-2,
                     explained_variance=float(pca.explained_variance_ratio_.sum()), labels_used_for_PCA=False)
    elif method == 'leverage_weighted':
        # Training-only ridge leverage; no outcomes used to decide weights.
        b = x[:, 2:]
        mean = b.mean(axis=0)
        centered = b-mean
        cov = centered.T@centered/len(b)+.1*np.eye(b.shape[1])
        inverse = np.linalg.inv(cov)
        d2 = np.maximum(np.einsum('ij,ij->i', centered@inverse, centered), 0.)
        threshold = float(np.quantile(d2, .95))
        weights = np.clip(np.sqrt(max(threshold, 1e-12)/np.maximum(d2, 1e-12)), .25, 1.)
        head['coefficients'], audit = solve(weighted_bce, head['coefficients'], (x, z, y, penalty, weights))
        model.update(leverage_mean=mean, leverage_inverse=inverse, leverage_threshold=threshold)
        audit.update(weight_min=float(weights.min()), downweighted_rows=int(np.sum(weights < 1.)),
                     weight_definition='clip(sqrt(training_p95_d2/max(d2,1e-12)),.25,1)',
                     query_downweighting=False, class_balance_changed=True,
                     weighted_event_rate=float(np.dot(weights, y)/weights.sum()))
    elif method == 'brier_loss':
        # Brier proper scoring loss under a logistic parameterization is not
        # globally convex; use one prespecified BCE initialization, no search.
        initial_loss = float(brier_objective(head['coefficients'], x, z, y, penalty)[0])
        model['head']['coefficients'], audit = solve(brier_objective, head['coefficients'], (x, z, y, penalty))
        audit.update(nonconvex_parameterization=True, initializer='same-fold BCE head',
                     loss_before=initial_loss)
        assert audit['objective'] <= initial_loss+1e-10
    else:
        assert method == 'natural_cubic'
    model['optimization'] = audit
    return model, audit, count


def probability(model, frame, k, base_head):
    if model['kind'] == 'spline_pca':
        x, z = design(model['head'], frame, k)
        xp = np.column_stack([x[:, :2], model['pca'].transform(x[:, 2:])])
        delta = xp@model['coefficients']
        return expit(z+.5*delta), delta
    return corrected_probability(model['head'], frame, k)


def preflight():
    rng = np.random.default_rng(61027)
    n = 1000
    k = expit(rng.normal(-1.5, .8, n))
    h = pd.DataFrame(dict(application_id=[f'SYNTHETIC_{i}' for i in range(n)],
                          a=rng.lognormal(size=n), b=rng.normal(size=n), constant=np.ones(n),
                          employment_type=np.resize(['salaried', 'self_employed', 'variable_income'], n),
                          target=rng.binomial(1, k)))
    h.loc[h.index[::7], 'a'] = np.nan
    h['all_missing'] = np.nan
    columns = ['a', 'b', 'constant', 'all_missing', 'employment_type']
    base = fit_head(h, k, columns, .003, .5)
    f = dict(h=h, kh=k, head=base, synthetic_preflight=True)
    x = rng.normal(size=(100, 6))
    x[:, 0] = 1.
    z = rng.normal(size=len(x))
    y = rng.binomial(1, expit(z))
    beta = rng.normal(0., .1, x.shape[1])
    penalty = np.full(x.shape[1], .003)
    penalty[0] = 0.
    gradients = []
    for name, fun, args in [
            ('weighted_BCE', weighted_bce, (x, z, y, penalty, rng.uniform(.25, 1., len(y)))),
            ('Brier', brier_objective, (x, z, y, penalty))]:
        error = check_grad(lambda b: fun(b, *args)[0], lambda b: fun(b, *args)[1], beta)
        assert error < 1e-5, (name, error)
        gradients.append(dict(method=name, gradient_error=float(error)))
    group = [np.array([1, 2]), np.array([3, 4, 5])]
    sparse, opt = solve_group_lasso(x, z, y, penalty, group, .01)
    objective = lambda b: weighted_bce(b, x, z, y, penalty, np.ones(len(y)))[0]+group_penalty(b, group, .01)
    assert objective(sparse) <= objective(np.zeros(x.shape[1]))+1e-10
    # An affine sequence outside each boundary must have zero second difference.
    cubic = NaturalCubicBasis().fit(np.arange(100., dtype=float)[:, None])
    for values in [np.array([-20., -10., 0.]), np.array([100., 110., 120.])]:
        assert np.allclose(np.diff(cubic.transform(values[:, None]), n=2, axis=0), 0., atol=1e-9)
    branches = []
    for spec in json.loads((V/'structured_head_plan.json').read_text()):
        model, audit, count = fit_adapter(f, spec, columns)
        query = h.iloc[:100].copy()
        query['employment_type'] = 'UNSEEN'
        query['a'] = np.linspace(-10., 1000., len(query))
        p, _ = probability(model, query, k[:100], base)
        assert np.isfinite(p).all() and np.all((p > 0.) & (p < 1.))
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        again, _ = probability(joblib.load(buffer), query, k[:100], base)
        assert np.array_equal(p, again)
        branches.append(dict(method=spec['method'], parameters=count, checkpoint_replay_exact=True,
                             training_only_preprocessing=True, optimization=audit))
    dump(dict(status='PASS', synthetic_only=True, gradient_checks=gradients,
              natural_cubic_linear_tail_check=True, branches=branches, source_sha256=sha(__file__)),
         V/'structured_head/reports/PREFLIGHT.json')
    print('STRUCTURED_HEAD_PREFLIGHT_PASS', flush=True)


def main(ident):
    spec = next(s for s in json.loads((V/'structured_head_plan.json').read_text()) if s['id'] == ident)
    pre = json.loads((V/'structured_head/reports/PREFLIGHT.json').read_text())
    assert pre['status'] == 'PASS' and pre['source_sha256'] == sha(__file__)
    return evaluate(ident, plan_file='structured_head_plan.json', adapter_fit=fit_adapter,
                    adapter_probability=probability, method_formulas={spec['method']: spec['formula']},
                    spectral_methods=[], worker_source='structured_heads.py', postprocess=postprocess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        if args.preflight:
            preflight()
        else:
            assert args.run
            main(args.run)
    except Exception:
        if args.run:
            dump(dict(status='FAILED', traceback=traceback.format_exc()), V/'runs'/args.run/'STATUS.json')
        raise
