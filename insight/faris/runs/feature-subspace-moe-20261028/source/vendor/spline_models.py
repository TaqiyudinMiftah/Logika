"""Exact historical spline200 functions, extracted without executing the old campaign.

Only predictions/aggregate optimizer evidence are persisted by the caller.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.preprocessing import SplineTransformer, StandardScaler
pop_cols = ["pop_A", "pop_B", "pop_C1", "pop_C2"]

def fit_logistic_irls(X_tr, y_tr, C=0.002, sample_weight=None, offset_tr=None, n_iter=4):
    """Exact L2-penalized logistic regression via fast Cholesky Newton-IRLS."""
    n, d = X_tr.shape
    lam = 1.0 / max(C, 1e-8)
    sw = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    off = np.zeros(n, dtype=np.float64) if offset_tr is None else np.asarray(offset_tr, dtype=np.float64)
    # Initialize intercept at log-odds of y_tr if no offset, else 0
    b0 = 0.0 if offset_tr is not None else float(logit(np.clip(np.average(y_tr, weights=sw), 1e-4, 1.0 - 1e-4)))
    w = np.zeros(d, dtype=np.float64)
    Xb = np.zeros(n, dtype=np.float64)

    for _ in range(n_iter):
        eta = off + b0 + Xb
        p = expit(eta)
        hess_w = np.maximum(sw * p * (1.0 - p), 1e-6)
        grad_r = sw * (y_tr - p)
        # Update intercept b0
        b0_step = np.sum(grad_r) / max(np.sum(hess_w), 1e-6)
        b0 += b0_step
        eta += b0_step
        p = expit(eta)
        hess_w = np.maximum(sw * p * (1.0 - p), 1e-6)
        grad_r = sw * (y_tr - p)

        # Solve (X^T diag(hess_w) X + lam I) dw = X^T grad_r - lam * w
        Xw = X_tr * np.sqrt(hess_w)[:, None].astype(np.float32)
        H = (Xw.T @ Xw).astype(np.float64)
        H[np.diag_indices_from(H)] += lam
        rhs = (X_tr.T @ grad_r.astype(np.float32)).astype(np.float64) - lam * w
        dw = np.linalg.solve(H, rhs)
        w += dw
        Xb = (X_tr @ w.astype(np.float32)).astype(np.float64)
        if np.max(np.abs(dw)) < 1e-5:
            break
    return b0, w.astype(np.float32)

def predict_irls(X, b0, w, offset=None):
    eta = b0 + (X @ w).astype(np.float64)
    if offset is not None:
        eta = eta + offset
    return expit(eta), eta

def make_scaled_spline_matrix(df, fit_rows, feature_cols, n_knots=6, degree=2, top_inter_n=30):
    pops = df[pop_cols].to_numpy(dtype=np.float32)
    num_vecs = []
    cat_vecs = [pops]
    miss_vecs = []

    for c in feature_cols:
        if c not in df.columns or c in pop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            v = df[c].to_numpy(dtype=np.float64)
            v = np.sign(v) * np.log1p(np.abs(v))
            miss = ~np.isfinite(v)
            med = np.nanmedian(v[fit_rows])
            if not np.isfinite(med):
                med = 0.0
            vf = np.where(miss, med, v)
            mu = np.mean(vf[fit_rows])
            sd = max(np.std(vf[fit_rows]), 1e-4)
            z = np.clip((vf - mu) / sd, -6.0, 6.0).astype(np.float32)
            num_vecs.append(z)
            m_rate = np.mean(miss[fit_rows])
            if 0.005 <= m_rate <= 0.995:
                miss_vecs.append(miss.astype(np.float32)[:, None])
        else:
            s = df[c].astype("string").fillna("__MISSING__")
            for cat in sorted(s.iloc[fit_rows].unique()):
                cat_vecs.append((s == cat).to_numpy(dtype=np.float32)[:, None])

    num_mat = np.column_stack(num_vecs)
    n_sp = min(60, num_mat.shape[1])
    st = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", extrapolation="constant", include_bias=False)
    st.fit(num_mat[fit_rows, :n_sp])
    sp_mat = st.transform(num_mat[:, :n_sp]).astype(np.float32)

    inter_list = [pops * num_mat[:, j:j+1] for j in range(min(top_inter_n, num_mat.shape[1]))]
    X_raw = np.hstack([num_mat, sp_mat] + cat_vecs + miss_vecs + inter_list).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_raw[fit_rows])
    X_scaled = np.clip(scaler.transform(X_raw), -8.0, 8.0).astype(np.float32)
    return X_scaled


def fixed_spline_fit(training, columns, prepare):
    """Historical fixed spline77/88 learner; permitted fit-size extension to70k."""
    import warnings
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    assert not set(columns) & {'target', 'customer_id', 'application_id', 'split'}
    cats = training[columns].select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numeric = [c for c in columns if c not in cats]
    continuous = [c for c in numeric if training[c].nunique(dropna=True) > 10]
    linear = [c for c in numeric if c not in continuous]
    encoder = ColumnTransformer([
        ('spline', Pipeline([('fill', SimpleImputer(strategy='median')),
                            ('basis', SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant')),
                            ('scale', StandardScaler())]), continuous),
        ('linear', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), linear),
        ('missing', MissingIndicator(features='all', sparse=False, error_on_new=False), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
    ])
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        design = encoder.fit_transform(prepare(training, columns, cats))
        model = LogisticRegression(C=.05, solver='newton-cholesky', max_iter=80, tol=1e-5)
        model.fit(design, training.target.astype(int), sample_weight=np.ones(len(training)))
    messages = [str(w.message) for w in captured]
    assert not any('converge' in message.lower() for message in messages), messages
    return {'encoder': encoder, 'model': model, 'columns': columns, 'categorical': cats,
            'fit_rows': len(training), 'resolved_parameters': model.get_params(), 'warnings': messages}


def direct_spline_objective(delta, design, offset, target, threshold):
    """H016: same anchored smooth-Youden objective and fixed constants."""
    displacement = .5 * (design @ delta)
    smooth = expit((offset + displacement - threshold) / .5)
    weights = np.where(target == 1, -1 / int(target.sum()), 1 / int((target == 0).sum()))
    value = float(weights @ smooth + .5 * np.mean(displacement**2) + .5 * .003 * (delta @ delta))
    gradient = .5 * (design.T @ (weights * smooth * (1 - smooth) / .5 + displacement / len(target))) + .003 * delta
    return value, gradient


def direct_spline_update(old, training, query, prepare):
    """Fit H016 delta in memory and emit predictions plus aggregate diagnostics."""
    from scipy.optimize import minimize, check_grad
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        basis = old['encoder'].transform(prepare(training, old['columns'], old['categorical']))
        offset = old['model'].decision_function(basis)
        design = np.column_stack([np.ones(len(training)), basis])
        y = training.target.to_numpy(int)
        delta0 = np.zeros(design.shape[1]); threshold = float(logit(y.mean()))
        small = np.r_[np.flatnonzero(y == 0)[:96], np.flatnonzero(y == 1)[:32]]
        args = (design, offset, y, threshold)
        smallargs = (design[small], offset[small], y[small], threshold)
        error = float(check_grad(lambda v: direct_spline_objective(v, *smallargs)[0],
                                 lambda v: direct_spline_objective(v, *smallargs)[1], delta0,
                                 direction='random', seed=2026102516))
        assert error < 2e-5, error
        initial = direct_spline_objective(delta0, *args)[0]
        result = minimize(direct_spline_objective, delta0, args=args, method='L-BFGS-B', jac=True,
                          options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-7})
        final, gradient = direct_spline_objective(result.x, *args)
        assert result.success and np.max(np.abs(gradient)) < 1e-4, result.message
        assert final <= initial + 1e-10
        q_basis = old['encoder'].transform(prepare(query, old['columns'], old['categorical']))
        q_design = np.column_stack([np.ones(len(query)), q_basis])
        prediction = expit(old['model'].decision_function(q_basis) + .5 * (q_design @ result.x))
    return prediction, {'derivative_error': error, 'initial_objective': initial, 'final_objective': final,
                        'gradient_max_abs': float(np.max(np.abs(gradient))), 'iterations': int(result.nit),
                        'threshold': threshold, 'fit_rows': len(training), 'strength': .5,
                        'temperature': .5, 'trust': 1., 'ridge': .003}
