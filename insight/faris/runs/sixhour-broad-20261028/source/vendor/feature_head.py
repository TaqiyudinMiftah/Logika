"""The teammate's additive correction estimator with explicit factor settings."""
import numpy as np
from scipy.special import logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder
from threadpoolctl import threadpool_limits
import baseline as b


def fit(frame, p, columns, penalty, strength):
    assert not set(columns) & {"target", "application_id", "customer_id", "split"}
    cats = frame[columns].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [c for c in columns if c not in cats]
    encoder = ColumnTransformer([
        ("smooth", Pipeline([
            ("fill", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("basis", SplineTransformer(n_knots=3, degree=2, knots="quantile", extrapolation="constant")),
            ("scale", StandardScaler()),
        ]), nums),
        ("missing", MissingIndicator(features="all", sparse=False, error_on_new=False), nums),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])
    with threadpool_limits(limits=2):
        basis = encoder.fit_transform(b.prepare(frame, columns, cats))
        offset = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
        mean = float(offset.mean()); sd = max(float(offset.std()), 1e-8)
        x = np.column_stack([np.ones(len(frame)), (offset - mean) / sd, basis])
        assert np.isfinite(x).all()
        y = frame.target.to_numpy(int); weights = np.ones(len(y))
        result = minimize(
            b.objective, np.zeros(x.shape[1]),
            args=(x, offset, y, weights, penalty, 0.),
            method="L-BFGS-B", jac=True,
            options=dict(maxiter=500, ftol=1e-11, gtol=1e-7),
        )
        value, gradient = b.objective(result.x, x, offset, y, weights, penalty, 0.)
    assert result.success and np.max(abs(gradient)) < 1e-4, result.message
    return dict(
        encoder=encoder, columns=columns, categorical=cats,
        offset_mean=mean, offset_sd=sd, coefficients=result.x,
        spec=dict(correction_strength=strength, penalty=penalty), positive_weight=1.,
        objective=value, gradient_max_abs=float(np.max(abs(gradient))),
        iterations=int(result.nit),
    )
