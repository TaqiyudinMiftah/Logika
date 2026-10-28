"""Small conditional corrections to the frozen teammate Kumo model."""
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits
import baseline as b

PAIRS = {
    "control": [],
    "credit_agreement": [
        ("re_bureau_score_posterior", "re_late_30_posterior"),
    ],
    "consumption_obligations": [
        ("re_price_log_posterior", "v2_existing_total_burden_observed"),
    ],
    "measurement_reliability": [
        ("re_price_log_posterior", "re_price_posterior_sd"),
        ("re_bureau_score_posterior", "re_bureau_posterior_sd"),
    ],
}


def objective(beta, x, offset, y, penalties):
    eta = offset + x @ beta
    loss = np.mean(np.logaddexp(0, eta) - y * eta)
    loss += .5 * np.dot(penalties, beta ** 2)
    gradient = x.T @ (expit(eta) - y) / len(y) + penalties * beta
    return float(loss), gradient


def tensor_values(block, frame):
    values = b.prepare(frame, block["columns"], [])
    u = block["spline"].transform(block["imputer"].transform(values))
    width = u.shape[1] // 2
    u = u - block["mean"]
    return (u[:, :width, None] * u[:, None, width:]).reshape(len(frame), -1)


def fit_geometry(frame, x, pairs):
    blocks, raw = [], []
    for pair in pairs:
        columns = list(pair)
        values = b.prepare(frame, columns, [])
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        filled = imputer.fit_transform(values)
        spline = SplineTransformer(
            n_knots=3, degree=2, knots="quantile", extrapolation="constant"
        )
        u = spline.fit_transform(filled)
        block = dict(columns=columns, imputer=imputer, spline=spline, mean=u.mean(0))
        blocks.append(block)
        raw.append(tensor_values(block, frame))
    raw = np.column_stack(raw)
    projection = np.linalg.lstsq(x, raw, rcond=1e-10)[0]
    residual = raw - x @ projection
    values, vectors = np.linalg.eigh(residual.T @ residual / len(residual))
    keep = values > max(float(values.max()) * 1e-6, 1e-10)
    rotation = vectors[:, keep] / np.sqrt(values[keep])
    z = residual @ rotation
    assert np.isfinite(z).all() and z.shape[1] > 0
    orthogonality = float(np.max(np.abs(x.T @ z / len(x))))
    assert orthogonality < 1e-5, orthogonality
    return dict(blocks=blocks, projection=projection, rotation=rotation,
                dimensions=z.shape[1], orthogonality=orthogonality), z


def interaction_design(geometry, frame, x):
    raw = np.column_stack([tensor_values(block, frame) for block in geometry["blocks"]])
    return (raw - x @ geometry["projection"]) @ geometry["rotation"]


def fit(control, frame, raw_kumo, name):
    """frame contains only correction-training rows; no evaluation argument."""
    assert name in PAIRS
    if name == "control":
        return dict(name=name, additive=copy.deepcopy(control), geometry=None)
    with threadpool_limits(limits=4):
        x, offset = b.design(control, frame, raw_kumo)
        geometry, z = fit_geometry(frame, x, PAIRS[name])
        full = np.column_stack([x, z])
        penalties = np.r_[0., np.full(x.shape[1] - 1, .003),
                          np.full(z.shape[1], .03)]
        initial = np.r_[control["coefficients"], np.zeros(z.shape[1])]
        y = frame.target.to_numpy(int)
        result = minimize(
            objective, initial, args=(full, offset, y, penalties),
            method="L-BFGS-B", jac=True,
            options=dict(maxiter=500, ftol=1e-11, gtol=1e-7),
        )
        loss, gradient = objective(result.x, full, offset, y, penalties)
    assert result.success and np.max(np.abs(gradient)) < 1e-4, result.message
    return dict(
        name=name, additive=copy.deepcopy(control), geometry=geometry,
        coefficients=result.x, objective=loss, gradient_max=float(np.max(abs(gradient))),
        iterations=int(result.nit), interaction_penalty=.03,
    )


def predict(model, frame, raw_kumo):
    # Both b.design and tensor_values explicitly select named feature columns.
    # Evaluation labels/identifiers cannot become predictors.
    with threadpool_limits(limits=4):
        x, offset = b.design(model["additive"], frame, raw_kumo)
        if model["geometry"] is None:
            delta = x @ model["additive"]["coefficients"]
        else:
            z = interaction_design(model["geometry"], frame, x)
            delta = np.column_stack([x, z]) @ model["coefficients"]
    out = expit(offset + .5 * delta)
    assert np.isfinite(out).all()
    return out
