"""Cross-fitted monotone calibration and a small, regularized soft gate."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax


def calibrate(p, y):
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    y = np.asarray(y, float)

    def objective(theta):
        eta = theta[0] + theta[1] * z
        error = expit(eta) - y
        loss = np.mean(np.logaddexp(0, eta) - y * eta) + .005 * (theta[0]**2 + (theta[1] - 1)**2)
        gradient = np.array([np.mean(error), np.mean(error * z)]) + .01 * (theta - [0., 1.])
        return float(loss), gradient
    fit = minimize(objective, [0., 1.], jac=True, method='L-BFGS-B', bounds=[(-5., 5.), (0., 5.)],
                   options=dict(maxiter=400, ftol=1e-12, gtol=1e-8))
    assert fit.success, fit.message
    return fit.x


def calibrated(theta, p):
    return expit(theta[0] + theta[1] * logit(np.clip(p, 1e-6, 1-1e-6)))


class GateEncoder:
    def fit(self, frame):
        self.numeric = list(frame.select_dtypes(include=[np.number]).columns)
        self.categories = {c: sorted(frame[c].astype('string').fillna('MISSING').unique().tolist())
                           for c in frame if c not in self.numeric}
        a = frame[self.numeric].to_numpy(float)
        a = np.sign(a) * np.log1p(abs(a))
        self.median = np.nanmedian(np.where(np.isfinite(a), a, np.nan), axis=0)
        self.median[~np.isfinite(self.median)] = 0.
        a = np.where(np.isfinite(a), a, self.median)
        self.mean = a.mean(0)
        self.sd = np.maximum(a.std(0), .001)
        return self

    def transform(self, frame):
        a = frame[self.numeric].to_numpy(float)
        a = np.sign(a) * np.log1p(abs(a))
        missing = ~np.isfinite(a)
        a = np.where(missing, self.median, a)
        z = [np.ones((len(frame), 1)), np.clip((a-self.mean)/self.sd, -8, 8), missing.astype(float)]
        for c, categories in self.categories.items():
            value = frame[c].astype('string').fillna('MISSING')
            z.append(np.column_stack([(value == x).to_numpy(float) for x in categories]))
        result = np.column_stack(z)
        assert np.isfinite(result).all()
        return result

    def recipe(self):
        return dict(numeric=self.numeric, categories=self.categories, median=self.median.tolist(),
                    mean=self.mean.tolist(), sd=self.sd.tolist())


def fit_gate(z, probabilities, y, prior, penalty=.01):
    """No outer scores, residual features, or query labels enter this fit."""
    p = np.clip(np.asarray(probabilities, float), 1e-6, 1-1e-6)
    y = np.asarray(y, float)
    base = np.log(np.asarray(prior, float))
    n, experts = p.shape
    columns = z.shape[1]

    def objective(flat):
        beta = flat.reshape(columns, experts-1)
        score = np.column_stack([np.zeros(n), z @ beta]) + base
        weights = softmax(score, axis=1)
        mixed = np.clip((weights*p).sum(1), 1e-9, 1-1e-9)
        loss = -np.mean(y*np.log(mixed)+(1-y)*np.log1p(-mixed)) + .5*penalty*np.sum(beta**2)
        derivative = (mixed-y)/(mixed*(1-mixed))/n
        score_gradient = derivative[:,None]*weights*(p-mixed[:,None])
        gradient = z.T @ score_gradient[:,1:] + penalty*beta
        return float(loss), gradient.ravel()

    fit = minimize(objective, np.zeros(columns*(experts-1)), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
    assert fit.success, fit.message
    beta = fit.x.reshape(columns, experts-1)
    return beta, dict(iterations=int(fit.nit), objective=float(fit.fun),
                      gradient_max=float(np.max(abs(fit.jac))), parameters=beta.size,
                      penalty=penalty, prior=list(prior), fitted_weights_saved=False)


def predict_gate(z, p, beta, prior):
    weights = softmax(np.column_stack([np.zeros(len(z)), z@beta]) + np.log(prior), axis=1)
    return (weights*p).sum(1), weights


def synthetic_checks():
    import pandas as pd
    rng = np.random.default_rng(20261028)
    frame = pd.DataFrame(dict(depth=rng.normal(size=320), status=['a','b']*160))
    encoder = GateEncoder().fit(frame.iloc[:240])
    changed = frame.copy(); changed.loc[240:, 'depth'] = 1e80; changed.loc[240:, 'status'] = 'unseen'
    other = GateEncoder().fit(changed.iloc[:240])
    assert encoder.recipe() == other.recipe()
    z = encoder.transform(frame); p = rng.uniform(.03,.5,(320,5)); y = rng.binomial(1,.14,320)
    prior = [.9,.025,.025,.025,.025]
    beta, audit = fit_gate(z[:240], p[:240], y[:240], prior)
    result, weights = predict_gate(z[240:], p[240:], beta, prior)
    assert np.isfinite(result).all() and np.allclose(weights.sum(1),1.)
    assert np.all((result>=0)&(result<=1))
    a = calibrate(p[:240,1], y[:240]); assert a[1]>=0
    assert np.isfinite(calibrated(a,p[240:,1])).all()
    return dict(gate_solver=audit['iterations'], heldout_mutation_preserves_preprocessor=True,
                mixture_finite_bounded=True, weights_sum_to_one=True, monotone_calibration=True)
