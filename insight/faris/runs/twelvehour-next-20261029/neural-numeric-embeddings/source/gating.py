"""Small held-out-meta calibration and stacking; learned coefficients stay in RAM."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def calibrate(p, y, penalty=.005):
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    y = np.asarray(y, float)
    assert z.shape == y.shape and np.isfinite(z).all()

    def objective(theta):
        eta = theta[0] + theta[1] * z
        error = expit(eta) - y
        centered = theta - [0., 1.]
        loss = np.mean(np.logaddexp(0., eta) - y * eta) + penalty * np.dot(centered, centered)
        gradient = np.array([np.mean(error), np.mean(error * z)]) + 2 * penalty * centered
        return float(loss), gradient

    fit = minimize(objective, [0., 1.], jac=True, method='L-BFGS-B',
                   bounds=[(-5., 5.), (0., 5.)],
                   options=dict(maxiter=400, ftol=1e-12, gtol=1e-8))
    assert fit.success and np.isfinite(fit.x).all(), str(fit.message)
    return fit.x


def calibrated(theta, p):
    return expit(theta[0] + theta[1] * logit(np.clip(p, 1e-6, 1 - 1e-6)))


def fit_stack(anchor, candidate, y, penalty=.03):
    x = np.column_stack([np.ones(len(anchor)),
                         logit(np.clip(anchor, 1e-6, 1 - 1e-6)),
                         logit(np.clip(candidate, 1e-6, 1 - 1e-6))])
    y = np.asarray(y, float)
    prior = np.array([0., 1., 0.])

    def objective(theta):
        eta = x @ theta
        difference = theta - prior
        loss = np.mean(np.logaddexp(0., eta) - y * eta) + .5 * penalty * np.dot(difference, difference)
        gradient = x.T @ (expit(eta) - y) / len(y) + penalty * difference
        return float(loss), gradient

    fit = minimize(objective, prior, jac=True, method='L-BFGS-B',
                   bounds=[(-5., 5.), (0., 5.), (0., 5.)],
                   options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
    assert fit.success and np.isfinite(fit.x).all(), str(fit.message)
    audit = dict(method='ridge Bernoulli logistic stack of two logits and intercept',
                 penalty=penalty, prior=[0., 1., 0.], nonnegative_channel_coefficients=True,
                 iterations=int(fit.nit), objective=float(fit.fun),
                 learned_coefficients_serialized=False)
    return fit.x, audit


def stacked(theta, anchor, candidate):
    x = np.column_stack([np.ones(len(anchor)),
                         logit(np.clip(anchor, 1e-6, 1 - 1e-6)),
                         logit(np.clip(candidate, 1e-6, 1 - 1e-6))])
    return expit(x @ theta)
