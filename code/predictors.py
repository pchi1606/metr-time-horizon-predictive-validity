"""Three predictors for the horizon shootout.

P1 (horizon/spline, Nguyen & Fithian Model 1):
    logit p_j(t) = alpha_j - beta * f(log2 t), beta > 0,
    f monotone cubic (PCHIP) through strictly increasing knot values,
    4 interior knots at train quantiles; f(min)=min, f(max)=max.
P2 (mean accuracy): per-AI weighted train mean, constant on held-out tasks.
P3 (IRT ability): 1PL Rasch logit p_ij = alpha_j - beta*theta_i on train tasks,
    then OLS theta_i ~ log2(T_i); held-out tasks use predicted theta.
All fits minimize weighted binary cross-entropy (weights = invsqrt_task_weight).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

EPS = 1e-12


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bce(y, p, w):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))


class SplineHorizon:
    """P1: shared monotone spline success curves (Nguyen & Fithian Model 1)."""

    def __init__(self):
        self.knots_ = None
        self.f_ = None
        self.alpha_ = None
        self.beta_ = None
        self.ai_index_ = None

    def _make_f(self, knots, u):
        # strictly increasing knot values via exp-gap parameterization;
        # endpoints pinned for identifiability: f(min)=min, f(max)=max
        gaps = np.exp(u)
        frac = np.cumsum(gaps) / np.sum(gaps)
        xmin, xmax = knots[0], knots[-1]
        values = xmin + (xmax - xmin) * np.concatenate([[0.0], frac[:-1], [1.0]])
        return PchipInterpolator(knots, values, extrapolate=True), values

    def fit(self, x, y, w, j, n_ai):
        xs = np.asarray(x, float)
        xmin, xmax = xs.min(), xs.max()
        qs = np.quantile(xs, [0.2, 0.4, 0.6, 0.8])
        knots = np.concatenate([[xmin], qs, [xmax]])
        # ensure strictly increasing knots (nudge ties)
        for k in range(1, len(knots)):
            if knots[k] <= knots[k - 1]:
                knots[k] = knots[k - 1] + 1e-9
        n_u = len(knots) - 1  # gaps between consecutive knots
        # init: f = identity -> u_k = log(knot gap)
        u0 = np.log(np.diff(knots))
        # init alphas at per-AI logit means
        a0 = np.zeros(n_ai)
        for jj in range(n_ai):
            m = (j == jj)
            pj = np.average(y[m], weights=w[m])
            pj = min(max(pj, 1e-3), 1 - 1e-3)
            a0[jj] = math_logit(pj)
        x0 = np.concatenate([a0, [0.0], u0])  # log_beta=0 -> beta=1

        def obj(theta):
            alpha = theta[:n_ai]
            beta = np.exp(theta[n_ai])
            f, _ = self._make_f(knots, theta[n_ai + 1:])
            p = _sigmoid(alpha[j] - beta * f(xs))
            return _bce(y, p, w)

        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 20000, "maxfun": 200000})
        assert res.success, f"P1 fit failed: {res.message}"
        alpha = res.x[:n_ai]
        beta = np.exp(res.x[n_ai])
        f, values = self._make_f(knots, res.x[n_ai + 1:])
        # monotonicity assertion on a fine grid
        grid = np.linspace(xmin, xmax, 2001)
        assert np.all(np.diff(f(grid)) >= -1e-9), "fitted f not monotone"
        self.knots_, self.f_, self.alpha_, self.beta_ = knots, f, alpha, beta
        return self

    def predict(self, x, j):
        return _sigmoid(self.alpha_[j] - self.beta_ * self.f_(np.asarray(x, float)))

    def horizon(self, jj, q=0.5):
        """50% (or q) time horizon in minutes for AI jj via numerical inverse."""
        from scipy.optimize import brentq
        target = math_logit(q)
        xmin, xmax = self.knots_[0], self.knots_[-1]
        # solve f(x) = alpha_j / beta
        rhs = self.alpha_[jj] / self.beta_
        if rhs <= self.f_(xmin):
            return 2.0 ** xmin  # below range: report floor
        if rhs >= self.f_(xmax):
            return 2.0 ** xmax  # above range: report ceiling
        x_star = brentq(lambda t: self.f_(t) - rhs, xmin, xmax)
        return 2.0 ** x_star


def math_logit(p):
    import math
    return math.log(p / (1 - p))


class MeanPredictor:
    """P2: per-AI weighted train mean success rate."""

    def fit(self, y, w, j, n_ai):
        self.mean_ = np.zeros(n_ai)
        for jj in range(n_ai):
            m = (j == jj)
            self.mean_[jj] = np.sum(w[m] * y[m]) / np.sum(w[m])
        return self

    def predict(self, j):
        return np.clip(self.mean_[j], EPS, 1 - EPS)


class IrtPredictor:
    """P3: 1PL Rasch + explanatory regression of difficulty on log human time."""

    def fit(self, x, y, w, j, i, n_ai, n_task):
        xs = np.asarray(x, float)
        # center theta inside the objective for identifiability
        def obj(theta):
            alpha = theta[:n_ai]
            beta = np.exp(theta[n_ai])
            phi = theta[n_ai + 1:]
            th = phi - phi.mean()
            p = _sigmoid(alpha[j] - beta * th[i])
            return _bce(y, p, w)

        a0 = np.zeros(n_ai)
        for jj in range(n_ai):
            m = (j == jj)
            pj = np.average(y[m], weights=w[m])
            pj = min(max(pj, 1e-3), 1 - 1e-3)
            a0[jj] = math_logit(pj)
        x0 = np.concatenate([a0, [0.0], np.zeros(n_task)])
        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 20000, "maxfun": 200000})
        assert res.success, f"P3 fit failed: {res.message}"
        phi = res.x[n_ai + 1:]
        theta = phi - phi.mean()
        self.alpha_ = res.x[:n_ai]
        self.beta_ = np.exp(res.x[n_ai])
        # explanatory regression: theta_i ~ a + b * x_i (train tasks only)
        # x per task = its (single) log2 human time
        task_x = np.array([xs[i == t][0] for t in range(n_task)])
        X = np.column_stack([np.ones(n_task), task_x])
        coef, *_ = np.linalg.lstsq(X, theta, rcond=None)
        self.a_, self.b_ = coef
        return self

    def predict(self, x, j):
        theta_hat = self.a_ + self.b_ * np.asarray(x, float)
        return _sigmoid(self.alpha_[j] - self.beta_ * theta_hat)
