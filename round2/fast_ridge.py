"""Analytic-gradient drop-in for r11_stage1_ridge.fit_irt_ident_ridge.

Identical objective, parameterisation, initialisation and L-BFGS-B
options as the deposited code; the only change is that the gradient is
supplied analytically instead of by finite differences. Equivalence is
NOT assumed: validate_fast_ridge.py checks this implementation against
the deposited construction on the locked partition (selected lambda per
fold, inner means, outer log-losses from output/r26_p3r_bracket.json).
"""
import numpy as np
from scipy.optimize import minimize

from predictors import _sigmoid, math_logit
from r11_stage1_ridge import fit_irt_ident_ridge as _deposited_fit


def fit_irt_ident_ridge_fast(x, y, w, j, i, n_ai, n_task, lam=0.0):
    xs = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    def obj_grad(theta):
        alpha = theta[:n_ai]
        beta = np.exp(theta[n_ai])
        phi = theta[n_ai + 1:]
        th = phi - phi.mean()
        z = alpha[j] - beta * th[i]
        p = _sigmoid(z)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        val = -np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc)))
        if lam > 0:
            val += lam * np.sum((beta * th) ** 2)
        # gradient: d val / d z = w (p - y)
        gz = w * (p - y)
        g_alpha = np.bincount(j, weights=gz, minlength=n_ai)
        # d/d th_i = sum over task i's runs of gz * (-beta)
        g_th = -beta * np.bincount(i, weights=gz, minlength=n_task)
        # d/d log-beta = sum gz * (-th_i) [+ penalty]
        g_logbeta = float(np.sum(gz * (-th[i])))
        if lam > 0:
            g_th = g_th + 2.0 * lam * beta ** 2 * th
            g_logbeta += 2.0 * lam * np.sum((beta * th) ** 2)
        # th = phi - mean(phi): d/d phi = g_th - mean(g_th)
        g_phi = g_th - g_th.mean()
        grad = np.concatenate([g_alpha, [g_logbeta], g_phi])
        return val, grad

    a0 = np.zeros(n_ai)
    for jj in range(n_ai):
        m = (j == jj)
        pj = min(max(np.average(y[m], weights=w[m]), 1e-3), 1 - 1e-3)
        a0[jj] = math_logit(pj)
    x0 = np.concatenate([a0, [0.0], np.zeros(n_task)])
    res = minimize(obj_grad, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": 20000, "maxfun": 200000})
    if not res.success:
        # The exact gradient can trip L-BFGS-B's line search where the
        # deposited finite-difference path succeeds (observed: fold 0
        # inner split, ABNORMAL_TERMINATION_IN_LNSRCH). Fall back to the
        # deposited implementation for that fit — same objective, init
        # and options — so the hybrid's outputs remain the deposited
        # construction's outputs; validate_fast_ridge.py verifies this
        # end-to-end against the deposited bracket record.
        return _deposited_fit(x, y, w, j, i, n_ai, n_task, lam=lam)
    phi = res.x[n_ai + 1:]
    theta = phi - phi.mean()
    beta = np.exp(res.x[n_ai])
    alpha = res.x[:n_ai]
    task_x = np.array([xs[i == t][0] for t in range(n_task)])
    X = np.column_stack([np.ones(n_task), task_x])
    (a, b), *_ = np.linalg.lstsq(X, theta, rcond=None)
    return {"alpha": alpha, "beta": beta, "theta": theta, "a": a, "b": b,
            "task_x": task_x}
