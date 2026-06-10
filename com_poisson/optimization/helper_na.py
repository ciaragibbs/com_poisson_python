"""Helper function for CMP process-noise optimisation."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from ..smoother.ppafilt_compoisson_fisher_na import ppafilt_compoisson_fisher_na


def helper_na(
    Q: np.ndarray,
    theta0: np.ndarray,
    N: np.ndarray,
    X_lam: np.ndarray,
    G_nu: np.ndarray,
    W0: np.ndarray,
    F: np.ndarray,
) -> float:
    """Compute negative predicted log-likelihood for CMP model.

    Used as the objective function for optimising the process-noise
    variance parameters ``Q``.

    Parameters
    ----------
    Q : np.ndarray, shape (d,)
        Log-diagonal elements of the block process-noise matrix.
        Layout depends on the dimensions of ``X_lam`` and ``G_nu``:
        - If ``n_lam >= 2``: Q[0] is the intercept variance for lambda,
          Q[1] is the common variance for the remaining lambda coefficients.
          If ``n_nu >= 2``: Q[2] intercept, Q[3] common for nu; else Q[2] for nu.
        - If ``n_lam == 1``: Q[0] for lambda; same pattern for nu.
    theta0 : np.ndarray, shape (p,)
        Initial parameter vector.
    N : np.ndarray, shape (1, T)
        Observed spike counts.
    X_lam : np.ndarray, shape (T, n_lam)
        Design matrix for lambda.
    G_nu : np.ndarray, shape (T, n_nu)
        Design matrix for nu.
    W0 : np.ndarray, shape (p, p)
        Initial posterior covariance.
    F : np.ndarray, shape (p, p)
        State transition matrix.

    Returns
    -------
    neg_llhd_pred : float
        Negative predicted log-likelihood (to be minimised).
    """
    n_lam = X_lam.shape[1]
    n_nu = G_nu.shape[1]

    if n_lam >= 2:
        Q_lam = np.concatenate([[Q[0]], Q[1] * np.ones(n_lam - 1)])
        if n_nu >= 2:
            Q_nu = np.concatenate([[Q[2]], Q[3] * np.ones(n_nu - 1)])
        else:
            Q_nu = np.array([Q[2]])
    else:
        Q_lam = np.array([Q[0]])
        if n_nu >= 2:
            Q_nu = np.concatenate([[Q[1]], Q[2] * np.ones(n_nu - 1)])
        else:
            Q_nu = np.array([Q[1]])

    Qmatrix = np.diag(np.concatenate([Q_lam, Q_nu]))

    result = ppafilt_compoisson_fisher_na(theta0, N, X_lam, G_nu, W0, F, Qmatrix)
    _, _, lam, nu, log_Zvec, _, _, _ = result

    T = N.shape[1]
    if len(log_Zvec) == T:
        n_arr = N[0, :].astype(float)
        lam_safe = np.where(lam == 0, 1.0, lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            llhd_pred = float(
                np.nansum(
                    n_arr * np.log(lam_safe)
                    - nu * gammaln(n_arr + 1)
                    - log_Zvec
                )
            )
        print(f"llhd {llhd_pred:.2f}...")
        return -llhd_pred
    else:
        return np.inf
