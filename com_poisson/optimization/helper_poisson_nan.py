"""Helper function for Poisson process-noise optimisation."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from ..smoother.ppafilt_poissexp_nan import ppafilt_poissexp_nan


def helper_poisson_nan(
    Q: np.ndarray,
    b0: np.ndarray,
    N: np.ndarray,
    X: np.ndarray,
    W0: np.ndarray,
    F: np.ndarray,
) -> float:
    """Compute negative predicted log-likelihood for the Poisson model.

    Used as the objective for optimising the process-noise variance ``Q``.

    Parameters
    ----------
    Q : np.ndarray, shape (d,)
        Variance parameters.  If ``X`` has 2+ columns, ``Q[0]`` is the
        intercept variance and ``Q[1]`` is the shared variance for the
        remaining columns.  If ``X`` has 1 column, ``Q`` is a scalar.
    b0 : np.ndarray, shape (p,)
        Initial parameter vector.
    N : np.ndarray, shape (T,)
        Observed spike counts.
    X : np.ndarray, shape (T, p)
        Design matrix.
    W0 : np.ndarray, shape (p, p)
        Initial posterior covariance.
    F : np.ndarray, shape (p, p)
        State transition matrix.

    Returns
    -------
    neg_llhd_pred : float
        Negative predicted log-likelihood (to be minimised).
    """
    p = X.shape[1]
    if p >= 2:
        Qmatrix = np.diag(np.concatenate([[Q[0]], Q[1] * np.ones(p - 1)]))
    else:
        Qmatrix = np.array([[Q[0]]])

    _, _, lam = ppafilt_poissexp_nan(N, X, b0, W0, F, Qmatrix)

    n_arr = N.astype(float)
    lam_safe = np.where(lam == 0, 1.0, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        llhd_pred = float(
            np.nansum(
                -lam + np.log(lam_safe) * n_arr - gammaln(n_arr + 1)
            )
        )
    print(f"llhd {llhd_pred:.2f}...")
    return -llhd_pred
