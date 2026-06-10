"""Gradient and Hessian of the CMP log-posterior w.r.t. theta (free nu)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from ..distribution.cmp_moment import CMPmoment
from .blktridiag import blktridiag


def gradHessTheta_na(
    vecTheta: np.ndarray,
    X_lam: np.ndarray,
    G_nu: np.ndarray,
    theta0_tmp: np.ndarray,
    W0_tmp: np.ndarray,
    F: np.ndarray,
    Q_tmp: np.ndarray,
    spk_vec: np.ndarray,
) -> list:
    """Compute gradient and Hessian of the CMP log-posterior.

    Used inside ``newtonGH`` for MAP estimation of the full parameter
    trajectory when nu is free (jointly estimated with lambda).

    Parameters
    ----------
    vecTheta : np.ndarray, shape (p * T,)
        Flattened parameter matrix (column-major ordering).
    X_lam : np.ndarray, shape (T, n_lam)
        Design matrix for lambda.
    G_nu : np.ndarray, shape (T, n_nu)
        Design matrix for nu.
    theta0_tmp : np.ndarray, shape (p,)
        Prior mean on the initial state.
    W0_tmp : np.ndarray, shape (p, p)
        Prior covariance on the initial state.
    F : np.ndarray, shape (p, p)
        State transition matrix.
    Q_tmp : np.ndarray, shape (p, p)
        Process noise covariance.
    spk_vec : np.ndarray, shape (T,)
        Observed spike counts (NaN = missing).

    Returns
    -------
    GradHess : list of length 2
        ``[grad, hess]`` where ``grad`` is shape ``(p*T,)`` and ``hess``
        is a sparse ``(p*T, p*T)`` block tridiagonal matrix.
    """
    T = len(spk_vec)
    n_lam = X_lam.shape[1]
    n_nu = G_nu.shape[1]
    p = n_lam + n_nu
    maxSum = int(10 * np.nanmax(spk_vec))

    Theta = vecTheta.reshape(p, T, order="F")
    beta = Theta[:n_lam, :]
    gam = Theta[n_lam:, :]

    Q_inv_F = np.linalg.solve(Q_tmp, F)            # Q^{-1} F
    FT_Q_inv_F = F.T @ Q_inv_F                     # F' Q^{-1} F

    hessup = np.stack([Q_inv_F.T] * (T - 1), axis=-1)   # (p,p,T-1) super-diagonal
    hessub = np.stack([Q_inv_F] * (T - 1), axis=-1)     # (p,p,T-1) sub-diagonal
    hessmed = np.zeros((p, p, T))
    SCORE = np.zeros((p, T))

    W0_inv = np.linalg.inv(W0_tmp)
    Q_inv = np.linalg.inv(Q_tmp)

    for t in range(T):
        if not np.isnan(spk_vec[t]):
            lam = float(np.exp(X_lam[t, :] @ beta[:, t]))
            nu = float(np.exp(G_nu[t, :] @ gam[:, t]))
            mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, _ = CMPmoment(lam, nu, maxSum)

            x_t = X_lam[t, :, np.newaxis]
            g_t = G_nu[t, :, np.newaxis]

            SCORE[:n_lam, t] = (spk_vec[t] - mean_Y) * X_lam[t, :]
            SCORE[n_lam:, t] = nu * (-gammaln(spk_vec[t] + 1) + mean_logYfac) * G_nu[t, :]

            hess1 = -var_Y * (x_t @ x_t.T)
            hess2 = nu * cov_Y_logYfac * (x_t @ g_t.T)
            hess4 = -nu ** 2 * var_logYfac * (g_t @ g_t.T)

            hess = np.block([[hess1, hess2], [hess2.T, hess4]])
        else:
            SCORE[:, t] = 0.0
            hess = np.zeros((p, p))

        if t == 0:
            hessmed[:, :, t] = hess - W0_inv - FT_Q_inv_F
        elif t == T - 1:
            hessmed[:, :, t] = hess - Q_inv
        else:
            hessmed[:, :, t] = hess - Q_inv - FT_Q_inv_F

    # Build gradient
    grad = np.empty((p, T))
    # t=0
    grad[:, 0] = (
        SCORE[:, 0]
        - np.linalg.solve(W0_tmp, Theta[:, 0] - theta0_tmp)
        + F.T @ np.linalg.solve(Q_tmp, Theta[:, 1] - F @ Theta[:, 0])
    )
    # t=1..T-2
    for t in range(1, T - 1):
        grad[:, t] = (
            SCORE[:, t]
            - np.linalg.solve(Q_tmp, Theta[:, t] - F @ Theta[:, t - 1])
            + F.T @ np.linalg.solve(Q_tmp, Theta[:, t + 1] - F @ Theta[:, t])
        )
    # t=T-1
    grad[:, T - 1] = (
        SCORE[:, T - 1]
        - np.linalg.solve(Q_tmp, Theta[:, T - 1] - F @ Theta[:, T - 2])
    )

    GradHess = [
        grad.ravel(order="F"),
        blktridiag(hessmed, hessub, hessup),
    ]
    return GradHess
