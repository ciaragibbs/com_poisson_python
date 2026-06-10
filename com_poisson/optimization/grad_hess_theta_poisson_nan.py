"""Gradient and Hessian of the Poisson log-posterior."""

from __future__ import annotations

import numpy as np

from .blktridiag import blktridiag


def gradHessTheta_Poisson_nan(
    vecTheta: np.ndarray,
    X_lam: np.ndarray,
    theta0_tmp: np.ndarray,
    W0_tmp: np.ndarray,
    F: np.ndarray,
    Q_tmp: np.ndarray,
    spk_vec: np.ndarray,
) -> list:
    """Compute gradient and Hessian of the Poisson log-posterior.

    Parameters
    ----------
    vecTheta : np.ndarray, shape (n_lam * T,)
        Flattened parameter matrix.
    X_lam : np.ndarray, shape (T, n_lam)
        Design matrix for lambda (log-linear model).
    theta0_tmp : np.ndarray, shape (n_lam,)
        Prior mean on the initial state.
    W0_tmp : np.ndarray, shape (n_lam, n_lam)
        Prior covariance on the initial state.
    F : np.ndarray, shape (n_lam, n_lam)
        State transition matrix.
    Q_tmp : np.ndarray, shape (n_lam, n_lam)
        Process noise covariance.
    spk_vec : np.ndarray, shape (T,)
        Observed spike counts (NaN = missing).

    Returns
    -------
    GradHess : list of length 2
        ``[grad, hess]`` where ``grad`` is shape ``(n_lam*T,)`` and ``hess``
        is a sparse ``(n_lam*T, n_lam*T)`` block tridiagonal matrix.
    """
    T = len(spk_vec)
    n_lam = X_lam.shape[1]
    theta = vecTheta.reshape(n_lam, T, order="F")

    Q_inv_F = np.linalg.solve(Q_tmp, F)
    FT_Q_inv_F = F.T @ Q_inv_F

    hessup = np.stack([Q_inv_F.T] * (T - 1), axis=-1)
    hessub = np.stack([Q_inv_F] * (T - 1), axis=-1)
    hessmed = np.zeros((n_lam, n_lam, T))
    SCORE = np.zeros((n_lam, T))

    W0_inv = np.linalg.inv(W0_tmp)
    Q_inv = np.linalg.inv(Q_tmp)

    for t in range(T):
        if not np.isnan(spk_vec[t]):
            lam = np.exp(X_lam[t, :] @ theta[:, t])
            SCORE[:, t] = X_lam[t, :] * (spk_vec[t] - lam)
            x_t = X_lam[t, :, np.newaxis]
            hess = -x_t @ (lam * x_t.T)
        else:
            SCORE[:, t] = 0.0
            hess = np.zeros((n_lam, n_lam))

        if t == 0:
            hessmed[:, :, t] = hess - W0_inv - FT_Q_inv_F
        elif t == T - 1:
            hessmed[:, :, t] = hess - Q_inv
        else:
            hessmed[:, :, t] = hess - Q_inv - FT_Q_inv_F

    # Build gradient
    grad = np.empty((n_lam, T))
    grad[:, 0] = (
        SCORE[:, 0]
        - np.linalg.solve(W0_tmp, theta[:, 0] - theta0_tmp)
        + F.T @ np.linalg.solve(Q_tmp, theta[:, 1] - F @ theta[:, 0])
    )
    for t in range(1, T - 1):
        grad[:, t] = (
            SCORE[:, t]
            - np.linalg.solve(Q_tmp, theta[:, t] - F @ theta[:, t - 1])
            + F.T @ np.linalg.solve(Q_tmp, theta[:, t + 1] - F @ theta[:, t])
        )
    grad[:, T - 1] = (
        SCORE[:, T - 1]
        - np.linalg.solve(Q_tmp, theta[:, T - 1] - F @ theta[:, T - 2])
    )

    GradHess = [
        grad.ravel(order="F"),
        blktridiag(hessmed, hessub, hessup),
    ]
    return GradHess
