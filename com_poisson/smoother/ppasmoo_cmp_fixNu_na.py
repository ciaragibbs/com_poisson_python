"""Forward-backward smoother for CMP with fixed nu parameter."""

from __future__ import annotations

import warnings

import numpy as np

from ..distribution.cmp_moment import CMPmoment


def ppasmoo_cmp_fixNu_na(
    N: np.ndarray,
    X_lam: np.ndarray,
    nu: float,
    theta0: np.ndarray,
    W0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-backward smoother for CMP with a fixed nu parameter.

    Parameters
    ----------
    N : np.ndarray, shape (1, T)
        Observed spike counts.  NaN entries are treated as missing.
    X_lam : np.ndarray, shape (T, n_lam)
        Design matrix for the lambda component.
    nu : float
        Fixed dispersion parameter.
    theta0 : np.ndarray, shape (n_lam,)
        Initial parameter vector (log-lambda space).
    W0 : np.ndarray, shape (n_lam, n_lam)
        Initial posterior covariance.
    F : np.ndarray, shape (n_lam, n_lam)
        State transition matrix.
    Q : np.ndarray, shape (n_lam, n_lam)
        Process noise covariance.

    Returns
    -------
    theta : np.ndarray, shape (n_lam, T)
        Smoothed parameter estimates.
    W : np.ndarray, shape (n_lam, n_lam, T)
        Smoothed covariance matrices.
    lam_smoo : np.ndarray, shape (T,)
        Smoothed lambda at each time step.
    """
    n_spk = N.shape[1]
    maxSum = int(10 * np.nanmax(N))
    p = len(theta0)

    # Preallocate
    theta = np.zeros((p, n_spk))
    W = np.zeros((p, p, n_spk))
    lam = np.zeros(n_spk)

    # Initialise
    theta[:, 0] = theta0
    W[:, :, 0] = W0
    lam[0] = np.exp(X_lam[0, :] @ theta0)

    thetapred = theta.copy()
    Wpred = W.copy()

    I = np.eye(p)
    F_inv = np.linalg.inv(F)

    # Forward pass (filtering)
    for i in range(1, n_spk):
        thetapred[:, i] = F @ theta[:, i - 1]
        Wpred[:, :, i] = F @ W[:, :, i - 1] @ F.T + Q
        lam[i] = np.exp(X_lam[i, :] @ thetapred[:, i])

        if not np.isnan(N[0, i]):
            mean_Y, var_Y, _, _, _, _ = CMPmoment(lam[i], nu, maxSum)
            x_i = X_lam[i, :, np.newaxis]
            try:
                Wpostinv = np.linalg.inv(Wpred[:, :, i]) + var_Y * (x_i @ x_i.T)
                W[:, :, i] = np.linalg.inv(Wpostinv)
            except np.linalg.LinAlgError:
                warnings.warn("Singular matrix encountered.", stacklevel=2)
                return theta, W, lam
            theta[:, i] = thetapred[:, i] + W[:, :, i] @ ((N[0, i] - mean_Y) * X_lam[i, :])
        else:
            W[:, :, i] = Wpred[:, :, i]
            theta[:, i] = thetapred[:, i]

    lam_smoo = lam.copy()

    # Backward pass (RTS)
    for i in range(n_spk - 2, -1, -1):
        Wi = np.linalg.inv(Wpred[:, :, i + 1])
        Fsquig = F_inv @ (I - Q @ Wi)
        Ksquig = F_inv @ Q @ Wi

        theta[:, i] = Fsquig @ theta[:, i + 1] + Ksquig @ thetapred[:, i + 1]
        C = W[:, :, i] @ F.T @ Wi
        W[:, :, i] = W[:, :, i] + C @ (W[:, :, i + 1] - Wpred[:, :, i + 1]) @ C.T

        lam_smoo[i] = np.exp(X_lam[i, :] @ theta[:, i])

    return theta, W, lam_smoo
