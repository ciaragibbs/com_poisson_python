"""Forward-backward smoother for Poisson point-process with exponential link."""

from __future__ import annotations

import warnings

import numpy as np


def ppasmoo_poissexp_nan(
    n: np.ndarray,
    X: np.ndarray,
    b0: np.ndarray,
    W0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-backward smoother for Poisson GLM with log link.

    Performs forward filtering (Eden et al. 2004) followed by RTS smoothing.

    Parameters
    ----------
    n : np.ndarray, shape (T,)
        Observed spike counts.  NaN entries are treated as missing.
    X : np.ndarray, shape (T, p)
        Design (feature) matrix.
    b0 : np.ndarray, shape (p,)
        Initial parameter vector.
    W0 : np.ndarray, shape (p, p)
        Initial posterior covariance.
    F : np.ndarray, shape (p, p)
        State transition matrix.
    Q : np.ndarray, shape (p, p)
        Process noise covariance.

    Returns
    -------
    b : np.ndarray, shape (p, T)
        Smoothed parameter estimates.
    W : np.ndarray, shape (p, p, T)
        Smoothed covariance matrices.
    lam : np.ndarray, shape (T,)
        Filtered rate (pre-smoothing).
    lam_smoo : np.ndarray, shape (T,)
        Smoothed Poisson rate.
    """
    T = len(n)
    p = len(b0)
    offset = np.zeros(T)

    # Preallocate
    b = np.zeros((p, T))
    W = np.zeros((p, p, T))
    lam = np.zeros(T)

    # Initialise
    b[:, 0] = b0
    W[:, :, 0] = W0
    lam[0] = np.exp(X[0, :] @ b0 + offset[0])

    bpred = b.copy()
    Wpred = W.copy()

    I = np.eye(p)
    F_inv = np.linalg.inv(F)

    # Forward pass (filtering)
    for i in range(1, T):
        bpred[:, i] = F @ b[:, i - 1]
        Wpred[:, :, i] = F @ W[:, :, i - 1] @ F.T + Q
        lam[i] = np.exp(X[i, :] @ bpred[:, i] + offset[i])

        if not np.isnan(n[i]):
            x_i = X[i, :, np.newaxis]
            try:
                Wpostinv = np.linalg.inv(Wpred[:, :, i]) + lam[i] * (x_i @ x_i.T)
                W[:, :, i] = np.linalg.inv(Wpostinv)
            except np.linalg.LinAlgError:
                warnings.warn("Singular matrix encountered; stopping smoother.", stacklevel=2)
                return b, W, lam, lam.copy()
            b[:, i] = bpred[:, i] + W[:, :, i] @ (X[i, :] * (n[i] - lam[i]))
        else:
            W[:, :, i] = Wpred[:, :, i]
            b[:, i] = bpred[:, i]

    lam_smoo = lam.copy()

    # Backward pass (RTS)
    for i in range(T - 2, -1, -1):
        Wi = np.linalg.inv(Wpred[:, :, i + 1])
        Fsquig = F_inv @ (I - Q @ Wi)
        Ksquig = F_inv @ Q @ Wi

        b[:, i] = Fsquig @ b[:, i + 1] + Ksquig @ bpred[:, i + 1]
        C = W[:, :, i] @ F.T @ Wi
        W[:, :, i] = W[:, :, i] + C @ (W[:, :, i + 1] - Wpred[:, :, i + 1]) @ C.T
        lam_smoo[i] = np.exp(X[i, :] @ b[:, i] + offset[i])

    return b, W, lam, lam_smoo
