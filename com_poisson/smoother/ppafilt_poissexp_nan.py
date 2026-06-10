"""Forward filtering for Poisson point-process with exponential link (log-linear)."""

from __future__ import annotations

import warnings

import numpy as np


def ppafilt_poissexp_nan(
    n: np.ndarray,
    X: np.ndarray,
    b0: np.ndarray,
    W0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward filtering for Poisson GLM with log link.

    Implements the adaptive filter from Eden et al. Neural Computation (2004).

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
        Filtered parameter estimates.
    W : np.ndarray, shape (p, p, T)
        Filtered covariance matrices.
    lam : np.ndarray, shape (T,)
        Filtered Poisson rate (mean) at each step.
    """
    T = len(n)
    p = len(b0)

    # Preallocate
    b = np.zeros((p, T))
    W = np.zeros((p, p, T))
    lam = np.zeros(T)
    offset = np.zeros(T)

    # Initialise
    b[:, 0] = b0
    W[:, :, 0] = W0
    lam[0] = np.exp(X[0, :] @ b0 + offset[0])

    bpred = b.copy()
    Wpred = W.copy()

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
                warnings.warn("Singular matrix encountered; stopping filter.", stacklevel=2)
                return b, W, lam
            b[:, i] = bpred[:, i] + W[:, :, i] @ (X[i, :] * (n[i] - lam[i]))
        else:
            W[:, :, i] = Wpred[:, :, i]
            b[:, i] = bpred[:, i]

    return b, W, lam
