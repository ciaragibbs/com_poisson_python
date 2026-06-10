"""Forward-backward smoother for COM-Poisson with Fisher information.

Implements the Rauch-Tung-Striebel (RTS) backward pass following the
forward filter in ppafilt_compoisson_fisher_na.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import gammaln

from ..distribution.cmp_moment import CMPmoment
from ..distribution.logsum_calc import logsum_calc


def ppasmoo_compoisson_fisher_na(
    theta0: np.ndarray,
    N: np.ndarray,
    X_lam: np.ndarray,
    G_nu: np.ndarray,
    W0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Forward-backward smoother for CMP with Fisher information scoring.

    Performs forward filtering followed by the RTS backward smoothing pass.

    Parameters
    ----------
    theta0 : np.ndarray, shape (p,)
        Initial parameter vector.
    N : np.ndarray, shape (1, T)
        Observed spike counts.  NaN entries are treated as missing.
    X_lam : np.ndarray, shape (T, n_lam)
        Design matrix for the lambda component.
    G_nu : np.ndarray, shape (T, n_nu)
        Design matrix for the nu component.
    W0 : np.ndarray, shape (p, p)
        Initial posterior covariance.
    F : np.ndarray, shape (p, p)
        State transition matrix.
    Q : np.ndarray, shape (p, p)
        Process noise covariance.

    Returns
    -------
    theta : np.ndarray, shape (p, T)
        Smoothed parameter estimates.
    W : np.ndarray, shape (p, p, T)
        Smoothed covariance matrices.
    lam_pred : np.ndarray, shape (T,)
        Predicted lambda at each step (from forward pass).
    nu_pred : np.ndarray, shape (T,)
        Predicted nu at each step (from forward pass).
    log_Zvec_pred : np.ndarray, shape (T,)
        Log normalisation constants for predicted params.
    lam_filt : np.ndarray, shape (T,)
        Filtered lambda at each step.
    nu_filt : np.ndarray, shape (T,)
        Filtered nu at each step.
    log_Zvec_filt : np.ndarray, shape (T,)
        Log normalisation constants for filtered params.
    """
    n_spk = N.shape[1]
    maxSum = int(10 * np.nanmax(N))
    np_lam = X_lam.shape[1]
    p = len(theta0)

    # Preallocate
    theta = np.zeros((p, n_spk))
    W = np.zeros((p, p, n_spk))
    lam_pred = np.zeros(n_spk)
    nu_pred = np.zeros(n_spk)
    log_Zvec_pred = np.full(n_spk, np.nan)

    # Initialise
    theta[:, 0] = theta0
    W[:, :, 0] = W0

    lam_pred[0] = np.exp(X_lam[0, :] @ theta0[:np_lam])
    nu_pred[0] = np.exp(G_nu[0, :] @ theta0[np_lam:])
    logcum = logsum_calc(lam_pred[0], nu_pred[0], maxSum)
    log_Zvec_pred[0] = logcum[0]

    lam_filt = lam_pred.copy()
    nu_filt = nu_pred.copy()
    log_Zvec_filt = log_Zvec_pred.copy()

    thetapred = theta.copy()
    Wpred = W.copy()

    # ------------------------------------------------------------------ #
    # Forward pass (filtering)                                            #
    # ------------------------------------------------------------------ #
    for i in range(1, n_spk):
        thetapred[:, i] = F @ theta[:, i - 1]
        Wpred[:, :, i] = F @ W[:, :, i - 1] @ F.T + Q

        lam_pred[i] = np.exp(X_lam[i, :] @ thetapred[:np_lam, i])
        nu_pred[i] = np.exp(G_nu[i, :] @ thetapred[np_lam:, i])

        INFO = np.zeros((p, p))
        SCORE = np.zeros(p)

        if not np.isnan(N[0, i]):
            mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, log_Zvec_pred[i] = (
                CMPmoment(lam_pred[i], nu_pred[i], maxSum)
            )

            x_i = X_lam[i, :, np.newaxis]
            g_i = G_nu[i, :, np.newaxis]

            info1 = var_Y * (x_i @ x_i.T)
            info2 = -nu_pred[i] * cov_Y_logYfac * (x_i @ g_i.T)
            info4 = nu_pred[i] ** 2 * var_logYfac * (g_i @ g_i.T)

            INFO[:np_lam, :np_lam] += info1
            INFO[:np_lam, np_lam:] += info2
            INFO[np_lam:, :np_lam] += info2.T
            INFO[np_lam:, np_lam:] += info4

            n_val = N[0, i]
            SCORE[:np_lam] += (n_val - mean_Y) * X_lam[i, :]
            SCORE[np_lam:] += nu_pred[i] * (-gammaln(n_val + 1) + mean_logYfac) * G_nu[i, :]

        try:
            Wpostinv = np.linalg.inv(Wpred[:, :, i]) + INFO
            W[:, :, i] = np.linalg.inv(Wpostinv)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix encountered; stopping smoother.", stacklevel=2)
            return theta, W, lam_pred, nu_pred, log_Zvec_pred, lam_filt, nu_filt, log_Zvec_filt

        theta[:, i] = thetapred[:, i] + W[:, :, i] @ SCORE

        if not np.isnan(N[0, i]):
            lam_filt[i] = np.exp(X_lam[i, :] @ theta[:np_lam, i])
            nu_filt[i] = np.exp(G_nu[i, :] @ theta[np_lam:, i])
            logcum = logsum_calc(lam_filt[i], nu_filt[i], maxSum)
            log_Zvec_filt[i] = logcum[0]

    # ------------------------------------------------------------------ #
    # Backward pass (RTS smoothing)                                       #
    # ------------------------------------------------------------------ #
    I = np.eye(p)
    F_inv = np.linalg.inv(F)

    for i in range(n_spk - 2, -1, -1):
        Wi = np.linalg.inv(Wpred[:, :, i + 1])
        Fsquig = F_inv @ (I - Q @ Wi)
        Ksquig = F_inv @ Q @ Wi

        theta[:, i] = Fsquig @ theta[:, i + 1] + Ksquig @ thetapred[:, i + 1]
        C = W[:, :, i] @ F.T @ Wi
        W[:, :, i] = W[:, :, i] + C @ (W[:, :, i + 1] - Wpred[:, :, i + 1]) @ C.T

    return theta, W, lam_pred, nu_pred, log_Zvec_pred, lam_filt, nu_filt, log_Zvec_filt
