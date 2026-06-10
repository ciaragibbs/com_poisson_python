"""Parameter variance calculations for CMP distributions."""

from __future__ import annotations

import numpy as np

from .cmp_moment import CMPmoment


def varParam(
    X_lam: np.ndarray,
    G_nu: np.ndarray,
    theta_fit: np.ndarray,
    W_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute variance of the CMP rate parameter.

    Computes the variance of the mean firing rate using both exact
    (via CMP moments) and approximate (saddlepoint) methods via the
    delta method.

    Parameters
    ----------
    X_lam : np.ndarray, shape (T, n_lam)
        Design matrix for the lambda (rate) parameter.
    G_nu : np.ndarray, shape (T, n_nu)
        Design matrix for the nu (dispersion) parameter.
    theta_fit : np.ndarray, shape (n_lam + n_nu, T)
        Fitted parameter matrix at each time step.
    W_fit : np.ndarray, shape (n_lam + n_nu, n_lam + n_nu, T)
        Posterior covariance matrix at each time step.

    Returns
    -------
    var_rate_exact : np.ndarray, shape (T,)
        Variance of the rate using exact CMP moments.
    var_rate_app : np.ndarray, shape (T,)
        Variance of the rate using saddlepoint approximation.
    """
    maxSum = 1000
    T = theta_fit.shape[1]
    var_rate_exact = np.zeros(T)
    var_rate_app = np.zeros(T)

    nLam = X_lam.shape[1]
    nNu = G_nu.shape[1]

    for k in range(T):
        x_tmp = X_lam[k, :]          # (nLam,)
        g_tmp = G_nu[k, :]            # (nNu,)

        # Build the 2 x (nLam+nNu) linear map Z_tmp
        Z_tmp = np.zeros((2, nLam + nNu))
        Z_tmp[0, :nLam] = x_tmp
        Z_tmp[1, nLam:] = g_tmp

        Mu_tmp = Z_tmp @ theta_fit[:, k]           # (2,)
        Sig_tmp = Z_tmp @ W_fit[:, :, k] @ Z_tmp.T  # (2,2)

        # Delta-method variance of the covariance block
        # V[i,j] = (exp(Sig[i,j]) - 1) * exp(Mu[i] + Mu[j] + (Sig[i,i]+Sig[j,j])/2)
        mu_outer = np.add.outer(Mu_tmp, Mu_tmp)            # (2,2)
        sig_diag = np.diag(Sig_tmp)
        sig_diag_outer = np.add.outer(sig_diag, sig_diag) / 2.0  # (2,2)
        V = (np.exp(Sig_tmp) - 1.0) * np.exp(mu_outer + sig_diag_outer)

        lam_tmp = np.exp(Mu_tmp[0])
        nu_tmp = np.exp(Mu_tmp[1])

        # Exact gradient using CMP moments
        _, var_Y, _, _, cov_Y_logYfac, _ = CMPmoment(lam_tmp, nu_tmp, maxSum)
        grad_exact = np.array([var_Y / lam_tmp, -cov_Y_logYfac])
        var_rate_exact[k] = grad_exact @ V @ grad_exact

        # Approximate gradient (saddlepoint)
        alpha = lam_tmp ** (1.0 / nu_tmp)
        grad_app = np.array([
            (1.0 / nu_tmp) * (lam_tmp ** (1.0 / nu_tmp - 1)),
            -alpha * np.log(lam_tmp) / nu_tmp ** 2 - 1.0 / (2.0 * nu_tmp ** 2),
        ])
        var_rate_app[k] = grad_app @ V @ grad_app

    return var_rate_exact, var_rate_app
