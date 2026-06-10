"""Moments of the Conway-Maxwell-Poisson distribution."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .logsum_calc import logsum_calc


def CMPmoment(
    lam: float,
    nu: float,
    maxSum: int = 1000,
) -> tuple[float, float, float, float, float, float]:
    """Compute moments of the Conway-Maxwell-Poisson distribution.

    Uses a saddlepoint approximation when ``lam >= 2`` and ``nu <= 2``
    (the large-count regime), and numerical summation otherwise.

    Parameters
    ----------
    lam : float
        Lambda parameter of the CMP distribution (lam > 0).
    nu : float
        Nu (dispersion) parameter (nu > 0).
    maxSum : int, optional
        Maximum summation index used in numerical computation. Default 1000.

    Returns
    -------
    mean_Y : float
        E[Y]
    var_Y : float
        Var[Y]
    mean_logYfac : float
        E[log(Y!)]
    var_logYfac : float
        Var[log(Y!)]
    cov_Y_logYfac : float
        Cov(Y, log(Y!))
    log_Z : float
        Log of the normalisation constant log(Z(lambda, nu)).
    """
    if lam >= 2 and nu <= 2:
        alpha = lam ** (1.0 / nu)
        c1 = (nu ** 2 - 1) / 24.0
        c2 = (nu ** 2 - 1) / 48.0 + c1 ** 2 / 2.0

        log_Z = (
            nu * alpha
            - (nu - 1) * np.log(lam) / (2.0 * nu)
            - (nu - 1) * np.log(2 * np.pi) / 2.0
            - np.log(nu) / 2.0
            + np.log(1.0 + c1 / (nu * alpha) + c2 / (nu * alpha) ** 2)
        )

        mean_Y = (
            alpha
            - (nu - 1) / (2.0 * nu)
            - (nu ** 2 - 1) / (24.0 * nu ** 2 * alpha)
            - (nu ** 2 - 1) / (24.0 * nu ** 3 * alpha ** 2)
        )

        var_Y = (
            alpha / nu
            + (nu ** 2 - 1) / (24.0 * nu ** 3 * alpha)
            + (nu ** 2 - 1) / (12.0 * nu ** 4 * alpha ** 2)
        )

        mean_logYfac = (
            alpha * (np.log(lam) / nu - 1)
            + np.log(lam) / (2.0 * nu ** 2)
            + 1.0 / (2.0 * nu)
            + np.log(2 * np.pi) / 2.0
            - (1.0 / (24.0 * alpha))
            * (1 + 1.0 / nu ** 2 + np.log(lam) / nu - np.log(lam) / nu ** 3)
            - (1.0 / (24.0 * alpha ** 2))
            * (1.0 / nu ** 3 + np.log(lam) / nu ** 2 - np.log(lam) / nu ** 4)
        )

        var_logYfac = (
            alpha * np.log(lam) ** 2 / nu ** 3
            + np.log(lam) / nu ** 3
            + 1.0 / (2.0 * nu ** 3)
            + (1.0 / (24.0 * nu ** 5 * alpha))
            * (-2 * nu ** 2 + 4 * nu * np.log(lam) + (nu ** 2 - 1) * np.log(lam) ** 2)
            + (1.0 / (24.0 * nu ** 6 * alpha ** 2))
            * (
                -3 * nu ** 2
                - 2 * nu * (nu ** 2 - 3) * np.log(lam)
                + 2 * (nu ** 2 - 1) * np.log(lam) ** 2
            )
        )

        cov_Y_logYfac = (
            alpha * np.log(lam) / nu ** 2
            + 1.0 / (2.0 * nu ** 2)
            + (1.0 / (24.0 * alpha))
            * (2.0 / nu ** 3 + np.log(lam) / nu ** 2 - np.log(lam) / nu ** 4)
            - (1.0 / (24.0 * alpha ** 2))
            * (1.0 / nu ** 2 - 3.0 / nu ** 4 - 2 * np.log(lam) / nu ** 3 + 2 * np.log(lam) / nu ** 5)
        )

    else:
        logcum_app = logsum_calc(lam, nu, maxSum)

        log_Z = logcum_app[0]
        mean_Y = np.exp(logcum_app[1] - log_Z)
        var_Y = np.exp(logcum_app[2] - log_Z) - mean_Y ** 2
        mean_logYfac = np.exp(logcum_app[3] - log_Z)
        var_logYfac = np.exp(logcum_app[4] - log_Z) - mean_logYfac ** 2
        cov_Y_logYfac = np.exp(logcum_app[5] - log_Z) - np.exp(
            logcum_app[1] + logcum_app[3] - 2 * log_Z
        )

    return mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, log_Z
