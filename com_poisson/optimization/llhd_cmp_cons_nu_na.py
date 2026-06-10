"""CMP log-likelihood with constant nu."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from ..distribution.cmp_moment import CMPmoment


def llhdCMP_consNu_na(
    nu: float,
    X_lam: np.ndarray,
    theta: np.ndarray,
    spk_vec: np.ndarray,
) -> float:
    """Compute the CMP log-likelihood for constant nu.

    Parameters
    ----------
    nu : float
        Fixed dispersion parameter.
    X_lam : np.ndarray, shape (T, n_lam)
        Design matrix for lambda.
    theta : np.ndarray, shape (n_lam, T)
        Parameter matrix at each time step.
    spk_vec : np.ndarray, shape (T,)
        Observed spike counts (NaN = missing).

    Returns
    -------
    llhd : float
        Log-likelihood value.
    """
    maxSum = int(10 * np.nanmax(spk_vec))
    llhd = 0.0

    for t in range(spk_vec.shape[0]):
        if not np.isnan(spk_vec[t]):
            loglamTmp = float(X_lam[t, :] @ theta[:, t])
            lamTmp = np.exp(loglamTmp)
            _, _, _, _, _, logZtmp = CMPmoment(lamTmp, nu, maxSum)
            llhd += (
                spk_vec[t] * loglamTmp
                - nu * gammaln(spk_vec[t] + 1)
                - logZtmp
            )

    return llhd
