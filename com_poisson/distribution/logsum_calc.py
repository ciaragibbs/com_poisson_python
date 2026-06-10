"""Log-sum calculations for numerical stability in CMP distribution computations."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def logsum_calc(lam: float, nu: float, n: int) -> np.ndarray:
    """Compute log-sum terms needed for CMP distribution moments.

    This function computes six log-sum quantities used in the exact
    computation of CMP moments via numerical summation.

    Parameters
    ----------
    lam : float
        Lambda parameter of the CMP distribution (lam > 0).
    nu : float
        Nu (dispersion) parameter of the CMP distribution (nu > 0).
    n : int
        Maximum summation index (truncation point).

    Returns
    -------
    cum_log_app : np.ndarray, shape (6,)
        Array of log-sum values:
        [0] log(Z)         - log normalisation constant
        [1] log(sum Y)     - for computing E[Y]
        [2] log(sum Y^2)   - for computing E[Y^2]
        [3] log(sum log(j!)) - for computing E[log(Y!)]
        [4] log(sum log(j!)^2) - for computing E[(log(Y!))^2]
        [5] log(sum Y*log(j!)) - for computing Cov(Y, log(Y!))
    """
    cum_log_app = np.zeros(6)

    js = np.arange(0, n + 1, dtype=float)

    # log of unnormalised probabilities: j*log(lam) - nu*log(j!)
    log_Zi = js * np.log(lam) - nu * gammaln(js + 1)

    # weighted terms
    with np.errstate(divide="ignore", invalid="ignore"):
        log_js = np.where(js > 0, np.log(js), -np.inf)
        log2_js = np.where(js > 0, 2 * np.log(js), -np.inf)

    log_Ai = log_js + js * np.log(lam) - nu * gammaln(js + 1)
    log_Bi = log2_js + js * np.log(lam) - nu * gammaln(js + 1)

    # Terms involving log(j!) – start from j=2 (index 2 onward) since log(0!)=log(1!)=0
    idx = np.arange(2, n + 1, dtype=float)
    log_gammaln_idx = np.log(gammaln(idx + 1))  # log(log(j!)) only valid for j>=2

    log_Ci = log_Zi[2:] + log_gammaln_idx
    log_Di = log_Zi[2:] + 2 * log_gammaln_idx
    log_Ei = log_Ai[2:] + log_gammaln_idx

    def _logsumexp(log_vals: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        finite_vals = log_vals[np.isfinite(log_vals)]
        if len(finite_vals) == 0:
            return -np.inf
        log_max = np.max(finite_vals)
        return log_max + np.log(np.sum(np.exp(finite_vals - log_max)))

    cum_log_app[0] = _logsumexp(log_Zi)
    cum_log_app[1] = _logsumexp(log_Ai)
    cum_log_app[2] = _logsumexp(log_Bi)
    cum_log_app[3] = _logsumexp(log_Ci)
    cum_log_app[4] = _logsumexp(log_Di)
    cum_log_app[5] = _logsumexp(log_Ei)

    return cum_log_app
