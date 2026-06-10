"""Conway-Maxwell-Poisson probability density function."""

import numpy as np
from scipy.special import gammaln


def com_pdf(n: int | np.ndarray, lam: float, nu: float) -> float | np.ndarray:
    """Compute the Conway-Maxwell-Poisson probability density function.

    The CMP distribution has PMF:
        P(Y=n) = (lambda^n / (n!)^nu) / Z(lambda, nu)

    where Z is the normalisation constant.

    Parameters
    ----------
    n : int or array-like
        Count value(s) at which to evaluate the PMF (non-negative integers).
    lam : float
        Lambda parameter (lam > 0).
    nu : float
        Nu (dispersion) parameter (nu > 0).
        nu=1 gives Poisson, nu->0 gives geometric, nu->inf gives Bernoulli.

    Returns
    -------
    pdf : float or np.ndarray
        Probability P(Y=n) for each value in n.

    References
    ----------
    Kadane et al. (2003), "Conjugate Analysis of the Conway-Maxwell-Poisson
    Distribution", Carnegie Mellon University.
    """
    summax = 1000
    termlim = 1e-12

    Z = 0.0
    for js in range(1, summax + 1):
        term = np.exp((js - 1) * np.log(lam) - nu * gammaln(js))
        if js > 3 and (term / Z) < termlim:
            break
        Z += term

    n_arr = np.asarray(n, dtype=float)
    return np.exp(n_arr * np.log(lam) - nu * gammaln(n_arr + 1)) / Z
