"""Random sampling from the Conway-Maxwell-Poisson distribution."""

from __future__ import annotations

import numpy as np

from .com_pdf import com_pdf


def com_rnd(
    lam: float | np.ndarray,
    nu: float | np.ndarray,
    N: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw random samples from a Conway-Maxwell-Poisson distribution.

    Parameters
    ----------
    lam : float or array-like
        Lambda parameter(s).  When ``N`` is None the length of ``lam``
        determines how many samples are drawn (one per element).
    nu : float or array-like
        Nu (dispersion) parameter(s).  Must be broadcastable with ``lam``.
    N : int, optional
        Number of samples to draw.  When provided, ``lam`` and ``nu`` must
        be scalars and ``N`` samples are returned.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.  If None, uses
        ``np.random.default_rng()``.

    Returns
    -------
    x : np.ndarray
        Sampled counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    max_val = 1000
    counts = np.arange(0, max_val + 1)

    if N is None:
        lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
        nu_arr = np.atleast_1d(np.asarray(nu, dtype=float))
        x = np.empty(len(lam_arr), dtype=int)
        for i in range(len(lam_arr)):
            probs = com_pdf(counts, lam_arr[i], nu_arr[i])
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            cdf = np.cumsum(probs)
            u = rng.uniform()
            x[i] = int(np.searchsorted(cdf, u))
        return x
    else:
        lam_scalar = float(lam)
        nu_scalar = float(nu)
        probs = com_pdf(counts, lam_scalar, nu_scalar)
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        cdf = np.cumsum(probs)
        u = rng.uniform(size=int(N))
        return np.searchsorted(cdf, u).astype(int)
