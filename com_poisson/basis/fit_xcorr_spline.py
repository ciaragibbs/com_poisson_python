"""Cross-correlation fitting with B-splines."""

from __future__ import annotations

import numpy as np

from .fast_bspline import fastBSpline


def fitXcorrSpline(
    x: np.ndarray,
    y: np.ndarray,
    nknots: int,
) -> np.ndarray:
    """Fit cross-correlation data with a smoothness-penalised B-spline.

    Parameters
    ----------
    x : array-like, shape (n,)
        Lag values (normalised to ``[-0.5, 0.5]``).
    y : array-like, shape (n,)
        Cross-correlation values at each lag.
    nknots : int
        Number of interior knots.

    Returns
    -------
    yhat : np.ndarray, shape (n,)
        Smoothed cross-correlation values.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Augmented evaluation grid (MATLAB adds -1 and 2 as endpoint guards)
    x0 = np.concatenate([[-1.0], x, [2.0]])

    # Knot sequence: symmetrically spaced log-scale knots around 0.5
    log_knots = np.logspace(-3, 0, nknots) / 2.0
    knots = np.sort(np.concatenate([[-1.0], -log_knots + 0.5, log_knots + 0.5, [1.0]]))

    # Augmented observations with zero boundary conditions
    y0 = np.concatenate([[0.0], y, [0.0]])

    sp = fastBSpline.pspline(knots, 3, x0, y0, lam=2.0)
    yhat = sp.evalAt(x)
    return yhat
