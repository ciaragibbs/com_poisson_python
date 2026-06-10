"""Cubic B-spline basis generation."""

from __future__ import annotations

import numpy as np

from .fast_bspline import fastBSpline


def getCubicBSplineBasis(
    x: np.ndarray,
    nknots: int,
    isCirc: bool = False,
) -> np.ndarray:
    """Generate a cubic B-spline design matrix.

    Parameters
    ----------
    x : array-like, shape (n,)
        Evaluation points.
        - If ``isCirc=False``: points in ``[0, 1]``.
        - If ``isCirc=True``: points in ``[0, 2*pi]``.
    nknots : int
        Number of interior knots.  For ``isCirc=False`` the number of basis
        functions is ``nknots + 1`` (intercept column prepended); for
        ``isCirc=True`` it is ``nknots``.
    isCirc : bool, optional
        If True, generate a circular (periodic) B-spline basis. Default False.

    Returns
    -------
    b : np.ndarray, shape (n, n_basis)
        Design matrix of B-spline basis values.
    """
    x = np.asarray(x, dtype=float).ravel()

    if not isCirc:
        nknots_inner = nknots - 1  # match MATLAB's nknots-1 adjustment
        weights = np.ones(nknots_inner + 1)
        knots = np.linspace(-2.0 / nknots_inner, 1.0 + 2.0 / nknots_inner, nknots_inner + 5)
        s = fastBSpline(knots, weights)
        b = s.getBasis(x)
        # Prepend intercept column
        intercept = np.ones((len(x), 1))
        b = np.hstack([intercept, b])
    else:
        knots = np.linspace(-2 * 2 * np.pi / nknots, 2 * np.pi + 2 * 2 * np.pi / nknots, nknots + 5)
        weights = np.ones(nknots + 1)
        s = fastBSpline(knots, weights)
        b = np.zeros((len(x), nknots + 1))
        for k in range(-4, 5):
            xk = x + k * 2 * np.pi
            b += s.getBasis(xk)
        b = b[:, :-1]

    return b
