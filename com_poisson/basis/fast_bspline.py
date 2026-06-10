"""Fast B-spline implementation using scipy.interpolate.

This module provides a Python translation of the MATLAB fastBSpline class,
using ``scipy.interpolate.BSpline`` as the computational backend.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate as si
import scipy.linalg as sl


class fastBSpline:
    """Lightweight B-spline class with an interface similar to the MATLAB fastBSpline.

    A B-spline is defined by:

        y(x) = sum_i  B_i(x, knots) * weights_i

    The order of the spline is ``len(knots) - len(weights) - 1``.

    Parameters
    ----------
    knots : array-like
        Non-decreasing knot sequence.
    weights : array-like
        Coefficient (weight) vector.

    Attributes
    ----------
    knots : np.ndarray
    weights : np.ndarray
    order : int
        Polynomial order (``len(knots) - len(weights) - 1``).
    """

    ZERO = "zero"
    CONSTANT = "constant"

    def __init__(self, knots: np.ndarray, weights: np.ndarray) -> None:
        self.knots = np.asarray(knots, dtype=float)
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.outOfRange = self.ZERO

    @property
    def order(self) -> int:
        return len(self.knots) - len(self.weights) - 1

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evalAt(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the spline at points *x*.

        Parameters
        ----------
        x : array-like
        Returns
        -------
        Sx : np.ndarray, same shape as *x*
        """
        x = np.asarray(x, dtype=float)
        shape = x.shape
        x_flat = x.ravel()

        if self.outOfRange == self.CONSTANT:
            x_flat = np.clip(x_flat, self.knots[0], self.knots[-1])

        sp = si.BSpline(self.knots, self.weights, self.order, extrapolate=False)
        Sx = sp(x_flat)
        if self.outOfRange == self.ZERO:
            Sx = np.where(np.isnan(Sx), 0.0, Sx)
        return Sx.reshape(shape)

    def getBasis(self, x: np.ndarray) -> np.ndarray:
        """Return the B-spline design matrix at points *x*.

        Parameters
        ----------
        x : array-like, shape (n,)

        Returns
        -------
        B : np.ndarray, shape (n, len(weights))
            Each column *i* is ``B_i(x)``.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = len(x)
        nw = len(self.weights)
        B = np.zeros((n, nw))
        k = self.order
        knots = self.knots

        for i in range(nw):
            w_unit = np.zeros(nw)
            w_unit[i] = 1.0
            sp = si.BSpline(knots, w_unit, k, extrapolate=False)
            col = sp(x)
            B[:, i] = np.where(np.isnan(col), 0.0, col)
        return B

    def Btimesy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute ``getBasis(x).T @ y`` efficiently.

        Parameters
        ----------
        x : array-like, shape (n,)
        y : array-like, shape (n,)

        Returns
        -------
        result : np.ndarray, shape (len(weights),)
        """
        B = self.getBasis(x)
        return B.T @ np.asarray(y, dtype=float).ravel()

    def dx(self) -> "fastBSpline":
        """Return a new fastBSpline that is the derivative of this spline."""
        k = self.order
        knots = self.knots
        weights = self.weights

        # Derivative formula for B-splines
        new_weights = np.zeros(len(weights) - 1)
        for i in range(len(new_weights)):
            denom = knots[i + k + 1] - knots[i + 1]
            if denom == 0:
                new_weights[i] = 0.0
            else:
                new_weights[i] = k * (weights[i + 1] - weights[i]) / denom
        new_knots = knots[1:-1]
        return fastBSpline(new_knots, new_weights)

    # ------------------------------------------------------------------ #
    # Static constructors                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def lsqspline(
        knots: np.ndarray,
        order: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "fastBSpline":
        """Fit a least-squares B-spline to observations.

        Parameters
        ----------
        knots : array-like
            Knot sequence.
        order : int
            Polynomial order (usually 3 for cubic).
        x : array-like
            Observation locations.
        y : array-like
            Observed values.

        Returns
        -------
        sp : fastBSpline
        """
        knots = np.asarray(knots, dtype=float)
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        nw = len(knots) - order - 1

        # Build design matrix and solve
        sp0 = fastBSpline(knots, np.zeros(nw))
        B = sp0.getBasis(x)
        weights, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
        return fastBSpline(knots, weights)

    @staticmethod
    def pspline(
        knots: np.ndarray,
        order: int,
        x: np.ndarray,
        y: np.ndarray,
        lam: float = 1.0,
        pdegree: int = 2,
    ) -> "fastBSpline":
        """Fit a smoothness-penalized B-spline (P-spline) to observations.

        Parameters
        ----------
        knots : array-like
        order : int
            Polynomial order.
        x : array-like
            Observation locations.
        y : array-like
            Observed values.
        lam : float
            Penalty weight (larger = smoother).
        pdegree : int
            Order of the difference penalty. Default 2 (penalise curvature).

        Returns
        -------
        sp : fastBSpline
        """
        knots = np.asarray(knots, dtype=float)
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        nw = len(knots) - order - 1

        sp0 = fastBSpline(knots, np.zeros(nw))
        B = sp0.getBasis(x)

        # Build the difference-penalty matrix D
        D = np.diff(np.eye(nw), n=pdegree, axis=0)
        P = lam * D.T @ D

        weights = np.linalg.solve(B.T @ B + P, B.T @ y)
        return fastBSpline(knots, weights)
