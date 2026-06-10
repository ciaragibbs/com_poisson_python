"""Newton-Raphson optimiser using gradient and Hessian information."""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def newtonGH(
    fdf: Callable[[np.ndarray], list],
    x0: np.ndarray,
    TolX: float = 1e-10,
    MaxIter: int = 1000,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newton-Raphson optimiser using gradient and Hessian (or its inverse).

    Parameters
    ----------
    fdf : callable
        Function that takes ``x`` and returns ``[grad, hess]`` where
        ``grad`` is the gradient vector and ``hess`` is the Hessian matrix
        (or sparse Hessian).  The solver minimises ``-log-likelihood``, so
        the Hessian should be negative definite at the minimum.
    x0 : np.ndarray, shape (p,)
        Initial parameter vector.
    TolX : float, optional
        Convergence tolerance on the step size. Default 1e-10.
    MaxIter : int, optional
        Maximum number of iterations. Default 1000.
    verbose : bool, optional
        If True, print iteration progress.

    Returns
    -------
    x : np.ndarray, shape (p,)
        Solution (converged parameter vector).
    fx : np.ndarray
        Gradient at the solution.
    dfdx : np.ndarray or sparse matrix
        Hessian at the solution.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within ``MaxIter`` steps.
    ValueError
        If the iterate becomes NaN or Inf.
    """
    TolFun = np.finfo(float).eps
    xx = [x0.copy()]
    dh = fdf(x0)
    fx = dh[0]
    dfdx = dh[1]

    k = 0
    for k in range(1, MaxIter + 1):
        dfdx = dh[1]

        # Solve dfdx * dx = -fx  (Newton step)
        if sp.issparse(dfdx):
            try:
                dx, _ = spla.minres(dfdx, -fx)
            except Exception:
                dx = spla.spsolve(dfdx, -fx)
        else:
            try:
                dx = np.linalg.solve(dfdx, -fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(dfdx, -fx, rcond=None)[0]

        x_new = xx[-1] + dx
        xx.append(x_new)

        dh = fdf(x_new)
        fx = dh[0]

        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            raise ValueError("Newton iteration diverged (NaN/Inf in iterate).")

        if verbose:
            print(f"iter: {k}, norm(fx) = {np.linalg.norm(fx):.6g}")

        if np.linalg.norm(fx) < TolFun or np.linalg.norm(dx) < TolX:
            break

    if k >= MaxIter:
        raise RuntimeError(
            f"newtonGH did not converge in {MaxIter} iterations. "
            "Consider increasing MaxIter or TolX."
        )

    x = xx[-1]
    dfdx = dh[1]
    return x, fx, dfdx
