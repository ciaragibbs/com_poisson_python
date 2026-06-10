"""Block tridiagonal sparse matrix constructor."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def blktridiag(
    Amd: np.ndarray,
    Asub: np.ndarray,
    Asup: np.ndarray,
    n: int | None = None,
) -> sp.csr_matrix:
    """Construct a sparse block tridiagonal matrix.

    Two calling conventions are supported:

    **Mode 1** – replicated blocks::

        A = blktridiag(Amd, Asub, Asup, n)

    ``Amd``, ``Asub``, ``Asup`` are 2-D arrays of shape ``(p, q)``, replicated
    ``n``, ``n-1``, ``n-1`` times on the respective diagonals.

    **Mode 2** – distinct blocks::

        A = blktridiag(Amd, Asub, Asup)

    ``Amd`` has shape ``(p, q, n)``, ``Asub`` and ``Asup`` have shape
    ``(p, q, n-1)``.

    Parameters
    ----------
    Amd : np.ndarray
        Main diagonal blocks (``p x q`` or ``p x q x n``).
    Asub : np.ndarray
        Sub-diagonal blocks (``p x q`` or ``p x q x (n-1)``).
    Asup : np.ndarray
        Super-diagonal blocks (``p x q`` or ``p x q x (n-1)``).
    n : int, optional
        Number of blocks (Mode 1 only).

    Returns
    -------
    A : scipy.sparse.csr_matrix
        ``(n*p) x (n*q)`` sparse block tridiagonal matrix.
    """
    if n is not None:
        # Mode 1: replicated blocks
        Amd = np.atleast_2d(Amd)
        Asub = np.atleast_2d(Asub)
        Asup = np.atleast_2d(Asup)
        p, q = Amd.shape
        if n == 1:
            return sp.csr_matrix(Amd)
        Amd_3d = np.stack([Amd] * n, axis=-1)
        Asub_3d = np.stack([Asub] * (n - 1), axis=-1)
        Asup_3d = np.stack([Asup] * (n - 1), axis=-1)
    else:
        # Mode 2: distinct blocks
        if Amd.ndim == 2:
            Amd = Amd[:, :, np.newaxis]
        if Asub.ndim == 2:
            Asub = Asub[:, :, np.newaxis]
        if Asup.ndim == 2:
            Asup = Asup[:, :, np.newaxis]
        Amd_3d = Amd
        Asub_3d = Asub
        Asup_3d = Asup

    p, q, n_blocks = Amd_3d.shape

    rows = []
    cols = []
    vals = []

    # Main diagonal blocks
    for k in range(n_blocks):
        for r in range(p):
            for c in range(q):
                rows.append(k * p + r)
                cols.append(k * q + c)
                vals.append(Amd_3d[r, c, k])

    # Sub-diagonal blocks (below main diagonal, block row k+1, col k)
    for k in range(n_blocks - 1):
        for r in range(p):
            for c in range(q):
                rows.append((k + 1) * p + r)
                cols.append(k * q + c)
                vals.append(Asub_3d[r, c, k])

    # Super-diagonal blocks (above main diagonal, block row k, col k+1)
    for k in range(n_blocks - 1):
        for r in range(p):
            for c in range(q):
                rows.append(k * p + r)
                cols.append((k + 1) * q + c)
                vals.append(Asup_3d[r, c, k])

    total_rows = n_blocks * p
    total_cols = n_blocks * q
    A = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(total_rows, total_cols),
        dtype=float,
    )
    return A
