"""Spike matrix and spike vector utilities."""

from __future__ import annotations

import numpy as np


def getSpkVec(
    tsp: np.ndarray,
    dt: float,
    T: float,
    isLogical: bool = False,
) -> np.ndarray:
    """Build a spike count vector from spike times.

    Parameters
    ----------
    tsp : array-like
        Spike times (positive floats, in seconds or same units as *dt*/*T*).
    dt : float
        Bin size.
    T : float
        Total duration.
    isLogical : bool, optional
        If True, return a binary (0/1) vector with at most one spike per bin.

    Returns
    -------
    S : np.ndarray, shape (ceil(T/dt),)
    """
    tsp = np.asarray(tsp, dtype=float).ravel()
    tsp = tsp[(tsp > 0) & (tsp < T)]
    n_bins = int(np.ceil(T / dt))

    if not isLogical:
        S = np.zeros(n_bins)
        bin_indices = (np.ceil(tsp / dt) - 1).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        for idx in bin_indices:
            S[idx] += 1
    else:
        S = np.zeros(n_bins, dtype=bool)
        bin_indices = (np.ceil(tsp / dt) - 1).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        S[bin_indices] = True

    return S


def getSpkMat(
    Tlist: list | np.ndarray,
    dt: float,
    T: float | None = None,
    isLogical: bool = False,
) -> np.ndarray:
    """Build a spike matrix from a list of spike time arrays.

    Parameters
    ----------
    Tlist : list of array-like, or array-like
        If a list, each element is a spike-time array for one neuron.
        If a single array, returns a 1-D spike vector.
    dt : float
        Bin size.
    T : float, optional
        Total duration.  Inferred from the maximum spike time if not given.
    isLogical : bool, optional
        If True, return binary rows.

    Returns
    -------
    S : np.ndarray
        Shape ``(n_neurons, n_bins)`` if *Tlist* is a list, else ``(n_bins,)``.
    """
    if T is None or T == 0:
        if isinstance(Tlist, (list, tuple)):
            T = max(np.max(np.asarray(t, dtype=float)) for t in Tlist if len(t) > 0)
        else:
            T = float(np.max(np.asarray(Tlist, dtype=float)))

    if isinstance(Tlist, (list, tuple)):
        rows = [getSpkVec(t, dt, T, isLogical) for t in Tlist]
        return np.vstack(rows)
    else:
        return getSpkVec(np.asarray(Tlist), dt, T, isLogical)
