"""Basis function subpackage for COM-Poisson package.

Public API
----------
fastBSpline              : B-spline class
getCubicBSplineBasis     : generate cubic B-spline design matrix
getSpkMat                : build spike matrix from spike times
fitXcorrSpline           : smooth cross-correlations with B-splines
"""

from .fast_bspline import fastBSpline
from .fit_xcorr_spline import fitXcorrSpline
from .get_cubic_bspline_basis import getCubicBSplineBasis
from .get_spk_mat import getSpkMat, getSpkVec

__all__ = [
    "fastBSpline",
    "getCubicBSplineBasis",
    "getSpkMat",
    "getSpkVec",
    "fitXcorrSpline",
]
