"""COM-Poisson Python Package.

A Python 3.9+ package implementing adaptive filtering for Conway-Maxwell-Poisson
(CMP) distributions, primarily for neuronal spike data analysis.

Subpackages
-----------
distribution   : CMP distribution functions (PDF, RNG, moments)
smoother       : Adaptive filtering and smoothing algorithms
optimization   : Newton-Raphson optimisation and helpers
basis          : B-spline basis functions

Usage
-----
>>> from com_poisson.distribution import CMPmoment, com_pdf, com_rnd
>>> from com_poisson.smoother import ppasmoo_compoisson_fisher_na
>>> from com_poisson.basis import getCubicBSplineBasis
"""

from . import basis, distribution, optimization, smoother, utils

__version__ = "0.1.0"
__all__ = ["distribution", "smoother", "optimization", "basis", "utils"]
