"""Optimization subpackage for COM-Poisson adaptive filtering.

Public API
----------
blktridiag                    : sparse block tridiagonal matrix
newtonGH                      : Newton-Raphson optimiser
gradHessTheta_na              : CMP gradient/Hessian (free nu)
gradHessTheta_CMP_fixNu_na    : CMP gradient/Hessian (fixed nu)
gradHessTheta_Poisson_nan     : Poisson gradient/Hessian
llhdCMP_consNu_na             : CMP log-likelihood with constant nu
helper_na                     : CMP Q-optimisation helper
helper_poisson_nan            : Poisson Q-optimisation helper
"""

from .blktridiag import blktridiag
from .grad_hess_theta_cmp_fixnu_na import gradHessTheta_CMP_fixNu_na
from .grad_hess_theta_na import gradHessTheta_na
from .grad_hess_theta_poisson_nan import gradHessTheta_Poisson_nan
from .helper_na import helper_na
from .helper_poisson_nan import helper_poisson_nan
from .llhd_cmp_cons_nu_na import llhdCMP_consNu_na
from .newton_gh import newtonGH

__all__ = [
    "blktridiag",
    "newtonGH",
    "gradHessTheta_na",
    "gradHessTheta_CMP_fixNu_na",
    "gradHessTheta_Poisson_nan",
    "llhdCMP_consNu_na",
    "helper_na",
    "helper_poisson_nan",
]
