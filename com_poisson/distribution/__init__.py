"""Conway-Maxwell-Poisson distribution subpackage.

Public API
----------
CMPmoment   : compute CMP distribution moments
com_pdf     : CMP probability mass function
com_rnd     : random sampling from CMP
logsum_calc : log-sum helper for numerical stability
varParam    : parameter variance calculations
"""

from .cmp_moment import CMPmoment
from .com_pdf import com_pdf
from .com_rnd import com_rnd
from .logsum_calc import logsum_calc
from .var_param import varParam

__all__ = [
    "CMPmoment",
    "com_pdf",
    "com_rnd",
    "logsum_calc",
    "varParam",
]
