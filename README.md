# COM_POISSON Python Package

A Python implementation of adaptive filtering for Conway-Maxwell-Poisson (CMP) distributions.

This package converts the original MATLAB codebase (https://github.com/weigcdsb/COM_POISSON) to Python 3.9+.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from com_poisson.distribution import CMPmoment, com_pdf, com_rnd
from com_poisson.smoother import ppasmoo_compoisson_fisher_na
```

## Features

- Conway-Maxwell-Poisson distribution functions
- Adaptive filtering and smoothing algorithms
- Parameter optimization using Newton-Raphson methods
- B-spline basis functions for flexible modeling
- Designed for neuronal spike data analysis

