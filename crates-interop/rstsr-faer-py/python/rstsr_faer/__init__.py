"""rstsr_faer — Python array-API binding of rstsr (faer device).

Thin package: the native layer is ``rstsr_faer.rstsr_faer`` (pyo3), and the
array-API-graded namespace is :mod:`rstsr_faer.api`.  No algorithms live on
the Python side; Python only shuffles handles and mirrors signatures.
"""

from . import rstsr_faer as _native
from . import api

__version__ = "0.9.0"
