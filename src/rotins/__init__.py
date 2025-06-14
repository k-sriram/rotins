"""
rotins - A module for rotational and instrumental broadening of stellar spectra
============================================================================

This module provides functionality to perform rotational and instrumental
broadening on stellar spectra.
"""

"""
rotins - A module for rotational and instrumental broadening of stellar spectra
============================================================================

This module provides functionality to perform rotational and instrumental
broadening on stellar spectra.
"""

from rotins.core import (
    Broadening,
    InsKernel,
    RotIns,
    __version__,
    rotins,
)

__all__ = ["RotIns", "InsKernel", "Broadening", "rotins"]
__version__ = __version__  # Re-export from core
