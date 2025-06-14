# rotins.py
# A module to perform rotational and instrumental broadening of synthetic
# stellar spectra. This is written as a python translation of the FORTRAN
# program used in TLUSTY by Ivan Hubeny et al.
# Author: Sriram Krishna
# Created: 2022-01-28

# MIT License

# Copyright (c) 2022 Sriram Krishna

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


r"""
rotins
======

This module provides functionality to perform rotational and instrumental
broadening on stellar spectra. It offers both object-oriented and functional
programming interfaces to accommodate different programming styles.

Interfaces
---------
    1. Class-based (RotIns):
        Provides an object-oriented interface with full configuration options.
        Suitable for complex workflows and when you need to customize the
        broadening process.

    2. Functional (rotins):
        Provides a functional programming interface that returns a closure.
        Ideal for simple workflows and when you need to apply the same
        broadening parameters to multiple spectra.

Examples
--------
    Using the class-based interface:
        >>> from rotins import RotIns
        >>> broadener = RotIns(vsini=50.0, fwhm=0.1)
        >>> conv_wl, conv_spec = broadener.broaden(wl, spec)

    Using the functional interface:
        >>> from rotins import rotins
        >>> broaden = rotins(vsini=50.0, fwhm=0.1)
        >>> conv_wl1, conv_spec1 = broaden(wl1, spec1)
        >>> conv_wl2, conv_spec2 = broaden(wl2, spec2)

    Using spectral resolution instead of FWHM:
        >>> broadener = RotIns(vsini=50.0, fwhm=50000, fwhm_type="res")
        >>> conv_wl, conv_spec = broadener.broaden(wl, spec)

Notes
-----
    The input spectra need not have uniform spacing. This module will first
    cast the input spectra into a uniform grid with an appropriate step size.

    If either of vsini or fwhm is given as `None` or 0.0, the corresponding
    broadening will be skipped.

    For normalized spectra, use base_flux=1.0 (default).
    For non-normalized spectra, use base_flux=0.0.

    The calculations are done according to the book:
    Gray, D.F., 2008, The Observation and Analysis of Stellar Photospheres,
    Cambridge University Press, Cambridge, 3rd edition.

Theory
------
    For rotational broadening the kernel is:
    .. math::
    G(\Delta\lambda) = \frac{2(1-\epsilon)\[1-(\Delta\lambda/\Delta\lambda_0)^2\]^{1/2}+(\pi\epsilon/2)\[1-(\Delta\lambda/\Delta\lambda_0)^2\]}{\pi\Delta\lambda_0(1-\epsilon/3)}
    where,
    .. math::
    \Delta\lambda_0 = \frac{\lambda v \sin i}{c}

    For instrumental broadening the kernel is a Gaussian with standard
    deviation given by the FWHM:
    .. math::
    \sigma = \frac{\textrm{FWHM}}{2\sqrt{2\ln 2}}

Module Attributes
---------------
    char_step : float = 0.01
        Minimum characteristic step size for the convolution kernels.
        The default value is good for optical spectra when described in
        Angstrom.
"""  # noqa: E501

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

# dunders

__version__ = "1.0.0"
__all__ = ["convolve", "Kernel", "RotKernel", "InsKernel", "Broadening", "RotIns"]

# Constants

DEFAULT_LIMB_COEFF = 0.6
SPEED_LIGHT_KMS = 2.99792e5


# Module level variables

char_step = 0.01
should_convolve_pad = True

# Utility functions


def convolve(
    spec: npt.NDArray[np.floating],
    kernel: npt.NDArray[np.floating],
    pad: Optional[bool] = None,
) -> npt.NDArray[np.floating]:
    """Convolves the spectrum with the kernel.

    Optionally, pads the spectrum with edge values before convolution.
    """
    if pad is None:
        pad = should_convolve_pad
    if pad:
        kw = len(kernel)
        spec = np.pad(spec, (kw, kw), "edge")
        conv = np.convolve(spec, kernel, "same")
        return conv[kw:-kw]
    return np.convolve(spec, kernel, "same")


def _get_basis(step: float, limit: float) -> npt.NDArray[np.floating]:
    """Generates a basis array for the convolution kernel.

    The generated array is symmetric about the y-axis. It will overshoot the
    limit by one step.
    """
    numsteps = int(np.ceil(limit / step))
    basis = np.array([step * i for i in range(-numsteps, numsteps + 1)])
    return basis


def _linspace_stepped(
    start: float, stop: float, step: float
) -> npt.NDArray[np.floating]:
    """Like np.linspace but with a step size instead of array size.

    Uses the interval [start, stop).
    """
    numsteps = int(np.floor((stop - start) / step)) + 1
    return np.array([start + step * i for i in range(numsteps)])


def _interpolate_spec(
    wl: npt.NDArray[np.floating],
    spec: npt.NDArray[np.floating],
    basis: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Interpolates the spectrum onto the basis array."""
    inter_f = interp1d(wl, spec, "cubic", assume_sorted=True)
    return inter_f(basis)


def _sort(
    x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sort x and y according to x."""
    if not np.all(x[:-1] <= x[1:]):
        xinds = x.argsort()
        x, y = x[xinds], y[xinds]
    return x, y


def _get_section(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    lim: Optional[Tuple[float, float]] = None,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Truncate the spectrum to the given limits."""

    if lim is None:
        return x.copy(), y.copy()

    if lim[0] < x[0] or lim[1] > x[-1]:
        raise ValueError("lim must be within x")

    limbool = np.logical_and(x >= lim[0], x <= lim[1])

    # Include an extra element if possible
    if not limbool[0]:
        limbool[np.searchsorted(x, lim[0]) - 1] = True
    if not limbool[-1]:
        limbool[np.searchsorted(x, lim[-1])] = True

    return x[limbool], y[limbool]


# Classes


class Kernel(ABC):
    """Abstract base class for convolution kernels.

    The behaviour of the kernel should be defined by the following methods:

    Abstract methods
    ----------------

    prof(wl) -> Profile function
        Returns the profile function for the given wavelength array.

    step(wl) -> step size
        Returns the an ideal step size for the kernel. The shape of the kernel
        should be well-defined on an array of this step size.

    get_default_limits(wl) -> limits
        Returns an ideal size limit of the kernel. The profile function
        should be negligible outside this limit.
    """

    @abstractmethod
    def prof(
        self, wl_mid: float
    ) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
        pass

    @abstractmethod
    def step(self, wl_mid: float) -> float:
        pass

    @abstractmethod
    def get_default_limits(self, wl_mid: float) -> float:
        pass

    def kernel(
        self, wl_mid: float, step: Optional[float] = None, limit: Optional[float] = None
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Returns the kernel at the given wavelength array.

        The kernel is calculated using the `prof` method. The step size and
        limit can be given as arguments. If not, values are calculated using the
        `step` and `get_default_limits` method. The kernel is always normalized.

        Parameters
        ----------
        wl_mid : float
            The central wavelength of the kernel.
        step : float, optional
            The step size of the kernel.
        limit : float, optional
            The width limit upto which the kernel is evaluated. Make sure
            that the profile function is negligible outside this limit.

        Returns
        -------
        ndarray[float]
            The wavelength array of the kernel.
        ndarray[float]
            The calculated kernel.
        """
        if step is None:
            step = self.step(wl_mid)
        if limit is None:
            limit = self.get_default_limits(wl_mid)
        basis = _get_basis(step, limit)
        kernel = self.prof(wl_mid)(basis)
        kernel /= np.sum(kernel)
        return basis, kernel


class RotKernel(Kernel):
    """A convolution kernel for broadening due to rotation of the star."""

    def __init__(
        self,
        vsini: float,
        limb_coeff: float = DEFAULT_LIMB_COEFF,
    ):
        """Initializes the RotKernel.
        This broadening can be mainly characterized using a single
        parameter: v sin i. Where v is the rotational veocity at the surface of
        the star and i is the angle which the angle of rotation of the star
        makes with the line of sight.

        However the limbs of the stars emit less light than the central
        regions. This effect is taken care of by the limb darkening
        coefficient. This program assumes a default value of 0.6 as used in
        Ivan Hubeny's SYNSPEC program.

        Parameters
        ----------
            vsini : float (km s-1)
                The component of rotational velocity of the star along the line
                of sight in km s-1.
            limb_coeff : float, default 0.6
                The limb darkening coefficient."""
        self.vsini = vsini
        self.limb_coeff = limb_coeff

    def _get_dl0(self, wl_mid: float) -> float:
        """Characteristic scale of the rotational kernel."""
        return wl_mid * self.vsini / SPEED_LIGHT_KMS

    def get_default_limits(self, wl_mid: float) -> float:
        return self._get_dl0(wl_mid)

    def step(self, wl_mid: float) -> float:
        return self._get_dl0(wl_mid) / 5

    def prof(
        self, wl_mid: float
    ) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
        dl0 = self._get_dl0(wl_mid)
        c_1 = 2 * (1 - self.limb_coeff)
        c_2 = np.pi * self.limb_coeff / 2
        den = np.pi * dl0 * (1 - self.limb_coeff / 3)

        def prof_func(
            x: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            x_1 = 1 - np.square(x / dl0)
            z = np.zeros(len(x_1))
            x_1 = np.maximum(x_1, z)
            return (c_1 * np.sqrt(x_1) + c_2 * x_1) / den

        return prof_func


class InsKernel(Kernel):
    """A convolution kernel for broadening due to instrumental effects."""

    def __init__(self, param: float, paramtype: Literal["fwhm", "res"] = "fwhm"):
        """Initializes the InsKernel.

        This broadening is modeled as a Gaussian. The width of the Gaussian can
        be specified by using either the fwhm or the spectral resolution.

        If Full Width at Half Maximum (FWHM) is used, the parameter should be
        given in the same units as the wavelength array. If spectral resolution
        is used, the parameter should be the ratio of the wavelength and the
        FWHM at the given wavelength.

        Parameters
        ----------
            param : float (wavelength units (FWHM) or ratio (resolution))
                The parameter which describes the width of the Gaussian.
            paramtype : "fwhm" or "res", default "fwhm"
                The type of the parameter. Either "fwhm" or "res".
        """
        if paramtype == "fwhm":
            self.fwhm = param
        elif paramtype == "res":
            self.res = param
        else:
            raise ValueError(f"Invalid value for paramtype: {paramtype}")
        self.paramtype: Literal["fwhm", "res"] = paramtype

    def get_fwhm(self, wl_mid: float) -> float:
        if self.paramtype == "fwhm":
            return self.fwhm
        elif self.paramtype == "res":
            return wl_mid / self.res
        raise NotImplementedError(
            f"get_fwhm not implemented for paramtype: {self.paramtype}"
        )

    def get_default_limits(self, wl_mid: float) -> float:
        return self.get_dli(self.get_fwhm(wl_mid)) * 4

    def step(self, wl_mid: float) -> float:
        return self.get_fwhm(wl_mid) / 10.0

    @staticmethod
    def get_dli(fwhm: float) -> float:
        """Returns the characteristic scale of the instrumental kernel.
        Also stdev / sqrt(2)"""
        return fwhm / (2 * np.sqrt(np.log(2)))

    def prof(
        self, wl_mid: float
    ) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
        fwhm = self.get_fwhm(wl_mid)
        dli = self.get_dli(fwhm)
        c1 = 1 / (np.sqrt(np.pi) * dli)

        def prof_func(
            x: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            return c1 * np.exp(-np.square(x / dli))

        return prof_func


class Broadening:
    """A class which can be used to do broadening of the spectrum.

    This class acts as a container for `Kernels` and provides the broaden method
    which does the convolution.
    """

    def __init__(self, kernels: list[Kernel], base_flux: float = 1.0):
        """Initializes the Broadening class.

        Parameters
        ----------
            kernels : list[Kernel]
                The list of kernels which will be used to do broadening.
            base_flux : float, default 1.0
                The flux level corresponding to the base level of the spectrum.
                For normalized spectra this should be 1.0. If not using
                normalized spectra, this is best left at 0.0.
        """
        self.kernels = kernels
        self.base_flux = base_flux

    def broaden(
        self,
        wl: npt.NDArray[np.floating],
        spec: npt.NDArray[np.floating],
        lim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Broadens the spectrum. One can optionally truncate the spectrum
        before broadening.

        Parameters
        ----------
            wl : ndarray[float]
                The wavelength array of the spectrum. It can be in any units. Be
                sure to set `module.char_step` appropriately. Also should be
                consistent with kernel parameters. Can take unsorted values.
            spec : ndarray[float]
                The flux array of the spectrum.
            lim : tuple[float, float], optional
                Optional limits to which the spectrum will be truncated before
                broadening. If not given, the entire spectrum will be used.

        Returns
        -------
            ndarray[float]
                The wavelength array of the broadened spectrum.
            ndarray[float]
                The flux array of the broadened spectrum.
        """

        wl, spec = _sort(wl, spec)  # Sort the arrays
        wl, spec = _get_section(wl, spec, lim)  # Truncate to limits

        spec = spec - self.base_flux  # Normalize to base level

        wl_mid = (wl[0] + wl[-1]) / 2
        step_in = np.diff(wl).min()

        # Currently we use the minimum step size of the kernels. An argument
        # could be made to use the maximum instead.
        step = min((k.step(wl_mid) for k in self.kernels))
        step = max(step, char_step)
        step = min(step, step_in)  # Don't go larger than original step size

        if lim is None:
            wl_lin = _linspace_stepped(wl[0], wl[-1], step)
        else:
            wl_lin = _linspace_stepped(lim[0], lim[1], step)

        # Interpolate the spectrum to the new wavelength array
        spec_lin = _interpolate_spec(wl, spec, wl_lin)

        # Convolve the spectrum with the kernels
        for k in self.kernels:
            _, kernel = k.kernel(wl_mid, step)
            spec_lin = convolve(spec_lin, kernel)

        spec_lin = spec_lin + self.base_flux  # Unnormalize
        return wl_lin, spec_lin


class RotIns(Broadening):
    # DRY violated here and in __init__. Make sure to update this docstring if
    # any parent class is modified.
    """Class to perform rotational and instrumental broadening of spectra.

    It can be initialized with parameters for the rotational and instrumental
    components. Thereafter the `broaden` method can be used to perform the
    broadening.

    Example
    -------


        >>> wl, spec = np.loadtxt("spectrum.xy", unpack=True)
        >>> vsini = 20.0  # km s-1
        >>> res = 50000
        >>> conv_wl, conv_spec = RotIns(vsini, res, "res").broaden(wl, spec)
        conv_wl and conv_spec now describe the broadened spectra.

    Notes
    -----
        The input spectra need not have uniform spacing. This module will first
        cast the input spectra into a uniform grid with an appropriate step
        size. Nor does the input spectra need to be sorted.
    """

    def __init__(
        self,
        vsini: Optional[float] = None,
        fwhm: Optional[float] = None,
        fwhm_type: Literal["fwhm", "res"] = "fwhm",
        limb_coeff: float = DEFAULT_LIMB_COEFF,
        base_flux: float = 1.0,
    ):
        """Initializes the RotIns class.

        After light leaves the photosphere and before being observed by us, the
        stellar spectrum experiences broadening due to the rotation of the star
        and instrumental effects. This class can be used to perform both of
        these on a synthetic spectrum.

        The rotational broadening can be mainly characterized using a single
        parameter: v sin i. Where v is the rotational veocity at the surface of
        the star and i is the angle which the angle of rotation of the star
        makes with the line of sight.

        However the limbs of the stars emit less light than the central
        regions. This effect is taken care of by the limb darkening
        coefficient. This program assumes a default value of 0.6 as used in
        Ivan Hubeny's SYNSPEC program.

        The instrumental broadening is modeled as a Gaussian. The width of the
        Gaussian can be specified by using either the fwhm or the spectral
        resolution.

        If Full Width at Half Maximum (FWHM) is used, the parameter should be
        given in the same units as the wavelength array. If spectral resolution
        is used, the parameter should be the ratio of the wavelength and the
        FWHM at the given wavelength.

        Parameters
        ----------
            vsini : float (km s-1) or None
                The component of rotational velocity of the star along the line
                of sight in km s-1. If None, no rotational broadening will be
                performed.
            fwhm : float (wavelength units (FWHM) or ratio (resolution)) | None
                The parameter which describes the width of the instrumental
                Gaussian. If None, no instrumental broadening will be performed.
            fwhm_type : "fwhm" or "res", default "fwhm"
                The type of the parameter. Either "fwhm" or "res".
            limb_coeff : float, default 0.6
                The limb darkening coefficient.
            base_flux : float, default 1.0
                The flux level corresponding to the base level of the spectrum.
                For normalized spectra this should be 1.0. If not using
                normalized spectra, this is best left at 0.0.
        """
        kernels: list[Kernel] = []
        if vsini is not None and vsini != 0.0:
            kernels.append(RotKernel(vsini, limb_coeff))
        if fwhm is not None and fwhm != 0.0:
            kernels.append(InsKernel(fwhm, fwhm_type))
        super().__init__(kernels, base_flux)


def rotins(
    vsini: Optional[float] = None,
    fwhm: Optional[float] = None,
    fwhm_type: Literal["fwhm", "res"] = "fwhm",
    limb_coeff: float = DEFAULT_LIMB_COEFF,
    base_flux: float = 1.0,
) -> Callable[..., Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
    """Creates a function that applies rotational and instrumental broadening to spectra.

    This is a functional programming interface to the RotIns class. It returns a closure
    that can be used to broaden multiple spectra with the same parameters.

    Parameters
    ----------
    vsini : Optional[float], optional
        The component of rotational velocity of the star along the line
        of sight in km s-1. If None, no rotational broadening will be
        performed.
    fwhm : Optional[float], optional
        The parameter which describes the width of the instrumental
        Gaussian. If None, no instrumental broadening will be performed.
    fwhm_type : "fwhm" or "res", default "fwhm"
        The type of the parameter. Either "fwhm" or "res".
    limb_coeff : float, default 0.6
        The limb darkening coefficient.
    base_flux : float, default 1.0
        The flux level corresponding to the base level of the spectrum.
        For normalized spectra this should be 1.0. If not using
        normalized spectra, this is best left at 0.0.

    Returns
    -------
    Callable
        A function that takes wavelength and flux arrays and optional limits,
        and returns the broadened spectrum.

    Example
    -------
    >>> # Create a broadening function with specific parameters
    >>> broaden = rotins(vsini=100.0, fwhm=0.1)
    >>> # Apply it to multiple spectra
    >>> conv_wl1, conv_spec1 = broaden(wl1, spec1)
    >>> # Can optionally specify limits
    >>> conv_wl2, conv_spec2 = broaden(wl2, spec2, lim=(4000, 5000))
    >>> # Different base flux for non-normalized spectra
    >>> broaden_raw = rotins(vsini=100.0, fwhm=0.1, base_flux=0.0)
    >>> conv_wl3, conv_spec3 = broaden_raw(wl3, spec3)
    """
    broadener = RotIns(vsini, fwhm, fwhm_type, limb_coeff, base_flux)

    def broadening_function(
        wl: npt.NDArray[np.floating],
        spec: npt.NDArray[np.floating],
        lim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return broadener.broaden(wl, spec, lim)

    return broadening_function
