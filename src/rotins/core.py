from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


DEFAULT_LIMB_COEFF = 0.6
SPEED_LIGHT_KMS = 2.99792e5
CHARD = 0.01
EPSILON = 1e-6


def get_basis(step: float, limit: float) -> npt.NDArray[np.floating]:
    numsteps = int(np.ceil(limit / step))
    basis = np.array([step * i for i in range(-numsteps, numsteps + 1)])
    return basis


def linspace_stepped(
    start: float, stop: float, step: float
) -> npt.NDArray[np.floating]:
    numsteps = int(np.floor((stop - start) / step)) + 1
    return np.array([start + step * i for i in range(numsteps)])


def interpolate_spec(
    wl: npt.NDArray[np.floating],
    spec: npt.NDArray[np.floating],
    basis: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    inter_f = interp1d(wl, spec, "cubic", assume_sorted=True)
    return inter_f(basis)


def normalize_kernel(
    basis: npt.NDArray[np.floating], kernel: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    step = (basis[-1] - basis[0]) / (len(basis) - 1)
    return kernel * step


def convolve(
    spec: npt.NDArray[np.floating],
    kernel: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.convolve(spec, kernel, "same")


def get_section(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    lim: tuple[float, float] = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    # Sort if not sorted
    if not np.all(x[:-1] <= x[1:]):
        xinds = x.argsort()
        x, y = x[xinds], y[xinds]

    if lim is None:
        return x.copy(), y.copy()

    if lim[0] < x[0] or lim[1] > x[-1]:
        raise ValueError("lim must be within x")

    limbool = np.logical_and(x >= lim[0], x <= lim[1])
    # Include an extra element if possible
    if limbool[0] is False and (mini := np.searchsorted(x, lim[0])) > 0:
        limbool[mini - 1] = True
    if limbool[-1] is False and (maxi := np.searchsorted(x, lim[-1])) < len(x):
        limbool[maxi] = True

    return x[limbool], y[limbool]


class Kernel(ABC):
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
        self, wl_mid: float, step: float = None, limit: float = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        if step is None:
            step = self.step(wl_mid)
        if limit is None:
            limit = self.get_default_limits(wl_mid)
        basis = get_basis(step, limit)
        kernel = self.prof(wl_mid)(basis)
        return basis, kernel


class RotKernel(Kernel):
    def __init__(
        self,
        vsini: float,
        limb_coeff: float = DEFAULT_LIMB_COEFF,
    ):
        self.vsini = vsini
        self.limb_coeff = limb_coeff

    def _get_dl0(self, wl_mid: float) -> float:
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
    def __init__(self, param: float, paramtype: Literal["fwhm", "res"] = "fwhm"):
        if paramtype == "fwhm":
            self.fwhm = param
        elif paramtype == "res":
            self.res = param
        else:
            raise ValueError(f"Invalid value for paramtype: {paramtype}")
        self.paramtype = paramtype

    def get_fwhm(self, wl_mid: float) -> float:
        if self.paramtype == "fwhm":
            return self.fwhm
        elif self.paramtype == "res":
            return wl_mid / self.res

    def get_default_limits(self, wl_mid: float) -> float:
        return self.get_fwhm(wl_mid) / (2 * np.log(2)) * 4

    def step(self, wl_mid: float) -> float:
        return self.get_fwhm(wl_mid) / 10.0

    def prof(
        self, wl_mid: float
    ) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
        fwhm = self.get_fwhm(wl_mid)
        dli = fwhm / (2 * np.log(2))
        c1 = 1 / (np.sqrt(np.pi) * dli)

        def prof_func(
            x: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            return c1 * np.exp(-np.square(x / dli))

        return prof_func


class Broadening:
    def __init__(self, kernels: list[Kernel]):
        self.kernels = kernels

    def broaden(
        self,
        wl: npt.NDArray[np.floating],
        spec: npt.NDArray[np.floating],
        lim: tuple[float, float] = None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        wl, spec = get_section(wl, spec, lim)

        spec = 1 - spec

        wl_mid = (wl[0] + wl[-1]) / 2
        step_in = (wl[-1] - wl[0]) / (len(wl) - 1)

        step = min((k.step(wl_mid) for k in self.kernels))
        step = max(step, CHARD)
        step = min(step, step_in)

        if lim is None:
            wl_lin = linspace_stepped(wl[0], wl[-1], step)
        else:
            wl_lin = linspace_stepped(lim[0], lim[1], step)
        spec_lin = interpolate_spec(wl, spec, wl_lin)
        for k in self.kernels:
            basis, kernel = k.kernel(wl_mid, step)
            nkernel = normalize_kernel(basis, kernel)
            spec_lin = convolve(spec_lin, nkernel)

        spec_lin = 1 - spec_lin
        return wl_lin, spec_lin


class RotIns(Broadening):
    def __init__(
        self,
        vsini: float,
        fwhm: float,
        fwhm_type: Literal["fwhm", "res"] = "fwhm",
        limb_coeff: float = DEFAULT_LIMB_COEFF,
    ):
        kernels: list[Kernel] = []
        if vsini > EPSILON:
            kernels.append(RotKernel(vsini, limb_coeff))
        if fwhm > EPSILON:
            kernels.append(InsKernel(fwhm, fwhm_type))
        super().__init__(kernels)
