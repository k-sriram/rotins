from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import scipy.stats
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import rotins.core as core

EPSILON = 1e-6
WL_MID = 4000.0
SAMPLE_SIZE = 601
FloatArray = core.FloatArray


# utils


def measure_fwhm(x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]) -> float:
    amax = np.argmax(y)

    xmax = x[amax]
    hmax = y[amax] / 2

    rfy = interp1d(x, y - hmax, "cubic")

    lhm = root_scalar(rfy, method="bisect", bracket=[x[0], xmax])
    rhm = root_scalar(rfy, method="bisect", bracket=[xmax, x[-1]])

    if not (lhm.converged and rhm.converged):
        raise ValueError("Couldn't find roots for half maximum")

    return rhm.root - lhm.root


def gaussian(
    mean: float = 0.0, std: float = 1.0, size: int = SAMPLE_SIZE
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    x = np.linspace(-10.0, 10.0, size)
    y = scipy.stats.norm(mean, std).pdf(x)
    return x, y


def sample_spectrum(wl=WL_MID, fwhm=1.0, scale=1.0):
    std = fwhm / 2 / np.sqrt(2 * np.log(2))
    x, y = gaussian(0.0, std)
    x = x + wl
    y = 1 - (y * scale)
    return x, y


# Fixtures


@pytest.fixture
def sample_ins_kernel():
    return core.InsKernel(50000, "res")


@pytest.fixture
def sample_rot_kernel():
    return core.RotKernel(30.0)


# Unit tests


def test_measure_fwhm():
    x, y = gaussian()
    fwhm = measure_fwhm(x, y)
    expected = 2 * np.sqrt(2 * np.log(2))
    assert abs((fwhm / expected) - 1) < 1e-6


@pytest.mark.parametrize(
    "step, limit",
    [
        (0.15, 2.0),
        (0.2, 1.0),
    ],
)
def test_get_basis(step, limit):
    basis = core.get_basis(step, limit)
    assert np.all(np.diff(basis) - step < EPSILON)
    assert -limit - step < basis[0] <= -limit
    assert limit <= basis[-1] < limit + step


@pytest.mark.parametrize(
    "start, stop, step",
    [
        (4083.123, 4109.383, 0.12),
        (5000.0, 5010.0, 0.1),
    ],
)
def test_linspace_stepped(start, stop, step):
    linspace = core.linspace_stepped(start, stop, step)
    assert linspace[0] == start
    assert linspace[-1] <= stop
    assert linspace[-1] > stop - step
    assert np.all(np.diff(linspace) - step < EPSILON)


@pytest.mark.parametrize(
    "kernel, args",
    [
        (core.RotKernel, [30.0]),
        (core.InsKernel, [50000.0, "res"]),
    ],
)
def test_kernel(kernel, args):
    kx, ky = kernel(*args).kernel(4000.0)
    assert ky[0] < EPSILON
    assert ky[-1] < EPSILON
    assert trapz(ky, kx) - 1 < EPSILON


# Integration tests


@pytest.mark.parametrize(
    "wl, fwhm, scale, kernel_type, kernel_args",
    [
        (4000.0, 1.0, 0.4, core.InsKernel, [50000.0, "res"]),
        (5000.0, 0.2, 0.05, core.InsKernel, [0.1, "fwhm"]),
        (4623.45, 0.5, 0.2, core.RotKernel, [50.0]),
    ],
)
def test_broad(wl, fwhm, scale, kernel_type, kernel_args, makeplots):
    x, y = sample_spectrum(wl, fwhm, scale)
    ifwhm = measure_fwhm(x, 1 - y)

    kernel: core.Kernel = kernel_type(*kernel_args)
    kx, ky = kernel.kernel(wl)
    kfwhm = measure_fwhm(kx, ky)

    efwhm = np.sqrt(np.square([ifwhm, kfwhm]).sum())

    Ins = core.Broadening([kernel])
    rx, ry = Ins.broaden(x, y)

    rfwhm = measure_fwhm(rx, 1 - ry)
    print(f"{ifwhm=:.3f}, {kfwhm=:.3f}, {efwhm=:.3f}, {rfwhm=:.3f}")

    if makeplots:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.plot(x, y, "b", label="Input", lw=0.5)
        plt.plot(
            kx + wl, 1 - ky, "g", label="Kernel", lw=0.5
        )
        plt.plot(rx, ry, "r", label="Convolved", lw=0.5)
        plt.legend()
        plt.xlim([wl - rfwhm * 2, wl + rfwhm * 2])
        plt.title(f"{ifwhm=:.3f}, {kfwhm=:.3f}, {efwhm=:.3f}, {rfwhm=:.3f}")
        kernel_type_str = "ins" if kernel_type == core.InsKernel else "rot"
        plt.savefig(f"tests/figure_{kernel_type_str}_{fwhm}.png")

    if kernel_type == core.InsKernel:
        assert abs(efwhm - rfwhm) < 1e-3

    ieqw = trapz(1 - y, x)
    reqw = trapz(1 - ry, rx)

    print(f"{ieqw*1000=:.2f}, {reqw*1000=:.2f}")
    assert abs(ieqw - reqw) < 1e-4


@pytest.mark.parametrize("testcase", ["1", "2"])
@pytest.mark.parametrize(
    "outfile, fwhm, vsini",
    [
        ("fwhm01", 0.1, 0.0),
        ("fwhm05", 0.5, 0.0),
        ("fwhm2", 2.0, 0.0),
        ("vrot10", 0.0, 10.0),
        ("vrot50", 0.0, 50.0),
        ("vrot200", 0.0, 200.0),
        ("vrot25fwhm02", 0.2, 25.0),
    ],
)
def test_rotins(outfile, fwhm, vsini, makeplots, testcase):
    TESTDATA = Path(__file__).parent / "data"
    SPEC = f"t{testcase}_norm"
    outfile = f"t{testcase}_{outfile}"
    wl, flux = np.loadtxt(TESTDATA / SPEC, unpack=True)

    cwl, cflux = core.RotIns(vsini, fwhm).broaden(wl, flux)
    ewl, eflux = np.loadtxt(TESTDATA / outfile, unpack=True)

    lim = (max(ewl[0], cwl[0]), min(ewl[-1], cwl[-1]))
    eflux = 1 - eflux
    cflux = 1 - cflux
    n = max(len(ewl), len(cwl))
    x = np.linspace(lim[0], lim[1], n)

    cflux = interp1d(cwl, cflux, kind="cubic")(x)
    eflux = interp1d(ewl, eflux, kind="cubic")(x)

    stdev = np.sqrt(np.square(eflux - cflux).sum())
    stdev /= n

    if makeplots:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.plot(wl, flux, "k", label="Input", lw=0.5)
        plt.plot(x, 1 - eflux, "g", label="Expected", lw=0.5)
        plt.plot(x, 1 - cflux, "r", label="Convolved", lw=0.5)
        plt.legend()
        plt.xlim(lim)
        plt.title(f"{fwhm=:.2f}, {vsini=:.1f}, {stdev=:.3f}")
        plt.savefig(f"tests/rotins_{outfile}.png")

    if vsini == 0.0:
        ifwhm = measure_fwhm(wl, 1 - flux)
        efwhm = measure_fwhm(x, eflux)
        cfwhm = measure_fwhm(x, cflux)
        tfwhm = np.sqrt(np.square([ifwhm, fwhm]).sum())
        print(f"\n{ifwhm=:.3f}, {cfwhm=:.3f} {fwhm=:.3f}, {efwhm=:.3f}, {tfwhm=:.3f}")
        # input, convolved, kernel, expected, theoretical
        assert abs(cfwhm - efwhm) / efwhm < 1e-3

    assert stdev < 1e-4
