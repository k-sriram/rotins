# rotins

A Python package for rotational and instrumental broadening of stellar spectra. This is a Python translation of the FORTRAN program used in TLUSTY by Ivan Hubeny et al.

## Installation

```bash
pip install rotins
```

## Features

- Rotational broadening with configurable limb darkening coefficient
- Instrumental broadening using Gaussian profiles
- Support for both FWHM and spectral resolution parameters
- Both object-oriented and functional programming interfaces
- Support for normalized and non-normalized spectra
- Efficient convolution with proper edge handling

## Quick Start

```python
import numpy as np
from rotins import RotIns, rotins

# Load your spectrum (wavelength and flux arrays)
wl = np.loadtxt("spectrum_wl.txt")
flux = np.loadtxt("spectrum_flux.txt")

# Using the class-based interface
broadener = RotIns(vsini=50.0, fwhm=0.1)  # vsini in km/s, fwhm in wavelength units
conv_wl, conv_flux = broadener.broaden(wl, flux)

# Using the functional interface
broaden = rotins(vsini=50.0, fwhm=0.1)
conv_wl, conv_flux = broaden(wl, flux)
```

## Documentation

### Class-based Interface

```python
from rotins import RotIns

broadener = RotIns(
    vsini=50.0,           # Rotational velocity in km/s
    fwhm=0.1,            # FWHM in wavelength units
    fwhm_type="fwhm",    # Either "fwhm" or "res"
    limb_coeff=0.6,      # Limb darkening coefficient
    base_flux=1.0,       # Base flux level
)

# Broaden the spectrum
conv_wl, conv_flux = broadener.broaden(
    wl,                  # Wavelength array
    flux,               # Flux array
    lim=(4000, 5000),   # Optional wavelength limits
)
```

### Functional Interface

```python
from rotins import rotins

# Create a broadening function
broaden = rotins(
    vsini=50.0,           # Rotational velocity in km/s
    fwhm=0.1,            # FWHM in wavelength units
    fwhm_type="fwhm",    # Either "fwhm" or "res"
    limb_coeff=0.6,      # Limb darkening coefficient
    base_flux=1.0,       # Base flux level
)

# Apply broadening to multiple spectra
conv_wl1, conv_flux1 = broaden(wl1, flux1)
conv_wl2, conv_flux2 = broaden(wl2, flux2, lim=(4000, 5000))
```

### Parameters

- **vsini**: Rotational velocity in km/s. If None or 0.0, no rotational broadening is applied.
- **fwhm**: Full Width at Half Maximum in wavelength units (if fwhm_type="fwhm") or resolving power (if fwhm_type="res"). If None or 0.0, no instrumental broadening is applied.
- **fwhm_type**: Type of the FWHM parameter. Either "fwhm" for direct wavelength units or "res" for spectral resolution.
- **limb_coeff**: Limb darkening coefficient. Default is 0.6 (from SYNSPEC).
- **base_flux**: Base flux level. Use 1.0 for normalized spectra, 0.0 for non-normalized spectra.
- **lim**: Optional tuple of (min_wavelength, max_wavelength) to limit the broadening range.

## Scientific Details

The calculations follow the formulas from:

Gray, D.F., 2008, The Observation and Analysis of Stellar Photospheres, Cambridge University Press, Cambridge, 3rd edition.

### Rotational Broadening

The rotational broadening kernel is given by:

```math
G(\Delta\lambda) = \frac{2(1-\epsilon)[1-(\Delta\lambda/\Delta\lambda_0)^2]^{1/2}+(\pi\epsilon/2)[1-(\Delta\lambda/\Delta\lambda_0)^2]}{\pi\Delta\lambda_0(1-\epsilon/3)}
```

where:
- `\epsilon` is the limb darkening coefficient
- `\Delta\lambda_0 = \frac{\lambda v \sin i}{c}`
- `c` is the speed of light

### Instrumental Broadening

Instrumental broadening uses a Gaussian kernel with:

```math
\sigma = \frac{\textrm{FWHM}}{2\sqrt{2\ln 2}}
```

## Contributing

Pull requests are welcome. Please ensure that tests pass and add new tests for new features.

## License

MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{rotins,
  author = {Krishna, Sriram},
  title = {rotins: A Python package for stellar spectral broadening},
  year = {2022},
  url = {https://github.com/k-sriram/rotins}
}