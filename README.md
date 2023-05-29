# cryoeraser

[![License](https://img.shields.io/pypi/l/cryoeraser.svg?color=green)](https://github.com/teamtomo/cryoeraser/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cryoeraser.svg?color=green)](https://pypi.org/project/cryoeraser)
[![Python Version](https://img.shields.io/pypi/pyversions/cryoeraser.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/cryoeraser/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/cryoeraser/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/cryoeraser/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/cryoeraser)

*cryoeraser* is a Python package for erasing features in cryo electron microscopy data.

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/7307488/241686174-f660589b-5a72-4ede-9302-9aa55a7c840f.png">
</p>

Removing high-contrast image features such as ice contamination or gold fiducial markers
can be useful for downstream processing.

Masked image regions are replaced with noise drawn from a normal distribution matching
the local mean and a global standard deviation of an image.

## Installation

*cryoeraser* is available on the Python package index.

```shell
pip install cryoeraser
```

We recommend working in a virtual environment.

## Usage

```shell
import tifffile
from cryoeraser import erase_2d

image = tifffile.imread('examples/data/TS_01_0.0.tiff')
mask = tifffile.imread('examples/data/TS_01_0.0_mask.tiff')

erased = erase_2d(image=image, mask=mask)
```

An [example notebook](docs/erase_2d_example.ipynb) is also provided.

