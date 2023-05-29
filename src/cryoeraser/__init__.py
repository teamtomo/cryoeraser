"""Erase high-contrast features in cryo electron microscopy data"""
from importlib.metadata import PackageNotFoundError, version
from .erase import erase_2d

try:
    __version__ = version("cryoeraser")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"
