"""Functions to automate building a cluster expansion model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("CEAuto")
except PackageNotFoundError:
    # package is not installed
    pass
