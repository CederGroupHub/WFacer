"""Functions to automate building a cluster expansion model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("WFacer")
except PackageNotFoundError:
    # package is not installed
    pass
