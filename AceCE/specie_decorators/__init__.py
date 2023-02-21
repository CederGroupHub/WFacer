"""Decorators to turn structure of Element into structure of decorated Species."""

from collections import defaultdict

from smol.utils import get_subclasses

from .base import BaseDecorator, decorator_factory
from .charge import (
    FixedChargeDecorator,
    MagneticChargeDecorator,
    PmgGuessChargeDecorator,
)

allowed_decorators = defaultdict(lambda: [])
for subclass in get_subclasses(BaseDecorator).values():
    prop = subclass.decorated_prop_name
    if subclass.required_prop_names is not None:
        allowed_decorators[prop].append(subclass.__name__)

__all__ = [
    "MagneticChargeDecorator",
    "PmgGuessChargeDecorator",
    "FixedChargeDecorator",
    "decorator_factory",
    "allowed_decorators",
]
