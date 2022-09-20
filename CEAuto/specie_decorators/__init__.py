from collections import defaultdict
from smol.utils import get_subclasses
from .base import BaseDecorator, decorator_factory
from .charge import MagneticChargeDecorator, PmgGuessChargeDecorator

allowed_decorators = defaultdict(lambda: [])
for subclass in get_subclasses(BaseDecorator).values():
    prop = subclass.decorated_prop_name
    allowed_decorators[prop].append(subclass.__name__)

__all__ = ['MagneticChargeDecorator',
           'PmgGuessChargeDecorator',
           'decorator_factory',
           "allowed_decorators"]
