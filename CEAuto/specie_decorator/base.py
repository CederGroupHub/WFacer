"""Decorate properties to a structure composed of Element.

Currently, we can only decorate charge. Plan to allow decorating
spin in the future updates.
"""

__author__ = 'Fengyu Xie'

from abc import ABCMeta, abstractmethod
import warnings
from monty.json import MSONable
from collections import defaultdict
from sklearn.mixture import GaussianMixture

from smol.cofe.space.domain import get_species
from smol.utils import derived_class_factory, class_name_from_str


class BaseDecorator(MSONable, metaclass=ABCMeta):
    """Abstract decorator class.

    1, Each decorator should only be used to decorate one property.
    2, Can not decorate entries with partially disordered structures.
    """
    # Edit this as you implement new child classes.
    decorated_prop_name = None
    required_props = []

    def __init__(self, labels=None):
        """Initialize.

        Args:
           labels(dict{str:list}): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are decoration values,
               such as oxidation states, magnetic spin directions.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               Not necessary for some sub-classes, such as:
               GuessChargeDecorator.
        """
        labels = labels or {}
        self.labels = {get_species(key): val
                       for key, val in labels.items()}

    @staticmethod
    def group_site_by_species(entries):
        """Group required properties on sites by species.

        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
        Return:
            (Entry index, site index) occupied by each species:
            defaultdict
        """
        groups_by_species = defaultdict(lambda: [])

        # These entries should not contain Vacancy.
        for e_id, entry in enumerate(entries):
            for s_id, site in enumerate(entry.structure):
                sp = site.species
                groups_by_species[sp] += [(e_id, s_id)]

        return groups_by_species

    @property
    @abstractmethod
    def is_trained(self):
        """Gives whether this decorator is trained before.

        If trained, will be blocked from training again.
        Returns:
            bool.
        """
        return

    @abstractmethod
    def train(self, entries, reset=False):
        """Train the decoration model.

        Model or model parameters should be stored in a property of the
        object.

        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
            reset(Boolean): optional
                If you want to re-train the decorator model, set this value
                to true. Otherwise, we will skip training if the model is
                trained before. Default to false.
        """
        return

    @abstractmethod
    def decorate(self, entries):
        """Give decoration to entries based on trained model.

        If an assigned entry is not valid,
        for example, in charge assignment, if an assigned structure is not
        charge neutral, then this entry will be returned as None.
        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
        Returns:
            List[NoneType|ComputedStructureEntry]
        """
        return

    # TODO: change these to non-abstract methods.
    @abstractmethod
    def as_dict(self):
        """Serialization method."""
        return

    @classmethod
    @abstractmethod
    def from_dict(cls, d):
        """Deserialization."""
        return


class MixtureGaussianDecorator(BaseDecorator):
    """Mixture of Gaussians (MoGs) decorator class.

    Uses mixture of gaussian to label each species.
    """
    decorated_prop_name = ""
    required_props = []
    gaussian_model_keys = ('weights_', 'means_', 'covariances_',
                           'precisions_', 'precisions_cholesky_',
                           'converged_', 'n_iter_', 'lower_bound_')

    def __init__(self, labels, gaussian_models=None):
        """Initialize.

        Args:
           labels(dict{str:list}): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are decoration values,
               such as oxidation states, magnetic spin directions.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               Not necessary for some sub-classes, such as:
               GuessChargeDecorator.
           gaussian_models(dict{str:GaussianMixture}):
               Gaussian models corresponding to each key in labels.
        """
        super(MixtureGaussianDecorator, self).__init__(labels)
        if gaussian_models is None:
            gaussian_models = {}
        for key in self.labels:
            if key not in gaussian_models:
                warnings.warn(f"A gaussian model for {key} is missing! "
                              "Initializing from empty.")
                gaussian_models[key] = GaussianMixture(n_components=
                                                       len(self.labels[key]))
        self._gms = gaussian_models

    @staticmethod
    def serialize_gaussian_model(model):
        """Serialize gaussian model into dict.

        Note:
            Do not serialize an un-trained model!
        """
        data = {'init_params': model.get_params(),
                'model_params': {}}
        for p in MixtureGaussianDecorator.gaussian_model_keys:
            data['model_params'][p] = getattr(model, p)
            # Not always json.dump-able.
        return data

    @staticmethod
    def deserialize_gaussian_model(data):
        """Recover gaussian model from dict."""
        model = GaussianMixture(**data["init_params"])
        for p, v in data["model_params"].items():
            setattr(model, p, v)

    @staticmethod
    def is_trained_gaussian_model(model):
        """Whether a gaussian model is trained."""
        return all(k in model.__dict__
                   for k in MixtureGaussianDecorator.gaussian_model_keys)

    @property
    def is_trained(self):
        return all(MixtureGaussianDecorator.is_trained_gaussian_model(m)
                   for m in self._gms.values())

    def train(self, entries, reset=False):
        """Train the decoration model.

        Model or model parameters should be stored in a property of the
        object.

        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
            reset(Boolean): optional
                If you want to re-train the decorator model, set this value
                to true. Otherwise, we will skip training if the model is
                trained before. Default to false.
        """
        if self.is_trained and not reset:
            return
        else:
            for entry in entries:



class GpOptimizedDecorator(BaseDecorator):
    """Gaussian process decorator class.

    Uses Gaussian optimization process described by J. Yang
    et.al. Can only handle decoration from a single scalar
    property.
    """
    # Edit this as you implement new child classes.
    decorated_prop_name = None
    required_props = []

    def __init__(self, labels, ):
        """Initialize.

        Args:
           labels(dict{str:list}): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are decoration values,
               such as oxidation states, magnetic spin directions.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               Not necessary for some sub-classes, such as:
               GuessChargeDecorator.
            cutting_bounds
        """
        super(GaussianOptimizedDecorator, self).__init__(labels)

    @staticmethod
    def group_site_by_species(entries):
        """Group required properties on sites by species.

        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
        Return:
            (Entry index, site index) occupied by each species:
            defaultdict
        """
        groups_by_species = defaultdict(lambda: [])

        # These entries should not contain Vacancy.
        for e_id, entry in enumerate(entries):
            for s_id, site in enumerate(entry.structure):
                sp = site.species
                groups_by_species[sp] += [(e_id, s_id)]

        return groups_by_species

    @property
    @abstractmethod
    def is_trained(self):
        """Gives whether this decorator is trained before.

        If trained, will be blocked from training again.
        Returns:
            bool.
        """
        return

    @abstractmethod
    def train(self, entries, reset=False):
        """Train the decoration model.

        Model or model parameters should be stored in a property of the
        object.

        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
            reset(Boolean): optional
                If you want to re-train the decorator model, set this value
                to true. Otherwise, we will skip training if the model is
                trained before. Default to false.
        """
        return

    @abstractmethod
    def decorate(self, entries):
        """Give decoration to entries based on trained model.

        If an assigned entry is not valid,
        for example, in charge assignment, if an assigned structure is not
        charge neutral, then this entry will be returned as None.
        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
        Returns:
            List[NoneType|ComputedStructureEntry]
        """
        return

    @abstractmethod
    def as_dict(self):
        """Serialization method."""
        return

    @classmethod
    @abstractmethod
    def from_dict(cls, d):
        """Deserialization."""
        return

def decorator_factory(decorator_type, *args, **kwargs):
    """Create a species decorator with given name.

    Args:
        decorator_type(str):
            Name of a BaseDecorator subclass.
        *args, **kwargs:
            Arguments used to intialize the class.
    """
    name = class_name_from_str(decorator_type)
    return derived_class_factory(name, BaseDecorator, *args, **kwargs)
