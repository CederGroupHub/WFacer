"""Decorate properties to a structure composed of Element.

Currently, we can only decorate charge. Plan to allow decorating
spin in the future updates.
"""

__author__ = 'Fengyu Xie, Julia H. Yang'

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from monty.json import MSONable
from collections import defaultdict
from copy import deepcopy
import functools

from pymatgen.core import Composition, Element, Species, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from sklearn.mixture import GaussianMixture
from skopt import gp_minimize

from smol.cofe.space.domain import get_species
from smol.utils import derived_class_factory, class_name_from_str


class BaseDecorator(MSONable, metaclass=ABCMeta):
    """Abstract decorator class.

    1, Each decorator should only be used to decorate one property.
    2, Currently, only supports assigning labels from one scalar site property,
    which should be sufficient for most purposes.
    3, Can not decorate entries with partially disordered structures.
    """
    # Edit this as you implement new child classes.
    decorated_prop_name = None
    required_prop_name = None

    def __init__(self, labels=None):
        """Initialize.

        Args:
           labels(dict{str:list}): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are possible decorated property
               values, such as oxidation states, magnetic spin directions.
               Values are sorted such that the cluster centers in the
               required property is increasing. For example, in Mn(2, 3, 4)+
               all high spin, the magnetic moments is sorted as [Mn4+, Mn3+, Mn2+],
               thus you should provide labels as {Element("Mn"):[4, 3, 2]}.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               This argument may not be necessary for some sub-classes, such as:
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are the cause of your own error!
        """
        labels = labels or {}
        for key in labels:
            if len(labels[key]) < 1:
                raise ValueError(f"Trying to decorate {key} but no decoration "
                                 "label given!")
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

    def _process(self, entries, decorate_rules):
        """Decorate entries with rules."""
        entries_decor = []
        for struct_id, entry in enumerate(entries):
            s_undecor = entry.structure
            species_decor = []
            for site_id, site in enumerate(s_undecor):
                sp = site.species
                if isinstance(sp, Composition):
                    raise ValueError("Can not decorate partially disordered site: "
                                     f"{site}")
                if (struct_id in decorate_rules
                        and site_id in decorate_rules[struct_id]):
                    label = decorate_rules[struct_id][site_id]
                    if self.decorated_prop_name == "oxi_state":
                        if isinstance(sp, Element):
                            sp_decor = Species(sp.symbol, oxidation_state=label)
                        else:
                            sp_decor = deepcopy(sp)
                            sp_decor._oxi_state = label
                    else:
                        if self.decorated_prop_name not in sp.supported_properties:
                            raise ValueError("Pymatgen Species does not support "
                                             f"property {self.decorated_prop_name}!")
                        if isinstance(sp, Element):
                            sp_decor = Species(sp.symbol,
                                               properties={self.decorated_prop_name:
                                                           label})
                        else:
                            sp_decor = deepcopy(sp)
                            sp_decor._properties[self.decorated_prop_name] = label

                    species_decor.append(sp_decor)

                else:  # Undecorated sites might continue to be Element.
                    species_decor.append(sp)

            s_decor = Structure(s_undecor.lattice,
                                species_decor,
                                s_undecor.frac_coords)
            energy_adjustments = (entry.energy_adjustments
                                  if len(entry.energy_adjustments) != 0
                                  else None)
            # Constant energy adjustment is set as a manual class object.
            entry_decor = ComputedStructureEntry(s_decor,
                                                 energy=entry.uncorrected_energy,
                                                 energy_adjustments=energy_adjustments,
                                                 parameters=entry.parameters,
                                                 data=entry.data,
                                                 entry_id=entry.entry_id)
            entries_decor.append(entry_decor)
        return entries_decor

    @abstractmethod
    def _filter(self, entries):
        """Filter out entries by some criteria.

        Must be implemented for every decorator class.
        For entries that does not satisfy criteria, will
        be replaced with None.
        """
        return entries

    # Should save and load dicts with Monty.
    def as_dict(self):
        """Serialization method."""
        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "labels": self.labels}

    @classmethod
    @abstractmethod
    def from_dict(cls, d):
        """Deserialization."""
        return


class MixtureGaussianDecorator(BaseDecorator, metaclass=ABCMeta):
    """Mixture of Gaussians (MoGs) decorator class.

    Uses mixture of gaussian to label each species.
    """
    decorated_prop_name = ""
    required_prop_name = ""
    gaussian_model_keys = ('weights_', 'means_', 'covariances_',
                           'precisions_', 'precisions_cholesky_',
                           'converged_', 'n_iter_', 'lower_bound_')

    def __init__(self, labels, gaussian_models=None):
        """Initialize.

        Args:
           labels(dict{str:list}): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are possible decorated property
               values, such as oxidation states, magnetic spin directions.
               Values are sorted such that the cluster centers in the
               required property is increasing. For example, in Mn(2, 3, 4)+
               all high spin, the magnetic moments is sorted as [Mn4+, Mn3+, Mn2+],
               thus you should provide labels as {Element("Mn"):[4, 3, 2]}.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               This argument may not be necessary for some sub-classes, such as:
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are the cause of your own error!
           gaussian_models(dict{str|Element|Species:GaussianMixture}):
               Gaussian models corresponding to each key in labels.
        """
        super(MixtureGaussianDecorator, self).__init__(labels)
        if gaussian_models is None:
            gaussian_models = {}
        for key in self.labels:
            if key not in gaussian_models:
                warnings.warn(f"Gaussian model for {key} is missing! "
                              "Initializing from empty.")
                gaussian_models[key] = GaussianMixture(n_components=len(
                    self.labels[key]))
        self._gms = gaussian_models

    @staticmethod
    def serialize_gaussian_model(model):
        """Serialize gaussian model into dict."""
        data = {'init_params': model.get_params(),
                'model_params': {}}
        for p in MixtureGaussianDecorator.gaussian_model_keys:
            if p in model.__dict__:
                data['model_params'][p] = getattr(model, p)
            # Contains np.array, not directly json.dump-able.
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
            groups = self.group_site_by_species(entries)
            # Train model for each species in the labels dict.
            for species in self.labels:
                structure_sites = groups[species]
                # Need some pre-processing to make sure entries data
                # include the required properties, and the properties
                # should be in the form of 1D arrayLike per entry.
                props = [entries[struct_id]
                         .data[self.required_prop_name][site_id]
                         for struct_id, site_id in structure_sites]
                props = np.array(props)
                if len(props.shape) == 1:
                    props = props.reshape(-1, 1)
                elif len(props.shape) > 2:
                    raise ValueError("Can not train on tensor properties! "
                                     "Convert to scalar or vector before "
                                     "training!")
                _ = self._gms[species].fit(props)

    def decorate(self, entries):
        """Give decoration to entries based on trained model.

        If an assigned entry is not valid,
        for example, in charge assignment, if an assigned structure is not
        charge neutral, then this entry will be returned as None.
        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.
        Returns:
            Entries with structures decorated. Returns None
            List[NoneType|ComputedStructureEntry]
        """
        if not self.is_trained:
            raise ValueError("Can not make predictions from un-trained"
                             " models!")
        groups = self.group_site_by_species(entries)
        decoration_rule = {}
        for species in groups:
            structure_sites = groups[species]
            model = self._gms[species]
            centers = getattr(model, "means_")
            centers_argsort = np.argsort(centers)
            props = [entries[struct_id]
                     .data[self.required_prop_name][site_id]
                     for struct_id, site_id in structure_sites]
            props = np.array(props)
            if len(props.shape) == 1:
                props = props.reshape(-1, 1)
            elif len(props.shape) > 2:
                raise ValueError("Can not train on tensor properties! "
                                 "Convert to scalar or vector before "
                                 "training!")
            cluster_ids = model.predict(props)
            label_ids = centers_argsort[cluster_ids]
            labels = np.array(self.labels[species])
            assigned_labels = labels[label_ids]
            for i, (struct_id, site_id) in enumerate(structure_sites):
                if struct_id not in decoration_rule:
                    decoration_rule[struct_id] = {}
                decoration_rule[struct_id][site_id] = \
                    assigned_labels[i]
        entries_processed = self._process(entries, decoration_rule)
        return self._filter(entries_processed)

    def as_dict(self):
        """Serialize to dict."""
        d = super(MixtureGaussianDecorator, self).as_dict()
        d["models"] = {species: self.serialize_gaussian_model(model)
                       for species, model in self._gms.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        """Load from dict."""
        # Please load dict with monty.
        models = {species: cls.deserialize_gaussian_model(data)
                  for species, data in d["models"].items()}
        return cls(d["labels"], models)


class GpOptimizedDecorator(BaseDecorator, metaclass=ABCMeta):
    """Gaussian process decorator class.

    Uses Gaussian optimization process described by J. Yang
    et.al. Can only handle decoration from a single scalar
    property up to now.
    """
    # Edit this as you implement new child classes.
    decorated_prop_name = ""
    required_prop_name = ""

    def __init__(self, labels, cuts=None):
        """Initialize.

        Args:
           labels(dict{str:list}): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are possible decorated property
               values, such as oxidation states, magnetic spin directions.
               Values are sorted such that the cluster centers in the
               required property is increasing. For example, in Mn(2, 3, 4)+
               all high spin, the magnetic moments is sorted as [Mn4+, Mn3+, Mn2+],
               thus you should provide labels as {Element("Mn"):[4, 3, 2]}.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               This argument may not be necessary for some sub-classes, such as:
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are the cause of your own error!
            cuts(dict{str: list}): optional
               Cuts to divide required property value into sectors, so as
               to decide the label they belong to. Keys are the same
               as argument "labels".
               For example, if labels={Element("Mn"):[4, 3, 2]} and
               cuts={Element("Mn"):[0.5, 1.0]}, and the required property
               is "total_magmom", then Mn atoms with magnetic moment < 0.5
               will be assigned label 4, atoms with 0.5 <= magnetic moment
               < 1.0 will be assigned label 3, and atoms with magnetic
               moment >= 1.0 will be assigned label 2.
               If provided:
               1, Must be monotonically ascending,
               2, Must be len(labels[key]) = len(cuts[key]) + 1 for any key.
        """
        super(GpOptimizedDecorator, self).__init__(labels)
        if cuts is not None:
            for species in self.labels:
                if species not in cuts:
                    raise ValueError(f"Cuts not provided for species {species}!")
                if len(cuts[species]) + 1 != len(self.labels[species]):
                    raise ValueError(f"Number of cuts for species {species} "
                                     f"does not match the number of its labels!")
        self._cuts = cuts

    @property
    def is_trained(self):
        """Gives whether this decorator is trained before.

        If trained, will be blocked from training again.
        Returns:
            bool.
        """
        return self._cuts is not None

    def _decoration_rules_from_cuts(self, entries, cuts):
        """Get decoration rules from cuts."""

        def get_sector_id(x, sector_cuts):
            # sector_cuts must be ascending.
            assert np.allclose(sector_cuts, np.sort(sector_cuts))
            y = np.append(sector_cuts, np.inf)
            for yid, yy in enumerate(y):
                if x < yy:
                    return yid

        groups = self.group_site_by_species(entries)
        decoration_rule = {}
        for species in groups:
            structure_sites = groups[species]
            props = [entries[struct_id]
                     .data[self.required_prop_name][site_id]
                     for struct_id, site_id in structure_sites]
            props = np.array(props)
            if len(props.shape) != 1:
                raise ValueError("GpOptimizedDecorator can only be used "
                                 "on scalar properties!")
            label_ids = [get_sector_id(p, cuts[species]) for p in props]
            labels = np.array(self.labels[species])
            assigned_labels = labels[label_ids]
            for i, (struct_id, site_id) in enumerate(structure_sites):
                if struct_id not in decoration_rule:
                    decoration_rule[struct_id] = {}
                decoration_rule[struct_id][site_id] = \
                    assigned_labels[i]
        return decoration_rule

    def _evaluate_objective(self, entries, cuts_flatten):
        """Evaluate the objective function as count of filtered entries."""
        # De-flatten.
        cuts = {}
        n_cuts = 0
        for species in sorted(self.labels.keys()):
            cuts[species] = cuts_flatten[n_cuts:
                                         n_cuts + len(self.labels[species]) - 1]
            n_cuts = n_cuts + len(self.labels[species]) - 1
        decoration_rules = self._decoration_rules_from_cuts(entries, cuts)
        entries_processed = self._process(entries, decoration_rules)
        return len([entry for entry in self._filter(entries_processed)
                    if entry is None])

    def _form_initial_guesses(self, entries):
        """Form initial guesses (flatten)."""
        groups = self.group_site_by_species(entries)
        cuts_flatten_init = []
        domains_flatten_init = []
        for species in sorted(self.labels.keys()):
            structure_sites = groups[species]
            # Need some pre-processing to make sure entries data
            # include the required properties, and the properties
            # are in the form of 1D arrayLike per entry.
            props = [entries[struct_id]
                     .data[self.required_prop_name][site_id]
                     for struct_id, site_id in structure_sites]
            props = np.array(props)
            if len(props.shape) != 1:
                raise ValueError("GpOptimizedDecorator can only be used "
                                 "on scalar properties!")
            model = GaussianMixture(n_components=len(self.labels[species]))
            _ = model.fit(props.reshape(-1, 1))
            centers = getattr(model, "means_")
            centers_argsort = np.argsort(centers)
            lin_space = np.linspace(np.min(props) - 0.1, np.max(props) + 0.1, 1000)
            cluster_ids = model.predict(lin_space.reshape(-1, 1))
            label_ids = centers_argsort[cluster_ids]
            cuts_species = []
            last_label_id = label_ids[0]
            for label_id, p in zip(label_ids, lin_space):
                if label_id == last_label_id + 1:
                    cuts_species.append(p)
                    last_label_id = label_id
            if len(cuts_species) > 1:
                delta = np.min([cuts_species[i] - cuts_species[i - 1]
                                for i in range(1, len(cuts_species))]) * 0.3
                # Only allow a small amount of tuning with gp-minimize.
                # This also keeps ascending order between cutting points.
            else:
                delta = (np.max(props) - np.min(props)) * 0.2
            domains_species = [(c - delta, c + delta) for c in cuts_species]
            cuts_flatten_init.extend(cuts_species)
            domains_flatten_init.extend(domains_species)
        return cuts_flatten_init, domains_flatten_init

    def train(self, entries, reset=False):
        """Train the decoration model.

        First initialize with mixture of gaussian, then
        optimize some objective function with gaussian process.
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
            cuts_flatten_init, domains_flatten_init = \
                self._form_initial_guesses(entries)
            objective = functools.partial(self._evaluate_objective, entries)
            result = gp_minimize(objective, domains_flatten_init,
                                 n_calls=50,
                                 acq_optimizer='sampling',
                                 noise=0.001)
            if result is not None and result.x is not None:
                cuts_flatten_opt = result.x
            else:
                warnings.warn("Decorator model can't be optimized properly "
                              "with Bayesian process, use mixture of gaussian "
                              "clustering prediction instead!")
                cuts_flatten_opt = cuts_flatten_init
            # De-flatten.
            cuts = {}
            n_cuts = 0
            for species in sorted(self.labels.keys()):
                cuts[species] = cuts_flatten_opt[n_cuts:
                                                 n_cuts
                                                 + len(self.labels[species]) - 1]
                n_cuts += len(self.labels[species]) - 1
            self._cuts = cuts

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
        decoration_rules = self._decoration_rules_from_cuts(entries, self._cuts)
        entries_decorated = self._process(entries, decoration_rules)
        return self._filter(entries_decorated)

    def as_dict(self):
        """Serialization method."""
        d = super(GpOptimizedDecorator, self).as_dict()
        d["cuts"] = self._cuts
        return d

    @classmethod
    def from_dict(cls, d):
        """Deserialization."""
        return cls(d["labels"], d.get("cuts"))


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
