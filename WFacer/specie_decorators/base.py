"""Decorate properties to a structure composed of Element.

This module offers generic classes and functions for defining an algorithm
used to map VASP calculated site properties into the label of species. For
example, :class:`BaseDecorator`, :class:`MixtureGaussianDecorator`,
:class:`GpOptimizedDecorator` and :class:`NoTrainDecorator`. These abstract
classes are meant to be inherited by any decorator class that maps specific
site properties.

Currently, we can only decorate charge. Plan to allow decorating
spin in the future updates.

.. note:: All entries should be re-decorated and all decorators
 should be retrained after an iteration.
"""

__author__ = "Fengyu Xie, Julia H. Yang"

import functools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from warnings import warn

import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Species, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from sklearn.mixture import GaussianMixture
from skopt import gp_minimize
from smol.cofe.space.domain import get_species
from smol.utils.class_utils import (
    class_name_from_str,
    derived_class_factory,
    get_subclasses,
)

# Add here if you implement more decorators.
valid_decorator_types = {
    "oxi_state": (
        "pmg-guess-charge",
        "magnetic-charge",
    )
}


def _get_required_site_property(entry, site_id, prop_name):
    """Find required site property.

    If prop_name contains dot ("."), site property will be treated as a dictionary.
    """

    def explore_key_path(path, d):
        d_last = d.copy()
        for k in path:
            d_last = d_last.get(k, {})
        if isinstance(d_last, dict) and len(d_last) == 0:
            return None
        return d_last

    key_path = prop_name.split(".")
    return explore_key_path(key_path, entry.structure[site_id].properties)


class BaseDecorator(MSONable, metaclass=ABCMeta):
    """Abstract decorator class.

    #. Each decorator should only be used to decorate one property.
    #. Currently, only supports assigning labels from one scalar site property,
       and requires that the site property can be accessed from
       :class:`ComputedStructureEntry`, which should be sufficient for most
       purposes.
    #. Can not decorate entries with partial disorder.
    """

    # Edit these if you implement new child classes.
    decorated_prop_name = None
    required_prop_names = None

    def __init__(self, labels=None, **kwargs):
        """Initialize.

        Args:
           labels(dict of str or Species to list): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are possible decorated property
               values, such as oxidation states, magnetic spin directions.
               Values are sorted such that the corresponding cluster centers of the
               required property is increasing. For example, in Mn(2, 3, 4)+
               (high spin), the magnetic moments is sorted as [Mn4+, Mn3+, Mn2+],
               thus you should provide labels as {Element("Mn"):[4, 3, 2]}.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               If you have multiple required properties, or required properties
               have multiple dimensions, the labels order must match the sort
               in the order of self.required_properties. Properties are sorted
               lexicographically.
               This argument may not be necessary for some decorator, such as
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are responsible for your own error!
        """
        labels = labels or {}
        self.labels = {get_species(key): val for key, val in labels.items()}

    @staticmethod
    def group_site_by_species(entries):
        """Group required properties on sites by species.

        Args:
            entries(list of ComputedStructureEntry):
                Entries of computed structures.

        Return:
            defaultdict:
               (Entry index, site index) belonging to each species.
        """
        groups_by_species = defaultdict(lambda: [])

        # These entries should not contain Vacancy.
        for e_id, entry in enumerate(entries):
            for s_id, site in enumerate(entry.structure):
                # site.species is always a Composition object.
                # site.specie is an Element or Species.
                sp = site.specie
                groups_by_species[sp] += [(e_id, s_id)]

        return groups_by_species

    @property
    @abstractmethod
    def is_trained(self):
        """Gives whether this decorator is trained before.

        If trained, will be blocked from training again.

        Returns:
            bool:
               Whether the model has been trained.
        """
        return

    @abstractmethod
    def train(self, entries, reset=False):
        """Train the decoration model.

        Model or model parameters should be stored in a property of the
        object.

        Args:
            entries(list of ComputedStructureEntry):
                Entries of computed structures.
            reset(bool): optional
                If you want to re-train the decorator model, set this value
                to true. Otherwise, will skip training if the model is
                trained. Default to false.
        """
        return

    @abstractmethod
    def decorate(self, entries):
        """Give decoration to entries based on trained model.

        If an assigned entry is not valid,
        for example, in charge assignment, if a decorated structure is not
        charge neutral, this entry will be returned as None.

        Args:
            entries(list of ComputedStructureEntry):
                Entries of computed, undecorated structures.

        Returns:
            list of NoneType or ComputedStructureEntry:
               Entries with decorated structures or failed structures.
        """
        return

    def _load_props(self, species, entries, groups):
        """Load required properties from entries."""
        # If required_prop_name can be parsed by emmet into
        # the structure of ComputedStructureEntry. Otherwise,
        # need some pre-processing to make sure entry data
        # include the required properties, and the properties
        # should be in the form of 1D arrayLike per entry.
        # Properties are concatenated in the order given by
        # cls.required_properties.
        props = []
        structure_sites = groups[species]
        for struct_id, site_id in structure_sites:
            site_props = []
            for prop_name, taskdoc_query in self.required_prop_names:
                p = _get_required_site_property(entries[struct_id], site_id, prop_name)
                if hasattr(p, "__iter__"):
                    raise ValueError(
                        "Cannot train assignment on non-scalar" " property."
                    )
                else:
                    site_props.append(p)
            props.append(site_props)
        return props

    def _process(self, entries, decorate_rules):
        """Decorate entries with rules."""
        entries_decor = []
        for struct_id, entry in enumerate(entries):
            s_undecor = entry.structure
            species_decor = []
            for site_id, site in enumerate(s_undecor):
                if not hasattr(site, "specie"):
                    raise ValueError(
                        "Can not decorate partially disordered site: " f"{site}"
                    )
                sp = site.specie
                if struct_id in decorate_rules and site_id in decorate_rules[struct_id]:
                    label = decorate_rules[struct_id][site_id]
                    if self.decorated_prop_name == "oxi_state":
                        if isinstance(sp, Element):
                            sp_decor = Species(sp.symbol, oxidation_state=label)
                        else:
                            sp_decor = deepcopy(sp)
                            sp_decor._oxi_state = label
                    else:
                        # After pymatgen 2023.07.20, properties dictionary is deprecated.
                        # Only spin will be supported.
                        if self.decorated_prop_name.lower() != "spin":
                            raise ValueError(
                                "Pymatgen Species does not support "
                                f"property {self.decorated_prop_name}!"
                            )

                        # Atomic species with spin is treated as zero oxidation state.
                        if isinstance(sp, Element):
                            sp_decor = Species(
                                sp.symbol,
                                oxidation_state=0,
                                spin=label,
                            )
                        else:
                            sp_decor = Species(
                                sp.symbol,
                                oxidation_state=sp.oxi_state,
                                spin=label,
                            )

                    species_decor.append(sp_decor)

                else:  # Undecorated sites might continue to be Element.
                    species_decor.append(sp)

            # Preserve all information.
            site_properties = defaultdict(lambda: [])
            for site in s_undecor:
                for p in site.properties:
                    site_properties[p].append(site.properties[p])
            s_decor = Structure(s_undecor.lattice, species_decor, s_undecor.frac_coords)
            for prop, values in site_properties.items():
                s_decor.add_site_property(prop, values)
            energy_adjustments = (
                entry.energy_adjustments
                if (
                    len(entry.energy_adjustments) != 0
                    and entry.energy_adjustments is not None
                )
                else None
            )
            # Constant energy adjustment is set as a manual class object.
            entry_decor = ComputedStructureEntry(
                s_decor,
                energy=entry.uncorrected_energy,
                energy_adjustments=energy_adjustments,
                parameters=entry.parameters,
                data=entry.data,
                entry_id=entry.entry_id,
            )
            entries_decor.append(entry_decor)
        return entries_decor

    @abstractmethod
    def _filter(self, entries):
        """Filter out entries by some criteria.

        Must be implemented for every decorator class.
        The entries that fail to satisfy the specific criteria
        defined here will be returned as None.
        """
        return entries

    # Should save and load dicts with Monty.
    def as_dict(self):
        """Serialize the decorator."""
        labels = {str(key): val for key, val in self.labels.items()}
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "labels": labels,
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, d):
        """Deserialization."""
        return


class MixtureGaussianDecorator(BaseDecorator, metaclass=ABCMeta):
    """Mixture of Gaussians (MoGs) decorator class.

    Uses mixture of Gaussians method to label each species.

    .. note:: No test has been added for this specific class yet.
    """

    decorated_prop_name = None
    required_prop_names = None
    gaussian_model_keys = (
        "weights_",
        "means_",
        "covariances_",
        "precisions_",
        "precisions_cholesky_",
        "converged_",
        "n_iter_",
        "lower_bound_",
    )

    def __init__(self, labels, gaussian_models=None, **kwargs):
        """Initialize.

        Args:
           labels(dict of str to list): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are possible decorated property
               values, such as oxidation states, magnetic spin directions.
               Values are sorted such that the cluster centers in the
               required property is increasing. For example, in Mn(2, 3, 4)+
               all high spin, the magnetic moments is sorted as [Mn4+, Mn3+, Mn2+],
               thus you should provide labels as {Element("Mn"):[4, 3, 2]}.
               If you have multiple required properties, or required properties
               have multiple dimensions, the labels order must match the sort
               in the order of self.required_properties. Properties are sorted
               lexicographically.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               This argument may not be necessary for some sub-classes, such as:
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are the cause of your own error!
           gaussian_models(dict of str or Element or Species to GaussianMixture):
               Gaussian models corresponding to each key in argument **labels**.
        """
        super().__init__(labels, **kwargs)
        if gaussian_models is None:
            gaussian_models = {}
        gaussian_models = {
            get_species(key): val for key, val in gaussian_models.items()
        }
        for key in self.labels:
            if key not in gaussian_models:
                warn(
                    f"Gaussian model for {key} is missing! " "Initializing from empty."
                )
                gaussian_models[key] = GaussianMixture(
                    n_components=len(self.labels[key])
                )
        self._gms = gaussian_models

    @staticmethod
    def serialize_gaussian_model(model):
        """Serialize gaussian model into dict."""
        data = {"init_params": model.get_params(), "model_params": {}}
        for k in MixtureGaussianDecorator.gaussian_model_keys:
            if k in model.__dict__:
                data["model_params"][k] = getattr(model, k)
            # Contains np.array, not directly json.dump-able. Use monty.
        return data

    @staticmethod
    def deserialize_gaussian_model(data):
        """Recover gaussian model from dict."""
        model = GaussianMixture(**data["init_params"])
        for k, v in data["model_params"].items():
            setattr(model, k, v)
        return model

    @staticmethod
    def is_trained_gaussian_model(model):
        """Whether a gaussian model is trained."""
        return all(
            [k in model.__dict__ for k in MixtureGaussianDecorator.gaussian_model_keys]
        )

    @property
    def is_trained(self):
        """Determine whether the decorator has been trained.

        Returns:
            bool:
              Whether the model has been trained.
        """
        return all([self.is_trained_gaussian_model(m) for m in self._gms.values()])

    def train(self, entries, reset=False):
        """Train the decoration model.

        Model or model parameters should be stored in a property of the
        object.

        Args:
            entries(list of ComputedStructureEntry):
                Entries of computed structures.
            reset(bool): optional
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
                props = self._load_props(species, entries, groups)
                props = np.array(props)
                if len(props.shape) > 2:
                    raise ValueError(
                        "Can not train on tensor properties! "
                        "Convert to scalar or vector before "
                        "training!"
                    )
                self._gms[species] = self._gms[species].fit(props)

    def decorate(self, entries):
        """Give decoration to entries based on trained model.

        If an assigned entry is not valid,
        for example, in charge assignment, if an assigned structure is not
        charge neutral, then this entry will be returned as None.

        Args:
            entries(list of ComputedStructureEntry):
                Entries of computed, undecorated structures.

        Returns:
            List of NoneType or ComputedStructureEntry:
                Entries with decorated structures or failed structures.
        """
        if not self.is_trained:
            raise ValueError("Can not make predictions from un-trained" " models!")
        groups = self.group_site_by_species(entries)
        decoration_rule = {}
        for species in groups:
            structure_sites = groups[species]
            model = self._gms[species]
            centers = getattr(model, "means_")
            # Lex sort on dimensions,
            # flip to make sure that first dim sorted first.
            marks = np.flip(np.array(centers).transpose(), axis=0)
            centers_argsort = np.lexsort(marks).tolist()
            props = self._load_props(species, entries, groups)
            props = np.array(props)
            if len(props.shape) > 2:
                raise ValueError(
                    "Can not train on tensor properties! "
                    "Convert to scalar or vector before "
                    "training!"
                )
            cluster_ids = model.predict(props)
            label_ids = [centers_argsort.index(c) for c in cluster_ids]
            labels = np.array(self.labels[species])
            assigned_labels = labels[label_ids]
            for i, (struct_id, site_id) in enumerate(structure_sites):
                if struct_id not in decoration_rule:
                    decoration_rule[struct_id] = {}
                decoration_rule[struct_id][site_id] = assigned_labels[i]
        entries_processed = self._process(entries, decoration_rule)
        return self._filter(entries_processed)

    def as_dict(self):
        """Serialize to dict."""
        d = super().as_dict()
        d["models"] = {
            str(species): self.serialize_gaussian_model(model)
            for species, model in self._gms.items()
        }
        return d

    @classmethod
    def from_dict(cls, d):
        """Load from dict."""
        # Please load dict with monty.
        models = d.get("models")
        if models is not None:
            models = {k: cls.deserialize_gaussian_model(v) for k, v in models.items()}
        return cls(d["labels"], models)


class GpOptimizedDecorator(BaseDecorator, metaclass=ABCMeta):
    """Gaussian process decorator class.

    Uses Gaussian optimization process described by `J. H. Yang
    et al. <https://www.nature.com/articles/s41524-022-00818-3>`_

    Up to now, this class can only take as input a single scalar
    property per site.
    """

    # Edit this as you implement new child classes.
    decorated_prop_name = ""
    required_prop_names = []

    def __init__(self, labels, cuts=None, **kwargs):
        """Initialize.

        Args:
           labels(dict of str to list): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are possible decorated property
               values, such as oxidation states, magnetic spin directions.
               Values are sorted such that the cluster centers in the
               required property is increasing. For example, in Mn(2, 3, 4)+
               all high spin, the magnetic moments is sorted as [Mn4+, Mn3+, Mn2+],
               thus you should provide labels as {Element("Mn"):[4, 3, 2]}.
               Keys can be either Element and Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               If you have multiple required properties, or required properties
               have multiple dimensions, the labels order must match the sort
               in the order of self.required_properties. Properties are sorted
               lexicographically.
               This argument may not be necessary for some sub-classes, such as:
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are the cause of your own error!
           cuts(dict of str or Species over list): optional
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

               #. Cut values must be monotonically increasing,
               #. Must satisfy len(labels[key]) = len(cuts[key]) + 1 for any key.
        """
        super().__init__(labels, **kwargs)
        if cuts is not None:
            cuts = {get_species(key): val for key, val in cuts.items()}
            for species in self.labels:
                if species not in cuts:
                    raise ValueError(f"Cuts not provided for species {species}!")
                if len(cuts[species]) + 1 != len(self.labels[species]):
                    raise ValueError(
                        f"Number of cuts for species {species} "
                        f"does not match the number of its labels!"
                    )
        self._cuts = cuts

    @property
    def is_trained(self):
        """Gives whether this decorator is trained before.

        If trained, will be blocked from training again.

        Returns:
            bool:
                Whether the model is trained.
        """
        return self._cuts is not None

    def _decoration_rules_from_cuts(self, entries, cuts):
        """Get decoration rules from cuts."""

        def get_sector_id(x, sector_cuts):
            # sector_cuts must be ascending.
            y = np.append(sector_cuts, np.inf)
            for yid, yy in enumerate(y):
                if x <= yy:
                    return yid

        groups = self.group_site_by_species(entries)
        decoration_rule = {}
        for species in groups:
            structure_sites = groups[species]
            props = self._load_props(species, entries, groups)
            props = np.array(props)
            if props.shape[1] != 1:
                raise ValueError(
                    "GpOptimizedDecorator can only be trained "
                    "on one scalar property!"
                )
            props = props.flatten()
            label_ids = [get_sector_id(p, cuts[species]) for p in props]
            labels = np.array(self.labels[species])
            assigned_labels = labels[label_ids]
            for i, (struct_id, site_id) in enumerate(structure_sites):
                if struct_id not in decoration_rule:
                    decoration_rule[struct_id] = {}
                decoration_rule[struct_id][site_id] = assigned_labels[i]
        return decoration_rule

    def _evaluate_objective(self, entries, cuts_flatten):
        """Evaluate the objective function as count of filtered entries."""
        # De-flatten.
        cuts = {}
        n_cuts = 0
        for species in sorted(self.labels.keys()):
            cuts[species] = cuts_flatten[
                n_cuts : n_cuts + len(self.labels[species]) - 1
            ]
            n_cuts = n_cuts + len(self.labels[species]) - 1
        decoration_rules = self._decoration_rules_from_cuts(entries, cuts)
        entries_processed = self._process(entries, decoration_rules)
        return len(
            [entry for entry in self._filter(entries_processed) if entry is None]
        )  # To be minimized.

    def _form_initial_guesses(self, entries):
        """Form initial guesses (flatten)."""
        groups = self.group_site_by_species(entries)
        cuts_flatten_init = []
        domains_flatten_init = []
        for species in sorted(self.labels.keys()):
            # Need some pre-processing to make sure entries data
            # include the required properties, and the properties
            # are in the form of 1D arrayLike per entry.
            props = self._load_props(species, entries, groups)
            props = np.array(props)
            if props.shape[1] != 1:
                raise ValueError(
                    "GpOptimizedDecorator can only be used " "on scalar properties!"
                )
            props = props.flatten()
            model = GaussianMixture(n_components=len(self.labels[species]))
            _ = model.fit(props.reshape(-1, 1))
            centers = getattr(model, "means_")
            # Lex sort on dimensions,
            # flip to make sure that first dim sorted first.
            marks = np.flip(np.array(centers).transpose(), axis=0)
            centers_argsort = np.lexsort(marks).tolist()
            lin_space = np.linspace(np.min(props) - 0.5, np.max(props) + 0.5, 2000)
            cluster_ids = model.predict(lin_space.reshape(-1, 1))
            label_ids = [centers_argsort.index(c) for c in cluster_ids]
            cuts_species = []
            last_label_id = label_ids[0]
            for label_id, p in zip(label_ids, lin_space):
                if label_id != last_label_id:
                    # assert label_id == last_label_id + 1
                    cuts_species.append(p)
                    last_label_id = label_id

            if len(cuts_species) > 1:
                delta = (
                    np.min(
                        [
                            cuts_species[i] - cuts_species[i - 1]
                            for i in range(1, len(cuts_species))
                        ]
                    )
                    * 0.3
                )
                # Only allow a small amount of tuning with gp-minimize.
                # This also keeps ascending order between cutting points.
            else:
                delta = (np.max(props) - np.min(props)) * 0.2
            domains_species = [(c - delta, c + delta) for c in cuts_species]
            cuts_flatten_init.extend(cuts_species)
            domains_flatten_init.extend(domains_species)
        return cuts_flatten_init, domains_flatten_init

    def train(self, entries, reset=False, n_calls=50):
        """Train the decoration model.

        First initialize with mixture of gaussian, then
        optimize some objective function with gaussian process.

        Args:
            entries(list of ComputedStructureEntry):
                Entries of computed structures.
            reset(bool): optional
                If you want to re-train the decorator model, set this value
                to true. Otherwise, training will be skipped if the model is
                trained. Default to false.
            n_calls(int): optional
                The number of iterations to be used by :func:`gp_minimize`.
                Default is 50.
        """
        if self.is_trained and not reset:
            return
        else:
            cuts_flatten_init, domains_flatten_init = self._form_initial_guesses(
                entries
            )
            objective = functools.partial(self._evaluate_objective, entries)
            result = gp_minimize(
                objective,
                domains_flatten_init,
                n_calls=n_calls,
                acq_optimizer="sampling",
                noise=0.001,
            )
            if result is not None and result.x is not None:
                cuts_flatten_opt = result.x
            else:
                warn(
                    "Decorator model can't be optimized properly "
                    "with Bayesian process, use mixture of gaussian "
                    "clustering prediction instead!"
                )
                cuts_flatten_opt = cuts_flatten_init
            # De-flatten.
            cuts = {}
            n_cuts = 0
            for species in sorted(self.labels.keys()):
                cuts[species] = cuts_flatten_opt[
                    n_cuts : n_cuts + len(self.labels[species]) - 1
                ]
                cuts[species] = np.array(cuts[species]).tolist()
                n_cuts += len(self.labels[species]) - 1
            self._cuts = cuts

    def decorate(self, entries):
        """Give decoration to entries based on trained model.

        If an assigned entry is not valid,
        for example, in charge assignment, if a decorated structure is not
        charge neutral, then its corresponding entry will be returned as None.

        Args:
            entries(list of ComputedStructureEntry):
                Entries of computed, undecorated structures.

        Returns:
            list of NoneType or ComputedStructureEntry:
                Entries with decorated structures or failed structures.
        """
        decoration_rules = self._decoration_rules_from_cuts(entries, self._cuts)
        entries_decorated = self._process(entries, decoration_rules)
        return self._filter(entries_decorated)

    def as_dict(self):
        """Serialize the decorator."""
        d = super().as_dict()
        # Species serialized to string directly. Many other properties
        # might not be supported. Wait for pymatgen update.
        if self.is_trained:
            d["cuts"] = {str(key): val for key, val in self._cuts.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        """Deserialization."""
        return cls(d["labels"], d.get("cuts"))


class NoTrainDecorator(BaseDecorator):
    """Decorators that does not need training."""

    def __init__(self, labels, **kwargs):
        """Initialize.

        Args:
           labels(dict of str or Species to list}): optional
               A table of labels to decorate each element with.
               keys are species symbol, values are possible decorated property
               values, such as oxidation states, magnetic spin directions.
               Values are sorted such that the corresponding cluster centers of the
               required property is increasing. For example, in Mn(2, 3, 4)+
               (high spin), the magnetic moments is sorted as [Mn4+, Mn3+, Mn2+],
               thus you should provide labels as {Element("Mn"):[4, 3, 2]}.
               Keys can be either Element and Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               If you have multiple required properties, or required properties
               have multiple dimensions, the labels order must match the sort
               in the order of self.required_properties. Properties are sorted
               lexicographically.
               This argument may not be necessary for some decorator, such as
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are responsible for your own error!
        """
        super().__init__(labels, **kwargs)

    @property
    def is_trained(self):
        """Always considered trained."""
        return True

    def train(self, entries=None, reset=False):
        """Train the model.

        This decorator does not require training at all. Keep
        this method just for consistency.
        """
        return


def decorator_factory(decorator_type, *args, **kwargs):
    """Create a BaseDecorator with its subclass name.

    Args:
        decorator_type(str):
            The name of a subclass of :class:`BaseDecorator`.
        *args, **kwargs:
            Arguments used to initialize the class.

    Returns:
        BaseDecorator:
            The initialized decorator.
    """
    if "decorator" not in decorator_type and "Decorator" not in decorator_type:
        decorator_type += "-decorator"
    name = class_name_from_str(decorator_type)
    return derived_class_factory(name, BaseDecorator, *args, **kwargs)


def get_site_property_query_names_from_decorator(decname):
    """Get the required properties from a decorator name.

    Args:
        decname(str):
            Decorator name.

    Returns:
        list of str:
            The list of names of required site properties by the
            decorator.
    """
    if "decorator" not in decname and "decorator" not in decname:
        decname += "-decorator"
    clsname = class_name_from_str(decname)
    dec_class = get_subclasses(BaseDecorator).get(clsname)
    if dec_class is None:
        raise ValueError(f"required decorator {clsname} is not implemented!")
    return dec_class.required_prop_names
