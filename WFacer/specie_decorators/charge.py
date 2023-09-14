"""For charge assignment.

Charges will be assigned by magnitudes of magnetic moments.
"""

from pymatgen.entries.computed_entries import ComputedStructureEntry

from .base import BaseDecorator, GpOptimizedDecorator, NoTrainDecorator

__author__ = "Fengyu Xie"


class ChargeDecorator(BaseDecorator):
    """A type of decorators to assign charge."""

    decorated_prop_name = "oxi_state"
    required_prop_names = None

    def __init__(self, labels=None, max_allowed_abs_charge=0):
        """Initialize.

        Args:
            labels(dict{str|Species:list}): optional
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
            max_allowed_abs_charge(float): optional
               Maximum allowed absolute value of charge in a decorated structure
               entry. If abs(structure.charge) exceeds this value, the entry
               will be filtered and returned as a NoneType.
               Default to 0, which means we require absolute charge balance.
        """
        super().__init__(labels=labels)
        self.max_allowed_abs_charge = max_allowed_abs_charge

    def _filter(self, entries):
        """Filter out entries with imbalanced charge."""
        return [
            (
                entry
                if abs(entry.structure.charge) <= self.max_allowed_abs_charge
                else None
            )
            for entry in entries
        ]

    def as_dict(self):
        """Serialize to dict."""
        d = super().as_dict()
        d["max_allowed_abs_charge"] = self.max_allowed_abs_charge
        return d

    @classmethod
    def from_dict(cls, d):
        """De-serialize."""
        return cls(d.get("labels"), d.get("max_allowed_abs_charge", 0))


class PmgGuessChargeDecorator(ChargeDecorator, NoTrainDecorator):
    """Assign charges from pymatgen auto guesses.

    Notice: This class does not need labels at all.
    Warning: This Decorator should not be used with
    structures that include multi-valent elements!
    """

    decorated_prop_name = "oxi_state"
    required_prop_names = []

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

    def decorate(self, entries):
        """Decorate entries by guessed charges.

        Warning: Do not use this with multi-valent
        elements, unless you know what you want
        clearly!!!

        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.

        Returns:
            List[NoneType|ComputedStructureEntry]
        """
        entries_decor = []
        for entry in entries:
            s_decor = entry.structure.copy()
            s_decor.add_oxidation_state_by_guess()
            energy_adjustments = (
                entry.energy_adjustments if len(entry.energy_adjustments) != 0 else None
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
        return self._filter(entries_decor)


class FixedChargeDecorator(ChargeDecorator, NoTrainDecorator):
    """Assign fixed charge to each element from setting.

    Warning: This Decorator should not be used with
    structures that include multi-valent elements!
    """

    decorated_prop_name = "oxi_state"
    required_prop_names = []

    def decorate(self, entries):
        """Decorate entries by guessed charges.

        Warning: Do not use this with multi-valent
        elements, unless you know what you want
        clearly!!!

        Args:
            entries(List[ComputedStructureEntry]):
                Entries of computed structures.

        Returns:
            List[NoneType|ComputedStructureEntry]
        """
        entries_decor = []
        for entry in entries:
            s_decor = entry.structure.copy()
            oxi_states = [self.labels[site.specie] for site in entry.structure]
            s_decor.add_oxidation_state_by_site(oxi_states)
            energy_adjustments = (
                entry.energy_adjustments if len(entry.energy_adjustments) != 0 else None
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
        return self._filter(entries_decor)


class MagneticChargeDecorator(GpOptimizedDecorator, ChargeDecorator):
    """Assign charges from magnitudes of total magentic moments on sites.

    Is a sub-class of GPOptimizedDecorator.
    """

    decorated_prop_name = "oxi_state"
    # [(name of the site property to use in pymatgen.structure,
    #  the string path used to query TaskDoc)]
    required_prop_names = [
        ("magmom", "calcs_reversed.0-output" + ".outcar.magnetization.^tot")
    ]

    def __init__(self, labels, cuts=None, max_allowed_abs_charge=0):
        """Initialize.

        Args:
            labels(dict{str: List[int|float]...}):
               A table of species as key, and charges to decorate to the
               species in the key. Values of a key should be sorted as
               the decorated species should have increasing magnetic
               moment.
               For example, in Mn(2, 3, 4)+ all high spin, the magnetic
               moments is sorted as [Mn4+, Mn3+, Mn2+], therefore,
               you should provide labels as {Element("Mn"):[4, 3, 2]}.
               Keys can be either Element|Species object, or their
               string representations. Currently, do not support decoration
               of Vacancy.
               This argument may not be necessary for some sub-classes, such as:
               GuessChargeDecorator.
               Be sure to provide labels for all the species you wish to assign
               a property to, otherwise, you are the cause of your own error!
            cuts(dict{str: List[int|float]...}): optional
               A table of species and cutting points of the magnetic moments,
               so that a magnetic moment is compared with each of these cutting
               values, and decided which charge label it should be assigned with.
            max_allowed_abs_charge(float): optional
               Maximum allowed absolute value of charge in a decorated structure
               entry. If abs(structure.charge) exceeds this value, the entry
               will be filtered and returned as a NoneType.
               Default to 0, which means we require absolute charge balance.
        """
        super().__init__(
            labels=labels, cuts=cuts, max_allowed_abs_charge=max_allowed_abs_charge
        )
