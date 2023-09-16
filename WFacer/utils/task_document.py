"""Handle atomate2 taskdocument output.

Extend the methods in this file if more site-wise properties other than magmom should
be extracted.
"""

from pymatgen.entries.computed_entries import ComputedStructureEntry

from ..specie_decorators.base import get_site_property_query_names_from_decorator
from .query import get_property_from_object


def _merge_computed_structure_entry(entry, structure):
    """Merge a structure into :class:`ComputedEntry`.

    Args:
        entry(ComputedEntry):
            A computed Entry extracted from a :class:`TaskDoc`.
        structure(Structure):
            A structure from the same :class:`TaskDoc`.

    Return:
        ComputedStuctureEntry.
    """
    return ComputedStructureEntry(
        structure,
        entry.uncorrected_energy,  # Use uncorrected to init.
        entry.correction,
        entry.composition,
        entry.energy_adjustments,
        entry.parameters,
        entry.data,
        entry.entry_id,
    )


def get_entry_from_taskdoc(taskdoc, property_and_queries=None, decorator_names=None):
    """Get the computed structure entry from :class:`TaskDoc`.

    Args:
        taskdoc(TaskDoc):
            A task document generated as vasp task output by emmet-core.
        property_and_queries(list[(str, str)|str]): optional
            A list of property names to be retrieved from taskdoc,
            and the query string to retrieve them, paired in tuples.
            If only strings are given, will also query with the given
            string.
            These are properties that you wish to record besides
            "energy" and "uncorrected_energy", etc. By default,
            will not record any other property.
        decorator_names(list[str]): optional
            The name of decorators used in this CE workflow, used to
            determine what site properties to retrieve from
            TaskDoc and to include in the returned entry.

    Returns:
        ComputedStructureEntry, dict:
            The computed structure entry, with each site having the site
            property required by decorator, and the properties
            dict ready to be inserted into a :class:`CeDataWangler`.
    """
    # Final optimized structure.
    structure = taskdoc.structure
    # The computed entry, not including the structure.
    computed_entry = taskdoc.entry
    prop_dict = {}
    if property_and_queries is not None:
        for p in property_and_queries:
            if isinstance(p, (tuple, list)):
                prop_dict[p[0]] = get_property_from_object(taskdoc, p[1])
            elif isinstance(p, str):
                prop_dict[p] = get_property_from_object(taskdoc, p)
            else:
                raise ValueError(
                    "Property names and their query strings"
                    " must either be in tuples or be in"
                    " strings!"
                )
    site_props = {}
    if decorator_names is not None:
        for d in decorator_names:
            site_property_query_names = get_site_property_query_names_from_decorator(d)
            for sp, query in site_property_query_names:
                # Total magnetization on each site is already read and added to
                # structure by atomate2. It should be overwritten.
                if sp != "magmom":
                    site_props[sp] = get_property_from_object(taskdoc, query)
    for sp, prop in site_props.items():
        structure.add_site_property(sp, prop)
    return _merge_computed_structure_entry(computed_entry, structure), prop_dict
