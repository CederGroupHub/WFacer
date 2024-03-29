"""Handle atomate2 task document output.

Extend the methods in this file if more site-wise properties other than magmom should
be extracted.

.. note:
 Currently only supports reading from class :class:`emmet.core.tasks.TaskDoc`,
 class :class:`atomate2.cp2k.schemas.task.TaskDocument` and
 class :class:`atomate2.forcefields.schemas.ForceFieldTaskDocument`.
"""

from atomate2.cp2k.schemas.task import TaskDocument
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from emmet.core.tasks import TaskDoc
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

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
        ComputedStuctureEntry:
            A :class:`ComputedStructureEntry` created from
            class :class:`ComputedEntry`.
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
        taskdoc(StructureMetadata):
            A task document generated as vasp task output by emmet-core, CP2K
            or force fields.
        property_and_queries(list of (str, str) or list of str): optional
            A list of property names to be retrieved from taskdoc,
            and the query string to retrieve them, paired in tuples.
            If only strings are given, will also query with the given
            string.
            These are properties that you wish to record besides
            "energy" and "uncorrected_energy", etc. By default,
            will not record any other property.
        decorator_names(list of str): optional
            The name of decorators used in this CE workflow, used to
            determine what site properties to retrieve from
            TaskDoc and to include in the returned entry.

    Returns:
        ComputedStructureEntry, dict:
            The computed structure entry, with each site having the site
            property required by decorator, and the properties
            dict ready to be inserted into a :class:`CeDataWangler`.
    """
    if not isinstance(taskdoc, (TaskDoc, TaskDocument, ForceFieldTaskDocument)):
        raise ValueError(f"Document type {type(taskdoc)} not supported!")
    # Final optimized structure.
    structure = taskdoc.structure
    # The computed entry, not including the structure.
    if isinstance(taskdoc, (TaskDoc, TaskDocument)):
        computed_entry = taskdoc.entry
    # In ForcefieldTaskdocument, Need to retrieve the entry from ionic steps.
    else:
        computed_entry = ComputedEntry(
            composition=taskdoc.structure.composition,
            energy=taskdoc.output.energy,
            correction=0.0,
        )

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
                # structure by atomate2 for all kinds of makers (VASP, CP2K, force fields).
                # There is no need to do it again.
                if sp != "magmom":
                    site_props[sp] = get_property_from_object(taskdoc, query)
    for sp, prop in site_props.items():
        structure.add_site_property(sp, prop)
    return _merge_computed_structure_entry(computed_entry, structure), prop_dict
