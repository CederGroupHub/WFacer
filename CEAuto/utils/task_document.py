"""Handle atomate2 taskdocument output.

Extend the methods in this file if more site-wise properties other than magmom should
be extracted.
"""
from pymatgen.entries.computed_entries import ComputedStructureEntry

from ..specie_decorators.base import get_site_property_name_from_decorator


def merge_computed_structure_entry(entry, structure):
    """Merge structure into ComputedEntry.

    Args:
        entry(ComputedEntry):
            A computed Entry given by taskdoc.
        structure(Structure):
            A structure given by taskdoc.
    Return:
        ComputedStuctureEntry.
    """
    return ComputedStructureEntry(structure,
                                  entry.uncorrected_energy,  # Use uncorrected to init.
                                  entry.correction,
                                  entry.composition,
                                  entry.energy_adjustments,
                                  entry.parameters,
                                  entry.data,
                                  entry.entry_id)


def get_property_from_taskdoc(taskdoc, property_name):
    """Get structure properties from taskdoc.

    Note: currently, will naively extract from taskdoc's computed entry,
    calcs_reversed[0], outputsummary data dict and attributes. This will
    work for most properties and site-wise properties.
    If in the future more properties are needed, revise this function.
    Args:
        taskdoc(TaskDocument):
            A task document generated as vasp task output by atomate2.
        property_name(str):
            A property names to be retrieved from taskdoc.
            These are properties that you wish to record besides
            "energy" and "uncorrected_energy", etc.
        Returns:
            float: value of the queried property.
    """
    entry = taskdoc.entry
    last_calc = taskdoc.calcs_reversed[0]
    output = taskdoc.output
    # Add more special conversion rules if needed.
    query = property_name

    if query in entry.data:
        return entry.data[query]
    elif query in vars(entry):
        return getattr(entry, query)
    elif query in vars(last_calc):
        return getattr(last_calc, query)
    elif query in vars(last_calc.output):
        return getattr(last_calc.output, query)
    elif query in last_calc.output.outcar:
        return last_calc.output.outcar[query]
    elif query in vars(output):
        return getattr(last_calc.output, query)
    else:
        raise ValueError(f"{query} can not be found"
                         f" in task document!")


def get_entry_from_taskdoc(taskdoc, properties=None, decorator_names=None):
    """Get the computed structure entry from taskdoc.

    Args:
        taskdoc(TaskDocument):
            A task document generated as vasp task output by atomate2.
        properties(list[str]): optional
            A list of property names to be retrieved from taskdoc.
            These are properties that you wish to record besides
            "energy" and "uncorrected_energy", etc. By default,
            will not record any other property.
        decorator_names(list[str]): optional
            The name of decorators used in this CE workflow, used to
            determine what site-wise properties to retrieve from
            TaskDocument, and to include in the returned entry.
    Returns:
        ComputedStructureEntry, dict:
            The computed structure entry, with each site having the site
            property required by decorator, and the properties
            dict for insertion into CeDataWangler.
    """
    # Final optimized structure.
    structure = taskdoc.structure
    # The computed entry, not including the structure.
    computed_entry = taskdoc.entry
    prop_dict = {}
    if properties is not None:
        for p in properties:
            prop_dict[p] = get_property_from_taskdoc(taskdoc, property_name=p)
    site_props = {}
    if decorator_names is not None:
        for d in decorator_names:
            site_property_names = get_site_property_name_from_decorator(d)
            for sp in site_property_names:
                # Total magnetization on each site is already read by atomate2,
                # and added to structure site properties.
                if sp != "magmom":
                    site_props[sp] = get_property_from_taskdoc(taskdoc, sp)
    for sp, prop in site_props.items():
        structure.add_site_property(sp, prop)
    return (merge_computed_structure_entry(computed_entry, structure),
            prop_dict)

