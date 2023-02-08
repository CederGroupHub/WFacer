"""Utilities related to min energies and convex hull.

Notice: when generating and adding training structures, distinguish
element oxidation states. But when generating hulls for comparing
convergence, will not distinguish oxidation states.
"""
import numpy as np
from collections import defaultdict

from pymatgen.core import Composition

from smol.cofe.wrangling.tools import _energies_above_hull


def get_min_energy_structures_by_composition(wrangler, max_iter_id=None):
    """Get minimum energy and structure at each composition.

    This function provides quick tools to compare minimum DFT energies.
    Remember this is NOT hull!
    Sublattice and oxidation state degrees of freedom in compositions
    are not distinguished in generating hull.

    Args:
        wrangler(CeDataWrangler):
            Datawangler object.
        max_iter_id(int): optional
            Maximum iteration index included in the energy comparison.
            If none given, will read existing maximum iteration number.
    Returns:
        defaultdict:
            element compositions as keys, energy per site and structure
            as values.
    """
    min_e = defaultdict(lambda: (np.inf, None))
    prim_size = len(wrangler.cluster_subspace.structure)
    if max_iter_id is None:
        max_iter_id = wrangler.max_iter_id
    for entry in wrangler.entries:
        if entry.data["properties"]["spec"]["iter_id"] <= max_iter_id:
            # Normalize composition and energy to eV per site.
            comp = Composition({k: v / entry.data["size"] / prim_size
                                for k, v
                                in entry.structure.composition
                               .element_composition.items()})
            e = entry.energy / entry.data["size"] / prim_size  # eV/site.
            s = entry.structure
            if e < min_e[comp][0]:
                min_e[comp] = (e, s)
    return min_e


def get_hull(wrangler, max_iter_id=None):
    """Get the energies and compositions on the convex hull.

    Sublattice and oxidation state degrees of freedom in compositions
    are not distinguished in generating hull.

    Args:
        wrangler(CeDataWrangler):
            Datawangler object.
        max_iter_id(int): optional
            Maximum iteration index included in the energy comparison.
            If none given, will read existing maximum iteration number.

    Returns:
        dict:
            element compositions as keys, energy per site and structure
            as values.
    """
    if max_iter_id is None:
        max_iter_id = wrangler.max_iter_id
    data = [(entry.structure, entry.energy) for entry in wrangler.entries
            if entry.data["properties"]["spec"]["iter_id"] <= max_iter_id]
    structures, energies = list(zip(*data))
    e_above_hull = _energies_above_hull(structures, energies,
                                        wrangler.cluster_subspace.structure)

    hull = {}
    prim_size = len(wrangler.cluster_subspace.structure)
    for entry, energy, on_hull in zip(wrangler.entries, energies,
                                      np.isclose(e_above_hull, 0)):
        if on_hull:
            comp = Composition({k: v / entry.data["size"] / prim_size
                                for k, v
                                in entry.structure.composition
                               .element_composition.items()})
            e = energy / entry.data["size"] / prim_size  # eV/site
            hull[comp] = (e, entry.structure)
    return hull
