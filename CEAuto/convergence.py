"""Convergence checks."""
import numpy as np

from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe import ClusterExpansion


def compare_min_energy_structures_by_composition(min_e1, min_e2, matcher=None):
    """Compare minimum energy and structure by composition for convergence check.

     We will only compare keys that exist in both older and newer iterations.
     If one composition appears in the older one but not the newer one, we will not
     claim convergence.

    Args:
        min_e1 (defaultdict):
            Minimum energies and structures from an earlier iteration.
        min_e2 (defaultdict):
            Minimum energies and structures from a later iteration.
            See docs in CEAuto.wrangling.
        matcher (StructureMatcher): optional
            A StructureMatcher used compare structures.
            wrangler.cluster_subspace._site_matcher is recommended.
    Return:
        float, bool:
            maximum energy difference in eV/site, and whether a new GS appeared.
    """
    diffs = []
    matches = []
    matcher = matcher or StructureMatcher()
    for comp in min_e2:
        if comp not in min_e1:
            return np.inf, True  # New composition appears.
        if not (min_e2[comp][0] == np.inf and min_e1[comp][0] == np.inf):
            diffs.append(np.abs(min_e2[comp] - min_e1[comp]))
            matches.append(matcher.fit(min_e2[comp][1], min_e1[comp][1]))
    if len(diffs) == 0:
        return np.inf, True
    return np.max(diffs), not(np.all(matches))


def compare_fitted_coefs(cluster_subspace, coefs_prev, coefs_now):
    """Compare fitted coefficients for convergence.

    Args:
        cluster_subspace(ClusterSubspace):
            The cluster subspace used in fitting.
        coefs_prev(1d arrayLike):
            Cluster coefficients fitted in the previous iteration.
            Not ECIs because not divided by multiplicity!
        coefs_now(1d arrayLike):
            Cluster coefficeints fitted in the latest iteration.
    Returns:
        float:
            || ECI' - ECI ||_1 / ||ECI||_1.
    """
    # Get ECIs from coefficients.
    eci_prev = ClusterExpansion(cluster_subspace, coefficients=coefs_prev).eci
    eci_now = ClusterExpansion(cluster_subspace, coefficients=coefs_now).eci

    return np.linalg.norm(eci_prev - eci_now, ord=1) / np.linalg.norm(eci_prev, ord=1)
