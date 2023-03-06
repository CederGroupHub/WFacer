"""Convergence checks."""
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from smol.cofe import ClusterExpansion

from .utils.convex_hull import get_min_energy_structures_by_composition


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
            See docs in WFacer.wrangling.
        matcher (StructureMatcher): optional
            A StructureMatcher used compare structures.
            wrangler.cluster_subspace._site_matcher is recommended.
    Return:
        float, bool:
            maximum energy difference in eV/site,
            and whether a new ground state structure appeared.
    """
    diffs = []
    matches = []
    matcher = matcher or StructureMatcher()
    for comp in min_e2:
        if comp not in min_e1:
            return np.inf, True  # New composition appears.
        if not (min_e2[comp][0] == np.inf and min_e1[comp][0] == np.inf):
            diffs.append(np.abs(min_e2[comp][0] - min_e1[comp][0]))
            matches.append(matcher.fit(min_e2[comp][1], min_e1[comp][1]))
    if len(diffs) == 0:
        return np.inf, True
    return np.max(diffs), not (np.all(matches))


def compare_fitted_coefs(cluster_subspace, coefs_prev, coefs_now):
    """Compare fitted coefficients for convergence.

    Args:
        cluster_subspace(ClusterSubspace):
            The cluster subspace used in fitting.
        coefs_prev(1d arrayLike):
            Cluster coefficients fitted in the previous iteration.
            Not ECIs because not divided by multiplicity!
        coefs_now(1d arrayLike):
            Cluster coefficients fitted in the latest iteration.
    Returns:
        float:
            || ECI' - ECI ||_1 / ||ECI||_1.
    """
    # Get ECIs from coefficients.
    eci_prev = ClusterExpansion(cluster_subspace, coefficients=coefs_prev).eci
    eci_now = ClusterExpansion(cluster_subspace, coefficients=coefs_now).eci

    return np.linalg.norm(eci_prev - eci_now, ord=1) / np.linalg.norm(eci_prev, ord=1)


def ce_converged(
    coefs_history, cv_history, cv_std_history, wrangler, convergence_options
):
    """Check whether the ce workflow has converged.

    Args:
        coefs_history(list[list[float]]):
            CE coefficients from all past iterations.
        cv_history(list[float]):
            Past cross validation errors.
        cv_std_history(list[float]):
            Past cross validation standard deviations.
            The length of the first three arguments must
            be equal.
        wrangler(CeDataWrangler):
            A wrangler storing all past training data.
            Maximum recorded iteration index in wrangler
            must be equal to that of current iter_id - 1.
        convergence_options(dict):
            Pre-processed convergence criterion.
    Returns:
        bool.
    """
    # Wrangler is not empty, but its maximum iteration index does not match the
    # last iteration.
    coefs_history = coefs_history or []
    if len(coefs_history) < 2:
        return False
    iter_id = wrangler.max_iter_id

    cv_converged = (
        cv_history[-1] <= convergence_options["cv_tol"]
        and (
            convergence_options["std_cv_rtol"] is None
            or cv_std_history[-1] / cv_history[-1] <= convergence_options["std_cv_rtol"]
        )
        and abs(cv_history[-1] - cv_history[-2]) / cv_history[-2]
        <= convergence_options["delta_cv_rtol"]
    )

    eci_converged = (
        convergence_options["delta_eci_rtol"] is None
        or compare_fitted_coefs(
            wrangler.cluster_subspace, coefs_history[-2], coefs_history[-1]
        )
        <= convergence_options["delta_eci_rtol"]
    )

    min_e1 = get_min_energy_structures_by_composition(wrangler, max_iter_id=iter_id - 1)
    min_e2 = get_min_energy_structures_by_composition(wrangler, max_iter_id=iter_id)
    matcher = wrangler.cluster_subspace._site_matcher
    max_diff, new_gs_found = compare_min_energy_structures_by_composition(
        min_e1, min_e2, matcher
    )
    # eV/site to meV/site.
    min_e_converged = 1000 * max_diff / cv_history[-1] <= convergence_options[
        "delta_min_e_rtol"
    ] and (not convergence_options["continue_on_finding_new_gs"] or not new_gs_found)

    max_iter = (
        convergence_options["max_iter"]
        if convergence_options["max_iter"] is not None
        else np.inf
    )

    real_convereged = cv_converged and eci_converged and min_e_converged

    return real_convereged or iter_id >= max_iter - 1
