"""DataWrangler.

This file includes a modified version of StructureWrangler.
"""

__author__ = "Fengyu Xie"

import numpy as np
import warnings
from collections import defaultdict

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition

from smol.cofe.wrangling.wrangler import StructureWrangler


def collinear(a1, a2):
    """Check if 2 arraylikes are collinear.

    Args:
        a1, a2 (1D arraylike[float]):
            Arrays to check.
    """
    return np.isclose(np.dot(a1, a2),
                      np.linalg.norm(a1) * np.linalg.norm(a2))


class DataWrangler(StructureWrangler):
    """DataWrangler class.

    Interfaces CEAuto generated data, does insertion and deletion,
    but will not generate any data.
    It cannot do charge assignment, etc, as well.

    Note: This DataWrangler is not serializable with legacy versions of smol.
    """
    def _check_duplicacy(self, entry, sm=StructureMatcher()):
        """Whether an entry symmetrically duplicates with existing ones"""
        for eid, entry_old in self.entries:
            if sm.fit(entry_old.data["refined_structure"],
                      entry.data["refined_structure"]):
                return eid
        return None

    @property
    def max_iter_id(self):
        """Maximum index of iteration existing.

        Iteration counted from 0.
        """
        return max(entry.data["iter_id"] for entry in self.entries)

    def process_entry(
        self,
        entry,
        properties=None,
        weights=None,
        supercell_matrix=None,
        site_mapping=None,
        verbose=False,
        raise_failed=False,
        iter_id=0,
    ):
        """Process a ComputedStructureEntry to be added to StructureWrangler.

        Checks if the structure for this entry can be matched to the
        ClusterSubspace prim structure to obtain its supercell matrix,
        correlation, and refined structure. If so, the entry will be updated by adding
        these to its data dictionary.

        Args:
            entry (ComputedStructureEntry):
                A ComputedStructureEntry corresponding to a training strucutre and
                properties
            properties (dict): optional
                A dictionary with a keys describing the property and the target
                value for the corresponding structure. Energy and corrected energy
                should already be in the ComputedStructureEntry so there is no need
                to pass it here.
                You are free to make up the keys for each property but make
                sure you are consistent for all structures that you add.
            weights (dict): optional
                The weight given to the structure when doing the fit. The key
                must match at least one of the given properties.
            supercell_matrix (ndarray): optional
                if the corresponding structure has already been matched to the
                ClusterSubspace prim structure, passing the supercell_matrix
                will use that instead of trying to re-match. If using this
                the user is responsible to have the correct supercell_matrix.
                Here you are the cause of your own bugs.
            site_mapping (list): optional
                site mapping as obtained by
                :code:`StructureMatcher.get_mapping`
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. If you pass this
                option, you are fully responsible that the mappings are correct!
            verbose (bool):
                if True, will raise warning for structures that fail in
                StructureMatcher, and structures that have duplicate corr vectors.
            raise_failed (bool): optional
                if True, will raise the thrown error when adding a structure
                that fails. This can be helpful to keep a list of structures that
                fail for further inspection.
            iter_id (int): optional
                Number of iteration when the structure is inserted. Default to 0.

        Returns:
            ComputedStructureEntry: entry with CE pertinent properties
        """
        processed_entry = super(DataWrangler, self)\
            .process_entry(entry, properties, weights,
                           supercell_matrix, site_mapping,
                           verbose, raise_failed)
        processed_entry.data["iter_id"] = iter_id
        return processed_entry

    def add_entry(
        self,
        entry,
        properties=None,
        weights=None,
        supercell_matrix=None,
        site_mapping=None,
        verbose=True,
        raise_failed=False,
        iter_id=0
    ):
        """Add a structure and measured property to the DataWrangler.

        The energy and properties need to be extensive (i.e. not normalized per atom
        or unit cell, directly from DFT).

        An attempt to compute the correlation vector is made and if successful the
        structure is succesfully added. Otherwise the structure is ignored.
        Usually failures are caused by the StructureMatcher in the given
        ClusterSubspace failing to map structures to the primitive structure.

        Same as StructureWrangler but refuses to insert symmetrically equivalent
        entries. It also records the iteration number when then entry was added.

        Args:
            entry (ComputedStructureEntry):
                A ComputedStructureEntry with a training structure, energy and
                properties
            properties (dict):
                Dictionary with a key describing the property and the target
                value for the corresponding structure. For example if only a
                single property {'energy': value} but can also add more than
                one, i.e. {'total_energy': value1, 'formation_energy': value2}.
                You are free to make up the keys for each property but make
                sure you are consistent for all structures that you add.
            weights (dict):
                the weight given to the structure when doing the fit. The key
                must match at least one of the given properties.
            supercell_matrix (ndarray): optional
                if the corresponding structure has already been matched to the
                ClusterSubspace prim structure, passing the supercell_matrix
                will use that instead of trying to re-match. If using this,
                the user is responsible for having the correct supercell_matrix.
                Here you are the cause of your own bugs.
            site_mapping (list): optional
                site mapping as obtained by StructureMatcher.get_mapping
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. If you pass this
                option, you are fully responsible that the mappings are correct!
            verbose (bool): optional
                if True, will raise warning regarding  structures that fail in
                StructureMatcher, and structures that have duplicate corr vectors.
            raise_failed (bool): optional
                if True, will raise the thrown error when adding a structure
                that  fails. This can be helpful to keep a list of structures that
                fail for further inspection.
            iter_id (int): optional
                Iteration number when this entry was inserted. Default to 0.
                Recommend providing.
        """
        processed_entry = self.process_entry(
            entry,
            properties,
            weights,
            supercell_matrix,
            site_mapping,
            verbose,
            raise_failed,
            iter_id
        )
        if processed_entry is not None:
            dupe_eid = self._check_duplicacy(entry,
                                             sm=self._subspace._site_matcher)
            if dupe_eid is None:
                self._entries.append(processed_entry)
                if verbose:
                    self._corr_duplicate_warning(self.num_structures - 1)
            else:
                if verbose:
                    warnings.warn("Provided entry duplicates with existing entry"
                                  f" number {dupe_eid}. Skipped.")

    def get_min_energy_by_composition(self, max_iter_id=None):
        """Get minimum energy by composition.

        This function provides quick tools to compare minimum DFT energies.
        Remember this is NOT energy above hull!

        Args:
            max_iter_id(int): optional
                Maximum iteration index included in the energy comparison. If none
                given, will read existing maximum iteration number.
        Returns:
            defaultdict.
        """
        min_e = defaultdict(lambda: np.inf)
        if max_iter_id is None:
            max_iter_id = self.max_iter_id
        for entry in self.entries:
            if entry.iter_id <= max_iter_id:
                # Normalize composition and energy to per prim.
                comp = Composition({k: v / entry.data["size"] for k, v
                                   in entry.structure.composition.items()})
                e = entry.energy / entry.data["size"]
                if e < min_e[comp]:
                    min_e[comp] = e
        return min_e

    def min_energies_difference_by_composition(self, iter_id1, iter_id2):
        """Compare minimum energy by composition.

        This is used to show whether the minimum energy per comp has converged
        or not, as a part of convergence criteria. We will only compare keys
        that exist in both older and newer iterations. If one composition
        appears in the older one but not the newer one, we will not claim
        convergence.

        Args:
            iter_id1, iter_id2(int):
                Records of minimum energy by composition, in 2 subsequent
                iterations.
                iter_id1 should be from a previous iteration, iter_id2 should be
                from a newer iteration.
        Return:
            float: maximum energy difference in eV/atom.
        """
        if iter_id2 <= iter_id1:
            raise ValueError("The 2nd arg must be larger than 1st arg!")
        min_e1 = self.get_min_energy_by_composition(max_iter_id=iter_id1)
        min_e2 = self.get_min_energy_by_composition(max_iter_id=iter_id2)
        diffs = []
        for comp in min_e2:
            if comp not in min_e1:
                return np.inf  # New composition appears.
            if not min_e2[comp] == np.inf and min_e1[comp] == np.inf:
                diffs.append(np.abs(min_e2[comp] - min_e1[comp]))
        if len(diffs) == 0:
            return np.inf
        return np.max(diffs) / len(self.cluster_subspace.structure)

