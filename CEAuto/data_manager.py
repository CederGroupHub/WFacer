"""DataManager.

This file includes a class to manage read and write in calculation 
dataframes, including sc_df, comp_df, fact_df.
It also handles conversion between formats of data, such as occupancy
to structure, to correlation vector, etc.

This file does not generate any data. It only manages data!
"""

__author__ = "Fengyu Xie"

import numpy as np
from copy import deepcopy

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .utils.occu_utils import (get_dim_ids_table,
                               occu_to_species_n)
from .utils.comp_utils import normalize_compstat
from .utils.frame_utils import load_dataframes, save_dataframes

from .wrappers import InputsWrapper
from .config_paths import (WRAPPER_FILE, OPTIONS_FILE,
                           SC_FILE, COMP_FILE, FACT_FILE)
from .comp_space import CompSpace

from smol.cofe.space.domain import get_allowed_species, get_site_spaces
from smol.cofe.wrangling.wrangler import StructureWrangler
from smol.moca.sublattice import Sublattice

import logging
log = logging.getLogger(__name__)


# TODO: rewrite this to attach StructureWrangler.
class DataManager(StructureWrangler):
    """DataManger class.

    Interfaces all CEAuto generated data, does insertion and deletion,
    but will not generate data.
    """
    def __init__(self, cluster_subspace):
        """Initialize DataManager

        Args:
            cluster_subspace(ClusterSubspace):
                An smol cluster subspace object to expand on.
        """
        super(DataManager, self).__init__(cluster_subspace)
        self.prim = cluster_subspace.structure

        self._dim_ids_table = {}  # Used for computing compositions.

    def _get_dim_ids_table(self, sc_mat):
        """Get a table of dimension indices of each species on each site."""
        sc_mat = np.array(sc_mat, dtype=int)
        sc_mat = tuple([tuple(row) for row in sc_mat.tolist()])
        if sc_mat not in self._dim_ids_table:
            unique_spaces = tuple(set(get_site_spaces(self.prim)))

            sc = self.prim.copy()
            sc.make_supercell(sc_mat)
            allowed_species = get_allowed_species(sc)
            sublattices = [Sublattice(site_space,
                           np.array([i for i, sp in enumerate(allowed_species)
                                     if sp == list(site_space.keys())]))
                           for site_space in unique_spaces]
            table = get_dim_ids_table(sublattices, active_only=False)
            self._dim_ids_table[sc_mat] = table
        return self._dim_ids_table[sc_mat]

    def _get_sublatt_resolved_composition(self, occupancy, sc_mat):
        """Get sublattice resolved composition (n format)."""
        table = self._get_dim_ids_table(sc_mat)
        return occu_to_species_n(occupancy, table)

    # Update these after ComputedStructureEntry update.
    def process_data(
        self,
        structure,
        properties,
        normalized=False,
        weights=None,
        supercell_matrix=None,
        site_mapping=None,
        verbose=False,
        raise_failed=False,
    ):
