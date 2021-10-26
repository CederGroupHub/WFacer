__author__="Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.

Algorithm based on: 

Ground state structures will also be added to the structure pool, but 
they are not added here. They will be added in the convergence checker
module.
"""

import logging
log = logging.getLogger(__name__)

import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product, chain

from .utils.sc_utils import enumerate_matrices
from .utils.math_utils import select_rows
from .utils.comp_utils import (check_comp_restriction,
                               get_Noccus_of_compstat)

from .ce_handler import CanonicalmcHandler


class StructureEnumerator:
    """Structure enumeration class."""
    def __init__(self, data_manager, history_wrapper):

        """Initialization.

        Args:
            data_manager(DataManager):
                The datamanager object to socket enumerated data.
            history_wrapper(HistoryWrapper):
                Wrapper containing previous CE fits.
        """

        self._dm = data_manager

        self.prim = self.inputs_wrapper.prim
        self.bits = self.inputs_wrapper.bits
        self.sublat_list = self.inputs_wrapper.sublat_list
        self.sl_sizes = self.inputs_wrapper.sl_sizes
        self.is_charged_ce = self.inputs_wrapper.is_charged_ce

        self.n_strs_enum = (self.inputs_wrapper.
                            enumerator_options['n_strs_enum'])
            
        self.transmat = self.inputs_wrapper.enumerator_options['transmat']
        sc_size = self.inputs_wrapper.enumerator_options['sc_size']
        if isinstance(sc_sizes, int):
            self.sc_sizes = [sc_size]
        self.max_sc_cond = (self.inputs_wrapper.
                            enumerator_options['max_sc_cond'])
        self.min_sc_angle = (self.inputs_wrapper.
                             enumerator_options['min_sc_angle'])

        # TODO: improve compspace to allow more sophisticated composition
        # constraints. (Smol edit work, not for here now.)
        self.comp_restrictions = (self.inputs_wrapper.
                                  enumerator_options['comp_restrictions'])
        self.comp_enumstep = (self.inputs_wrapper.
                              enumerator_options['comp_enumstep'])
        self.select_method = (self.inputs_wrapper.
                              enumerator_options['select_method'])
        self.n_strs_init = (self.inputs_wrapper.
                            enumerator_options['n_strs_init'])
        self.n_strs_add = (self.inputs_wrapper.
                           enumerator_options['n_strs_add'])

        self.handler_args = (self.inputs_wrapper.
                             enumerator_options['handler_args_enum'])

        self.ce = history_wrapper.last_ce

        if self.sc_size % self.comp_enumstep != 0:
            raise ValueError("Composition enumeration step can't divide " +
                             "supercell size.")

    @property
    def n_strs(self):
        """Number of enumerated structures.

        Returns:
           int.
        """
        return len(self.fact_df)

    @property
    def data_manager(self):
        """DataManager object.

        Returns:
           DataManager.
        """
        return self._data_manager

    @property
    def inputs_wrapper(self):
        """Inputs wrapper object.

        Returns:
           InputsWrapper.
        """
        return self.data_manager._iw

    @property
    def compspace(self):
        """Constained composition space."""
        if self._compspace is None:
            self._compspace = CompSpace(self.bits, self.sl_sizes)
        return self._compspace

    @property
    def sc_df(self):
        """Supercell matrices dataframe.

        If none yet, shall be enumerated.
        Return:
            pd.Dataframe:
                Supercell matrix dimension table.
        """
        return self._dm.sc_df

    @property
    def comp_df(self):
        """Compositions dataframe. 

        Gives a dimension table with sc_id, comp_id, and the composition.
        Return:
            pd.DataFrame:
                Dimension table containing all enumerated compositions, and
                their correponding supercell matrices.
        """
        return self._dm.comp_df

    @property
    def fact_df(self):
        """Fact dataframe.

        Stores all generated structures and their caluclated properties.
        Return:
            pd.Dataframe
        """
        return self._dm.fact_df

    def set_sc_matrices(self, matrices=[]):
        """Set supercell matrices in DataManager.

        Recommended not to call after 1st cycle.
        Args:
            matrices(List[3*3 ArrayLike[int]]): 
                Supercell matrices to insert
        """
        for m in matrices:
            if isinstance(m,np.ndarray):
                m = m.tolist()
            _ = self._dm.insert_one_supercell(m)

    def enumerate_sc_matrices(self):
        """Enumerate supercell matrices if nothing is present.

        Don't call this after the 1st cycle!
        Return:
            enumerated supercell matrices(3*3 List)
        """
        log.critical("**Supercells enumeration.")
        mats = enumerate_matrices(self.sc_size, self.prim.lattice,
                                  transmat=self.transmat,
                                  max_sc_cond=self.max_sc_cond,
                                  min_sc_angle=self.min_sc_angle)
        log_str = "**Enumerated Supercell matrices: \n"
        for m in mats:
            log_str += "  {}\n".format(m)
        log.info(log_str)

        self.set_sc_matrices(matrices=mats)
        return mats

    def enumerate_comps(self):
        """Enumerate Compositions under supercells.

        Not recommended to call after 1st cycle.
        Return:
            List of enumerated compositions by sublattice.
        """
        if len(self.comp_df)>0:
            log.warning("Attempt to set composition after 1st cycle. " +
                        "Do this at your own risk!")
            return

        if len(self.sc_df) == 0:
            self.enumerate_sc_matrices()

        log.critical("**Compositions enumeration.")
        all_scs = []
        for m in self.sc_df.matrix:
            all_scs.append(int(round(abs(np.linalg.det(m)))))
        all_scs = sorted(list(set(all_scs)))

        for scs in all_scs:
            ucoords = self.compspace.frac_grids(sc_size=
                                                scs // self.comp_enumstep)
            comps = (self.compspace.frac_grids(
                     sc_size=scs // self.comp_enumstep,
                     form='composition'))

            log.critical("****Enumerated {} compositions under matrix size {}."
                         .format(len(ucoords), scs))

            # TODO: in the future version, move restriction to CompSpace.
            for ucoord, comp in zip(ucoords, comps):
                if check_comp_restriction(comp,
                                          self.sl_sizes,
                                          self.comp_restrictions):
                    _ = self._dm.insert_one_comp(ucoord)

        return comps

    def generate_structures(self, n_par=4, keep_gs=True,
                            weight_by_comp=True,
                            iter_id=0):
        """Enumerate structures under all different (sc_matrix, composition).

        The eneumerated structures will be deduplicated and selected based on
        CUR decomposition score to maximize sampling efficiency.

        Will run on initalization mode if no previous enumearation present,
        and on addition mode if previous enum present. These have different
        selection criterion.

        Note: Please check TimeKeeper to avoid duplicate generation in the
              same cycle.

        Args:
            n_par(int): optional
                Number of parallel handler processes. Default to 4.
            keep_gs(Bool): optional
                If true, will always add current ground states to pool.
                Default is True.
            weight_by_comp(Boolean): optional
                If true, will choose to add number of occupancies of each
                composition as proportional to the combinatoric configuration
                numbers under each composition.
                Default is True.
            iter_id(int):
                Iteration index. Use that from TimeKeeper unless you remember
                it yourself.

        Return:
            DataFrame, newly generated piece of fact_table.
        """
        # Generate Structures, and add to dataframes! (Takes in compstat)
        def raw_sample_under_sc_comp(ce, sc_mat, compstat, **handler_args):
            # Sample a single composition. Do selection later.

            sc_size = int(round(abs(np.linalg.det(sc_mat))))
            log.debug("****Supercell size: {}, compositon stat: {}."
                      .format(sc_size, compstat))
            tot_noccus = get_Noccus_of_compstat(compstat, scale_by=sc_size)
            # Handler will automatically initialize an occupation
            handler = CanonicalmcHandler(ce, sc_mat, compstat, **handler_args)
            # Handler will de-freeze from 0K, creating samples. for all runs.
            return handler.get_unfreeze_sample(), tot_noccus

        if len(self.comp_df) == 0:
            self.enumerate_comps()

        # Parallelize generation.
        pool = mp.Pool(n_par)

        comp_df = self.comp_df
        sc_df = self.sc_df
        # Some product of sc and comp may not allow integer composition.
        # Filter these out.
        all_sc_comps = []
        all_sc_comp_ids = []
        for (sc_id, mat), (comp_id, cstat) in \
            product(zip(sc_df.sc_id, sc_df.matrix),
                    zip(comp_df.comp_id, comp_df.cstat)):
            scs = int(round(abs(np.linalg.det(mat))))
            ucoords = np.array(list(chain(*[sl[: -1] for sl in cstat])))
            if np.allclose(ucoords, np.round(ucoords * scs) / scs, atol=1E-4):
                all_sc_comps.append((mat, cstat))
                all_sc_comp_ids.append((sc_id, comp_id))

        log.critical("**Generating structure samples under compositions.")
        all_occus_n = pool.map(lambda sc, comp:
                               raw_sample_under_sc_comp(self.ce, sc, comp,
                                                        **self.handler_args),
                               all_sc_comps)

        # Filter raw samples by weights.
        all_occus, comp_weights = list(zip(*all_occus_n))
        W_max = max(1, comp_weights)
        n_selects = [max(min(len(occus), 5),
                         int(round(w / W_max * len(occus))))
                     for occus, w in all_occus_n]
        # Take at least 3 samples each composition point.

        if weight_by_comp:
            if keep_gs:
                all_occus_filt = [[occus[0]] +
                                   random.sample(occus, n-1)
                                  for occus, n in
                                  zip(all_occus, n_selects)]
            else:
                all_occus_filt = [random.sample(occus, n)
                                  for occus, n in
                                  zip(all_occus, n_selects)]
        else:
            all_occus_filt = all_occus
        

        log.info("**Deduplicating structures by pre_insertion.")
        inserted_eids = []  # List of ints
        is_gs = []   # List of booleans

        # Fork out a new datamanager to deduplicate.
        dm_copy = self._dm.copy()
        for (sc_id, comp_id), occus in zip(all_sc_comp_ids, all_occus):
            for oid, occu in enumerate(occus):
                # Dedup
                old_eid = dm_copy.find_entry_id_from_occu(occu, sc_id=sc_id,
                                                          comp_id=comp_id)
                if old_eid is None:
                    _, _, new_eid = dm_copy.insert_one_occu(occu, sc_id=sc_id,
                                                            comp_id=comp_id,
                                                            iter_id=iter_id,
                                                            module_name='enum')
                    inserted_eids.append(new_eid)
                    is_gs.append((oid == 0))

        log.info("**Generated {} new deduplicated structures."
                 .format(len(inserted_eids)))

        # Compute indices to keep if keep_gs.
        keep = []
        if keep_gs:
            keep = [i for i in range(len(is_gs)) if is_gs[i]]

        # Selection.
        femat = dm_copy.fact_df.loc[dm_copy.entry_id.isin(inserted_eids),
                                    'ori_corr'].tolist()
        old_femat = self.fact_df.ori_corr.tolist()

        if iter_id <= 0:  # Do initialization
            n_enum = self.n_strs_init
        else:  # Do addition
            n_enum = self.n_strs_add
        if n_enum > len(femat):
            log.warning("**Number of deduplicated structures fewer than " +
                        "the number you wish to enumerate!")

        selected_rids = select_rows(femat,
                                    n_select=min(n_enum, len(femat)),
                                    old_femat=old_femat,
                                    method=self.select_method,
                                    keep=keep)
        unselected_eids = [inserted_eids[i] for i in range(len(femat))
                           if i not in selected_rids]
        log.critical("**Added {}/{} new structures."
                     .format(len(selected_rids), len(femat)))

        # Remove unselected entree.
        dm_copy.remove_entree_by_id(unselected_eids) 

        # Update to real DataManager.
        self._dm = dm_copy 
 
        filt = (self.fact_df.iter_id == iter_id)
        return self.fact_df.loc[filt_, :]

    # If you want to clear enumeration, use removal methods in DataManager.

# Don't save or load this object. Save or load data with DataManager,
# InputsWrapper, HistoryWrapper, TimeKeeper, etc.
