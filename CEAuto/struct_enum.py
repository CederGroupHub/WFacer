__author__="Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.
Ground state structures will also be added to the structure pool, but 
they are not added here. They will be added in the convergence checker
module.
"""

import warnings
import random
from copy import deepcopy
import numpy as np
import pandas as pd
import os
import json
import multiprocessing as mp
from tqdm import tqdm

from monty.json import MSONable

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe.extern.ewald import EwaldTerm
from smol.cofe.space.clusterspace import ClusterSubspace
from smol.cofe.space.domain import get_allowed_species, get_species, Vacancy
from smol.cofe.expansion import ClusterExpansion
from smol.moca import CanonicalEnsemble, Sampler, CompSpace

from .utils.sc_utils import enumerate_matrices
from .utils.math_utils import select_rows,combinatorial_number
from .utils.serial_utils import serialize_comp,deser_comp
from .utils.comp_utils import check_comp_restriction

from .data_manager import DataManager 
from .wrappers import InputsWrapper
from .ce_handler import CanonicalMCHandler

from .config_paths import *


class StructureEnumerator(MSONable):
    """Structure enumeration class."""
    def __init__(self, data_manager, history_wrapper):

        """Initialization. (Not recommended.)

        Args:
            data_manager(DataManager):
                The datamanager object to socket enumerated data.
            history_wrapper(Hisotory_wrapper):
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
        self.basis_type = self.inputs_wrapper.enumerator_options['basis_type']
            
        self.transmat = self.inputs_wrapper.enumerator_options['transmat']
        self.sc_size = self.inputs_wrapper.enumerator_options['sc_size']
        self.max_sc_cond = (self.inputs_wrapper.
                            enumerator_options['max_sc_cond'])
        self.min_sc_angle = (self.inputs_wrapper.
                             enumerator_options['min_sc_angle'])

        #TODO: improve compspace to allow more sophisticated composition
        # constraints.
        self.comp_restrictions = (self.inputs_wrapper.
                                  enumerator_options['comp_restrictions'])
        self.comp_enumstep = (self.inputs_wrapper.
                              enumerator_options['comp_enumstep'])
        self.select_method = (self.inputs_wrapper.
                              enumerator_options['select_method'])

        self.ce = history_wrapper.last_ce

        if self.sc_size % self.comp_enumstep!=0:
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
            self._compspace = CompSpace(self.bits,self.sl_sizes)
        return self._compspace

    @property
    def sc_df(self):
        """
        Supercell matrices used for structural enumeration. If none yet, shall be 
        enumerated.
        Return:
            Supercell matrix dimension table. Is a pd.Dataframe.
        """
        return self._dm.sc_df

    @property
    def comp_df(self):
        """
        Enumerate proper compositions under each supercell matrix, and gives a dimension table with
        sc_id, comp_id, and the exact composition.
        Return:
            Dimension table containing all enumerated compositions, and their correponding supercell
            matrices. Is a pandas.DataFrame.
        """
        return self._dm.comp_df

    @property
    def fact_df(self):
        """
        Fact dataframe, storing all generated structures and their caluclated properties.
        """
        return self._dm.fact_df

    def set_sc_matrices(self,matrices=[],add_new=False):
        """
        Interface method to preset supercell matrices before enumeration. If no preset
        is given, supercell matrices will be automatically enumerated. Otherwise will
        skip.

        Not frequently called.
        Args:
            matrices(List[3*3 ArrayLike]): 
                Supercell matrices to insert
            add_new(Boolean):
                If true, will add matices to the supercell dataframe, even if the
                current dataframe is not empty.
        """
        if len(self.sc_df)>0 and not add_new:
            warnings.warn("Attempt to set matrices after matrix enumeration. Skipping")
            return

        for m in matrices:
            if isinstance(m,np.ndarray):
                m = m.tolist()
            self._dm.insert_one_supercell(m)

    def enumerate_sc_matrices(self,add_new=False):
        """
        Enumerate supercell matrices if nothing is present.
        Args:
            add_new(Boolean):
                If true, will add matices to the supercell dataframe, even if the
                current dataframe is not empty.
        Return:
            enumerated supercell matrices(3*3 List)
        """
        print("**Start supercell enumeration.")
        mats = enumerate_matrices(self.sc_size,self.prim.lattice,\
                                  transmat=self.transmat,\
                                  max_sc_cond = self.max_sc_cond,\
                                  min_sc_angle = self.min_sc_angle)
        print("**Enumerated Supercell matrices: \n")
        for m in mat:
            print("  {}".format(m))
        if len(self.sc_df)==0:
            self.set_sc_matrices(matrices=mats,add_new = add_new)
        return mat

    def enumerate_comps(self,add_new=False):
        """
        Enumerate Compositions under supercells.
        Args:
            add_new(Boolean):
                If true, will add composition to the comp dataframe, even if the
                current dataframe is not empty.
        Return:
            List of enumerated compositions by sublattice.
        """
        if len(self.comp_df)>0 and not add_new:
            warnings.warn("Attempt to set composition after compositions enumeration. Skipping")
            return

        if len(self.sc_df) == 0:
            self.enumerate_sc_matrices()

        print("**Start compositions enumeration")
        for sc_id,m in zip(self.sc_df.sc_id,self.sc_df.matrix):
            scs = int(round(abs(np.linalg.det(m))))
            ucoords = self.compspace.frac_grids(sc_size=scs//self.comp_enumstep)
            comps = self.compspace.frac_grids(sc_size=scs//self.comp_enumstep,form='composition')

            print("****Enumerated {} compositions under matrix {}."\
                  .format(len(ucoords),m))
            for ucoord,comp in zip(ucoords,comps):
                if check_comp_restriction(comp,self.sl_sizes,self.comp_restrictions):
                    _,comp_id = self._dm.insert_one_comp(ucoord,sc_id=sc_id)

        return comps

    def generate_structures(self, n_par=4, keep_gs = True,weight_by_comp=True):
        """
        Enumerate structures under all different key = (sc_matrix, composition). The eneumerated structures
        will be deduplicated and selected based on CUR decomposition score to maximize sampling efficiency.
        Will run on initalization mode if no previous enumearation present, and on addition mode if previous
        enum present.
        Inputs:
            n_par(int):
                Number of parallel handler processes. Default to 4.
            keep_gs(Bool):
                If true, will always select current ground states. Default is True.
            weight_by_comp(Boolean):
                If true, will generate more raw samples for compositions with more possible occupations. The ratio of raw sample numbers is based on # of all occu possibilities.
                Default is True.
    
        Return:
            DataFrame, newly generated piece of fact_table.
        """
        if self._dm.schecker.after('enum'):       
            print("Currently at iteration number {}, already generated structures.") 
            filt = self.fact_df.iter_id == self.cur_iter_id
            return self.fact_df.loc[filt_,:]

        #Generate Structures, and add to dataframes!(Takes in compstat)
        def raw_sample_under_sc_comp(ce,sc_mat,compstat,**handler_args):
            """
            Sample a single composition point. Do selection later.
            """
            sc_size = int(round(abs(np.linalg.det(sc_mat))))
            print("****Supercell size: {}, compositon stat: {}.".format(sc_size,compstat))
            tot_noccus = get_Noccus_of_compstat(compstat,scale_by=sc_size)
            #Handler will automatically initialize an occupation
            handler = CanonicalMCHandler(ce,sc_mat,compstat,**handler_args)
            #Handler will de-freeze from 0K, creating samples. for all runs.
            return handler.get_unfreeze_sample(),tot_noccus

        if len(self.comp_df)==0:
            print("**Compositions not enumerated. Enumerate first.")
            self.enumerate_comps()

        #parallized.
        pool = mp.Pool(n_par)

        comp_df = self.comp_df.merge(self.sc_df,on='sc_id',how='left')
        all_sc_comps = list(zip(comp_df.matrix,comp_df.compstat))
        all_sc_comp_ids = list(zip(comp_df.sc_id,comp_df.comp_id))

        print("**Generating raw samples under compositions.")
        all_occus_n = pool.map(lambda sc,comp: raw_sample_under_sc_comp(self.ce,sc,comp,**self.handler_args),\
                             all_sc_comps)

        #Filter raw samples by weights.
        all_occus,comp_weights = list(zip(*all_occus_n))
        W_max = max(1,comp_weights)
        n_selects = [max(3,int(round(w/W_max*len(occus)))) for occus,w in all_occus_n]
        if weight_by_comp:
            if keep_gs:
                all_occus_filt = [[occus[0]]+random.sample(occus,min(n,len(occus))-1) \
                              for occus,n in zip(all_occus,n_selects)]
            else:
                all_occus_filt = [random.sample(occus,min(n,len(occus))) \
                              for occus,n in zip(all_occus,n_selects)]
        else:
            all_occus_filt = all_occus
        

        print("**Deduplicating structures by pre_insertion.")
        inserted_eids = []  #List of ints
        is_gs = []   #List of booleans

        #For safety, forking out a new datamanager
        dm_copy = deepcopy(self._dm)
        for (sc_id,comp_id),occus in tqdm(zip(all_sc_comp_ids,all_occus)):
            for oid,occu in enumerate(occus):
                #dedup
                old_eid = dm_copy.find_entry_id_from_occu(occu,sc_id=sc_id,\
                                                               comp_id=comp_id)
                if old_eid is None:
                    _,_,new_eid = dm_copy.insert_one_occu(occu,sc_id=sc_id,\
                                                               comp_id=comp_id,\
                                                               module_name='enum')
                    inserted_eids.append(new_eid)
                    is_gs.append((oid==0))

        print("**Generated {} new deduplicated structures.".format(len(inserted_eids)))

        #Compute indices to keep.
        keep = []
        if keep_gs:
            keep = [i for i in range(len(is_gs)) if is_gs[i]]

        #selection.
        femat = dm_copy.fact_df.loc[dm_copy.entry_id.isin(inserted_eids),'ori_corr'].tolist()
        old_femat = self.fact_df.ori_corr.tolist()
        if self.n_strs_enum>len(femat):
            warnings.warn("**Number of deduplicated structures fewer than the number you \
                           wish to enumerate!")
        selected_rids = select_rows(femat,n_select=min(self.n_strs_enum,len(femat)),\
                                          old_femat=old_femat,\
                                          method=self.select_method,\
                                          keep=keep)
        unselected_eids = [inserted_eids[i] for i in range(len(femat)) \
                           if i not in selected_rids]

        #Remove unselected entree.
        dm_copy.remove_entree_by_id(unselected_eids) 

        #Update to self attribute.
        self._dm = dm_copy 
 
        filt = self.fact_df.iter_id == self.cur_iter_id
        return self.fact_df.loc[filt_,:]      
            
    def clear_current_iter(self,flush_and_reload=True):
        """
        Clear all enumerated data of the current iteration. Can only be used with python interface.
        Use this at your own risk! 
        Args:
            flush_and_reload(Bool):
                If true, will flush the cleared dataframes and reload the status checker.
                Default to true.
        """
        print("All enumerated entree in the current iteration will be cleared. \
               Current iteration number: {}. Proceed?[y/n]".format(self.cur_iter_id))
        choice = raw_input().lower()
        if choice == 'y':
            self._dm.remove_entree_by_iters_modules(iter_ids=[self.cur_iter_id],\
                                                    modules=['enum'],\
                                                    flush_and_reload=flush_and_reload)
        elif choice != 'n':
            raise ValueError("Please respond with [Y/N] or [y/n]!")

    def clear_all(self):
        """
        Clear all data from all iterations. Use this at your own risk!!
        Args:
            flush_and_reload(Bool):
                If true, will flush the cleared dataframes and reload the status checker.
                Default to true.
        """
        print("Any enumerated entree in all iterations will be cleared. Proceed?[y/n]")
        choice = raw_input().lower()
        if choice == 'y':
            self._dm.reset(flush_and_reload=flush_and_reload)
        elif choice != 'n':
            raise ValueError("Please respond with [Y/N] or [y/n]!")


    def auto_save(self,sc_file=SC_FILE,comp_file=COMP_FILE,fact_file=FACT_FILE):
        """
        Automatically save dataframes into pre-configured paths.
        """
        self._dm.auto_save(sc_file=sc_file,comp_file=comp_file,fact_file=fact_file)

    @classmethod
    def auto_load(cls, data_manager,
                  options_file=OPTIONS_FILE,
                  ce_history_file=CE_HISTORY_FILE):
        """
        This method is the recommended way to initialize this object.
        It automatically reads all setting files with FIXED NAMES.
        YOU ARE NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!
        Args:
            data_manager(DataManager):
                Data manager object to read and save enumerated
                structures.
            options_file(str):
                path to options file. Options must be stored as yaml
                format. Default: 'options.yaml'
            ce_history_file(str):
                path to cluster expansion history file.
                Default: 'ce_history.json'
        Returns:
             Structure enumerator object.
        """
        options = InputsWrapper.auto_load(options_file=options_file,
                                          ce_history_file=ce_history_file)

        return cls(options.prim,
                   bits=options.bits,
                   sublat_list=options.sublat_list,
                   is_charged=options.is_charged_ce,
                   previous_ce=options.last_ce,
                   data_manager=data_manager,
                   **options.enumerator_options)
