"""
This file includes a class to manage read and write in calculation 
dataframes, including sc_df, comp_df, fact_df.
It also handles conversion between formats of data, such as occupancy
to structure, to correlation vector, etc.

This file does not generate any data. It only does data management!
"""

__author__ = "Fengyu Xie"

import numpy as np
import pandas as pd
import warnings 

from pymatgen.analysis.structure_matcher import StructureMatcher

from .utils.serial_utils import serialize_comp, deser_comp
from .utils.occu_utils import structure_from_occu, occu_to_species_stat,\
                              get_sc_sllist_from_prim
from .utils.comp_utils import normalize_compstat

from .comp_space import CompSpace
from .inputs_wrapper import InputsWrapper #Used in auto_load()
from .status_check import StatusChecker #Used in auto_load()

from .config_paths import *

class DataManager:
    """
    DataManger class to interface all CEAuto generated data.
    You are not recommended to call __init__ directly.
    """
    def __init__(self,prim,bits,sublat_list,subspace,schecker):
        #These attributes are used to convert between occupation/structure/composition.
        #InputsWrapper can read all settings from default or options file, and passes them
        #down to all other modules.
        """
        Args:
            prim(Structure):
                primitive cell containing all comp space info and lattice
                info. Make sure it is the same with the one used to initialize
                CE.
            bits(List[List[Specie|Vacancy]]):
                Species occupying each sublattice.
            sublat_list(List[List[int]]):
                Indices of PRIMITIVE CELL sites in the same sublattices. 
            subspace(Clustersubspace):
                The cluster subspace used to featurize structures.
            schecker(StatusChecker):
                A status checker object, can check current iteration
                number and which step of CE cycle we are in, based on 
                the current data and history files.
    
            Since this DataManager can not generate or calculate data, you must 
            provide this necessary arguments to initialize it.
        """

        self._prim = prim
        self._bits = bits
        self._sublat_list = sublat_list
        self._subspace = subspace

        self._compspace = CompSpace(self._bits,[len(sl) for sl in self._sublat_list])  

        self._sc_df = None
        self._comp_df = None
        self._fact_df = None

        self._schecker = schecker

        self._sc_load_path = SC_FILE
        self._comp_load_path = COMP_FILE
        self._fact_load_path = FACT_FILE

    @property
    def cur_iter_id(self):
        """
        Get the current iteration number from the fact table.
        """
        return self._schecker.cur_iter_id 

    @property
    def sc_df(self):
        """
        Supercell dataframe. Has two columns:
            sc_id(int):
                indices of supercells, starting from 0.
            matrix(List[List[int]]):
                Supercell matrices in 2D list.
        """
        if self._sc_df is None:
            self._sc_df = pd.DataFrame(columns=['sc_id','matrix'])
        return self._sc_df
 
    @property
    def comp_df(self):
        """
        Compositions dataframe. Has following columns:
            comp_id(int):
                indices of compositions, starting from 0
            sc_id(int):
                indices of supercells to which the compositions are bound to.
                For example, a supercell of size 2 can not have {'A':0.333,'B':0.667},
                etc.
            ucoord(List[float]):
                Normalized, unconstrained coordinates in the compositional space.
                (Normalized by supercell size)
            ccoord(List[float]):
                Normalized, constrained coordinates in the compositional space.
                (Normalized by supercell size)
            comp(List[Composition]):
                Normalized compositions by each sublattice.
                (Normalized by number of sublattice sites in supercell)
            cstat(List[List[float]]):
                Normalized species statistics table by each sublattice.
                (Normalized by supercell size)
            nondisc(List[float]):
                Non discriminative composition vector, sums same specie on
                different sublattices up. This is the only physical composition.
            eq_occu(List[int]):
                MC equilibrated occupation array in encoded form.
        """
        if self._comp_df is None:
            self._comp_df = pd.DataFrame(columns=['comp_id','sc_id',\
                                                  'ucoord','ccoord',\
                                                  'comp','cstat',\
                                                  'nondisc',\
                                                  'eq_occu'])
        return self.._comp_df

    @property
    def fact_df(self):
        """
        Fact table, containing all computed entrees.
        Columns:
            entry_id(int):
                Index of an entry, containing the computation result of a enumerated 
                structure.
                Starts from 0.
            sc_id(int):
                 Index of supercell matrix in the supercell matrix dimension table
            comp_id(int):
                Index of composition in the composition dimension table
            iter_id(int):
                Specifies in which iteration this structure is added into calculations.
                Both structure enumerator and ground state checker can add to fact
                table.
            module(str):
                Specifying the module name that generated and added this entry into the
                fact table. Can be 'enum' or 'gscheck'
            ori_occu(List of int):
                Original occupancy as it was enumerated. (Encoded, and turned into list)
            ori_corr(List of float):
                Original correlation vector computed from ori_occu. (Turned into list)
            calc_status(str):
                A string of two characters, specifying the calculation status of the current entry.
                'NC': not calculated.(not submitted or waiting.)
                'CC': Calculation written, but calculation not finished.
                'CL': calculation finished.
                'CF': calculated, and failed (or exceeded wall time).
                'AF': assignment failed. For example, not charge neutral after charge assignment.
                'MF': mapping failed, structure failed to map into original lattice or featurize.
                'SC': successfully calculated, mapped and featurized.
                Any other status other than 'SC' will cause the following columns to be None.
            map_occu(List of int or None):
                Mapped occupancy after DFT relaxation.(Encoded, and turned into List)
            map_corr(List of float or None):
                Mapped correlation vector computed from map_occu. (Turned into List)
            e_prim(float or None):
                DFT energies of structures, normalized to ev/prim.
            other_props(Dict or None):
                Other SCALAR properties to expand. If not specified, only e_prim will be expanded.
                In format: {'prop_name':prop_value,...}
        """
        if self._fact_df is None:
            self._fact_df = pd.DataFrame(columns=['entry_id','sc_id','comp_id',\
                                                  'iter_id','module',\
                                                  'ori_occu','ori_corr',\
                                                  'calc_status',\
                                                  'map_occu','map_corr',\
                                                  'e_prim','other_props'])
        return self._fact_df


    @property
    def fact_df_with_structures(self):
        """
        Returns a fact table with column 'ori_str' and 'map_str',
        storing the original, and the mapped structure object,
        respectively.
        This table will not be saved for the sake of disk space.
        """
        fact = self.fact_df.copy()
        fact = fact.merge(self.sc_df,how='left',on='sc_id')
        #pd.isna must be used, because np.nan may not be considered is None.
        fact['ori_str'] = fact.apply(lambda r: \
                                     structure_from_occu(r['ori_occu'],self._prim,r['matrix']) \
                                     if not pd.isna(r['ori_occu']) else None,axis=1)
        fact['map_str'] = fact.apply(lambda r: \
                                     structure_from_occu(r['map_occu'],self._prim,r['matrix']) \
                                     if not pd.isna(r['map_occu']) else None,axis=1)
        return fact

    def find_sc_id(self,sc_mat):
        """
        Find a supercell index from supercell dataframe by its matrix.
        If not present, will return None.
        Args:
            sc_mat(3*3 List[List[int]]):
                supercell matrix to query
        Returns:
            sc_id(int) or None.
        """
        new_sc = self.prim.copy()
        new_sc.make_supercell(sc_mat)

        sm = StructureMatcher()
        for old_id,old_mat in zip(self.sc_df.sc_id,self.sc_df.matrix):
            old_sc = self.prim.copy()
            old_sc.make_supercell(old_mat)
            if sm.fit(new_sc,old_sc):
                return old_id

        return None
        
    def insert_one_supercell(self,new_sc_mat):
        """
        Insert one supercell to the supercell dataframe.
        Will automatically de-duplicate before inserting.
        The new insertion will always have an index following the last one in dataframe.
        Args:
            new_sc_mat(List[List[int]):
                supercell matrix to insert. MUST BE LIST, NOT np.ndarray!
        Returns:
            int: sc_id of inserted supercell.
        """
        if np.array(new_sc_mat).shape!=(3,3):
            raise ValueError("Wrong supercell matrix input!")

        oid = self.find_sc_id(new_sc_mat)
        if oid is not None:
            warnings.warn("Matrix {} found in previous supercell with index: {}."\
                          .format(new_sc_mat,oid))
            return oid

        sc_id = self.sc_df.sc_id.max()+1 if len(self.sc_df)>0 else 0
        self._sc_df = self._sc_df.append({'sc_id':sc_id, 'matrix':new_sc_mat},\
                                          ignore_index=True)
        return sc_id


    def find_comp_id(self,comp,sc_id=None,sc_mat=None,comp_format='unconstr'):
        """
        Find index of a composition in the composition dataframe.
        If not present, will return None.
        Args:
            comp(Undefined format):
                composition by sublattices. Format defined by format
                parameter.
            sc_id(int):
                Bounded supercell matrix index. Optional, but can't be
                both None with sc_mat.
            sc_mat(List[List[int]]):
                Bounded supercell matrix. Optional, but can't be both
                None with sc_id.
            comp_format(str):
                Specifies format of the input composition. Can be:
                'unconstr','constr','compstat','composition'
        Returns:
            int: comp_id or None
        """
        if sc_id is None and sc_mat is None:
            raise ValueError("Arguments sc_id and sc_mat can not be both None.")       

        sc_id = sc_id or self.find_sc_id(sc_mat)
        if sc_id is None:
            return None

        ucoord = self._compspace.translate_format(comp,\
                                                  from_format=comp_format,\
                                                  to_format='unconstr').tolist()      

        dupe_filt_ = (self.comp_df.sc_id == sc_id) \
                     & (np.isclose(self.comp_df.ucoord.tolist(),ucoord).any(axis=1))

        for i,dupe in enumerate(dupe_filt_):
            if dupe:
                return self.comp_df.iloc[i]['comp_id']

        return None

    def insert_one_comp(self,new_comp,sc_id=None,sc_mat=None,comp_format='unconstr'):
        """
        Insert a composition into composition dataframe. If it is bound to a new 
        supercell matrix, the supercell will also be inserted.
        Will be deduplicated automatically upon insertion.
        Args:
            new_comp(Undefined format):
                composition to insert. Format depends on comp_format argument. Usaually
                an unconstrained composition coordinate.
                It is your responsibility to normalize this composition as required 
                in self.comp_df. Also, you are obliged to check whether this composition
                can be integer in your specified supercell size.
                The composition formats that satisfy the mentioned requirements can be
                generated by CompSpace.frac_grids()
            sc_id(int):
                Index of supercell matrix. Optional, but sc_id argument and sc_mat
                argument can't be both None. If you know the sc_id, you should provide
                it, because this saves time in parsing the supercells dataframe.
            sc_mat(List[List[int]):
                supercell matrix to insert. MUST BE LIST, NOT np.ndarray! Can not be
                both None with sc_id.
            comp_format(str):
                Specifies the input composition format. Can be 'unconstr','constr',
                'compstat','composition'.
        Returns:
            sc_id(int), comp_id(int):
                Indices of the inserted supercell and composition.
        """
        sc_id = sc_id or self.insert_one_supercell(sc_mat)
        o_cid = self.find_comp_id(new_comp,sc_id=sc_id,comp_format=comp_format)
        if o_cid is not None:
            warnings.warn("Composition {}({}) found in previous composition with index: {}."\
                          .format(new_comp,comp_format,o_cid))
            return sc_id, o_cid

        #Convert to list for proper serialization
        ucoord = self._compspace.translate_format(new_comp,\
                                                  from_format=comp_format,\
                                                  to_format='unconstr').tolist()      
        ccoord = self._compspace.translate_format(new_comp,\
                                                  from_format=comp_format,\
                                                  to_format='constr').tolist()      
        comp = self._compspace.translate_format(new_comp,\
                                                from_format=comp_format,\
                                                to_format='composition')
        cstat = self._compspace.translate_format(new_comp,\
                                                 from_format=comp_format,\
                                                 to_format='compstat').tolist()      
        nondisc = self._compsace.translate_format(new_comp,\
                                                  from_format=comp_format,\
                                                  to_format='nondisc').tolist()
       
        comp_id = self.comp_df.comp_id.max()+1 if len(self.comp_df)>0 else 0

        self._comp_df = self._comp_df.append({'comp_id':comp_id,'sc_id':sc_id,\
                                              'ucoord':ucoord,'coord':ccoord,\
                                              'cstat':cstat,'comp':comp,\
                                              'nondisc':nondisc,\
                                              'eq_occu':None},ignore_index=True)
        return sc_id, comp_id

    def find_entry_id_from_occu(self,occu,comp_id=None,comp=None,comp_format='unconstr',\
                          sc_id=None,sc_mat=None):
        """
        Find the entry index from an encoded occupation array.
        Args:
            occu(1D ArrayLike of ints):
                Encoded occupation array.
            comp_id(int):
                Compostion index in self.comp_df. Optional.
            comp(Undefined format):
                Normalized compositions by each sublattice.
                Format depends on comp_format parameter.
                It is your responsibility to check whether
                this is correctly formatted and normalized,
                as required in self.comp_df. Optional.
            comp_format(str):
                The format of given composition. Can be 
                'unconstr','constr','composition','compstat'.
                Optional. Should be provided together with 
                comp parameter.
            sc_id(int):
                Index of supercell matrix. Optional, but sc_id argument and sc_mat
                argument can't be both None. If you know the sc_id, you should provide
                it, because this saves time in parsing the supercells dataframe.
            sc_mat(List[List[int]):
                supercell matrix to insert. MUST BE LIST, NOT np.ndarray! Can not be
                both None with sc_id.          
            If you want to insert a calculated occupation or structure, you can pass
            down keys to calculated_info, and may overwrite iteration id with an additional
            iter_id argument.

        If you have the sc_id and the comp_id, you should provide them, because this
        will save a lot of time!!
        Returns:
            entry_id(int):
                index in fact_df of this new entry. If not found, will return None.
        """
        if sc_id is None and sc_mat is None:
            raise ValueError("Arguments sc_id and sc_mat can not be both none.")

        sc_id = sc_id or self.find_sc_id(sc_mat)
        sc_mat = sc_mat or self.sc_df[self.sc_df.sc_id==sc_id]\
                           .reset_index().iloc[0]['matrix']
        if sc_id is None:
            return None

        if comp_id is None and comp is None:
            #Get compositional statistics.(Must be normalized)
            sc_size = int(round(abs(np.linalg.det(sc_mat))))
            sc_sublat_list = get_sc_sllist_from_prim(self._sublat_list,sc_size=sc_size)
            cstat = occu_to_species_stat(occu,self._bits,sc_sublat_list)
            cstat = normalize_compstat(cstat,sc_size=sc_size)

            ucoords = self._compspace.translate_format(cstat,\
                                                       from_format='compstat',\
                                                       to_format='unconstr').tolist()

            comp_id = self.find_comp_id(ucoords,sc_id=sc_id)

        elif comp_id is None:
            ucoords = self._compspace.translate_format(comp,\
                                                       from_format=comp_format,\
                                                       to_format='unconstr').tolist()

            comp_id = self.find_comp_id(ucoords,sc_id=sc_id)      

        if comp_id is None:
            return None

        filt_ = (self.fact_df.sc_id == sc_id) & (self.fact_df.comp_id == comp_id)
        fact = self.fact_df[filt_].merge(self.sc_df,on='sc_id',how='left').reset_index()

        sm = StructureMatcher()
        s_new = structure_from_occu(occu,self._prim,sc_mat)

        for i in range(len(fact)):
            #Check duplicacy with structure matcher
            sc_mat_old = fact.iloc[i]['matrix']
            occu_old = fact.iloc[i]['ori_occu']
            s_old = structure_from_occu(occu_old,self._prim,sc_mat_old)

            if sm.fit(s_old,s_new):
                return fact.iloc[i]['entry_id']

        return None


    def insert_one_occu(self, occu, comp_id=None, comp=None, comp_format='ucoord',\
                              sc_id=None, sc_mat=None,\
                              module_name='enum',**calculated_info):
        """
        Insert one entry from an encoded occupation array.
        Args:
            occu(1D ArrayLike of ints):
                Encoded occupation array.
            comp_id(int):
                Compostion index in self.comp_df. Optional.
            comp(Undefined format):
                Normalized compositions by each sublattice.
                Format depends on comp_format parameter.
                It is your responsibility to check whether
                this is correctly formatted and normalized,
                as required in self.comp_df. Optional.
            comp_format(str):
                The format of given composition. Can be 
                'unconstr','constr','composition','compstat'.
                Optional. Should be provided together with 
                comp parameter.
            sc_id(int):
                Index of supercell matrix. Optional, but sc_id argument and sc_mat
                argument can't be both None. If you know the sc_id, you should provide
                it, because this saves time in parsing the supercells dataframe.
            sc_mat(List[List[int]):
                supercell matrix to insert. MUST BE LIST, NOT np.ndarray! Can not be
                both None with sc_id.          
            module_name(str):
                Name of the module that executes this insertion. For the current version,
                can be 'enum' or 'gs'. Default to enum.
            If you want to insert a calculated occupation or structure, you can pass
            down keys to calculated_info, and may overwrite iteration id with an additional
            iter_id argument.

        If you have the sc_id and the comp_id, you should provide them, because this
        will save a lot of time!!
        Returns:
            sc_id(int),comp_id(int),entry_id(int):
                indices in sc_df,comp_df,fact_df of this new entry
        """
        if sc_id is None and sc_mat is None:
            raise ValueError("Arguments sc_id and sc_mat can not be both none.")

        sc_id = sc_id or self.insert_one_supercell(sc_mat)
        sc_mat = sc_mat or self.sc_df[self.sc_df.sc_id==sc_id]\
                           .reset_index().iloc[0]['matrix']
       
        if comp_id is None and comp is None:
            #Get compositional statistics.(Must be normalized)
            sc_size = int(round(abs(np.linalg.det(sc_mat))))
            sc_sublat_list = get_sc_sllist_from_prim(self._sublat_list,sc_size=sc_size)
            cstat = occu_to_species_stat(occu,self._bits,sc_sublat_list)
            cstat = normalize_compstat(cstat,sc_size=sc_size)

            ucoords = self._compspace.translate_format(cstat,\
                                                       from_format='compstat',\
                                                       to_format='unconstr').tolist()

            _,comp_id = self.insert_one_comp(ucoords,sc_id=sc_id)

        elif comp_id is None:
            ucoords = self._compspace.translate_format(comp,\
                                                       from_format=comp_format,\
                                                       to_format='unconstr').tolist()

            _,comp_id = self.insert_one_comp(ucoords,sc_id=sc_id)      

        oid = self.find_entry_id_from_occu(occu,sc_id=sc_id,comp_id=comp_id) 
        if oid is not None:
            warnings.warn("Occupation {} found in previous table with index: {}."\
                          .format(occu,oid))
            return sc_id, comp_id, oid     
      
        eid = self.fact_df.entry_id.max()+1 if len(self.fact_df)>0 else 0

        iter_id = self.cur_iter_id

        matrix = self.sc_df[self.sc_df.sc_id==sc_id].reset_index().iloc[0]['matrix']
        s = structure_from_occu(occu,self._prim,matrix)
        corr = self._subspace.corr_from_structure(s,scmatrix=matrix)
        self._fact_df = self._fact_df.append({'entry_id':eid,'sc_id':sc_id,\
                                              'comp_id':comp_id,\
                                              'iter_id':calculated_info.get('iter_id',iter_id),\
                                              'module':module_name,\
                                              'ori_occu':occu,'ori_corr':corr,\
                                              'calc_status':calculated_info.get('calc_status','NC'),\
                                              'map_occu':calculated_info.get('map_occu'),\
                                              'map_corr':calculated_info.get('map_corr'),\
                                              'e_prim':calculated_info.get('e_prim'),\
                                              'other_props':calculated_info.get('other_props'),\
                                              ignore_index = True
                                             )

        return sc_id, comp_id, eid

    def insert_one_structure(self, s, comp_id=None, comp=None, comp_format='ucoord',\
                              sc_id=None, sc_mat=None,\
                              module_name='enum',**calculated_info):
        """
        Insert one entry from a structure in this cluster expansion system.
        Args:
            occu(1D ArrayLike of ints):
                Encoded occupation array.
            comp_id(int):
                Compostion index in self.comp_df. Optional.
            comp(Undefined format):
                Normalized compositions by each sublattice.
                Format depends on comp_format parameter.
                It is your responsibility to check whether
                this is correctly formatted and normalized,
                as required in self.comp_df. Optional.
            comp_format(str):
                The format of given composition. Can be 
                'unconstr','constr','composition','compstat'.
                Optional. Should be provided together with 
                comp parameter.
            sc_id(int):
                Index of supercell matrix. Optional. If you know the sc_id, you 
                should provide it, because this saves time in parsing the 
                supercells dataframe.
            sc_mat(List[List[int]):
                supercell matrix to insert. MUST BE LIST, NOT np.ndarray! Can not be
                both None with sc_id.          
            module_name(str):
                Name of the module that executes this insertion. ususally can be 'enum'
                or 'gs'. Default to enum.
            If you want to insert a calculated occupation or structure, you can pass
            down keys to calculated_info, and may overwrite iteration id with an additional
            iter_id argument.

        In structure insertion mode, you may not need to provide any of sc_id or
        sc_mat, but you are still recommended to do so.

        If you have the sc_id and the comp_id, you should provide them, because this
        will save a lot of time!!
        Returns:
            sc_id(int),comp_id(int),entry_id(int):
                indices in sc_df,comp_df,fact_df of this new entry
        """
        sc_mat = sc_mat or self._subspace.scmatrix_from_structure(s)
        if isinstance(sc_mat,np.ndarray):
            sc_mat = sc_mat.tolist()

        occu = self._subspace.occupancy_from_structure(s,scmatrix=sc_mat,encode=True)
        if isinstance(occu,np.ndarray):
            occu = occu.tolist()       

        return self.insert_one_occu(occu,comp_id=comp_id,comp=comp,comp_format=comp_format,\
                                         sc_id=sc_id,sc_mat=sc_mat,\
                                         module_name=module_name,**calculated_info)


    def remove_entree_by_id(self,entry_ids=[]):
        """
        Removes entree from the fact table. Does not change the supercell and the
        composition table. All indices will be re-assigned.
        Args:
            entry_ids(List[int]):
                The entry indices to remove. Default is empty list.
        """
        drop_ids = self.fact_df.index[self.fact_df.entry_id.isin(entry_ids)].tolist()
        self._fact_df = self._fact_df.drop(drop_ids)
        self._reassign_entry_ids()

    def remove_entree_by_iters_modules(self,iter_ids=[],modules=[],flush_and_reload=True):
        """
        Remove entree of specified iteration indices and modules.
        Args:
            iter_ids(List[int]):
                Iteration numbers to remove.
            modules(List[int]):
                Modules to remove.
            flush_and_reload(Boolean):
                If true, will re-save dataframes, and reload the status checker from the
                new saves.

        Note:
            Since this operation might change the status checker's return value,
            by default we will update the saved dataframes, and reload status checker.
        """
        filt = (self.fact_df.iter_id.isin(iter_ids)) & \
               (self.fact_df.module.isin(modules)
        eids = self.fact_df[filt].entry_id.tolist()
        self.remove_entree_by_id(eids)
        #Flush and read again.
        if flush_and_reload:
            self.auto_save(to_load_paths=True)
            self._schecker.re_load(from_load_paths=True)

    def remove_comps_by_id(self,comp_ids=[]):
        """
        Removes one composition from the composition table. Will also remove all entree
        in the fact table that have this composition. All indices will be reassigned.
        Does not change the supercell table.
        Args:
            comp_id(int):
                The composition index to remove. If not given, will remove the last composition
                in table.
        """   
        drop_ids_in_cdf = self.comp_df.index[self.comp_df.comp_id.isin(comp_ids)].tolist()
        drop_ids_in_fdf = self.fact_df.index[self.fact_df.comp_id.isin(comp_ids)].tolist()
        self._fact_df = self._fact_df.drop(drop_ids_in_fdf)
        self._comp_df = self._comp_df.drop(drop_ids_in_cdf)
        self._reassign_entry_ids()
        self._reassign_comp_ids()

    def remove_supercells_by_id(self,sc_id=[]):
        """
        Removes one supercell matrix from the supercell table. Will also remove all entree
        in the fact table that have this supercell, and all related compositions.
        All indices will be re-assigned.
        Args:
            sc_id(int):
                The entry index to remove. If not given, will remove the last entry
                in table.
        """      
        drop_ids_in_sdf = self.sc_df.index[self.sc_df.sc_id.isin(sc_ids)].tolist()   
        drop_ids_in_cdf = self.comp_df.index[self.comp_df.sc_id.isin(sc_ids)].tolist()
        drop_ids_in_fdf = self.fact_df.index[self.fact_df.sc_id.isin(sc_ids)].tolist()
        self._fact_df = self._fact_df.drop(drop_ids_in_fdf)
        self._comp_df = self._comp_df.drop(drop_ids_in_cdf)
        self._sc_df = self._sc_df.drop(drop_ids_in_sdf)
        self._reassign_entry_ids()
        self._reassign_comp_ids()
        self._reassign_sc_ids()

    def get_eid_w_status(self,status='NC'):
        """
        Get entree indices with a status.
        Args:
            status(str):
                The status that you wish to query.
        Returns: 
            List[int], indices in the fact table with that
            calc_status.
        """
        filt_ = self.fact_df.calc_status==status:
        return self.fact_df[filt_].entry_id.tolist()

    def set_status(self,eids = [],status='NC',flush_and_reload=True):
        """
        Set calc_status column of corresponding entree indices in the fact table.
        Args:
            eids(List[int]):
                entree indices to rewrite calc_status
            status(str):
                The status to rewrite with. See self.fact_df for allowed values.
            flush_and_reload(Boolean):
                If true, will re-save dataframes, and reload the status checker from the
                new saves. Default to true.
      
        """
        filt_ = self._fact_df.entry_id.isin(eids)
        self._fact_df.loc[filt_,'calc_status']=status

        #Flush and read again.
        if flush_and_reload:
            self.auto_save(to_load_paths=True)
            self._schecker.re_load(from_load_paths=True)

    def reset(self,flush_and_reload=True):
        """
        Reset all calculation data. Use this at your own risk!
        Args:
             flush_and_reload(Boolean):
                If true, will re-save dataframes, and reload the status checker from the
                new saves. Default to true.
        """
        print("Any enumerated entree in all iterations will be cleared. Proceed?[y/n]")
        choice = raw_input().lower()
        if choice == 'y':
            self.remove_supercells_by_id(sc_id=self.fact_df.sc_id.tolist())
        elif choice != 'n':
            raise ValueError("Please respond with [Y/N] or [y/n]!")

        #Flush and read again.
        if flush_and_reload:
            self.auto_save(to_load_paths=True)
            self._schecker.re_load(from_load_paths=True)

    def _reassign_entry_ids(self):
        """
        Reassign entry_id by the length of the fact dataframe.
        Order of rows will not be changed.
        """
        self._fact_df.entry_id = list(range(len(self._fact_df)))
        self._fact_df = self._fact_df.reset_index()

   def _reassign_comp_ids(self):
        """
        Reassign comp_id by the length of the comp dataframe. Both fact_df and the
        comp_df will be changed.
        Order of rows will not be changed.
        """
        old_cid = self._comp_df.comp_id.tolist()
        self._comp_df.comp_id = list(range(len(self._comp_df)))
        self._fact_df.comp_id = self._fact_df.comp_id.map(lambda ocid: old_cid.index(ocid))
        self._fact_df = self._fact_df.reset_index()
        self._comp_df = self._comp_df.reset_index()

   def _reassign_sc_ids(self):
        """
        Reassign sc_id by the length of the sc dataframe. All fact_df, comp_df and 
        sc_df will be changed.
        Order of rows will not be changed.
        """
        old_sid = self._comp_df.sc_id.tolist()
        self._sc_df.sc_id = list(range(len(self._sc_df)))
        self._fact_df.sc_id = self._fact_df.sc_id.map(lambda osid: old_sid.index(osid))
        self._comp_df.sc_id = self._comp_df.sc_id.map(lambda osid: old_sid.index(osid))
        self._sc_df = self._sc_df.reset_index()
        self._fact_df = self._fact_df.reset_index()
        self._comp_df = self._comp_df.reset_index()

    def _save_dataframes(self,sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Saving dimension tables and the fact table. Must set index=False, otherwise will always add
        One more row for each save and load.
        comp_df needs a little bit serialization.
        File names can be changed, but not recommended!
        """
        self.sc_df.to_csv(sc_file,index=False)
        comp_ser = self.comp_df.copy()
        comp_ser.comp = comp_ser.comp.map(lambda c: serialize_comp(c))
        comp_ser.to_csv(comp_file,index=False)
        if self.fact_df is not None:
            self.fact_df.to_csv(fact_file,index=False)

    def _load_dataframes(self,sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Loading dimension tables and the fact table. 
        comp_df needs a little bit de-serialization.
        File names can be changed, but not recommended!
        Notice: pandas loads lists as strings. You have to serialize them!
        """
        list_conv = lambda x: json.loads(x) if x is not None else None
        if os.path.isfile(sc_file):
            self._sc_df = pd.read_csv(sc_file,converters={'matrix':list_conv})
        if os.path.isfile(comp_file):
            #De-serialize compositions and list values
            self._comp_df = pd.read_csv(comp_file,
                                        converters={'ucoord':list_conv,
                                                    'ccoord':list_conv,
                                                    'cstat':list_conv,
                                                    'nondisc':list_conv,
                                                    'eq_occu':list_conv,
                                                    'comp':deser_comp
                                                   })
        if os.path.isfile(fact_file):
            self._fact_df = pd.read_csv(fact_file,
                                        converters={'ori_occu':list_conv,
                                                    'ori_corr':list_conv,
                                                    'map_occu':list_conv,
                                                    'map_corr':list_conv,
                                                    'other_props':list_conv
                                                   })


    @classmethod
    def auto_load(cls,options_file=OPTIONS_FILE,\
                      sc_file=SC_FILE,\
                      comp_file=COMP_FILE,\
                      fact_file=FACT_FILE,\
                      ce_history_file=CE_HISTORY_FILE):
        """
        This method is the recommended way to initialize this object.
        It automatically reads all setting files with FIXED NAMES.
        YOU ARE NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!
        Args:
            options_file(str):
                path to options file. Options must be stored as yaml
                format. Default: 'options.yaml'
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'
            comp_file(str):
                path to compositions file, in csv format.
                Default: 'comps.csv'             
            fact_file(str):
                path to enumerated structures dataframe file, in csv format.
                Default: 'data.csv'             
            ce_history_file(str):
                path to cluster expansion history file.
                Default: 'ce_history.json'
        Returns:
             DataManager object.
        """        
        options = InputsWrapper.auto_load(options_file=options_file,\
                                          ce_history_file=ce_history_file)

        schecker = StatusChecker.auto_load(sc_file=sc_file,\
                                           comp_file=comp_file,\
                                           fact_file=fact_file,\
                                           ce_history_file=ce_history_file)

        socket = cls(options.prim,\
                     options.bits,\
                     options.sublat_list,\
                     options.subspace,\
                     schecker)

        socket._load_dataframes(sc_file=sc_file,comp_file=comp_file,\
                                fact_file=fact_file)
        socket._sc_load_path = sc_file
        socket._comp_load_path = comp_file
        socket._fact_load_path = fact_file

        return socket


    def auto_save(self,sc_file=SC_FILE,comp_file=COMP_FILE,fact_file=FACT_FILE,\
                       to_load_paths=True):
        """
        Saves processed data to the dataframe csvs.
        """
        #Option file is read_only
        if to_load_paths:
            sc_file = self._sc_load_path
            comp_file = self._comp_load_path
            fact_file = self._fact_load_path

        self._save_dataframes(sc_file=sc_file,comp_file=comp_file,\
                              fact_file=fact_file)
        
