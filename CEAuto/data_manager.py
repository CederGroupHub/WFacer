"""DataManager.

This file includes a class to manage read and write in calculation 
dataframes, including sc_df, comp_df, fact_df.
It also handles conversion between formats of data, such as occupancy
to structure, to correlation vector, etc.

This file does not generate any data. It only manages data!
"""

__author__ = "Fengyu Xie"

import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from copy import deepcopy

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .utils.serial_utils import serialize_comp, deser_comp
from .utils.occu_utils import (structure_from_occu,
                               occu_from_structure,
                               occu_to_species_stat)
from .utils.comp_utils import normalize_compstat
from .utils.frame_utils import load_dataframes, save_dataframes

from .wrappers import InputsWrapper
from .config_paths import (WRAPPER_FILE, OPTIONS_FILE,
                           SC_FILE, COMP_FILE, FACT_FILE)
from .comp_space import CompSpace

from smol.cofe.space.domain import get_allowed_species, get_site_spaces
from smol.moca.ensemble.sublattice import Sublattice


class DataManager:
    """DataManger class.

    Interfaces all CEAuto generated data, does insertion and deletion,
    but will not generate data.
    """
    def __init__(self, inputs_wrapper,
                 sc_df=None, comp_df=None, fact_df=None):
        """Initialize.

        Args:
            inputs_wrapper(InputsWrapper):
                An inputs wrapper object containing all lattice and
                parsed options information.
                All other classes call it, but will not do operation
                on it.
            sc_df(pd.DataFrame):
                Supercell dataframe. Default to None.
            comp_df(pd.DataFrame):
                Composition dataframe. Default to None.
            fact_df(pd.DataFrame):
                Fact table, storing all computation data. Default to
                None.

            If you provide any DataFrame, it is your responsibility
            to check their formats.
        """
        self._iw = inputs_wrapper

        self._sc_df = sc_df
        self._comp_df = comp_df
        self._fact_df = fact_df

    @property
    def prim(self):
        """Primitive cell.

        Returns:
            pymatgen.Structure.
        """
        return self._iw.prim

    @property
    def compspace(self):
        """Composition space (constrained).

        Returns:
           smol.moca.CompSpace.
        """
        return self._iw.compspace

    @property
    def subspace(self):
        """Cluster subspace.

        Returns:
           smol.cofe.ClusterSubspace
        """
        return self._iw.subspace

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
            self._sc_df = pd.DataFrame(columns=['sc_id', 'matrix'])
        return self._sc_df
 
    @property
    def comp_df(self):
        """Compositions dataframe.

        Has following columns:
            comp_id(int):
                indices of compositions, starting from 0.
            ucoord(List[float]):
                Normalized, unconstrained coordinates in the compositional
                space.
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
                different sublattices up. This is the only physical
                composition.
        """
        if self._comp_df is None:
            self._comp_df = pd.DataFrame(columns=['comp_id',
                                                  'ucoord', 'ccoord',
                                                  'comp', 'cstat',
                                                  'nondisc'])
        return self._comp_df

    @property
    def fact_df(self):
        """Fact table, containing all computed entrees.

        Columns:
            entry_id(int):
                Index of an entry, containing the computation result of a
                enumerated structure.
                Starts from 0.
            sc_id(int):
                 Index of supercell matrix in the supercell matrix dimension
                 table.
            comp_id(int):
                Index of composition in the composition dimension table.
            iter_id(int):
                Specifies in which iteration this structure is added into
                calculations.
                Both structure enumerator and ground state checker can add
                to fact table.
            module(str):
                Specifying the module name that generated and added this entry
                into the fact table. Can be 'enum' or 'gs'.
            ori_occu(List of int):
                Original occupancy as it was enumerated. (Encoded, and turned
                into list)
            ori_corr(List of float):
                Original correlation vector computed from ori_occu. (Turned
                into list)
            calc_status(str):
                A string of two characters, specifying the calculation status
                of the current entry.
                'NC': not calculated.(not submitted or waiting.)
                'CC': Calculation written, but calculation not finished.
                'CL': calculation finished.
                'CF': calculated, and failed (or exceeded wall time).
                'AF': assignment failed. For example, not charge neutral after
                      charge assignment.
                'MF': mapping failed, structure failed to map into original
                      lattice or featurize.
                'SC': successfully calculated, mapped and featurized.
                Any other status other than 'SC' will cause the following
                columns to be None.
            map_occu(List of int or None):
                Mapped occupancy after DFT relaxation.(Encoded, and turned
                into List)
            map_corr(List of float or None):
                Mapped correlation vector computed from map_occu. (Turned
                into List)
            e_prim(float or None):
                DFT energies of structures, normalized to ev/prim.
            other_props(Dict or None):
                Other SCALAR properties to expand. If not specified, only
                e_prim will be expanded.
                In format: {'prop_name':prop_value,...}
        """
        if self._fact_df is None:
            self._fact_df = pd.DataFrame(columns=['entry_id', 'sc_id', 
                                                  'comp_id',
                                                  'iter_id', 'module',
                                                  'ori_occu', 'ori_corr',
                                                  'calc_status',
                                                  'map_occu', 'map_corr',
                                                  'e_prim', 'other_props'])
        return self._fact_df

    @property
    def fact_df_with_structures(self):
        """Returns a fact table with column 'ori_str' and 'map_str'.

        Storing the original, and the mapped structure object,
        respectively.
        This table will not be saved for the sake of disk space.
        """
        fact = self.fact_df.copy()
        fact = fact.merge(self.sc_df, how='left', on='sc_id')
        #pd.isna must be used, because np.nan may not be considered is None.
        fact['ori_str'] = fact.apply(lambda r:
                                     structure_from_occu(r['ori_occu'],
                                                         self.prim,
                                                         r['matrix'])
                                     if not np.any(pd.isna(r['ori_occu']))
                                     else None, axis=1)

        fact['map_str'] = fact.apply(lambda r:
                                     structure_from_occu(r['map_occu'],
                                                         self.prim,
                                                         r['matrix'])
                                     if not np.any(pd.isna(r['map_occu']))
                                   else None, axis=1)
        return fact

    def get_all_sublattices(self, sc_mat):
        """Get all sublattices from supercell matrix.

        Args:
            sc_mat(3*3 Arraylike[int]):
                Supercell matrix.

        Return:
            List[smol.moca.Sublattice]. All sublattices including inactive.
            Same ordering as smol.moca.Sublattice.
        """
        unique_spaces = tuple(set(get_site_spaces(self.prim)))

        sc = self.prim.copy()
        sc.make_supercell(sc_mat)
        allowed_species = get_allowed_species(sc)
        return [Sublattice(site_space,
                np.array([i for i, sp in enumerate(allowed_species)
                         if sp == list(site_space.keys())]))
                for site_space in unique_spaces]

    def find_sc_id(self, sc_mat):
        """Find a supercell index from supercell dataframe by its matrix.

        If not present, will return None. Will deduplicate symmetry equivalence.

        Args:
            sc_mat(3*3 ArrayLike):
                supercell matrix to query
        Returns:
            sc_id(int) or None.
        """
        for old_id, old_mat in zip(self.sc_df.sc_id, self.sc_df.matrix):
            old_lat = np.dot(old_mat, self.prim.lattice.matrix)
            new_lat = np.dot(sc_mat, self.prim.lattice.matrix)
            R = np.linalg.inv(old_lat) @ new_lat
            if (np.allclose(R @ R.T, np.identity(3)) and
                abs(abs(np.linalg.det(R)) - 1) < 1E-7):
                return old_id

        return None
        
    def insert_one_supercell(self, new_sc_mat):
        """Insert one supercell to the supercell dataframe.

        Will automatically de-duplicate before inserting.
        The new insertion will always have an index following
        the last one in dataframe.

        Args:
            new_sc_mat(3*3 ArrayLike of int):
                Supercell matrix to insert.
        Returns:
            int: sc_id of inserted supercell.
        """
        if np.array(new_sc_mat).shape!=(3, 3):
            raise ValueError("Wrong supercell matrix input!")

        if not np.allclose(new_sc_mat, np.round(new_sc_mat)):
            raise ValueError("Given supercell matrix is not all integer!")

        new_sc_mat = np.array(np.round(new_sc_mat), dtype=int).tolist()
        oid = self.find_sc_id(new_sc_mat)
        if not pd.isna(oid):
            return oid

        sc_id = self.sc_df.sc_id.max() + 1 if len(self.sc_df) > 0 else 0
        self._sc_df = self._sc_df.append({'sc_id': sc_id,
                                          'matrix': new_sc_mat},
                                         ignore_index=True)
        return sc_id

    def find_comp_id(self, comp, comp_format='unconstr'):
        """Find index of a composition in the composition dataframe.
        
        If not present, will return None.
        Args:
            comp(Undefined format):
                composition by sublattices. Format defined by format
                parameter.
            comp_format(str):
                Specifies format of the input composition. Can be:
                'unconstr','constr','compstat','composition'
        Returns:
            int: comp_id or None
        """
        ucoord = (self.compspace.translate_format(comp,
                                                  from_format=comp_format,
                                                  to_format='unconstr')
                  .tolist())

        for cid, x in zip(self.comp_df.comp_id,
                          self.comp_df.ucoord):
            if np.allclose(x, ucoord, atol=1E-6):
                return cid

        return None

    def insert_one_comp(self, new_comp, comp_format='unconstr'):
        """Insert a composition into composition dataframe.

        If it is bound to a new supercell matrix, the supercell will also be
        inserted.
        Will be deduplicated automatically upon insertion.
        Args:
            new_comp(Undefined format):
                composition to insert. Format depends on comp_format argument.
                Usaually an unconstrained composition coordinate.
                It is your responsibility to normalize this composition as
                required in self.comp_df. Also, you are obliged to check
                whether this composition can be integer in your specified
                supercell size.
                The composition formats that satisfy the mentioned requirements
                can be generated by CompSpace.frac_grids().
            comp_format(str):
                Specifies the input composition format. Can be 'unconstr',
                'constr','compstat','composition'.

        Returns:
            int, int:
              Indices of the inserted supercell and composition.
        """
        o_cid = self.find_comp_id(new_comp, comp_format=comp_format)

        if not pd.isna(o_cid):
            # If found previous match, no insertion.
            return o_cid

        # Convert to list for proper serialization
        ucoord = self.compspace.translate_format(new_comp,
                                                 from_format=comp_format,
                                                 to_format='unconstr').tolist()
        ccoord = self.compspace.translate_format(new_comp,
                                                 from_format=comp_format,
                                                 to_format='constr').tolist()      
        comp = self.compspace.translate_format(new_comp,
                                               from_format=comp_format,
                                               to_format='composition')
        cstat = self.compspace.translate_format(new_comp,
                                                from_format=comp_format,
                                                to_format='compstat')
        nondisc = (self.compspace.translate_format(new_comp,
                                                    from_format=comp_format,
                                                    to_format='nondisc')
                   .tolist())

        comp_id = (self.comp_df.comp_id.max() + 1
                   if len(self.comp_df) > 0 else 0)

        self._comp_df = self._comp_df.append({'comp_id': comp_id,
                                              'ucoord': ucoord,
                                              'ccoord': ccoord,
                                              'cstat': cstat,
                                              'comp': comp,
                                              'nondisc': nondisc},
                                             ignore_index=True)
        return comp_id

    def find_entry_id_from_occu(self, occu,
                                comp_id=None, comp=None,
                                comp_format='unconstr',
                                sc_id=None, sc_mat=None):
        """Find the entry index from an encoded occupation array.

        Args:
            occu(1D ArrayLike of ints):
                Encoded occupation array.
            comp_id(int):
                Compostion index in self.comp_df. Optional.
            comp(Undefined format):
                Normalized compositions by each sublattice. Format depends on
                comp_format parameter.
                It is your responsibility to check whether this is correctly
                formatted and normalized as required. Optional.
            comp_format(str):
                The format of given composition. Can be 'unconstr','constr',
                'composition','compstat'. Optional. Should be provided together
                with argument "comp".
            sc_id(int):
                Index of supercell matrix. Optional, but sc_id argument and
                sc_mat argument can't be both None. If you know the sc_id,
                you should provide it, because this saves time in parsing
                the supercells dataframe.
            sc_mat(3*3 Arralike of ints):
                supercell matrix to insert.

            If you want to insert a calculated occupation or structure, you
            can pass down keys to calculated_info, and may overwrite iteration
            id with an additional iter_id argument.

        If you have the sc_id and the comp_id, you should provide them.
        Note: Will remove decorations before deduplicating.

        Returns:
            entry_id(int):
                index in fact_df of this new entry. If not found, will return None.
        """
        if pd.isna(sc_id) and np.any(pd.isna(sc_mat)):
            raise ValueError("Arguments sc_id and sc_mat can not be both none.")

        sc_id = sc_id if not pd.isna(sc_id) else self.find_sc_id(sc_mat)
        sc_mat = (sc_mat if not np.any(pd.isna(sc_mat)) else
                  self.sc_df[self.sc_df.sc_id == sc_id]
                  .reset_index(drop=True).iloc[0]['matrix'])
        sc_size = int(round(abs(np.linalg.det(sc_mat))))

        if pd.isna(sc_id):
            return None

        # When you supercell matrices are only symmetrically equivalent,
        # but not identical, your occupancy array may have different mapping.
        # To avoid error, occupancies must be modified to the mapping in
        # one standard supercell matrix.
        s = structure_from_occu(occu, self.prim, sc_mat)
        sc_mat_std = (self.sc_df[self.sc_df.sc_id == sc_id]
                      .reset_index(drop=True).iloc[0]['matrix'])
        occu_std = occu_from_structure(s, self.prim, sc_mat_std)

        if pd.isna(comp_id) and comp is None:
            all_sublattices_std = self.get_all_sublattices(sc_mat_std)
            cstat = occu_to_species_stat(occu_std, all_sublattices_std)
            cstat = normalize_compstat(cstat, sc_size=sc_size)

            ucoords = (self.compspace.translate_format(cstat,
                                                       from_format='compstat',
                                                       to_format='unconstr')
                       .tolist())

            comp_id = self.find_comp_id(ucoords)

        elif pd.isna(comp_id):
            ucoords = (self.compspace.translate_format(comp,
                                                       from_format=comp_format,
                                                       to_format='unconstr')
                       .tolist())

            comp_id = self.find_comp_id(ucoords)      

        if pd.isna(comp_id):
            return None

        filt_ = ((self.fact_df.sc_id == sc_id) &
                 (self.fact_df.comp_id == comp_id))
        fact = self.fact_df[filt_].merge(self.sc_df,
                                         on='sc_id',
                                         how='left').reset_index(drop=True)

        # From now on, we check charge undecorated duplicacy.
        sm = StructureMatcher()
        s_new = structure_from_occu(occu_std, self.prim, sc_mat_std)
        s_new_clean = Structure(s_new.lattice,
                                [site.specie.symbol for site in s_new],
                                s_new.frac_coords)

        for i in range(len(fact)):
            # Check duplicacy with structure matcher
            sc_mat_old = fact.iloc[i]['matrix']
            occu_old = fact.iloc[i]['ori_occu']
            s_old = structure_from_occu(occu_old, self.prim, sc_mat_old)

            s_old_clean = Structure(s_old.lattice,
                                    [site.specie.symbol for site in s_old],
                                    s_old.frac_coords)

            if sm.fit(s_old_clean, s_new_clean):
                return fact.iloc[i]['entry_id']

        return None

    def insert_one_occu(self, occu, 
                        comp_id=None, comp=None, comp_format='ucoord',
                        sc_id=None, sc_mat=None,
                        iter_id=0, module_name='enum',
                        **calculated_info):
        """Insert one entry from an encoded occupation array.

        Args:
            occu(1D ArrayLike of ints):
                Encoded occupation array.
            comp_id(int):
                Compostion index in self.comp_df. Optional.
            comp(Undefined format):
                Normalized compositions by each sublattice. Format depends
                on comp_format parameter.
                It is your responsibility to check whether this is correctly
                formatted and normalized as required. Optional.
            comp_format(str):
                The format of given composition. Can be 'unconstr','constr',
                'composition','compstat'.
                Optional. Should be provided together with comp parameter.
            sc_id(int):
                Index of supercell matrix. Optional, but sc_id argument and
                sc_mat argument can't be both None.
            sc_mat(3*3 Arralike of int):
                Supercell matrix to insert.
            iter_id(int):
                Index of the iteration when this entry is generated. Default
                to 0. Must be the result of TimeKeeper.
            module_name(str):
                Name of the module that executes this insertion. For the
                current version, can be 'enum' or 'gs'. Default to enum.
            **calculated_info:
                If this entry has been calculated or mapped before, you can
                insert its DFT data here. Accepted keywords include:
                'calc_status', 'map_occu', 'map_corr', 'e_prim', 'other_props'.

        If you have the sc_id and the comp_id, you should provide them.

        Returns:
            sc_id(int), comp_id(int), entry_id(int):
                indices in sc_df, comp_df, fact_df of this new entry.
        """
        if pd.isna(sc_id) and np.any(pd.isna(sc_mat)):
            raise ValueError("Arguments sc_id and sc_mat can not be both none.")

        if not pd.isna(sc_id) and sc_id not in self.sc_df.sc_id:
            raise ValueError("Supercell index {} given, ".format(sc_id) +
                             "but not in supercell table.")

        if not pd.isna(comp_id) and comp_id not in self.comp_df.comp_id:
            raise ValueError("Composition index {} given, ".format(comp_id) +
                             "but not in composition table.")

        sc_id = (sc_id if not pd.isna(sc_id)
                 else self.insert_one_supercell(sc_mat))
        sc_mat = (sc_mat if not np.any(pd.isna(sc_mat)) else
                  self.sc_df[self.sc_df.sc_id == sc_id]
                  .reset_index(drop=True).iloc[0]['matrix'])
        sc_size = int(round(abs(np.linalg.det(sc_mat))))

        # When you supercell matrices are only symmetrically equivalent,
        # but not identical, your occupancy array may have different mapping.
        # To avoid error, occupancies must be modified to the mapping in
        # one standard supercell matrix.
        s = structure_from_occu(occu, self.prim, sc_mat)
        sc_mat_std = (self.sc_df[self.sc_df.sc_id == sc_id]
                      .reset_index(drop=True).iloc[0]['matrix'])
        occu_std = occu_from_structure(s, self.prim, sc_mat_std)

        if pd.isna(comp_id) and comp is None:
            all_sublattices_std = self.get_all_sublattices(sc_mat_std)
            cstat = occu_to_species_stat(occu_std, all_sublattices_std)
            cstat = normalize_compstat(cstat, sc_size=sc_size)

            ucoords = (self.compspace.translate_format(cstat,
                                                       from_format='compstat',
                                                       to_format='unconstr')
                       .tolist())

            comp_id = self.insert_one_comp(ucoords)

        elif pd.isna(comp_id):
            ucoords = (self.compspace.translate_format(comp,
                                                       from_format=comp_format,
                                                       to_format='unconstr')
                       .tolist())

            comp_id = self.insert_one_comp(ucoords)

        oid = self.find_entry_id_from_occu(occu_std, sc_id=sc_id,
                                           comp_id=comp_id)
        if not pd.isna(oid):
            log.debug("Occupancy {} found in previous table ".format(occu)
                      + "with index: {}.".format(oid))
            return sc_id, comp_id, oid

        eid = (self.fact_df.entry_id.max() + 1 if len(self.fact_df) > 0
               else 0)

        s = structure_from_occu(occu_std, self.prim, sc_mat_std)
        corr = self.subspace.corr_from_structure(s, scmatrix=sc_mat_std)

        # Standardize storage format.
        ori_corr = np.array(corr).tolist()
        calc_status = calculated_info.get('calc_status', 'NC')
        map_occu = calculated_info.get('map_occu')
        map_occu = (np.array(map_occu, dtype=int).tolist()
                    if map_occu is not None else None)
        map_corr = calculated_info.get('map_corr')
        map_corr = (np.array(map_corr).tolist()
                    if map_corr is not None else None)
        e_prim = calculated_info.get('e_prim')
        other_props = calculated_info.get('other_props')

        self._fact_df = self._fact_df.append({'entry_id': eid,
                                              'sc_id': sc_id,
                                              'comp_id': comp_id,
                                              'iter_id': iter_id,
                                              'module': module_name,
                                              'ori_occu': occu_std,
                                              'ori_corr': ori_corr,
                                              'calc_status': calc_status,
                                              'map_occu': map_occu,
                                              'map_corr': map_corr,
                                              'e_prim': e_prim,
                                              'other_props': other_props},
                                              ignore_index = True
                                             )

        return sc_id, comp_id, eid

    def insert_one_structure(self, s, 
                             comp_id=None, comp=None, comp_format='ucoord',
                             sc_id=None, sc_mat=None,
                             iter_id=0, module_name='enum',
                             **calculated_info):
        """Insert one entry from a structure.

        Args:
            s(pymatgen.Structure):
                Structure entry to insert.
            comp_id(int):
                Compostion index in self.comp_df. Optional.
            comp(Undefined format):
                Normalized compositions by each sublattice. Format depends
                on comp_format parameter.
                It is your responsibility to check whether this is correctly
                formatted and normalized as required in. Optional.
            comp_format(str):
                The format of given composition. Can be 'unconstr','constr',
                'composition','compstat'.
                Optional. Should be provided together with comp parameter.
            sc_id(int):
                Index of supercell matrix. Optional. If you know the sc_id,
                you should provide it.
            sc_mat(3*3 Arralike of int):
                Supercell matrix to insert.
            iter_id(int):
                Index of the iteration when this entry is generated. Default
                to 0. Must be the result of TimeKeeper.
            module_name(str):
                Name of the module that executes this insertion. For the
                current version, can be 'enum' or 'gs'. Default to enum.
            **calculated_info:
                If this entry has been calculated or mapped before, you can
                insert its DFT data here. Accepted keywords include:
                'calc_status', 'map_occu', 'map_corr', 'e_prim', 'other_props'.

        In structure insertion mode, you may not need to provide any of sc_id
        or sc_mat, but you are still recommended to do so, because mapping of 
        supercell can be very inaccurate.

        Returns:
            sc_id(int), comp_id(int), entry_id(int):
                indices in sc_df,comp_df,fact_df of this new entry
        """
        if pd.isna(sc_id) and np.any(pd.isna(sc_mat)):
            try:
                sc_mat = self.subspace.scmatrix_from_structure(s)
            except:
                raise ValueError("Arguments sc_id and sc_mat are both none, " +
                                 "but sc_mat can't be found.")

        if not(pd.isna(sc_id)) and sc_id not in self.sc_df.sc_id:
            raise ValueError("Supercell index {} given, but not in table."
                             .format(sc_id))

        sc_id = sc_id if not pd.isna(sc_id) else self.insert_one_supercell(sc_mat)
        sc_mat_std = (self.sc_df[self.sc_df.sc_id == sc_id].
                      reset_index(drop=True).iloc[0]['matrix'])

        occu = self.subspace.occupancy_from_structure(s, scmatrix=sc_mat_std,
                                                      encode=True)
        occu = np.array(occu, dtype=int).tolist()

        return self.insert_one_occu(occu, comp_id=comp_id, comp=comp,
                                    comp_format=comp_format,
                                    sc_id=sc_id, sc_mat=sc_mat,
                                    iter_id=iter_id,
                                    module_name=module_name,
                                    **calculated_info)

    def remove_entree_by_id(self, entry_ids=[]):
        """Removes entree from the fact table.

        Does not change the supercell and the composition table.
        All indices will be RE-assigned.
        Args:
            entry_ids(List[int]):
                The entry indices to remove. Default is empty list.
        """
        drop_ids = (self.fact_df.index[self.fact_df.entry_id.isin(entry_ids)]
                    .tolist())
        self._fact_df = self._fact_df.drop(drop_ids).reset_index(drop=True)
        self._reassign_entry_ids()

    def remove_entree_by_iters_modules(self, iter_ids=[], modules=[]):
        """Remove entree of specified iteration indices and modules.

        Entry indices will be RE-ASSIGNED!
        Args:
            iter_ids(List[int]):
                Iteration numbers to remove.
            modules(List[int]):
                Modules to remove.

        Note:
            This may affect the check point status of TimeKeeper. Be sure
            to reset TimeKeeper after this operation.
        """
        filt = ((self.fact_df.iter_id.isin(iter_ids)) &
                (self.fact_df.module.isin(modules)))
        eids = self.fact_df[filt].entry_id.tolist()
        self.remove_entree_by_id(eids)

    def remove_comps_by_id(self, comp_ids=[]):
        """Removes compositions from the composition table.

        Will also remove all entree in the fact table that have this
        composition. All indices will be reassigned.
        Does not change the supercell table.

        Indices will be RE-assigned!
        Args:
            comp_id(int):
                The composition index to remove. If not given, will remove
                the last composition in table.
        """   
        drop_ids_in_cdf = (self.comp_df.
                           index[self.comp_df.comp_id.isin(comp_ids)]
                           .tolist())
        drop_ids_in_fdf = (self.fact_df.
                           index[self.fact_df.comp_id.isin(comp_ids)]
                           .tolist())

        self._fact_df = (self._fact_df.drop(drop_ids_in_fdf)
                         .reset_index(drop=True))
        self._comp_df = (self._comp_df.drop(drop_ids_in_cdf)
                         .reset_index(drop=True))
        self._reassign_entry_ids()
        self._reassign_comp_ids()

    def remove_supercells_by_id(self, sc_ids=[]):
        """Removes one supercell matrix from the supercell table.

        Will also remove all entree in the fact table that is this supercell,
        and all related compositions.

        All indices will be Re-assigned!
        Args:
            sc_id(int):
                The entry index to remove. If not given, will remove the last
                entry in table.
        """      
        drop_ids_in_sdf = (self.sc_df.
                           index[self.sc_df.sc_id.isin(sc_ids)]
                           .tolist())
        drop_ids_in_fdf = (self.fact_df.
                           index[self.fact_df.sc_id.isin(sc_ids)]
                           .tolist())

        self._fact_df = (self._fact_df.drop(drop_ids_in_fdf)
                         .reset_index(drop=True))
        self._sc_df = (self._sc_df.drop(drop_ids_in_sdf)
                       .reset_index(drop=True))
        self._reassign_entry_ids()
        self._reassign_sc_ids()

    def get_eid_w_status(self, status='NC'):
        """Get entree indices with a status.

        Args:
            status(str):
                The status to query.

        Returns: 
            List[int], indices in the fact table with that
            calc_status.
        """
        filt_ = (self.fact_df.calc_status == status)
        return self.fact_df[filt_].entry_id.tolist()

    def set_status(self, eids = [], status='NC'):
        """Set calc_status of corresponding entree indices.

        Args:
            eids(List[int]):
                entree indices to rewrite calc_status
            status(str):
                The status to rewrite with. See self.fact_df for allowed
                values.
        """
        filt_ = self._fact_df.entry_id.isin(eids)
        self._fact_df.loc[filt_, 'calc_status'] = status

    def reset(self):
        """Reset all calculation data.

        Use this at your own risk!
        """
        log.warning("Clearing all previous records. " +
                    "Do this at your own risk!")
        self.remove_supercells_by_id(sc_ids=
                                     self.sc_df.sc_id.tolist())
        self.remove_comps_by_id(comp_ids=
                                self.comp_df.comp_id.tolist())

    def copy(self):
        """Deepcopy of DataManager."""
        sock = DataManager(self._iw.copy(),
                           self._sc_df.copy(),
                           self._comp_df.copy(),
                           self._fact_df.copy())

        return sock

    def _reassign_entry_ids(self):
        """Reassign entry_id by the length of the fact table.

        Ordering of rows will not be changed.
        """
        self._fact_df.entry_id = list(range(len(self._fact_df)))

    def _reassign_comp_ids(self):
        """Reassign comp_id by the length of the comp dataframe.

        Both fact_df and the comp_df will be re-indexed.

        Ordering of rows will not be changed.
        """
        old_cid = self._comp_df.comp_id.tolist()
        self._comp_df.comp_id = list(range(len(self._comp_df)))
        self._fact_df.comp_id = self._fact_df.comp_id.map(lambda ocid:
                                                          old_cid.index(ocid))

    def _reassign_sc_ids(self):
        """Reassign sc_id by the length of the sc dataframe.

        All fact_df and sc_df will be re-indexed.

        Order of rows will not be changed.
        """
        old_sid = self._sc_df.sc_id.tolist()
        self._sc_df.sc_id = list(range(len(self._sc_df)))
        self._fact_df.sc_id = self._fact_df.sc_id.map(lambda osid:
                                                      old_sid.index(osid))

    @classmethod
    def auto_load(cls,
                  wrapper_file=WRAPPER_FILE,
                  options_file=OPTIONS_FILE,
                  sc_file=SC_FILE,
                  comp_file=COMP_FILE,
                  fact_file=FACT_FILE):
        """Automatically loadsd DataManager from files.

        Recommended way to initialize this object.
        NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!
        Args:
            wrapper_file(str):
                path to inputs wrapper save file. Default:
                'inputs_wrapper.json'.
            options_file(str):
                path to options file. Options must be stored as yaml
                format. Default: 'options.yaml'.
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'.
            comp_file(str):
                path to compositions file, in csv format.
                Default: 'comps.csv'.
            fact_file(str):
                path to enumerated structures dataframe file, in csv format.
                Default: 'data.csv'.
        Returns:
             DataManager object.
        """        
        iw = InputsWrapper.auto_load(wrapper_file=wrapper_file,
                                     options_file=options_file)
        sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                                  comp_file=comp_file,
                                                  fact_file=fact_file)

        return cls(iw, sc_df=sc_df, comp_df=comp_df, fact_df=fact_df)

    def auto_save(self, wrapper_file=WRAPPER_FILE,
                  sc_file=SC_FILE, comp_file=COMP_FILE, fact_file=FACT_FILE):
        """Saves processed data to files.

        Args:
            wrapper_file(str):
                path to inputs wrapper save file. Default:
                'inputs_wrapper.json'.
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'.
            comp_file(str):
                path to compositions file, in csv format.
                Default: 'comps.csv'.
            fact_file(str):
                path to enumerated structures dataframe file, in csv format.
                Default: 'data.csv'.
        """
        self._iw.auto_save(wrapper_file=wrapper_file)
        save_dataframes(self.sc_df, self.comp_df, self.fact_df,
                        sc_file=sc_file, comp_file=comp_file,
                        fact_file=fact_file)
