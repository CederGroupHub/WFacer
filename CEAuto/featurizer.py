__author__ = 'Fengyu Xie'

"""Featurization module. Extracts feature vectors and scalar properties."""

import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from monty.json import MSONable
from copy import deepcopy
import warnings
import logging

from pymatgen.core import Structure
from pymatgen.core.periodic_table import (get_el_sp, Lattice, Element,
                                          Specie, DummySpecie, Species)

from smol.cofe.space.domain import get_allowed_species, Vacancy
from smol.cofe import ClusterSubspace, ClusterExpansion
from smol.cofe.extern.ewald import EwaldTerm

from .specie_decorator import *
from .data_manager import DataManager
from .calc_reader import *

from .utils.serial_utils import decode_from_dict

from .config_paths import *

def decorate_single(s, decor_keys, decor_values):
    """
    This function decorates a single, undecorated structure
    composed of pymatgen.Element into structure of pymatgen.Species.
    Vacancies not considered.

    Args:
        s(pymatgen.Structure):
            Structure to be decorated.
        decor_keys(list of str):
            Names of properties to be decorated onto the structure.
        decor_values(2D list, second dimension can be None):
            Values of properties to be assigned to each site. Shaped in:
            N_properties* N_sites.
            Charges will be stored in each specie.oxidation_state, while
            other properties will be stoered in specie._properties, if
            allowed by Species class.
            If any of the properties in the second dimension is None, will
            return None. (Decoration failed.)
    Returns:
        Pymatgen.Structure: Decorated structure.
    """
    for val in decor_values:
        if val is None:
            return None
    
    #transpose to N_sites*N_properties
    decors_by_sites = list(zip(*decor_values))
    species_new = []

    for sp,decors_of_site in zip(s.species,decor_by_sites):
        try:
            sp_new = Specie(sp.symbol)
        except:
            sp_new = DummySpecie(sp.symbol)
        
        other_props = {}
        for key,val in zip(decor_keys,decors_of_site):
            if key == 'charge':
                sp_new._oxi_state = val
            else:  # Other properties
                if key in Species.supported_properties:
                    other_props[key] = val
                else:
                    warnings.warn("{} is not a supported pymatgen property."
                                  .format(key))
        sp_new._properties = other_props
        species_new.append(sp_new)

    return Structure(s.lattice,species_new,s.frac_coords)
                

class Featurizer(MSONable):
    """Featurization of calculation results.

    Direct initialization is not recommended.
    """

    def __init__(self, prim, data_manager, calc_reader,
                 bits=None, sublat_list=None, is_charged=False,
                 previous_ce=None, decorators=[], other_props=[]):
        """Initialize Featurizer.
        Args:
            prim(Structure):
                primitive cell of the structure to do cluster expansion on.
            data_manager(DataManager):
                The database manager class to use for this instance.
            calc_reader(BaseCalcReader):
                Calculation reader, depends on your selected CalcWriter
                and CalcManager type.
            bits(List[List[Specie]]):
                Occupying species on each sublattice. Vacancy() should be
                included.
            sublat_list(List of lists on ints):
                Stores primitive cell indices of sites in the same sublattices.
                If none, sublattices will be automatically generated.
            is_charged(Boolean):
                If true, will do charged cluster expansion.
            previous_ce(smol.cofe.ClusterExpansion):
                A previous cluster expansion. By default, is None. If this is
                given, will featurize based on this cluster expansion indstead.
            decorators(List of .specie_decorator.Decorator objects):
                Decorators names called before mapping into feature vectors.
                For example, if we do cluster expansion with charge, since vasp
                calculated structures does not mark charges, we have to assign
                charges to atoms before mapping.
                All items in this list must be a class in .decorator. 
                If multiple decorators are given, decorations will be
                done in the order of this list.
                If None given, will check with prim, and see whether
                decorations are needed. If decorations are needed, but no
                decorator is given, will return an error. If multiple
                decorators are given on the same decoration type, for example,
                charge decoration by magnetization or bader charge, only the
                first one in list will be used.
                This duplication is not checked before model training and
                assignment, so you must check them on your own to avoid
                additional training cost.
                Currently, we only support optimized charge decoration from
                site magnetic moments.
            other_props(List of str):
                Calculated properties to extract for expansion. Currently
                none of other proerties than 'e_prim' is supported. You can
                add your own properties extractors in calc_reader classes.
                This class does not check whether a property name is allowed. 
                If not, error messages will be given by calc_reader class.
                Check calc_reader docs for detail.
                Any physical quantity will always be normalized by supercell
                size!
        """

        self.prim = prim
        self.bits = bits
        self.sublat_list = sublat_list
        self.sl_sizes = [len(sl) for sl in self.sublat_list]

        self.ce = previous_ce

        # Handling specie decoration types.
        self._decorators = decorators           

        self.other_props = other_props
        self._dm = data_manager
        self._reader = calc_reader

    @property
    def sc_df(self):
        """
        Supercell dataframe.
        """
        return self._dm.sc_df

    @property
    def comp_df(self):
        """
        Composition dataframe.
        """
        return self._dm.comp_df

    @property
    def fact_df(self):
        """
        Fact dataframe.
        """
        return self._dm.fact_df

    def featurize(self):
        """Load and featurize the fact table with vasp data.

        Will check previous CE flow status. If already featurized in this
        iteration, will not featurize again.
        """
        if self._dm.schecker.after('feat'):
            warnings.warn("**Featurization already done in iteration {}."
                          .format(self._dm.schecker.cur_iter_id))
            return

        sc_df = self.sc_df.copy()
        fact_table = self.fact_df.copy()
        calc_reader = self.calc_reader        

        logging.log("**Running featurization.")
        # Loading and decoration. If decorators not trained, train decorator.
        eid_unassigned = fact_table[fact_table.calc_status == 'CL'].entry_id
        # Check computation status, returns converged and failed indices.
        success_ids, fail_ids = (calc_reader.
                                 check_convergence_status(entry_ids =
                                 eid_unassigned))
        logging.log('**{}/{} converged computations in the last run.'
                    .format(len(success_ids), len(fact_unassigned)))

        fact_table.loc[fact_table.entry_id.isin(fail_ids),'calc_status'] = 'CF'

        fact_unassigned = (fact_table[fact_table.calc_status == 'CL']
                           .merge(sc_table, how = 'left',on = 'sc_id'))

        # Loading structures
        structures_unassign = (calc_reader.
                               load_structures(entry_ids =
                                               fact_unassigned.entry_id))
       
        # Loading properties and doing decorations
        if len(self._decorators) > 0:
            decorations = {}
            for decorator in self._decorators:
                decor_inputs = (calc_reader.
                                load_properties(entry_ids=
                                                fact_unassigned.entry_id,
                                                prop_names=
                                                decorator.required_props,
                                                include_pnames=True)
                if not decorator.trained:
                    logging.log('**Training decorator {}.'
                                .format(decorator.__class__.__name__))
                    decorator.train(structures_unassign, decor_inputs)

                decoration = decorator.assign(structures_unassign,
                                              decor_inputs)
                for prop_name,vals in decoration.items():
                    # Duplicacy removed here!
                    if prop_name not in decorations:
                        decorations[prop_name] = vals

            decor_keys = list(decorations.keys())
            decors_by_str = list(zip(*list(decorations.values())))

            sid_assign_fails = []
            structures_unmaped = []
            for sid, (s_unassign, decors_str) in
              enumerate(zip(structures_unassign, decors_by_str)):
                s_assign = decorate_single(s_unassign, decor_keys, decors_str)
                if s_assign is not None:
                    structures_unmaped.append(s_assign)
                else:
                    sid_assign_fails.append(sid)

            eid_assign_fails = fact_unassigned.iloc[sid_assign_fails].entry_id
            fact_table.loc[fact_table.entry_id.isin(eid_assign_fails),
                           'calc_status'] = 'AF'
        else:
            structures_unmaped = structures_unassign

        fact_unmaped = (fact_table[fact_table.calc_status == 'CL']
                        .merge(sc_table, how='left', on='sc_id'))

        warnings.warn('**{}/{} successful decorations.'
                      .format(len(fact_unmaped), len(fact_unassigned)))

        # Feature vectors.
        eid_map_fails = []
        occus_mapped = []
        corrs_mapped = []
        for eid, s_unmap, mat in
          zip(fact_unmaped.entry_id, structures_unmaped, fact_unmaped.matrix):
            try:
                #occupancies must be encoded
                occu = (self.ce.cluster_subspace.
                        occupancy_from_structure(s_unmap, scmatrix=mat,
                                                 encode=True))
                occu = occu.tolist()
                corr = (self.ce.cluster_subspace.
                        corr_from_structure(s_unmap, scmatrix=mat))
                corr = corr.tolist()
                occus_mapped.append(occu)
                corrs_mapped.append(corr)
            except:
                sid_map_fails.append(sid)

        # Failed structures
        eid_map_fails = fact_unmaped.iloc[sid_map_fails].entry_id
        fact_table.loc[fact_table.entry_id.isin(eid_map_fails),
                       'calc_status'] = 'MF'

        # Successfully extracted structures.
        fact_table.loc[fact_table.calc_status == 'CL',
                       'map_occu'] = occus_mapped
        fact_table.loc[fact_table.calc_status == 'CL',
                       'map_corr'] = corrs_mapped
        fact_table.loc[fact_table.calc_status == 'CL',
                       'calc_status'] = 'SC'

        logging.log("**{}/{} successful mappings in the last run."
                    .format(len(occus_mapped), len(fact_unmaped)))
        logging.log("**Featurization finished. Iter number: {}"
                    .format(self._dm.schecker.cur_iter_id))

        self._dm._fact_df = fact_table.copy()

    def get_properties(self):
        """Load expansion properties.

        By default, only loads energies. Properties will be noralized to
        per prim. All properties must be scalars!

        Always be called after featurization!
        """
        sc_table = self.sc_df.copy()
        fact_table = self.fact_df.copy()
        calc_reader = self.calc_reader

        fact_unchecked = (fact_table[(fact_table.calc_status == 'SC') &
                                     (fact_table.e_prim.isna())]
                          .merge(sc_table, how='left', on='sc_id'))

        eid_unchecked = fact_unchecked.entry_id

        # Loading un-normalized energies and oter properties
        e_norms = calc_reader.load_properties(entry_ids=eid_unchecked,
                                              prop_names='energy')
        other_props = calc_reader.load_properties(entry_ids=eid_unchecked,
                                                  prop_names=self.other_props,
                                                  include_pnames=True)

        # Normalize properties
        sc_sizes = fact_unchecked.matrix.map(lambda x: abs(np.linalg.det(x)))
        e_norms = (np.array(e_norms) / sc_sizes).tolist()

        for prop_name in other_props:
            other_props[prop_name] = (np.array(other_props[prop_name]) /
                                      sc_sizes).tolist()

        fact_table.loc[fact_table.entry_id.isin(eid_unchecked),
                       'e_prim'] = e_norms
        fact_table.loc[fact_table.entry_id.isin(eid_unchecked),
                       'other_props'] = [{p: other_props[p][s_id]
                                          for p in other_props}
                                         for s_id in
                                         range(len(eid_unchecked))]

        self._dm._fact_df = fact_table.copy()

    def auto_save(self, sc_file=SC_FILE, comp_file=COMP_FILE,
                  fact_file=FACT_FILE, decor_file=DECOR_FILE):
        """Save data in object.

        All paths optional, but I don't recommend to change.
        """
        self._dm.auto_save(sc_file=sc_file, comp_file=comp_file,
                           fact_file=fact_file)

        with open(decor_file, 'w') as fout:
            decorator_dicts = [decorator.as_dict()
                               for decorator in self._decorators]
            json.dump(decorator_dicts, fout)

    @classmethod
    def auto_load(cls, data_manager,
                  options_file=OPTIONS_FILE,
                  decor_file=DECOR_FILE,
                  ce_history_file=CE_HISTORY_FILE):
        """Load object. Recommended initialization.

        Change of paths not recommended unless specifically needed.
        Args:
            data_manager(DataManager):
                A DataManager object to read and write when featurizing.
            options_file(str):
                Path to options file. Options must be stored as yaml format.
                Default: 'options.yaml'
            decor_file(str):
                Decorators data file. Optional, default: 'decors.json'.
            ce_history_file(str):
                Path to cluster expansion history file.
                Default: 'ce_history.json'
        Returns:
             Featurizer object.
        """
        options = InputsWrapper.auto_load(options_file=options_file,
                                          ce_history_file=ce_history_file)

        if os.path.isfile(decor_file):
            with open(decor_file) as fin:
                decor_dicts = json.load(fin)
                decorators = [decode_from_dict(d) for d in decor_dicts]
        else:
            decorators_types = options.featurizer_options['decorators_types']
            decorators_args = options.featurizer_options['decorators_args']

            decorators = [globals()[name](**args) for name, args in
                          zip(decorators_types, decorators_args)]

        reader = options.calc_reader

        return cls(options.prim,
                   bits=options.bits,
                   sublat_list=options.sublat_list,
                   is_charged=options.is_charged_ce,
                   previous_ce=options.last_ce,
                   other_props=options.featurizer_options['other_props'],
                   decorators=decorators,
                   data_manager=data_manager,
                   calc_reader=reader)
