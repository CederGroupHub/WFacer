"""Featurization module. Extracts feature vectors and scalar properties."""

__author__ = 'Fengyu Xie'

import logging
log = logging.getLogger(__name__)

import numpy as np
import os

from monty.serialization import loadfn, dumpfn

from .specie_decorator import decorator_factory, decorate_single_structure
from .config_paths import DECOR_FILE


class Featurizer:
    """Featurization of calculation results.

    Direct initialization is not recommended.
    """
    def __init__(self, data_manager, history_wrapper,
                 decorators=None):
        """Initialize Featurizer.

        Args:
            data_manager(DataManager):
                The datamanager object to socket enumerated data.
            history_wrapper(HistoryWrapper):
                Wrapper containing previous CE fits.
            decorators(List[SpecieDecorator]):
                A list of specie decorators to use. If None, will
                initialize from options.
        """
        self._dm = data_manager
        self.prim = self.inputs_wrapper.prim
        self.bits = self.inputs_wrapper.bits
        self.sublat_list = self.inputs_wrapper.sublat_list
        self.sl_sizes = [len(sl) for sl in self.sublat_list]

        self.ce = history_wrapper.last_ce

        # Handling specie decoration types.
        if decorators is None:
            d_types = (self.inputs_wrapper
                       .featurizer_options['decorator_types'])
            d_args = (self.inputs_wrapper
                      .featurizer_options['decorator_args'])
            decorators = [decorator_factory(d_type, **d_args)
                          for d_type, d_arg in zip(d_types, d_args)]
                
        self._decorators = decorators

        self.other_props = (self.inputs_wrapper
                            .featurizer_options['other_props'])
        self.max_charge = (self.inputs_wrapper
                           .featurizer_options['max_charge'])
        self._reader = self.inputs_wrapper.calc_reader

    @property
    def inputs_wrapper(self):
        """InputsWrapper for project."""
        return self._dm._iw

    @property
    def data_manager(self):
        """DataManager for project."""
        return self._dm

    @property
    def decorators(self):
        """Species decorators to use."""
        return self._decorators

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

        Check your TimeKeeper and don't dupe run this module!
        """
        sc_df = self.sc_df.copy()
        fact_table = self.fact_df.copy()
        calc_reader = self.calc_reader

        log.critical("**Running featurization.")

        # Loading and decoration. If decorators not trained, train decorator.
        eids_unassigned = fact_table[fact_table.calc_status == 'CL'].entry_id
        # Check computation status, returns converged and failed indices.
        status = calc_reader.check_convergence_status(entry_ids=
                                                      eids_unassigned)
        success_ids = eids_unassigned[np.array(status)]
        fail_ids = eids_unassigned[~np.array(status)]
        log.info('**{}/{} converged computations in the last run.'
                 .format(len(success_ids), len(eids_unassigned)))

        fact_table.loc[fact_table.entry_id.isin(fail_ids),
                       'calc_status'] = 'CF'

        fact_unassigned = (fact_table[fact_table.calc_status == 'CL']
                           .merge(sc_table, how='left', on='sc_id'))

        # Loading structures
        structures_unassign = (calc_reader.
                               load_structures(entry_ids =
                                               fact_unassigned.entry_id))

        # Loading properties and doing decorations
        if len(self.decorators) > 0:
            decorations = {}
            for decorator in self.decorators:
                decor_inputs = (calc_reader.
                                load_properties(entry_ids=
                                                fact_unassigned.entry_id,
                                                prop_names=
                                                decorator.required_props,
                                                include_pnames=True))
                if not decorator.trained:
                    log.info('**Training decorator {}.'
                             .format(decorator.__class__.__name__))
                    decorator.train(structures_unassign, decor_inputs)

                decoration = decorator.assign(structures_unassign,
                                              decor_inputs)
                for prop_name, vals in decoration.items():
                    # Duplicacy removed here!
                    if prop_name not in decorations:
                        decorations[prop_name] = vals

            decor_keys = list(decorations.keys())
            decors_by_str = list(zip(*list(decorations.values())))

            sid_assign_fails = []
            structures_unmaped = []
            for sid, (s_unassign, decors_str) in \
              enumerate(zip(structures_unassign, decors_by_str)):
                s_assign = decorate_single_structure(s_unassign,
                                                     decor_keys,
                                                     decors_str,
                                                     max_charge=
                                                     self.max_charge)
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

        log.info('**{}/{} successful decorations.'
                    .format(len(fact_unmaped), len(fact_unassigned)))

        # Feature vectors.
        eid_map_fails = []
        occus_mapped = []
        corrs_mapped = []
        for eid, s_unmap, mat in zip(fact_unmaped.entry_id,
                                     structures_unmaped,
                                     fact_unmaped.matrix):
            try:
                # occupancies must be encoded
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

        log.info("**{}/{} successful mappings in the last run."
                 .format(len(occus_mapped), len(fact_unmaped)))
        log.critical("**Featurization finished.")

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

    def auto_save_decorators(self, decor_file=DECOR_FILE):
        """Serialize and save decorators to file."""
        dumpfn(self.decorators, decor_file)

    def auto_load_decorators(self, decor_file=DECOR_FILE):
        """Serialize and save decorators to file."""
        if os.path.isfile(decor_file):
            self._decorators = loadfn(decor_file)
        else:
            log.warning("Previous decorator file not found. Using default.")
