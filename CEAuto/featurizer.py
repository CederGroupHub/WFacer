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

    @staticmethod
    def get_unprocessed_eids(fact_df, base_status='CL'):
        """Get unprocessed entry ids from fact table."""
        return fact_df[fact_df.calc_stats == base_status].entry_id

    def query_sc_matrices(sc_df, fact_df, eids=[]):
        """Query supercell matrices from entree indices."""
        merge_df = fact_df.merge(sc_df, on='sc_id', how='left')
        return merge_df[merge_df.entry_id.isin(eids)].matrix

    @staticmethod
    def set_calc_status(fact_df, eids=[], status='CF'):
        """Set calculation status to eids.

        Args:
            fact_df(pd.DataFrame):
                Fact table.
            eids(Arraylike[int]):
                Entree ids to set status.
            status(str):
                The status to set.

        Returns:
            pd.DataFrame.
        """
        fact = fact_df.copy()
        fact.loc[fact.entry_id.isin(eids), 'calc_status'] = status
        return fact

    @staticmethod
    def decorate_structures(structs, decor_inputs=[], decorators=[]):
        """Decorate structures.

        Args:
            structs(List[pymatgen.Structure]):
                Undecorated structures, all composed of pymatgen.Element.
            decor_inputs(List[dict]):
                List of decorator input properties. See documentation
                of CEAuto.species_decorators.
            decorators(List[BaseDecorator]):
                List of decorator objects.

        Returns:
            List[pymatgen.Structure | None]:
                Decorated structures, composed of pymatgen.Species.
                If a structure is failed to assign, will give a 
                None.
            List[BaseDecorator]:
                Decorators after training.
        """
        # Loading properties and doing decorations
        if len(decorators) > 0:
            decorations = {}
            for decorator, d_input in zip(decorators, decor_inputs):
                if not decorator.trained:
                    log.info('**Training decorator {}.'
                             .format(decorator.__class__.__name__))
                    decorator.train(structs, d_input)

                decoration = decorator.assign(structs,
                                              d_input)
                for prop_name, vals in decoration.items():
                    # Duplicacy removed here!
                    if prop_name not in decorations:
                        decorations[prop_name] = vals

            decor_keys = list(decorations.keys())
            decors_by_str = list(zip(*list(decorations.values())))

            structs_decorated = []
            for s, decors in zip(structs, decors_by_str):
                s_decorated = decorate_single_structure(s,
                                                        decor_keys,
                                                        decors,
                                                        max_charge=
                                                        self.max_charge)
                structs_decorated.append(s_decorated)

        else:
            structs_decorated = structs

        return structs_decorated

    @staticmethod
    def map_structures(subspace, structs, scmatrices=None):
        """Map all structures into occu and corr.

        Args:
            subspace(smol.cofe.ClusterSubspace):
                Clutersubspace.
            structs(List[pymatgen.Structure]):
                Decorated structures.
            scmatrices(list[3*3 arraylike]): optional
                Supercell matrices for each structure. Better provided.

        Returns:
            List[List[int]|None], List[List[float]|None]:
                Mapped occupancies and correlation vectors.
                If match failed, will append a None.
        """
        # Feature vectors
        occus = []
        corrs = []
        scmatrices = scmatrices or [None for _ in range(len(structs))]

        for s, mat in zip(structs, scmatrices):
            try:
                # occupancies must be encoded
                occu = (subspace.
                        occupancy_from_structure(s, scmatrix=mat,
                                                 encode=True))
                occu = occu.tolist()
                corr = (subspace.
                        corr_from_structure(s, scmatrix=mat))
                corr = corr.tolist()
                occus.append(occu)
                corrs.append(corr)
            except:
                occus.append(None)
                corrs.append(None)

        return occus, corrs

    def featurize(self):
        """Load and featurize the fact table with vasp data.

        Check your TimeKeeper and don't dupe run this module!
        """
        sc_df = self.sc_df.copy()
        fact_df = self.fact_df.copy()
        reader = self.calc_reader
        subspace = self.ce.cluster_subspace

        log.critical("**Running featurization.")

        eids = self.get_unprocessed_eids(fact_df)
        status = reader.check_convergence_status(entry_ids=
                                                 eids)
        success_ids = eids[np.array(status)]
        fail_ids = eids[~np.array(status)]
        fact_df = self.set_calc_status(fact_df, fail_ids, 'CF')

        log.info('**{}/{} converged computations in the last run.'
                 .format(len(success_ids), len(eids)))

        # Decorations.
        eids = self.get_unprocessed_eids(fact_df)
        structs = reader.load_structures(entry_ids=eids)
        decorators = self.decorators
        decor_inputs = [calc_reader.
                        load_properties(entry_ids=eids,
                                        prop_names=
                                        decorator.required_props,
                                        include_pnames=True)
                        for decorator in decorators]
        structs_decor = self.decorate_structures(structs,
                                                 decor_inputs,
                                                 decorators)
        status = [(s is not None) for s in structs_decor]
        success_ids = eids[np.array(status)]
        fail_ids = eids[~np.array(status)]
        fact_df = self.set_calc_status(fact_df, fail_ids, 'AF')

        log.info('**{}/{} successful decorations.'
                 .format(len(success_ids), len(eids)))

        # Mappings.
        eids = self.get_unprocessed_eids(fact_df)
        fact_slice = fact_df[fact_df.entry_id.isin(eids)]
        fact_slice = fact_slice.merge(sc_df, on='sc_id', how='left')
        scmats = fact_slice.matrix
        structs_unmap = [s for s in structs_decor if s is not None]
        occus, corrs = self.map_structures(subspace, structs_unmap,
                                           scmatrices=scmats)
        status = [(o is not None and c is not None)
                  for o, c in zip(occus, corrs)]
        success_ids = eids[np.array(status)]
        fail_ids = eids[~np.array(status)]
        fact_df.loc[fact_df.entry_id.isin(eids), 'map_occu'] = occus
        fact_df.loc[fact_df.entry_id.isin(eids), 'map_corr'] = corrs
        fact_df = self.set_calc_status(fact_df, fail_ids, 'MF')

        log.info('**{}/{} successful mappings.'
                 .format(len(success_ids), len(eids)))

        fact_df = self.set_calc_status(fact_df, success_ids, 'SC')

        log.critical("**Featurization finished. Total success structures: {}."
                     .format(len(success_ids)))

        self._dm._fact_df = fact_df.copy()

    def get_properties(self):
        """Load expansion properties.

        By default, only loads energies. Properties will be noralized to
        per prim. All properties must be scalars!

        Always be called after featurization!
        """
        sc_df = self.sc_df.copy()
        fact_df = self.fact_df.copy()
        reader = self.calc_reader

        fact_unchecked = (fact_df[(fact_df.calc_status == 'SC') &
                                  (fact_df.e_prim.isna())]
                          .merge(sc_df, how='left', on='sc_id'))

        eid_unchecked = fact_unchecked.entry_id

        # Loading un-normalized energies and oter properties
        e_norms = reader.load_properties(entry_ids=eid_unchecked,
                                         prop_names='energy')
        other_props = reader.load_properties(entry_ids=eid_unchecked,
                                             prop_names=self.other_props,
                                             include_pnames=True)

        # Normalize properties
        sc_sizes = fact_unchecked.matrix.map(lambda x: abs(np.linalg.det(x)))
        e_norms = (np.array(e_norms) / sc_sizes).tolist()

        for prop_name in other_props:
            other_props[prop_name] = (np.array(other_props[prop_name]) /
                                      sc_sizes).tolist()

        fact_df.loc[fact_df.entry_id.isin(eid_unchecked),
                    'e_prim'] = e_norms
        fact_df.loc[fact_df.entry_id.isin(eid_unchecked),
                    'other_props'] = [{p: other_props[p][s_id]
                                       for p in other_props}
                                      for s_id in
                                      range(len(eid_unchecked))]

        self._dm._fact_df = fact_df.copy()

    def auto_save_decorators(self, decor_file=DECOR_FILE):
        """Serialize and save decorators to file."""
        dumpfn(self.decorators, decor_file)

    def auto_load_decorators(self, decor_file=DECOR_FILE):
        """Serialize and save decorators to file."""
        if os.path.isfile(decor_file):
            self._decorators = loadfn(decor_file)
        else:
            log.warning("Previous decorator file not found. Using default.")
