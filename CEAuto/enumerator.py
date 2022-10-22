__author__ = "Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.

Algorithm based on: 

Ground state structures will also be added to the structure pool, but 
they are not added here. They will be added in the convergence checker
module.
"""

import logging

import random
import numpy as np
import multiprocessing as mp
from itertools import product, chain
from functools import partial
import time

from monty.json import MSONable
from smol.moca import CompositionSpace
from smol.cofe import ClusterSubspace

from .utils.supercells import enumerate_matrices
from .utils.select_methods import select_initial_rows, select_added_rows

from .sample_geneators import CanonicalSampleGenerator


log = logging.getLogger(__name__)


def raw_sample_under_sc_comp(ce, handler_args, sc_mat_and_compstat):
    """Generate canonical raw sample from a composition.

    These are not de-duplicated.

    Args:
        ce(ClusterExpansion):
            Cluster expansion to sample with.
        handler_args(dict):
            Arguments to pass into CanonicalmcHandler.
        sc_mat_and_compstat(tuple(3*3 arraylike, List[List])):
            tuple of supercell matrix and composition in the form
            of compstat. (See CompSpace docs.)
    Returns:
        np.ndarray, int:
            Sampled occupancy strings and the number of all possible
            occupancies under this composition.
    """

    sc_mat, compstat = sc_mat_and_compstat
    sc_size = int(round(abs(np.linalg.det(sc_mat))))
    log.debug("****Supercell size: {}, compositon stat: {}."
              .format(sc_size, compstat))
    tot_noccus = get_Noccus_of_compstat(compstat, scale_by=sc_size)
    # Handler will automatically initialize an occupation
    handler = CanonicalmcHandler(ce, sc_mat, compstat, **handler_args)
    # Handler will de-freeze from 0K, creating samples. for all runs.
    return handler.get_unfreeze_sample(), tot_noccus


class SupercellMatrixEnumerator(MSONable):
    """Supercell matrix enumeration."""

    def __init__(self, cluster_space,
                 conv_mat=None,
                 supercell_from_conventional=True,
                 objective_sc_size=32,
                 max_sc_cond=8,
                 min_sc_angle=30,
                 sc_mats=None):
        """Initialize SupercellMatrixEnumerator.

        Args:
            cluster_space(ClusterSubspace):
                An un-trimmed ClusterSubspace as generated in
                InputsHandler.
            conv_mat(3*3 ArrayLike):
                Conventional matrix from a primitive cell to
                a conventional cell. Default to None.
            supercell_from_conventional(bool):
                Whether to find out primitive cell to conventional
                standard structure transformation matrix T, and enumerate
                super-cell matrices in the form of: M = M'T.
                Default to true. If not, will set T to eye(3).
            objective_sc_size(int):
                The Supercel sizes (in number of prims) to approach.
                Default to 32. Enumerated super-cell size will be
                a multiple of det(T) but the closest one to this objective
                size.
                Note: since super-cell matrices with too high a conditional
                number will be dropped, do not use a super-cell size whose
                decompose to 3 integer factors are different in scale.
                For example, 17 = 1 * 1 * 17 is the only possible factor
                decomposition for 17, whose matrix conditional number will
                always be larger than the cut-off (8).
            max_sc_cond(float):
                Maximum conditional number of the supercell lattice matrix.
                Default to 8, prevent overly slender super-cells.
            min_sc_angle(float):
                Minimum allowed angle of the supercell lattice.
                Default to 30, prevent overly skewed super-cells.
            sc_mats(List[3*3 ArrayLike[int]]):
                Supercell matrices. Will not enumerate super-cells if this
                is given. Default to None.
        """
        self._cluster_space = cluster_space
        self.supercell_from_conventional = supercell_from_conventional
        if conv_mat is not None and supercell_from_conventional:
            self.conv_mat = np.round(conv_mat).astype(int)
        else:
            self.conv_mat = np.eye(3, dtype=int)
        self.conv_size = int(round(abs(np.linalg.det(self.conv_mat))))
        self.sc_size = objective_sc_size // self.conv_size * self.conv_size
        self.max_sc_cond = max_sc_cond
        self.min_sc_angle = min_sc_angle
        self._sc_mats = sc_mats or self._enumerate_sc_mats()
        self._sc_mats = [np.round(m).astype(int) for m in self._sc_mats]

    def _enumerate_sc_mats(self):
        """Enumerate supercell matrices."""
        return enumerate_matrices(self.sc_size,
                                  self._cluster_space,
                                  conv_mat=self.conv_mat,
                                  max_sc_cond=self.max_sc_cond,
                                  min_sc_angle=self.min_sc_angle)

    @property
    def supercell_matrices(self):
        """Supercell matrices.

        If enumerated, their maximum conditional number will not exceed
        max_sc_cond, while their minimum angle will not be lower than
        min_sc_angle. 2 optimum super-cell matrices, one diagonal, one
        skew, will be chosen to minimize the degree of alias in
        cluster subspace.
        """
        return self._sc_mats

    def truncate_subspace(self):
        """Truncate aliased orbits.

        Aliased orbits with higher indices will be removed, only the orbit
        with minimal index in alias list will be kept.
        Only call this at the 1st iteration.
        """
        alias = []
        for m in self.supercell_matrices:
            alias_m = self._cluster_space.get_aliasd_orbits(m)
            alias_m = {sorted(sub_orbit)[0]: set(sorted(sub_orbit)[1:])
                       for sub_orbit in alias_m}
            alias.append(alias_m)
        to_remove = alias[0]
        for alias_m in alias[1:]:
            for key in to_remove:
                if key in alias_m:
                    to_remove[key] = to_remove[key].intersection(alias_m[key])
        to_remove = list(set(chain(*to_remove)))
        self._cluster_space.remove_orbits(to_remove)

    @property
    def cluster_subspace(self):
        """Cluster subspace.

        Will only be truncated until truncate_subspace is called.
        """
        return self._cluster_space

    def as_dict(self):
        """Serialize an object.

        Returns:
            dict.
        """
        return {
            "cluster_space": self.cluster_subspace.as_dict(),
            "conv_mat": self.conv_mat.tolist(),
            "supercell_from_conventional": self.supercell_from_conventional,
            "objective_sc_size": self.sc_size,
            "max_sc_cond": self.max_sc_cond,
            "min_sc_angle": self.min_sc_angle,
            "sc_mats": [m.tolist() for m in self.supercell_matrices],
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, d):
        """Reload from dictionary.

        Args:
            d(dict):
                Serialized dictionary of Enumerator.
        Returns:
            SupercellMatrixEnumerator.
        """
        d["cluster_space"] = ClusterSubspace.from_dict(d["cluster_space"])
        if "@module" in d:
            _ = d.pop("@module")
        if "@class" in d:
            _ = d.pop("@class")
        return cls(**d)


class CompositionEnumerator(MSONable):
    """Composition enumeration.

    Attributes:
        compositions(np.ndarray[int]):
            Enumerated compositions, in "n"-format of CompSpace,
            not normalized by super-cell size.
    """

    def __init__(self, sc_size,
                 comp_space=None,
                 bits=None,
                 sublattice_sizes=None,
                 charge_balanced=True,
                 other_constraints=None,
                 leq_constraints=None,
                 geq_constraints=None,
                 comp_enumeration_step=1,
                 compositions=None
                 ):
        """Initialize CompositionEnumerator.

        Args:
            sc_size(int):
                Size of super-cell in number of prim cells.
            comp_space(CompSpace):
                A composition space object.
            bits(List[List[Specie|Vacancy|Element]]): optional
                Species on each sub-lattice.
                Must at least give one of comp_space or bits.
                comp_space is used as priority.
            sublattice_sizes(1D ArrayLike[int]): optional
                Number of sites in each sub-lattice per primitive cell.
                If not given, assume one site for each sub-lattice.
                Better provide them as co-prime integers.
            charge_balanced(bool): optional
                Whether to add charge balance constraint. Default
                to true.
            other_constraints(List[tuple(1D arrayLike[float], float)]): optional
                Other equality type composition constraints except charge balance
                and site-number conservation. Should be given in the form of
                tuple(a, bb), each gives constraint np.dot(a, n)=bb. a and bb
                should be in the form of per primitive cell.
                For example, you may want to constrain n_Li + n_Vac = 0.5 per
                primitive cell.
            leq_constraints(List[tuple(1D arrayLike[float], float)]): optional
                Constraint np.dot(a, n)<=bb. a and bb should be in the form of
                per primitive cell.
            geq_constraints(List[tuple(1D arrayLike[float], float)]): optional
                Constraint np.dot(a, n)>=bb. a and bb should be in the form of
                per primitive cell.
                Both leq and geq constraints are only used when enumerating
                compositions. Table ergodicity code will only consider equality
                constraints, not leq and geqs.
            comp_enumeration_step(int): optional
                Step in returning the enumerated compositions.
                If step = N > 1, on each dimension of the composition space,
                we will only yield one composition every N compositions.
                Default to 1.
            compositions(2D ArrayLike[int]): optional
                Enumerated compositions. If given, will not do enumeration
                on compositions again.
        """
        if comp_space is None and bits is None:
            raise ValueError("Must at least provide one of comp_space "
                             "and bits.")
        self._comp_space = comp_space or\
            CompSpace(bits, sublattice_sizes,
                      charge_balanced=charge_balanced,
                      other_constraints=other_constraints,
                      leq_constraints=leq_constraints,
                      geq_constraints=geq_constraints)
        self.step = comp_enumeration_step
        self.sc_size = sc_size
        self._compositions = compositions or self._enumerate_compositions()
        self._compositions = np.round(self._compositions).astype(int)

    def _enumerate_compositions(self):
        """Enumerate compositions in n-format."""
        xs = self._comp_space.get_comp_grid(sc_size=self.sc_size,
                                            step=self.step)
        ns = [self._comp_space.translate_format(x, self.sc_size,
                                                from_format="x",
                                                to_format="n",
                                                rounding=True)
              for x in xs]
        return np.array(ns).astype(int)

    @property
    def compositions(self):
        """Enumerated compositions."""
        return self._compositions

    def as_dict(self):
        """Serialize an object.

        Returns:
            dict.
        """
        return {
            "comp_space": self._comp_space.as_dict(),
            "step": self.step,
            "sc_size": self.sc_size,
            "compositions": self._compositions.tolist(),
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, d):
        """Reload from dictionary.

        Args:
            d(dict):
                Serialized dictionary of Enumerator.
        Returns:
            CompositionEnumerator.
        """
        d["comp_space"] = CompSpace.from_dict(d["comp_space"])
        if "@module" in d:
            _ = d.pop("@module")
        if "@class" in d:
            _ = d.pop("@class")
        return cls(**d)


# TODO: finish StructureEnumerator.
class StructureEnumerator:
    """Structure enumeration class."""

    def __init__(self, data_manager, history_wrapper):

        """Initialize StructureEnumerator.

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

        self.transmat = self.inputs_wrapper.enumerator_options['transmat']
        sc_size = self.inputs_wrapper.enumerator_options['sc_size']
        self.sc_size = sc_size

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
        self.n_strs_per_comp_init = (self.inputs_wrapper.
            enumerator_options['n_strs_per_comp_init'])
        self.n_strs_per_comp_add = (self.inputs_wrapper.
            enumerator_options['n_strs_per_comp_add'])

        self.handler_args = (self.inputs_wrapper.
            enumerator_options['handler_args_enum'])

        self.ce = history_wrapper.last_ce

        if self.sc_size % self.comp_enumstep != 0:
            raise ValueError("Composition enumeration step can not" +
                             "divide supercell size {}.".format(self.sc_size))

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
        return self._dm

    @property
    def inputs_wrapper(self):
        """Inputs wrapper object.

        Returns:
           InputsWrapper.
        """
        return self.data_manager._iw

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

    @property
    def compspace(self):
        return self._dm.compspace

    def set_sc_matrices(self, matrices=[]):
        """Set supercell matrices in DataManager.

        Recommended not to call after 1st cycle.
        Args:
            matrices(List[3*3 ArrayLike[int]]): 
                Supercell matrices to insert
        """
        for m in matrices:
            if isinstance(m, np.ndarray):
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
        if len(self.comp_df) > 0:
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

        all_comps = []
        # Just in case you want to implement variable super cell size later.
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
                                          self.bits,
                                          self.sl_sizes,
                                          self.comp_restrictions):
                    comp_id = self._dm.insert_one_comp(ucoord)
                    all_comps.append(comp)

        return all_comps

    def generate_structures(self, n_par=4, keep_gs=True,
                            weight_by_comp=True,
                            iter_id=0):
        """Enumerate structures under all (sc_matrix, composition).

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
        time_init = time.time()

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

        log.info("**Generating structure samples under compositions.")
        all_occus_n = pool.map(partial(raw_sample_under_sc_comp,
                                       self.ce,
                                       self.handler_args),
                               all_sc_comps)
        pool.close()
        pool.join()

        log.info("**MC sample generation time: {} s."
                 .format(time.time() - time_init))
        # Filter raw samples by a composition weight.
        if iter_id <= 0:
            n_strs_per_comp = self.n_strs_per_comp_init
        else:
            n_strs_per_comp = self.n_strs_per_comp_add

        all_occus, weights = list(zip(*all_occus_n))
        weights = np.array(weights) / np.max(weights)
        W = np.sum(weights)
        N_pool = max(n_strs_per_comp * len(weights) * 3,
                     sum([len(occus) for occus in all_occus]))
        n_selects = [min(max(n_strs_per_comp,
                             int(round(w / W * N_pool))),
                         len(occus))
                     for occus, w in zip(all_occus, weights)]

        if weight_by_comp:
            if keep_gs:
                all_occus_filt = [[occus[0]] +
                                  random.sample(occus[1:], n - 1)
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
        is_gs = []  # List of booleans

        dm_copy = self._dm.copy()
        for (sc_id, comp_id), occus in zip(all_sc_comp_ids, all_occus_filt):
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

        log.info("**Generated {} new deduplicated structures in {} s."
                 .format(len(inserted_eids), time.time() - time_init))

        # Compute indices to keep if keep_gs.
        keep = []
        if keep_gs:
            keep = [i for i in range(len(is_gs)) if is_gs[i]]

        # Selection.
        femat_filt = dm_copy.fact_df.entry_id.isin(inserted_eids)
        femat = dm_copy.fact_df.loc[femat_filt,
                                    'ori_corr'].tolist()
        old_femat = self.fact_df.ori_corr.tolist()

        n_enum = n_strs_per_comp * len(weights)

        if n_enum > len(femat):
            log.warning("**Number of deduplicated structures " +
                        "fewer than the number you wish to " +
                        "enumerate! Consider increasing number of " +
                        "structures per composition.")

        selected_rids = select_rows(femat,
                                    n_select=min(n_enum, len(femat)),
                                    old_femat=old_femat,
                                    method=self.select_method,
                                    keep=keep)
        unselected_eids = [inserted_eids[i] for i in range(len(femat))
                           if i not in selected_rids]
        log.critical("**Added {}/{} new structures. Time: {} s."
                     .format(len(selected_rids), len(femat),
                             time.time() - time_init)
                     )

        # Remove unselected entree.
        dm_copy.remove_entree_by_id(unselected_eids)

        # Update to real DataManager.
        self._dm = dm_copy.copy()

        filt_ = (self.fact_df.iter_id == iter_id)
        return self.fact_df.loc[filt_, :]

    # If you want to clear enumeration, use remove() method in DataManager.

# Don't save or load this object. Save or load data with DataManager,
# InputsWrapper, HistoryWrapper, TimeKeeper, etc.
