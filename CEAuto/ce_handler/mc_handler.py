"""Monte-carlo handlers to compute ground states and sample structures."""

__author__="Fengyu Xie"

import logging
log = logging.getLogger(__name__)

import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
import random

from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe.space.domain import get_allowed_species
from smol.moca import CanonicalEnsemble, Sampler

from ..utils.comp_utils import scale_compstat
from ..utils.calc_utils import get_ewald_from_occu
from ..utils.class_utils import derived_class_factory
from ..utils.math_utils import GCD
from ..utils.occu_utils import get_all_sublattices


class MCHandler(ABC):
    """Base monte-carlo handler class.

    Provides ground states, defreeze sampling.

    Note: In the future, will support auto-equilibration.
    """
    def __init__(self, ce, sc_mat,
                 gs_occu=None,
                 anneal_series=[5000, 3200, 1600, 800, 400, 100],
                 unfreeze_series=[500, 1500, 5000],
                 n_runs_sa=300,
                 n_runs_unfreeze=600,
                 n_samples=100,
                 **kwargs):
        """Initialize.

        Args:
            ce(ClusterExpansion):
                A cluster expansion object to solve on.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
            anneal_series(List[float]):
                A series of temperatures to use in simulated annealing.
                Must be strictly decreasing.
            unfreeze_series(List[float]):
                A series of increasing temperatures to sample on.
                By default, will sample under 500, 1500 and 5000 K.
            n_runs_sa(int):
                Number of runs per simulated annealing step. 1 run = 
                # of sites in a supercell.
            n_runs_unfreeze(int):
                Number of runs per unfreezing step. 1 run = 
                # of sites in a supercell.
            n_samples(int):
                Number of random occupancies to slice each sampling run.
                Deduplicated sample will be smaller.
            gs_occu(List[int]):
                Encoded occupation array of previous ground states.
                 Optional, but if you have it, you can save the 
                 ground state solution time when sampling.
        """
        self.ce = ce
        self.sc_mat = np.array(sc_mat, dtype=int)

        self._ensemble = None
        self._sampler = None

        self.anneal_series = anneal_series
        self.unfreeze_series = unfreeze_series
        self.n_runs_sa = n_runs_sa
        self.n_runs_unfreeze = n_runs_unfreeze
        self.n_samples = n_samples

        self.is_indicator = (self.ce.cluster_subspace.orbits[0]
                             .basis_type == 'indicator')

        self.sc_size = int(round(abs(np.linalg.det(sc_mat))))

        self.prim = self.ce.cluster_subspace.structure

        self._gs_occu = (np.array(gs_occu, dtype=int)
                         if gs_occu is not None else None)

        self._all_sublattices = None

    @property
    def all_sublattices(self):
        """List of all active and inactive sublattices.

        Order determined as in smol.moca.get_sublattices.
        Returns:
            List[Sublattice]
        """
        if self._all_sublattices is None:
            self._all_sublattices = get_all_sublattices(self.
                                                        processor)
        return self._all_sublattices

    @property
    def sc_sublat_list(self):
        """List of sublattice sites in a supercell.

        Same ordering as smol.moca.ensemble.sublattice
        get_all_sublattices.
        """
        return [s.sites for s in self.all_sublattices]

    @property
    def sl_sizes(self):
        """Sublattice sizes in prim cell."""
        return [len(s) // self.sc_size for s in self.sc_sublat_list]

    @property
    def bits(self):
        """List of sublattice species in a supercell.

        Same ordering as smol.moca.ensemble.sublattice
        get_all_sublattices.
        """
        return [s.species for s in self.all_sublattices]

    @property
    @abstractmethod
    def ensemble(self):
        """Get ensemble to run."""
        return

    @property
    @abstractmethod
    def sampler(self):
        """Get sampler to run."""
        return

    @property
    def processor(self):
        """Processor."""
        return self.ensemble.processor

    def _get_min_occu_enthalpy(self):
        """Get minimum thermo function from the current ensemble's sampler.

        Different ensemble types have different thermo potentials. For 
        example, E for canonical ensemble, E-mu*x for semi-grand canonical 
        ensemble.
        In smol.moca, this quantity is called 'enthalpy'.
        """
        gs_occu = self.sampler.samples.get_minimum_enthalpy_occupancy()
        gs_enth = self.sampler.samples.get_minimum_enthalpy()
        return gs_occu, gs_enth

    def _initialize_occu_from_int_comp(self, int_comp):
        """Get an initial occupation array.

        Args:
            int_comp(List[List[int]]):
                integer composition, in compstat form.

        Output:
            init_occu:
                Arraylike of integers. Encoded occupation array.
        """
        rand_occus = []
        rand_ewalds = []

        for i in range(50):
            # Occupancy is coded
            occu = np.zeros(len(self.prim) * self.sc_size, dtype=int)
            for sl_id, (sl_int_comp, sl_sites) in \
              enumerate(zip(int_comp, self.sc_sublat_list)):
                if sum(sl_int_comp) != len(sl_sites):
                    raise ValueError("Num of sites can't match " +
                                     "composition on sublattice {}."
                                     .format(sl_id) +
                                     "Composition: {}, "
                                     .format(sl_int_comp) +
                                     "Number of sites: {}."
                                     .format(len(sl_sites)))

                sl_sites_shuffled = deepcopy(sl_sites)
                random.shuffle(sl_sites_shuffled)

                n_assigned = 0
                for sp_id, n_sp in enumerate(sl_int_comp):
                    sp_sites = sl_sites_shuffled[n_assigned : n_assigned + n_sp]
                    occu[sp_sites] = sp_id
                    n_assigned += n_sp
                assert n_assigned == len(sl_sites)

            rand_occus.append(occu)
            rand_ewalds.append(get_ewald_from_occu(occu, self.prim,
                               self.sc_mat))

        return rand_occus[np.argmin(rand_ewalds)]

    def get_ground_state(self):
        """
        Use simulated annealing to solve a ground state under the current
        condition.

        Returns:
            gs_occu, gs_e
        """
        n_steps_anneal = self.sc_size * len(self.prim) * self.n_runs_sa

        log.debug("****Annealing to the ground state. T series: {}."
                  .format(self.anneal_series))

        self.sampler.anneal(self.anneal_series, n_steps_anneal,
                            initial_occupancies=np.array([self._gs_occu],
                                                         dtype=int))

        log.info("****GS annealing finished!")
        gs_occu, gs_e = self._get_min_occu_enthalpy()
  
        # Updates
        self._gs_occu = gs_occu
        self.sampler.samples.clear()
        return gs_occu, gs_e

    def get_unfreeze_sample(self, progress=False):
        """
        Built in method to generate low-to medium energy occupancies
        under a supercell matrix and a fixed composition.
 
        Args:
            progress(Boolean):
                Whether or not to show progress bar during equilibration
                and generating. Default to False.

        Return:
            sample_occus(List[List[int]]):
                A list of sampled encoded occupation arrays. The first
                one will always be the one with lowest energy!
        """

        # Anneal n_atoms*100 per temp, Sample n_atoms*500, give 300 samples
        # for practical computation
        n_steps_sample = self.sc_size * len(self.prim) * self.n_runs_unfreeze
        thin_by = max(1, n_steps_sample // self.n_samples)

        sa_occu, sa_e = self.get_ground_state()
 
        # Will always contain GS structure at the first position in list
        rand_occus = [list(deepcopy(sa_occu))]

        # Sampling temperatures        
        for T in self.unfreeze_series:
            log.debug('******Getting samples under {} K.'.format(T))
            self.sampler.samples.clear()
            self.sampler._kernel.temperature = T

            # Equilibriate
            log.debug("******Equilibration run.")
            self.sampler.run(n_steps_sample,
                             initial_occupancies=np.array([sa_occu],
                                                          dtype=int),
                             thin_by=thin_by,
                             progress=progress)
            sa_occu = np.array(self.sampler.samples.get_occupancies()[-1],
                               dtype=int)
            self.sampler.samples.clear()

            # Sampling
            log.debug("******Generation run.")
            self.sampler.run(n_steps_sample,
                             initial_occupancies=np.array([sa_occu],
                                                          dtype=int),
                             thin_by=thin_by,
                             progress=progress)
            rand_occus.extend(np.array(self.sampler.samples.get_occupancies(),
                              dtype=int).tolist())

        self.sampler.samples.clear()
        rand_strs = [self.processor.structure_from_occupancy(occu)
                     for occu in rand_occus]

        # Internal deduplication
        sm = StructureMatcher()

        rand_dedup = []
        for s1_id, s1 in enumerate(rand_strs):
            duped = False
            for s2_id in rand_dedup:
                if sm.fit(s1, rand_strs[s2_id]):
                    duped = True
                    break
            if not duped:
                rand_dedup.append(s1_id)

        log.info("****{} unique structures generated."
                     .format(len(rand_dedup)))

        rand_occus_dedup = [rand_occus[s_id] for s_id in rand_dedup]

        return random.sample(rand_occus_dedup,
                             min(len(rand_occus_dedup),
                                 self.n_samples))


class CanonicalmcHandler(MCHandler):
    """MC handler in canonical ensemble."""
    def __init__(self, ce, sc_mat, compstat,
                 gs_occu=None,
                 anneal_series=[5000, 3200, 1600, 800, 400, 100],
                 unfreeze_series=[500, 1500, 5000],
                 n_runs_sa=300,
                 n_runs_unfreeze=600,
                 n_samples=300,
                 **kwargs):
        """Initialize.

        Args:
            ce(ClusterExpansion):
                A cluster expansion object to solve on.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
            compstat(2D List):
                Compositional statistics table, normalized by supercell size.
            anneal_series(List[float]):
                A series of temperatures to use in simulated annealing.
                Must be strictly decreasing.
            unfreeze_series(List[float]):
                A series of increasing temperatures to sample on.
                By default, will sample under 500, 1500 and 5000 K.
            n_runs_sa(int):
                Number of runs per simulated annealing step. 1 run = 
                # of sites in a supercell.
            n_runs_unfreeze(int):
                Number of runs per unfreezing step. 1 run = 
                # of sites in a supercell.
            n_samples(int):
                Number of random occupancies to slice each sampling run.
                Deduplicated sample will be smaller.
            gs_occu(List[int]):
                Encoded occupation array of previous ground states.
                Optional, but if provided, must have the same composition
                as compstat.
        """
        super().__init__(ce, sc_mat,
                         gs_occu=gs_occu,
                         anneal_series=anneal_series,
                         unfreeze_series=unfreeze_series,
                         n_runs_sa=n_runs_sa,
                         n_runs_unfreeze=n_runs_unfreeze,
                         n_samples=n_samples,
                         **kwargs)

        self.compstat = compstat
        self.int_comp = scale_compstat(compstat, by=self.sc_size)

        self._gs_occu = gs_occu or self._initialize_occu_from_int_comp(self.int_comp)

    @property
    def ensemble(self):
        """CanonicalEnsemble."""
        if self._ensemble is None:
            self._ensemble = (CanonicalEnsemble.
                              from_cluster_expansion(self.ce, self.sc_mat,
                                                     optimize_indicator=
                                                     self.is_indicator))
        return self._ensemble

    @property
    def sampler(self):
        """Sampler to run."""
        if self._sampler is None:
            self._sampler = Sampler.from_ensemble(self.ensemble,
                                                  temperature=5000,
                                                  nwalkers=1)
        return self._sampler



def mchandler_factory(mchandler_name, *args, **kwargs):
    """Create a MCHandler with given name.

    Args:
        mchandler_name(str):
            Name of a MCHandler class.
        *args, **kwargs:
            Arguments used to intialize the class.
    """
    name = mchandler_name.capitalize() + 'Handler'
    return derived_class_factory(name, MCHandler, *args, **kwargs)
