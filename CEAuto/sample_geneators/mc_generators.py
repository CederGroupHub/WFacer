"""Monte-carlo handlers to estimate ground states and sample structures."""

import numpy as np
from copy import deepcopy
from abc import ABCMeta, abstractmethod
import random
from monty.json import MSONable

from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.moca import CanonicalEnsemble, Sampler

__author__ = "Fengyu Xie"


class McSampleGenerator(MSONable, metaclass=ABCMeta):
    """Abstract monte-carlo sampler class.

    Provides unfreeze sampling, and saving of previously generated
    samples for the purpose of de-duplicating.
    """
    default_anneal_series = [5000, 3200, 1600, 800, 400, 100]
    default_unfreeze_series = [500, 1500, 5000]

    def __init__(self, ce, sc_mat,
                 anneal_series=None,
                 unfreeze_series=None,
                 n_steps_sa=50000,
                 n_steps_unfreeze=100000,
                 max_n_samples_per_iter=100,
                 past_occus=None,
                 **kwargs):
        """Initialize.

        Args:
            ce(ClusterExpansion):
                A cluster expansion object to enumerate with.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
            anneal_series(List[float]): optional
                A series of temperatures to use in simulated annealing.
                Must be mono-decreasing.
            unfreeze_series(List[float]): optional
                A series of increasing temperatures to sample on.
                Must be mono-increasing
            n_steps_sa(int): optional
                Number of steps to run per simulated annealing temperature.
            n_steps_unfreeze(int): optional
                Number of steps to run per unfreeze temperature.
            max_n_samples_per_iter(int): optional
                Maximum number of samples to draw per unfreezing
                run. Will generate as many structure candidates as possible,
                as long as they are not symmetrically equivalent with any
                past entries, but for each iteration will generate this
                many new structures at most.
            past_occus(2D ArrayLike):
                Occupancies enumerated in the past. For the purpose of
                de-duplication.
            kwargs:
                Not used. Just to mute excessive keywords.
        """
        self.ce = ce
        self.sc_mat = sc_mat
        self.sc_size = int(round(abs(np.linalg.det(sc_mat))))
        self.prim = self.ce.cluster_subspace.structure

        self.anneal_series = anneal_series or self.default_anneal_series
        self.unfreeze_series = unfreeze_series or self.default_unfreeze_series
        self.n_steps_sa = n_steps_sa
        self.n_steps_unfreeze = n_steps_unfreeze
        self.max_n_samples_per_iter = max_n_samples_per_iter

        self._gs_occu = None  # Cleared per-initialization.
        self._ensemble = None
        self._sampler = None

        self._past_occus = past_occus or []

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
        """Get Processor."""
        return self.ensemble.processor

    @property
    def sublattices(self):
        """Get sublattices in ensemble.

        Note: If you wish to do delicate operations such as sub-lattice
        splitting, please do it on self.ensemble.
        See docs of smol.moca.ensemble.
        """
        return self.ensemble.sublattices

    def _initialize_occu_from_n(self, n):
        """Get an initial occupancy string from n-format composition.

        Args:
            n(1D arrayLike[int]):
                An integer composition "n-format". For what is an
                "n-format", refer to comp_space.

        Output:
            Encoded occupancy array with composition n:
                np.ndarray[int]
        """
        n_species = 0
        occu = np.zeros(self.ensemble.num_sites, dtype=int) - 1
        for sublatt in self.sublattices:
            n_sublatt = n[n_species: n_species + len(sublatt.encoding)]
            if np.sum(n_sublatt) != len(sublatt.sites):
                raise ValueError(f"Composition n: {n} does not match "
                                 f"super-cell size on sub-lattice: {sublatt}")
            occu_sublatt = [code for code, nn in zip(sublatt.encoding, n_sublatt)
                            for _ in range(nn)]
            np.random.shuffle(occu_sublatt)
            occu[sublatt.sites] = occu_sublatt
            n_species += len(sublatt.encoding)
        assert n_species == len(n)
        assert not np.any(np.isclose(occu, -1))

    @abstractmethod
    def _get_init_occu(self):
        return []

    def get_ground_state(self, thin_by=1):
        """Use simulated anneal to solve for ground state.

        Args:
            thin_by(int):
                Steps to thin sampler by. See documentation
                for smol.moca.sampler. This is used to save
                memory space.
        Returns:
            ground state in encoded occupancy array:
                np.ndarray[int]
        """
        if self._gs_occu is None:
            init_occu = self._get_init_occu()

            self.sampler.anneal(self.anneal_series, self.n_steps_sa,
                                initial_occupancies=np.array([init_occu],
                                                             dtype=int),
                                thin_by=thin_by)

            # Save updates.
            self._gs_occu = (self.sample.samples
                             .get_minimum_enthalpy_occupancy().copy())
            self.sampler.samples.clear()
  
        return self._gs_occu

    # TODO: continue to refactor this.
    def get_unfreeze_sample(self):
        """Generate a sample of structures by unfreezing the ground state.

        Return:
            Samples in occupancy string:
                2D np.ndarray[int]
        """
        thin_by = max(1, self.n_steps_unfreeze // (self.max_n_samples_per_iter * 5))
        # Thin so we don't have to compare too many structures.

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
