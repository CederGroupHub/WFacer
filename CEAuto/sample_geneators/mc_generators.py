"""Monte-carlo handlers to estimate ground states and sample structures."""

import numpy as np
from abc import ABCMeta, abstractmethod
from monty.json import MSONable

from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.moca import CanonicalEnsemble, Sampler
from smol.utils import derived_class_factory, class_name_from_str

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
                 past_corrs=None):
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
            past_corrs(2D ArrayLike):
                Correlations enumerated in the past. For the purpose of
                de-duplication.
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
        self._past_corrs = past_corrs or []
        self._past_occus = [np.array(o, dtype=int) for o in self._past_occus]
        self._past_corrs = [np.array(c) for c in self._past_corrs]

    @property
    @abstractmethod
    def ensemble(self):
        """Get ensemble to run."""
        return

    @property
    def sampler(self):
        """Sampler to run."""
        if self._sampler is None:
            self._sampler = Sampler.from_ensemble(self.ensemble,
                                                  temperature
                                                  =self.anneal_series[0],
                                                  nwalkers=1)
        return self._sampler

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
        return np.array([], dtype=int)

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

    def get_unfreeze_sample(self):
        """Generate a sample of structures by unfreezing the ground state.

        Note: this function should not be called multiple times per iteration!
        Return:
            New samples in occupancy string and correlation vectors, and whether
            a new ground-state has been detected as the first inserted value:
                list np.ndarray[int], list np.ndarray, bool
        """
        thin_by = max(1, self.n_steps_unfreeze
                      // (self.max_n_samples_per_iter * 5))
        # Thin so we don't have to compare too many structures.

        gs_occu = self.get_ground_state()

        # Will always contain GS at the first position in list.
        rand_occus = [gs_occu.copy()]
        rand_corrs = [self.ensemble.compute_feature_vector(gs_occu)
                      / self.sc_size]  # Should add normalized feature vectors.

        # Sampling temperatures
        for T in self.unfreeze_series:
            self.sampler.samples.clear()
            self.sampler.mckernel.temperature = T

            # Equilibrate and sampling.
            self.sampler.run(2 * self.n_steps_unfreeze,
                             initial_occupancies=np.array([gs_occu],
                                                          dtype=int),
                             thin_by=thin_by)
            n_samples = self.sampler.samples.num_samples
            rand_occus.extend(np.array(self.sampler.samples.get_occupancies()
                                       [n_samples // 2:],
                                       dtype=int))  # Take the last half.
            rand_corrs.extend(self.sampler.samples.get_feature_vectors()
                              [n_samples // 2:] / self.sc_size)
            self.sampler.samples.clear()

        # Symmetry deduplication
        sm = StructureMatcher()

        n_occus_past = len(self._past_occus)
        assert n_occus_past == len(self._past_corrs)
        new_gs_detected = False
        for new_id, (new_occu, new_corr) \
                in enumerate(zip(rand_occus, rand_corrs)):
            for old_id, (old_occu, old_corr) \
                    in enumerate(zip(self._past_occus, self._past_corrs)):
                dupe = False
                if np.allclose(new_corr, old_corr):
                    s_new = self.processor.structure_from_occupancy(new_occu)
                    s_old = self.processor.structure_from_occupancy(old_occu)
                    if sm.fit(s_new, s_old):
                        dupe = True
                        break
                if not dupe:
                    self._past_occus.append(new_occu)
                    self._past_corrs.append(new_corr)
                    if new_id == 1:
                        new_gs_detected = True
                if len(self._past_occus) == (n_occus_past
                                             + self.max_n_samples_per_iter):
                    break

        return (self._past_occus[n_occus_past:],
                self._past_corrs[n_occus_past:],
                new_gs_detected)

    def as_dict(self):
        """Serialize into dict."""
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "ce": self.ce.as_dict(),
             "sc_mat": self.sc_mat,
             "anneal_series": self.anneal_series,
             "unfreeze_series": self.unfreeze_series,
             "n_steps_sa": self.n_steps_sa,
             "n_steps_unfreeze": self.n_steps_unfreeze,
             "max_n_samples_per_iter": self.max_n_samples_per_iter,
             "past_occus": [o.tolist() for o in self._past_occus],
             "past_corrs": [c.tolist() for c in self._past_corrs]
             }
        return d


class CanonicalSampleGenerator(McSampleGenerator):
    """Sample generator in canonical ensemble."""

    def __init__(self, ce, sc_mat, n,
                 anneal_series=None,
                 unfreeze_series=None,
                 n_steps_sa=50000,
                 n_steps_unfreeze=100000,
                 max_n_samples_per_iter=100,
                 past_occus=None,
                 past_corrs=None):
        """Initialize.

        Args:
            ce(ClusterExpansion):
                A cluster expansion object to enumerate with.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
            n(1D ArrayLike[int]):
                Composition in the n-format. (not normalized by
                super-cell size!!)
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
            past_corrs(2D ArrayLike):
                Correlations enumerated in the past. For the purpose of
                de-duplication.
        """
        super(CanonicalSampleGenerator, self)\
            .__init__(ce, sc_mat,
                      anneal_series=anneal_series,
                      unfreeze_series=unfreeze_series,
                      n_steps_sa=n_steps_sa,
                      n_steps_unfreeze=n_steps_unfreeze,
                      max_n_samples_per_iter=max_n_samples_per_iter,
                      past_occus=past_occus,
                      past_corrs=past_corrs)

        self.n = n

    @property
    def ensemble(self):
        """CanonicalEnsemble."""
        if self._ensemble is None:
            self._ensemble = (CanonicalEnsemble.
                              from_cluster_expansion(self.ce, self.sc_mat))
        return self._ensemble

    def _get_init_occu(self):
        """Get an initial occupancy for MC run."""
        return self._initialize_occu_from_n(self.n)

# TODO: add semi-grand generator in the future?
#  But not urgent. Do this only after smol cn-sgmc has been
#  merged with main.


def mcgenerator_factory(mcgenerator_name, *args, **kwargs):
    """Create a MCHandler with given name.

    Args:
        mcgenerator_name(str):
            Name of a McSampleGenerator sub-class.
        *args, **kwargs:
            Arguments used to intialize the class.
    """
    name = class_name_from_str(mcgenerator_name)
    return derived_class_factory(name, McSampleGenerator, *args, **kwargs)
