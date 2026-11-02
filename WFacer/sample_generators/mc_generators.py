"""Monte-carlo to estimate ground states and sample structures."""

import itertools
from abc import ABCMeta, abstractmethod
from warnings import warn

import numpy as np
from pymatgen.core import Element
from smol.cofe.space.domain import Vacancy
from smol.moca import CompositionSpace, Ensemble, Sampler
from smol.utils.class_utils import class_name_from_str, derived_class_factory

from ..utils.duplicacy import is_corr_duplicate, is_duplicate
from ..utils.occu import get_random_occupancy_from_counts

__author__ = "Fengyu Xie"


class McSampleGenerator(metaclass=ABCMeta):
    """Abstract monte-carlo sampler class.

    Allows finding the ground state, and then gradually heats up the ground state
    to generate an unfrozen sample.

    Each Generator should only handle one composition (canonical) or one set of
    chemical potentials (grand-canonical)!
    """

    default_anneal_temp_series = [5000, 3200, 1600, 1000, 800, 600, 400, 200, 100]
    # Allow up to 1 eV above composition should be reasonable. Would also give us more
    # choice of sample structures when interactions are high.
    default_heat_temp_series = [500, 2000, 5000, 12000]

    def __init__(
        self,
        ce,
        sc_matrix,
        anneal_temp_series=None,
        heat_temp_series=None,
        num_steps_anneal=None,
        num_steps_heat=None,
        duplicacy_criteria="correlations",
        remove_decorations_before_duplicacy=False,
    ):
        """Initialize.

        Args:
            ce(ClusterExpansion):
                A cluster expansion object to enumerate with.
            sc_matrix(3*3 ArrayLike):
                Supercell matrix to solve on.
            anneal_temp_series(list of float): optional
                A series of temperatures to use in simulated annealing.
                Must be strictly decreasing.
            heat_temp_series(list of float): optional
                A series of increasing temperatures to sample on.
                Must be strictly increasing
            num_steps_anneal(int): optional
                The number of MC steps to run per annealing temperature.
            num_steps_heat(int): optional
                The number of MC steps to run per heating temperature.
            duplicacy_criteria(str):
                The criteria when to consider two structures as the same and
                old to add one of them into the candidate training set.
                Default is "correlations", which means to assert duplication
                if two structures have the same correlation vectors. While
                "structure" means two structures must be symmetrically equivalent
                after being reduced. No other option is allowed.
                Note that option "structure" might be significantly slower since
                it has to attempt reducing every structure to its primitive cell
                before matching. It should be used with caution.
            remove_decorations_before_duplicacy(bool): optional
                Whether to remove all decorations from species (i.e,
                charge and other properties) before comparing duplicacy.
                Default to false. Only valid when duplicacy_criteria="structure".
        """
        self.ce = ce
        self.sc_matrix = np.array(sc_matrix, dtype=int)
        self.sc_size = int(round(abs(np.linalg.det(sc_matrix))))
        self.prim = self.ce.cluster_subspace.structure

        self.anneal_temp_series = anneal_temp_series or self.default_anneal_temp_series
        self.heat_temp_series = heat_temp_series or self.default_heat_temp_series
        self._num_steps_anneal = num_steps_anneal
        self._num_steps_heat = num_steps_heat
        self.duplicacy_criteria = duplicacy_criteria
        self.remove_decorations = remove_decorations_before_duplicacy

        self._gs_occu = None  # Cleared per-initialization.
        self._ensemble = None
        self._sampler = None

    @property
    @abstractmethod
    def ensemble(self):
        """Get ensemble to run."""
        return

    @property
    def sampler(self):
        """Sampler to run."""
        return

    @property
    def processor(self):
        """Get Processor."""
        return self.ensemble.processor

    @property
    def sublattices(self):
        """Get sublattices in ensemble.

        .. note:: If you wish to do delicate operations such as sub-lattice
         splitting, please do it on self.ensemble. Refer to
         :class:`smol.moca.ensemble` for further details.
        """
        return self.ensemble.sublattices

    @property
    def n_steps_per_scan(self):
        """Least number of steps required to span configurations."""
        return np.sum(
            [
                len(sublatt.encoding) * len(sublatt.active_sites)
                for sublatt in self.sublattices
            ]
        )

    @property
    def num_steps_anneal(self):
        """Number of steps to run at each temperature when annealing."""
        if self._num_steps_anneal is None:
            # By default, run over all configurations for 20 times.
            self._num_steps_anneal = self.n_steps_per_scan * 20
        return self._num_steps_anneal

    @property
    def num_steps_heat(self):
        """Number of steps to run at each temperature when annealing."""
        if self._num_steps_heat is None:
            # By default, run over all configurations for 40 times.
            self._num_steps_heat = self.n_steps_per_scan * 40
        return self._num_steps_heat

    @abstractmethod
    def _get_init_occu(self):
        return

    def get_ground_state_occupancy(self):
        """Use simulated annealing to solve the ground state occupancy.

        Returns:
            list of int:
             The ground-state occupancy string obtained through
             simulated annealing.

        """
        if self._gs_occu is None:
            init_occu = self._get_init_occu()

            self.sampler.anneal(
                self.anneal_temp_series,
                self.num_steps_anneal,
                initial_occupancies=np.array([init_occu], dtype=int),
            )

            # Save updates.
            self._gs_occu = (
                self.sampler.samples.get_minimum_enthalpy_occupancy()
                .astype(int)
                .tolist()
            )
            self.sampler.samples.clear()  # Free-up mem.

        return self._gs_occu

    def get_ground_state_structure(self):
        """Get the ground state structure.

        Returns:
            Structure.
        """
        return self.processor.structure_from_occupancy(
            self.get_ground_state_occupancy()
        )

    def get_ground_state_features(self):
        """Get the feature vector of the ground state.

        Returns:
            list of float.
        """
        gs_occu = self.get_ground_state_occupancy()
        return (
            self.processor.compute_feature_vector(np.array(gs_occu))
            / self.processor.size
        ).tolist()

    def get_unfrozen_sample(
        self,
        previous_sampled_structures=None,
        previous_sampled_features=None,
        num_samples=100,
    ):
        """Generate a sample of structures by heating the ground state.

        Args:
            previous_sampled_structures(list of Structure): optional
                Sample structures already calculated in past
                iterations.
            previous_sampled_features(list of ArrayLike): optional
                Feature vectors of sample structures already
                calculated in past iterations.
            num_samples(int): optional
                Maximum number of samples to draw per unfreezing
                run. Since Structures must be de-duplicated, the actual
                number of structures returned might be fewer than this
                threshold. Default to 100.

        Return:
            list of Structure, list of lists of int, list of lists of float:
                New samples structures, NOT including the ground-state,
                sampled occupancy arrays, and feature vectors of sampled
                structures.
        """
        previous_sampled_structures = previous_sampled_structures or []
        previous_sampled_features = previous_sampled_features or []
        if len(previous_sampled_features) != len(previous_sampled_structures):
            raise ValueError(
                "Must provide a feature vector for each" " structure passed in!"
            )

        thin_by = max(
            1, len(self.heat_temp_series) * self.num_steps_heat // (num_samples * 5)
        )
        # Thin so we don't have to de-duplicate too many structures.
        # Here we leave out 10 * num_samples to compare.

        gs_occu = self.get_ground_state_occupancy()
        gs_feature = self.get_ground_state_features()
        gs_str = self.get_ground_state_structure()

        # Will always contain GS at the first position in list.
        rand_occus = []
        init_occu = gs_occu.copy()

        # Sampling temperatures
        for T in self.heat_temp_series:
            self.sampler.samples.clear()
            self.sampler.mckernels[0].temperature = T

            # Equilibrate and sampling.
            self.sampler.run(
                2 * self.num_steps_heat,
                initial_occupancies=np.array([init_occu], dtype=int),
                thin_by=thin_by,
            )
            init_occu = self.sampler.samples.get_occupancies()[-1].astype(int)
            n_samples = self.sampler.samples.num_samples
            # Take the last half as equlibrated, only.
            rand_occus.extend(
                self.sampler.samples.get_occupancies(discard=n_samples // 2)
                .astype(int)
                .tolist()
            )

        # Symmetry deduplication
        rand_strs = [
            self.processor.structure_from_occupancy(occu) for occu in rand_occus
        ]
        rand_feats = [
            (
                self.processor.compute_feature_vector(np.array(occu))
                / self.processor.size
            ).tolist()
            for occu in rand_occus
        ]
        new_ids = []
        for new_id, new_str in enumerate(rand_strs):
            dupe = False
            old_strs = (
                previous_sampled_structures
                + [gs_str]
                + [rand_strs[ii] for ii in new_ids]
            )
            old_feats = (
                previous_sampled_features
                + [gs_feature]
                + [rand_feats[ii] for ii in new_ids]
            )
            for old_id, (old_str, old_feat) in enumerate(zip(old_strs, old_feats)):
                # Must remove decorations to avoid getting fully duplicate inputs.
                if self.duplicacy_criteria == "correlations":
                    dupe = is_corr_duplicate(
                        new_str, self.processor, features2=old_feat
                    )
                elif self.duplicacy_criteria == "structure":
                    dupe = is_duplicate(
                        old_str, new_str, remove_decorations=self.remove_decorations
                    )
                else:
                    raise ValueError(
                        f"{self.duplicacy_criteria} comparison not" f" supported!"
                    )
                if dupe:
                    break
            if not dupe:
                new_ids.append(new_id)

            if len(new_ids) == num_samples:
                break

        if len(new_ids) < num_samples:
            warn(
                f"Expected to enumerate {num_samples} structures,"
                f" but only {len(new_ids)} un-duplicate structures"
                f" could be generated!"
            )
        return (
            [rand_strs[i] for i in new_ids],
            [rand_occus[i] for i in new_ids],
            [rand_feats[i] for i in new_ids],
        )


class CanonicalSampleGenerator(McSampleGenerator):
    """Sample generator in canonical ensembles."""

    def __init__(
        self,
        ce,
        sc_matrix,
        counts,
        anneal_temp_series=None,
        heat_temp_series=None,
        num_steps_anneal=None,
        num_steps_heat=None,
        duplicacy_criteria="correlations",
        remove_decorations_before_duplicacy=False,
    ):
        """Initialize.

        Args:
            ce(ClusterExpansion):
                A cluster expansion object to enumerate with.
            sc_matrix(3*3 ArrayLike):
                Supercell matrix to solve on.
            counts(1D ArrayLike of int):
                Composition in the "counts " format, not normalized by
                number of primitive cells per super-cell. Refer to
                :mod:`smol.moca.composition` for explanation.
            anneal_temp_series(list of float): optional
                A series of temperatures to use in simulated annealing.
                Must be strictly decreasing.
            heat_temp_series(list of float): optional
                A series of increasing temperatures to sample on.
                Must be strictly increasing
            num_steps_anneal(int): optional
                The number of steps to run per simulated annealing temperature.
            num_steps_heat(int): optional
                The number of steps to run per heat temperature.
            duplicacy_criteria(str):
                The criteria when to consider two structures as the same and
                old to add one of them into the candidate training set.
                Default is "correlations", which means to assert duplication
                if two structures have the same correlation vectors. While
                "structure" means two structures must be symmetrically equivalent
                after being reduced. No other option is allowed.
                Note that option "structure" might be significantly slower since
                it has to attempt reducing every structure to its primitive cell
                before matching. It should be used with caution.
            remove_decorations_before_duplicacy(bool): optional
                Whether to remove all decorations from species (i.e,
                charge and other properties) before comparing duplicacy.
                Default to false. Only valid when duplicacy_criteria="structure".
        """
        super().__init__(
            ce,
            sc_matrix,
            anneal_temp_series=anneal_temp_series,
            heat_temp_series=heat_temp_series,
            num_steps_anneal=num_steps_anneal,
            num_steps_heat=num_steps_heat,
            duplicacy_criteria=duplicacy_criteria,
            remove_decorations_before_duplicacy=remove_decorations_before_duplicacy,
        )

        self.counts = np.round(counts).astype(int)

    @property
    def ensemble(self):
        """CanonicalEnsemble."""
        # Must use "expansion" as the processor type to give correlation vectors.
        if self._ensemble is None:
            self._ensemble = Ensemble.from_cluster_expansion(
                self.ce, self.sc_matrix, processor_type="expansion"
            )
        return self._ensemble

    @property
    def sampler(self):
        """A sampler to sample structures."""
        if self._sampler is None:
            # Check if charge balance is needed.
            self._sampler = Sampler.from_ensemble(
                self.ensemble, temperature=self.anneal_temp_series[0], nwalkers=1
            )
        return self._sampler

    def _get_init_occu(self):
        """Get an initial occupancy for MC run."""
        return get_random_occupancy_from_counts(self.ensemble, self.counts)


# Grand-canonical generator will not be used very often.
class SemigrandSampleGenerator(McSampleGenerator):
    """Sample generator in semi-grand canonical ensembles."""

    def __init__(
        self,
        ce,
        sc_matrix,
        chemical_potentials,
        anneal_temp_series=None,
        heat_temp_series=None,
        num_steps_anneal=None,
        num_steps_heat=None,
        duplicacy_criteria="correlations",
        remove_decorations_before_duplicacy=False,
    ):
        """Initialize.

        Args:
            ce(ClusterExpansion):
                A cluster expansion object to enumerate with.
            sc_matrix(3*3 ArrayLike):
                Supercell matrix to solve on.
            chemical_potentials(dict):
                Chemical potentials of each species. See documentation
                of :mod:`smol.moca.ensemble`.
            anneal_temp_series(list of float): optional
                A series of temperatures to use in simulated annealing.
                Must be strictly decreasing.
            heat_temp_series(list of float): optional
                A series of increasing temperatures to sample on.
                Must be strictly increasing.
            num_steps_anneal(int): optional
                The number of steps to run per simulated annealing temperature.
            num_steps_heat(int): optional
                The number of steps to run per heat temperature.
            duplicacy_criteria(str):
                The criteria when to consider two structures as the same and
                old to add one of them into the candidate training set.
                Default is "correlations", which means to assert duplication
                if two structures have the same correlation vectors. While
                "structure" means two structures must be symmetrically equivalent
                after being reduced. No other option is allowed.
                Note that option "structure" might be significantly slower since
                it has to attempt reducing every structure to its primitive cell
                before matching. It should be used with caution.
            remove_decorations_before_duplicacy(bool): optional
                Whether to remove all decorations from species (i.e,
                charge and other properties) before comparing duplicacy.
                Default to false. Only valid when duplicacy_criteria="structure".
        """
        super().__init__(
            ce,
            sc_matrix,
            anneal_temp_series=anneal_temp_series,
            heat_temp_series=heat_temp_series,
            num_steps_anneal=num_steps_anneal,
            num_steps_heat=num_steps_heat,
            duplicacy_criteria=duplicacy_criteria,
            remove_decorations_before_duplicacy=remove_decorations_before_duplicacy,
        )

        self.chemical_potentials = chemical_potentials

    @property
    def ensemble(self):
        """CanonicalEnsemble."""
        if self._ensemble is None:
            self._ensemble = Ensemble.from_cluster_expansion(
                self.ce,
                self.sc_matrix,
                chemical_potentials=self.chemical_potentials,
                processor_type="expansion",
            )
        return self._ensemble

    @property
    def sampler(self):
        """A sampler to sample structures."""
        if self._sampler is None:
            # Check if charge balance is needed.
            bits = [sl.species for sl in self.sublattices]
            charge_decorated = False
            for sp in itertools.chain(*bits):
                if not isinstance(sp, (Vacancy, Element)) and sp.oxi_state != 0:
                    charge_decorated = True
                    break
            if charge_decorated:
                step_type = "table-flip"
            else:
                step_type = "flip"
            self._sampler = Sampler.from_ensemble(
                self.ensemble,
                temperature=self.anneal_temp_series[0],
                step_type=step_type,
                nwalkers=1,
            )
        return self._sampler

    def _get_init_occu(self):
        """Get an initial occupancy for MC run."""
        bits = [sl.species for sl in self.sublattices]
        sublattice_sizes = np.array([len(sl.sites) for sl in self.sublattices])
        supercell_size = np.gcd.reduce(sublattice_sizes)
        sublattice_sizes = sublattice_sizes / supercell_size
        comp_space = CompositionSpace(bits, sublattice_sizes)
        center_coords = comp_space.get_centroid_composition(
            supercell_size=supercell_size
        )
        center_counts = comp_space.translate_format(
            center_coords, supercell_size, from_format="coordinates", to_format="counts"
        )
        return get_random_occupancy_from_counts(self.ensemble, center_counts)


def mcgenerator_factory(mcgenerator_name, *args, **kwargs):
    """Create a McSampleGenerator with its subclass name.

    Args:
        mcgenerator_name(str):
            The name of a subclass of :class:`McSampleGenerator`.
        *args, **kwargs:
            Arguments used to initialize the class.
    """
    if (
        "sample-generator" not in mcgenerator_name
        and "SampleGenerator" not in mcgenerator_name
    ):
        mcgenerator_name += "-sample-generator"
    name = class_name_from_str(mcgenerator_name)
    return derived_class_factory(name, McSampleGenerator, *args, **kwargs)
