"""Wrappers to load options and history files."""

__author__ = "Fengyu Xie"

import itertools
import numpy as np
import json
import os
import uuid

from monty.json import MSONable, MontyDecoder, MontyEncoder
from pymatgen.core import Structure, Lattice, Composition, DummySpecies
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from smol.cofe import ClusterSubspace
from smol.cofe.space.domain import (get_site_spaces, Vacancy,
                                    get_allowed_species,
                                    get_species)

from smol.cofe.extern import EwaldTerm  # Currently, only allow EwaldTerm.

from .utils.format_utils import merge_dicts
from .config_paths import (WRAPPER_FILE, OPTIONS_FILE, PRIM_FILE,
                           HISTORY_FILE)

import warnings


def construct_prim(bits, sublattice_sites, lattice, frac_coords):
    """Construct a primitive cell based on lattice info.

    Provides a toolkit method to initialize a primitive cell.
    Args:
        bits(List[List[Specie]]):
            Allowed species on each sublattice. No sorting
            required.
        sublattice_sites(List[List[int]]):
            Site indices in each sub-lattice of a primitive cell.
            Must include all site indices in range(len(frac_coords))
        lattice(Lattice):
            Lattice of the primitive cell.
        frac_coords(ArrayLike):
            Fractional coordinates of sites.

    Returns:
        a primitive cell structure (not necessarily charge neutral):
            Structure
    """
    n_sites = len(frac_coords)
    if not np.allclose(np.arange(n_sites),
                       list(itertools.chain(*sublattice_sites))):
        raise ValueError(f"Provided site indices: {sublattice_sites} "
                         f"does not match range of site count: {n_sites}")

    site_comps = [{} for _ in range(n_sites)]
    for sublatt_id, sublatt in enumerate(sublattice_sites):
        comp = Composition({sp: 1 / len(bits)
                            for sp in bits[sublatt_id]
                            if not isinstance(sp, Vacancy)})
        for i in sublatt:
            site_comps[i] = comp

    return Structure(lattice, site_comps, frac_coords)


class InputsWrapper(MSONable):
    """Wraps options into formats required to init other modules.

    Can be saved and reloaded.
    Direct initialization is not recommended. You are supposed to 
    initialize it with auto_load().
    """
    # Add here if you implement more decorators.
    valid_decorator_types = {"charge": ("guess-charge-decorator",
                                        "magnetic-charge-decorator",)}

    def __init__(self, prim, **kwargs):
        """Initialize.

        Args:
            prim(pymatgen.Structure):
                primitive cell. Partially occupied sites will be cluster
                expanded.
                If you already provide prim, do not provide the first 4
                arguments as they'll be generated from prim. Vice versa.
            **kwargs:
                Additional settings to CEAuto fittings and calculations.
        """
        self._options = kwargs
        self._options = self._process_options()

        self._prim = prim
        self._process_prim()

        # These will always be reconstructed.
        self._bits = None
        self._sublattice_sites = None

    def _process_prim(self):
        """Make primitive cell the real primitive."""
        sa = SpacegroupAnalyzer(self._prim, **self.space_group_options)
        # TODO: maybe we can re-define site_properties transformation
        # in the future.
        self._prim = sa.find_primitive(keep_site_properties=True)

    @property
    def prim(self):
        """Primitive cell (pre-processed).

        Returns:
            Structure
        """
        return self._prim

    @property
    def transmat(self):
        """Transformation matrix.

        If "supercell_from_conventional" set to True, will use the transformation
        matrix from processed primitive cell to the conventional cell. Otherwise,
        returns identity.
        The enumerator will enumerate super-cell matrices in the form of:
        M = M'T, where M' is another transformation matrix.
        Returns:
            np.ndarray[int]
        """
        if self.enumerator_options["supercell_from_conventional"]:
            sa = SpacegroupAnalyzer(self.prim, **self.space_group_options)
            t_inv = sa.get_conventional_to_primitive_transformation_matrix()
            return np.array(np.round(np.linalg.inv(t_inv)), dtype=int)
        return np.eye(3, dtype=int)

    @property
    def sc_size_enum(self):
        """Supercell size (in num of prim unit) to enumerate.

        Computed as self.enumerator_options["objective_sc_size"]
        // (det(transmat)) * det(transmat).
        Return:
            int
        """
        det = int(round(abs(np.linalg.det(self.transmat))))
        return self.enumerator_options["objective_sc_size"] // det * det

    @property
    def space_group_options(self):
        """Options to pre-process the primitive cell.

        Refer to kwargs of pymatgen's SpaceGroupAnalyzer.
        Returns:
            dict

        prim_symmetry_precision(float):
            precision to pass
        """
        return self._options.get("space_group_kwargs", {})

    @property
    def bits(self):
        """List of species on each sublattice.

        Returns:
            List[List[Specie]]
        """
        if self._bits is None:
            # By default, does NOT include measure!
            # Different sites with same species but different numbers
            # are considered the same sub-lattice!
            unique_spaces = tuple(set(get_site_spaces(self.prim)))

            # Same order as smol.moca.Sublattice
            self._bits = [list(space.keys()) for space in unique_spaces]

        return self._bits

    @property
    def sublattice_sites(self):
        """List of prim cell site indices in sub-lattices.

        Returns:
            List[List[int]]
        """
        if self._sublattice_sites is None:
            unique_spaces = tuple(set(get_site_spaces(self.prim)))
            allowed_species = get_allowed_species(self.prim)

            # Automatic sublattices, same rule as smol.moca.Sublattice
            self._sublattice_sites = [[i for i, sp in enumerate(allowed_species)
                                       if sp == list(space.keys())]
                                      for space in unique_spaces]

        # No sort of bits on sublattices! Order based on dict keys.
        return self._sublattice_sites

    @property
    def sublattice_sizes(self):
        """Sizes of each sub-lattice.

        Returns:
            List[int]
        """
        return [len(s) for s in self.sublattice_sites]

    @property
    def is_charged_ce(self):
        """Check whether the expansion is charge decorated.

        Will add EwaldTerm, and apply charge assignment.
        Returns:
            bool.
        """
        is_charged_ce = False
        for sp in itertools.chain(*self.bits):
            if (not isinstance(sp, (Vacancy, Element)) and
                    sp.oxi_state != 0):
                is_charged_ce = True
                break
        return is_charged_ce

    def get_cluster_subspace(self):
        """Cluster subspace of this system.

        If none provided, will be intialized from cutoffs.
        Returns:
            ClusterSubspace.
        """
        subspace = (ClusterSubspace.
                    from_cutoffs(self.prim,
                                 self.cluster_space_options['cutoffs'],
                                 basis=self.cluster_space_options['basis_type'],
                                 **self.cluster_space_options['matcher_kwargs'])
                    )

        if self.is_charged_ce:
            subspace.add_external_term(EwaldTerm(**self.cluster_space_options["ewald_kwargs"]))

        return subspace

    @property
    def project_name(self):
        """Name of CE project.

        Can be specified in options. If not specified, will
        generate an unique random UUID.
        Return:
            str
        """
        return self._options.get('project_name',
                                 str(uuid.uuid4()))

    @property
    def cluster_space_options(self):
        """ClusterSubspace options.

        Returns:
            Dict.

        Included keys:
            cutoffs(dict):
                Cutoffs used to initialize cluster subspace.
            basis_type(str):
                Basis to use for this cluster expansion project.
                Default to indicator.
            matcher_kwargs(dict):
                Keyword arguments to pass into cluster subspace's
                structure matchers. Default to empty.
            ewald_kwargs(dict):
                Keyword arguments to pass into cluster subspace's
                EwaldTerm, if it has one. Default to empty.
        """
        # Auto-generate cutoffs from neighbor distances.
        cutoffs = self._options.get('cutoffs')
        if cutoffs is None:
            d_nns = []
            for i, site1 in enumerate(self.prim):
                d_ij = []
                for j, site2 in enumerate(self.prim):
                    if j > i:
                        d_ij.append(site1.distance(site2))
                d_nns.append(min(d_ij
                                 + [self.prim.lattice.a,
                                    self.prim.lattice.b,
                                    self.prim.lattice.c]))
            d_nn = min(d_nns)

            # Empirical values from DRX, not necessarily true.
            cutoffs = {2: d_nn * 4.0, 3: d_nn * 2.0, 4: d_nn * 2.0}

        return {'cutoffs': cutoffs,
                'basis_type': self._options.get('basis_type', 'indicator'),
                'matcher_kwargs': self._options.get('matcher_kwargs', {}),
                'ewald_kwargs': self._options.get('ewald_kwargs', {})
                }

    @staticmethod
    def parse_comp_restriction(d):
        """Parse a restriction of composition.

        Arg:
            d(dict|list/tuple of dict):
                Dictionary of restrictions. Each key is a representation of a
                species, and each value can either be a tuple of lower and
                upper-limits, or a single float of upper-limit of the species
                atomic fraction. Number must be in [0, 1].
                If a list of dict provided, each dict in the list constrains
                on a sub-lattice composition.
        """
        comp_restriction = {}
        if isinstance(d, (list, tuple)):
            return [InputsWrapper.parse_comp_restriction(o) for o in d]

        for key, val in d.items():
            if isinstance(val, (list, tuple)):
                if len(val) != 2:
                    raise ValueError("Species concentration constraints provided "
                                     "as tuple, but length of tuple is not 2.")
                if val[1] < val[0]:
                    raise ValueError("Species concentration constraints provided "
                                     "as tuple, but lower-bound > upper_bound.")
                if val[1] < 0 or val[1] > 1 or val[0] < 0 or val[0] > 1:
                    raise ValueError("Provided species concentration limit must "
                                     "be in [0, 1]!")
                comp_restriction[get_species(key)] = tuple(val)
            else:
                if val < 0 or val > 1:
                    raise ValueError("Provided species concentration limit must "
                                     "be in [0, 1]!")
                comp_restriction[get_species(key)] = (0, val)

        return comp_restriction

    # TODO: More generic comp_restriction?
    @property
    def enumerator_options(self):
        """Get enumerator options.

        Return:
            Dict.

        Included keys:
        supercell_from_conventional(bool):
            Whether to find out primitive cell to conventional
            standard structure transformation matrix T, and enumerate
            super-cell matrices in the form of: M = M'T.
            Default to true. If not, will set T to eye(3).
        objective_sc_size(int):
            The Supercel sizes (in number of prims) to approach.
            Default to 32. Enumerated super-cell size will be
            a multiple of det(T) but the most close to this objective
            size.
        max_sc_cond(float):
            Maximum conditional number of the supercell lattice matrix.
            Default to 8, prevent overly slender super-cells.
        max_sc_angle(float):
            Minimum allowed angle of the supercell lattice.
            Default to 30, prevent overly skewed super-cells.
        sc_mats(List[3*3 ArrayLike[int]]):
            Supercell matrices. Will not enumerate super-cells if this
            is given. Default to None.
        comp_restrictions(List[dict]|dict):
            Restriction to enumerated concentrations of species.
            Given in the form of a single dict
                {Species: (min atomic percentage in structure,
                           max atomic percentage in structure)}
            or in list of dicts, each constrains a sub-lattice:
               [{Species: (min atomic percentage in sub-lattice,
                           max atomic percentage in sub-lattice)},
                ...]
        comp_enumeration_step (int):
            Spacing of enumerated compositions under sc_size, such
            that compositions will be enumerated as:
                comp_space.get_comp_grid(sc_size=
                sc_size // comp_enumeration_step).
            Default to det(transmat), but not always recommended.
        struct_to_comp_ratio (int|Dict):
            This gives the ratio between number of structures
            enumerated per iteration, and number of compositions
            enumerated. If an int given, will use the same value
            at iteration 0 ("init") and following iterations ("add").
            If in dict form, please give:
                {"init": ..., "add": ...}
            Default to: {"init": 4, "add": 2}
            It is not guaranteed that in an iteration, number of added
            structures equals to ratio * number of compositions, because
            selected structures may duplicate with existing ones thus
            not inserted.
        handler_args(Dict):
            kwargs to pass into CanonicalHander.
            See ce_handlers.CanonicalHandler.
        select_method(str):
            Structure selection method. Default is 'leverage'.
            Allowed options are: 'leverage' and 'random'.
        """
        det = int(round(abs(np.linalg.det(self.transmat))))
        comp_restriction = \
            self.parse_comp_restriction(self._options.get('comp_restriction', {}))
        return {'supercell_from_conventional':
                self._options.get('supercell_from_conventional', True),
                'objective_sc_size': self._options.get('objective_sc_size', 32),
                'max_sc_cond': self._options.get('max_sc_cond', 8),
                'min_sc_angle': self._options.get('min_sc_angle', 30),
                'sc_mats': self._options.get('sc_mats'),
                # If sc_mats is given, will overwrite matrices enumeration.
                'comp_restriction': comp_restriction,
                'comp_enumeration_step': self._options.get('comp_enumeration', det),
                'structs_to_comp_ratio': self._options.get('structs_to_comp_ratio',
                                                           {"init": 4, "add": 2}),
                'handler_args': self._options.get('handler_args', {}),
                'select_method': self._options.get('select_method', 'leverage')
                }

    # TODO: Maybe provide smart lattice parameters guessing when writing WF?
    @property
    def vasp_options(self):
        """Get structural relaxation job options.

        Returns:
            dict.

        Included keys:
        writer_strain(3*3 ArrayLike or 1D ArrayLike[float] of 3):
            Strain to apply in before relaxation.
            Default is [1.05, 1.03, 1.01], meaning 5%, 3% and 1% stretch
            along a, b, and c axis. This is set to
            break enforced symmetry before relaxation.
            You can turn it off by specifying [1, 1, 1].
        is_metal(bool):
            Whether this system is a metal or not. Default to False.
            Determines whether to use MPRelaxSet or MPMetalRelaxSet.
        pmg_set_setting(dict): optional
            Additional arguments to pass into a pymatgen set.
            Refer to documentation of pymatgen.io.sets.
        """
        writer_strain = \
            self._options.get('writer_strain',
                              [1.05, 1.03, 1.01])
        writer_strain = np.array(writer_strain)
        if len(writer_strain.shape) == 1:
            assert writer_strain.shape == (3,)
            writer_strain = np.diag(writer_strain)
        elif len(writer_strain.shape) == 2:
            assert writer_strain.shape == (3, 3)
        else:
            raise ValueError("Provided strain format is wrong. "
                             "Must be either 3*3 arraylike or "
                             "length 3 arraylike!")

        return {'writer_strain': writer_strain,
                'is_metal': self._options.get('is_metal', False),
                'pmg_set_setting': self._options.get('pmg_set_setting', {})
                }

    @property
    def decorating_options(self):
        """Get decorating options.

        Returns:
            dict.

        Included keys:
        decorated_properties(list(str)): optional
            Name of properties to decorate on vasp output structures.
            For example, "charge" is typically used for
        decorator_types(list(str)): optional
            Name of decorators to use for each property. If None, will
            choose the first one in valid_decorators.
        decorator_args(List[dict]): optional
            Arguments to pass into each decorator. See documentation of
            each decorator in species_decorator module.
        """

        # Update these pre-processing rules when necessary,
        # if you have new decorators implemented.
        def contains_transition_metal(s):
            for species in s.composition.keys():
                if (not isinstance(species, (DummySpecies, Vacancy))
                        and species.is_trainsition_metal):
                    return True
            return False

        decorated_properties = self._options.get('decorated_properties', [])
        decorator_types = self._options.get('decorator_types', [])
        decorator_args = self._options.get('decorator_args', [])
        if self.is_charged_ce and len(decorated_properties) == 0:
            decorated_properties.append("charge")
            if contains_transition_metal(self.prim):
                warnings.warn(f"Primitive cell: {self.prim}\n contains "
                              "transition metal, but we will apply the default "
                              "charge decorator based on pymatgen charge "
                              "guesses. Be sure you know what you are doing!")
            decorated_properties = ["charge"]
        assert (len(decorator_types) == 0 or
                len(decorator_types) == len(decorated_properties))
        assert (len(decorator_types) == 0 or
                len(decorator_types) == len(decorated_properties))
        if len(decorator_types) == 0:
            decorator_types = [self.valid_decorator_types[prop]
                               for prop in decorated_properties]
        elif len(decorator_types) != len(decorated_properties):
            raise ValueError("Provided decorator types, but length does not "
                             "match decorated properties.")
        if len(decorator_args) == 0:
            decorator_args = [{} for _ in decorated_properties]
        elif len(decorator_args) != len(decorated_properties):
            raise ValueError("Provided decorator args, but length does not "
                             "match decorated properties.")

        return {'decorated_properties': decorated_properties,
                'decorator_types': decorator_types,
                'decorator_args': decorator_args
                }

    @property
    def fitting_options(self):
        """Get fitting options.

        Returns:
            Dict.

        Included keys:
        regression_flavor(str):
            Choose regularization method from theorytoolkits. Default to
            'lasso'.
        weights_flavor(str):
            Weighting scheme to use. All available weighting schemes include
            "e_above_composition", "e_above_hull" and "unweighted" (default).
            See documentation for smol.cofe.wrangling.tool.
        use_hierarchy(str):
            Whether to use hierarchy in regularization fitting, when
            regression flavor is one of mixedL0.
            Default to True.
        fit_optimizer_kwargs(dict):
            Keyword arguments to pass into FitOptimizer class. See documentation
            of FitOptimizer.
        """
        return {'regression_flavor': self._options.get('regression_flavor',
                                                       'lasso'),
                'weights_flavor': self._options.get('weights_flavor',
                                                    'unweighted'),
                'use_hierarchy': self._options.get('use_hierarchy', True),
                'fit_optimizer_kwargs': self._options.get('fit_optimizer_kwargs',
                                                          {})
                }

    @property
    def convergence_options(self):
        """Get convergence criteria settings.

        Returns:
            dict.

        Included keys:
        cv_atol(float): optional
            Maximum allowed value of the CV. Unit in meV per site.
            (not eV per atom because some CE may contain Vacancies.)
            Default to 5 meV/site.
        cv_var_rtol(float): optional
            Maximum allowed square root variance of CV from 3
            parallel random validations, divided by CV value.
            This is another measure of model variance.
            Dimensionless, default to 1/3.
        delta_cv_rtol(float): optional
            Maximum CV difference between the latest 2 iterations,
            divided by the variance of CV value in the last iteration.
            Dimensionless, default to 1.
        delta_min_e_rtol(float): optional
            Tolerance of change to minimum energy under every composition
            between the last 2 iterations, divided by the value of CV.
            Dimensionless, default set to 1.
        """
        return {'cv_atol': self._options.get('cv_atol', 5),
                'cv_var_rtol': self._options.get('cv_var_rtol', 1 / 3),
                'delta_cv_rtol': self._options.get('delta_cv_rtol', 1),
                'delta_min_e_rtol': self._options.get('delta_min_e_rtol',
                                                      1)
                }

    def _process_options(self):
        """Pre-process options at initialization."""
        all_options = [{"project_name": self.project_name},
                       self.space_group_options,
                       self.cluster_space_options,
                       self.enumerator_options,
                       self.decorating_options,
                       self.fitting_options,
                       self.convergence_options]
        return merge_dicts(all_options, keep_all=False)

    @property
    def options(self):
        """Returns completed options dictionary for the user's reference.

        Returns:
            dict.
        """
        return self._options

    def copy(self):
        """Deepcopy InputsWrapper object."""
        socket = InputsWrapper(prim=self.prim.copy(),
                               **self.options)

        return socket

    @classmethod
    def from_dict(cls, d):
        """Deserialize from a dictionary.

        Better serialize and de-serialize with monty.
        Args:
            d(dict):
                Dictionary containing prim information and options.
        Returns:
            InputsWrapper object.
        """
        return cls(Structure.from_dict(d['prim']),
                   **d['options'])

    def as_dict(self):
        """Serialize all class data into a dictionary.

        Better serialize and de-serialize with monty.
        Returns:
            dict. Containing all serialized lattice data, options,
            and important attributes 
        """
        return {
            "prim": self.prim.as_dict(),
            "options": self.options,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    @classmethod
    def from_file(cls):
        """Automatically load object from json files.

        Recommended way to load.
        Returns:
            InputsWrapper.
        """
        if os.path.isfile(WRAPPER_FILE):
            with open(WRAPPER_FILE, encoding="utf-8") as fin:
                d = json.load(fin, cls=MontyDecoder)
                return cls.from_dict(d)
        elif os.path.isfile(PRIM_FILE) and os.path.isfile(OPTIONS_FILE):
            prim = Structure.from_file(PRIM_FILE)
            with open(OPTIONS_FILE) as fin:
                options = json.load(fin, cls=MontyDecoder)
            return cls(prim, **options)
        else:
            raise ValueError("No initial setting file provided! "
                             "Please provide at least one of: "
                             f"{WRAPPER_FILE} or {OPTIONS_FILE}.")

    def to_file(self):
        """Automatically save data into a json file."""
        with open(WRAPPER_FILE, 'w') as fout:
            json.dump(self.as_dict(), fout, cls=MontyEncoder)


class CEHistoryWrapper(MSONable):
    """History management.

    This class stores multiple all the past fitting results.
    Does not store past feature matrices. Matrices are stored in
    DataWrangler.
    """

    def __init__(self, cluster_subspace, history=None):
        """Initialize HistoryWrapper.

        Args:
            cluster_subspace(ClusterSubspace):
              Cluster subspace used in CE.
            history(List[dict]): optional
              A list of dict storing all past history info for CE models
              each iteration. Dict must contain a key "coefs".
              Currently, only supports expansion on energy.
        """
        if history is None:
            history = []
        self._history = history
        self._subspace = cluster_subspace

    @property
    def history(self):
        """History Dict."""
        return self._history

    @property
    def cluster_subspace(self):
        """Cluster subspace."""
        return self._subspace

    @property
    def existing_attributes(self):
        """All available attributes in history (besides coefs)."""
        all_attrs = set(key for record in self.history for key in record)
        all_attrs = all_attrs - {"coefs"}
        return all_attrs

    def get_coefs(self, iter_id=-1):
        """Get the cluster expansion coefficients n iterations ago.

        If none exist, will initialize a dummy CE, with all
        external terms with coef 1, and cluster terms with coef 0.
        Args:
            iter_id(int):
                Iteration index to get CE with. Default to -1.
        Returns:
            CE coefficients as described in smol.expansion:
                np.ndarray
        """
        if len(self.history) == 0:
            warnings.warn("No cluster expansion generated before, "
                          "returning a dummy CE instead.")

            coefs = np.zeros(self.cluster_subspace.num_corr_functions +
                             len(self.cluster_subspace.external_terms))
            if len(self.cluster_subspace.external_terms) > 0:
                coefs[-len(self.cluster_subspace.external_terms):] = 1.0
        else:
            coefs = np.array(self.history[iter_id]['coefs'])

        return coefs

    def get_attribute(self, name="cv", iter_id=-1):
        """Get any other historical attributes.

        If none exist, will return np.inf to block convergence.
        Args:
            name(str):
                Name of attribute to get. Can be "cv"
            iter_id(int):
                Iteration index to get CE with. Default to -1.
        Returns:
            float.
        """
        if len(self.history):
            return np.inf
        return self.history[iter_id][name]

    def add_record(self, record):
        """Add a record to history.

        Args:
             record(dict):
                 Dictionary containing CE fit infos. Must at least
                 include key "coefs".
        """
        if "coefs" not in record:
            raise ValueError("CE coefficients are required for a record!")
        record["coefs"] = np.array(record["coefs"])
        for attr in record:
            if attr not in self.existing_attributes and len(self.history) > 0:
                warnings.warn(f"Record contains a new key: {attr} "
                              f"not found in previous records!")

    def as_dict(self):
        """Serialize into dict.

        Returns:
           dict.
        """
        return {'cluster_subspace': self.cluster_subspace.as_dict(),
                'history': self.history,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d):
        """Initialize from dict.

        Args:
           d(Dict):
               Dict containing serialized object.

        Returns:
           CEHistoryWrapper.
        """
        return cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                   d.get('history'))

    def to_file(self):
        """Automatically save object to json file."""
        with open(HISTORY_FILE, 'w') as fout:
            json.dump(self.as_dict(), fout, cls=MontyEncoder)

    @classmethod
    def from_file(cls):
        """Automatically load object from json file.

        Recommended way to load.
        Returns:
            HistoryWrapper.
        """
        with open(HISTORY_FILE, 'r') as fin:
            d = json.load(fin, cls=MontyDecoder)

        return cls.from_dict(d)
