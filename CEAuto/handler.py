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
                                    get_allowed_species)

from smol.cofe.extern import EwaldTerm  # Currently, only allow EwaldTerm.

from .utils.format_utils import merge_dicts
from .utils.comp_constraint_utils import (parse_species_constraints,
                                          parse_generic_constraint)
from .config_paths import (INPUTS_FILE, OPTIONS_FILE, PRIM_FILE,
                           HISTORY_FILE)
from .specie_decorators import allowed_decorators

import warnings


def construct_prim(bits, sublattice_sites, lattice, frac_coords):
    """Construct a primitive cell based on lattice info.

    Provides a helper method to initialize a primitive cell.
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
                       sorted(list(itertools.chain(*sublattice_sites)))):
        raise ValueError(f"Provided site indices: {sublattice_sites} "
                         f"does not include all {n_sites} sites!")

    site_comps = [{} for _ in range(n_sites)]
    for sublatt_id, sublatt in enumerate(sublattice_sites):
        comp = Composition({sp: 1 / len(bits[sublatt_id])
                            for sp in bits[sublatt_id]
                            if not isinstance(sp, Vacancy)})
        for i in sublatt:
            site_comps[i] = comp

    return Structure(lattice, site_comps, frac_coords)


class InputsHandler(MSONable):
    """Wraps options into formats required to init other modules.

    Can be saved and reloaded.
    Direct initialization is not recommended. You are supposed to 
    initialize it with auto_load().
    """

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
                see docs of each property for details.
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
        #  in the future.
        self._prim = sa.find_primitive(keep_site_properties=True)

    @property
    def prim(self):
        """Primitive cell (pre-processed).

        Returns:
            Structure
        """
        return self._prim

    @property
    def prim_to_conv_transmat(self):
        """Matrix to transform primitive cell to a conventional cell.

        If "supercell_from_conventional" set to True, will enumerate super-cell
        matrices in the form of transmat @ enumerated_integer_matrix to preserve
        some conventional cell symmetry.
        Returns:
            np.ndarray[int]
        """
        sa = SpacegroupAnalyzer(self.prim, **self.space_group_options)
        t_inv = sa.get_conventional_to_primitive_transformation_matrix()
        return np.round(np.linalg.inv(t_inv)).astype(int)

    @property
    def space_group_options(self):
        """Options to pre-process the primitive cell.

        Refer to kwargs of pymatgen's SpaceGroupAnalyzer.
        Returns:
            dict

        Included keys:
            prim_symmetry_precision(float):
                precision to pass.
        """
        return self._options.get("space_group_kwargs", {})

    @property
    def bits(self):
        """List of species on each sublattice.

        Returns:
            List[List[Specie]]
        """
        if self._bits is None:
            # Now will have the same order as in processor
            unique_spaces = sorted(set(get_site_spaces(self.prim)))

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
            # Now will have the same order as in processor
            unique_spaces = sorted(set(get_site_spaces(self.prim)))
            allowed_species = get_allowed_species(self.prim)
            # Ordering between species in a sub-lattice is fixed.

            # Automatic sublattices, same rule as smol.moca.Sublattice
            self._sublattice_sites = [[i for i, sp in enumerate(allowed_species)
                                       if sp == list(space.keys())]
                                      for space in unique_spaces]

        return self._sublattice_sites

    @property
    def sublattice_sizes(self):
        """Num of sites in each sub-lattice per prim.

        Returns:
            List[int]
        """
        return [len(s) for s in self.sublattice_sites]

    @property
    def is_charged_ce(self):
        """Check whether the charge decoration is needed.

        If yes, will include EwaldTerm, and add charge decorators.
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
        """Cluster subspace.

        If none provided, will be intialized from cutoffs.
        This cluster subspace is not yet trimmed for aliasing!
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
            subspace.add_external_term(EwaldTerm(**self.cluster_space_options
            ["ewald_kwargs"]))

        return subspace

    @property
    def project_name(self):
        """Name of the CE project.

        Can be specified in options. If not specified, will
        be named with a random UUID.
        Return:
            str
        """
        return self._options.get('project_name',
                                 "ce-project-" + str(uuid.uuid4()))

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

            # Empirical values from DRX (744), not necessarily good for all.
            # It is highly recommended setting your own.
            cutoffs = {2: d_nn * 3.5, 3: d_nn * 2.0, 4: d_nn * 2.0}

        return {'cutoffs': cutoffs,
                'basis_type': self._options.get('basis_type', 'indicator'),
                'matcher_kwargs': self._options.get('matcher_kwargs', {}),
                'ewald_kwargs': self._options.get('ewald_kwargs', {})
                }

    @property
    def supercell_enumerator_options(self):
        """Get supercell enumerator options.

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
        return {'supercell_from_conventional':
                self._options.get('supercell_from_conventional', True),
                'objective_sc_size': self._options.get('objective_sc_size', 32),
                'max_sc_cond': self._options.get('max_sc_cond', 8),
                'min_sc_angle': self._options.get('min_sc_angle', 30),
                'sc_mats': self._options.get('sc_mats'),
                # If sc_mats is given, will overwrite matrices enumeration.
                }

    @property
    def composition_enumerator_options(self):
        """Get composition enumerator options.

        Return:
            Dict.

        Included keys:
        species_concentration_constraints(List[dict]|dict):
            Restriction to concentrations of species in enumerated
            structures.
            Given in the form of a single dict
                {Species: (min atomic percentage in structure,
                           max atomic percentage in structure)}
            or in list of dicts, each constrains a sub-lattice:
               [{Species: (min atomic percentage in sub-lattice,
                           max atomic percentage in sub-lattice)},
                ...]
            If values in dict are provided as a single float,
            Note: Species are saved as their __str__ method output.
            This will have some restriction on the allowed properties,
            but we have to bear with it until pymatgen is updated.
        eq_constraints(List[tuple(list[dict]|dict, float)]):
            Composition constraints with the form as: 1 N_A + 2 N_B = 3,
            etc. The number of species and the right side of the equation
            are normalized per primitive cell.
            Each constraint equation is given as a tupe.
            The dict (or list[dict]) encodes the left side of the equation.
            each key is a species or species string, while each value is
            the factor of the corresponding species in the constraint
            equation. If given as a list of dictionaries, each dictionary in the
            list specifically constrains the number of species on its corresponding
            sub-lattice. The float in the tuple encodes the right side of
            the equation. Refer to documentation of utils.comp_constraint_utils.
            For example, if a system contains A and B on sub-lattice 1, B and
            C on sub-lattice 2. If a constraint is given as:
            ({"A": 1, "B": 1, "C": 1}, 3), then the corresponding constraint
            equation should be:
                1 * N_A_sub1 + 1 * (N_B_sub1 + N_B_sub2) + 1 * N_C_sub2 == 3
            If the constraints is given as:
            ([{"A": 1, "B": 1}, {"B": 2, "C": 1}], 3), then the equation should be:
               1 * N_A_sub1 + 1 * N_B_sub1 + 2 *N_B_sub2 + 1 * N_C_sub2 == 3.
        leq_constraints(List[tuple(list[dict]|dict, float)]):
            Composition constraints with the form as: 1 N_A + 2 N_B <= 3.
            Dict format same as eq_constraints.
        geq_constraints(List[tuple(list[dict]|dict, float)]):
            Composition constraints with the form as: 1 N_A + 2 N_B >= 3.
            Dict format same as eq_constraints.
        All constraints will be parsed into a CompSpace readable format.
        comp_enumeration_step (int): optional
            Skip step in returning the enumerated compositions.
            If step > 1, on each dimension of the composition space,
            we will only yield one composition in every N compositions.
            Default to 1.
        compositions (2D arrayLike[int]): optional
            Fixed compositions with which to enumerate the structures. If
            given, will not enumerate other compositions.
            Should be provided in the "n"-format of
        """
        return {"comp_enumeration_step":
                self._options.get("comp_enumeration_step", 1),
                "species_concentration_constraints":
                self._options.get("species_concentration_constraints", []),
                "eq_constraints":
                self._options.get("eq_constraints", []),
                "leq_constraints":
                self._options.get("leq_constraints", []),
                "geq_constraints":
                self._options.get("geq_constraints", []),
                }

    @property
    def parsed_constraints(self):
        """Parsed constraints from enumerator options.

        Return:
            list(tuple(arrayLike, float)):
               Equality constraints, then leq and geq constraints,
               in the smol readable format.
        """
        leqs_species, geqs_species \
            = parse_species_constraints(self.composition_enumerator_options
                                        ["species_concentration_constraints"],
                                        self.bits, self.sublattice_sizes)
        eqs = [parse_generic_constraint(d, r, self.bits)
               for d, r in
               self.composition_enumerator_options["eq_constraints"]]
        leqs = [parse_generic_constraint(d, r, self.bits)
                for d, r in
                self.composition_enumerator_options["leq_constraints"]]
        geqs = [parse_generic_constraint(d, r, self.bits)
                for d, r in
                self.composition_enumerator_options["geq_constraints"]]

        return eqs, leqs + leqs_species, geqs + geqs_species

    @property
    def structure_enumerator_options(self):
        """Get structures enumerator options.

        Returns:
            dict.
        Included keys:
        num_structs_per_iter (int|tuple(int)):
            Number of new structures to enumerate per iteration.
            If given in a single int, will add the same amount of
            structures in any iteration.
            If given in a tuple of two ints, will add the amount of
            the first int in the first iteration (pool initialization),
            then the amount of the second int in the following
            iterations.
            It is recommended that in each iteration, at least 2~3
            structures are added for each composition.
            Default is (50, 30).
        sample_generator_kwargs(Dict):
            kwargs of CanonicalSampleGenerator.
        init_method(str):
            Structure selection method in the first iteration.
            Default is "CUR". Allowed options include: "CUR" and
            "random".
        add_method(str):
            Structure selection method in subsequent iterations.
            Default is 'leverage'. Allowed options are: 'leverage'
            and 'random'.
        keep_ground_states(bool):
            Whether to keep new ground states in the training set.
            Default to True.
        """
        return {"num_structs_per_iter":
                self._options.get("num_structs_per_iter", (50, 30)),
                "sample_generator_kwargs":
                self._options.get("sample_generator_kwargs", {}),
                "init_method":
                self._options.get("init_method", "CUR"),
                "add_method":
                self._options.get("add_method", "CUR"),
                "keep_ground_states":
                self._options.get("keep_ground_states", True)}

    @property
    def calculation_options(self):
        """Get the vasp calculation options.

        Returns:
            dict.

        Included keys:
        apply_strain(3*3 ArrayLike or 1D ArrayLike[float] of 3):
            Strain matrix to apply to the structure before relaxation,
            in order to break structural symmetry of forces.
            Default is [1.03, 1.02, 1.01], which means to
            stretch the structure by 3%, 2% and 1% along a, b, and c
            directions, respectively.
        relax_generator_kwargs(dict):
            Additional arguments to pass into an atomate2
            VaspInputGenerator
            that is used to initialize RelaxMaker.
        relax_maker_kwargs(dict):
            Additional arguments to pass into an atomate2
            RelaxMaker.
        add_tight_relax(bool):
            Whether to add a tight relaxation job after a coarse
            relaxation. Default to True.
            You may want to disable this if your system has
            difficulty converging forces or energies.
        tight_generator_kwargs(dict):
            Additional arguments to pass into an atomate2
            VaspInputGenerator
            that is used to initialize TightRelaxMaker.
        tight_maker_kwargs(dict):
            Additional arguments to pass into an atomate2
            TightRelaxMaker. A tight relax is performed after
            relaxation, if add_tight_relax is True.
       static_generator_kwargs(dict):
            Additional arguments to pass into an atomate2
            VaspInputGenerator
            that is used to initialize StaticMaker.
        static_maker_kwargs(dict):
            Additional arguments to pass into an atomate2
            StaticMaker.
        Refer to the atomate2 documentation for more information.
        Note: the default vasp sets in atomate 2 are not specifically
        chosen for specific systems. Using your own vasp set input
        settings is highly recommended!
        """
        writer_strain = \
            self._options.get('apply_strain',
                              [1.03, 1.02, 1.01])
        writer_strain = np.array(writer_strain)
        if len(writer_strain.shape) == 1:
            writer_strain = np.diag(writer_strain)

        if writer_strain.shape != (3, 3):
            raise ValueError("Provided strain format is wrong. "
                             "Must be either 3*3 arraylike or "
                             "an 1d arraylike of length 3!")

        return {"apply_strain": writer_strain.tolist(),
                "relax_generator_kwargs":
                    self._options.get("relax_generator_kwargs", {}),
                "relax_maker_kwargs":
                    self._options.get("relax_maker_kwargs", {}),
                "add_tight_relax":
                    self._options.get("add_tight_relax", True),
                "tight_generator_kwargs":
                    self._options.get("tight_generator_kwargs", {}),
                "tight_maker_kwargs":
                    self._options.get("tight_maker_kwargs", {}),
                "static_generator_kwargs":
                    self._options.get("static_generator_kwargs", {}),
                "static_maker_kwargs":
                    self._options.get("static_maker_kwargs", {}),
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
        decorator_kwargs(List[dict]): optional
            Arguments to pass into each decorator. See documentation of
            each decorator in species_decorator module.
        """
        # Update these pre-processing rules when necessary,
        # if you have new decorators implemented.

        decorated_properties = self._options.get('decorated_properties', [])
        decorator_types = self._options.get('decorator_types', [])
        decorator_args = self._options.get('decorator_kwargs', [])
        if self.is_charged_ce and len(decorated_properties) == 0:
            decorated_properties.append("oxi_state")
        if not len(decorator_types) == len(decorated_properties):
            raise ValueError("Number of properties to decorate does not"
                             " match the number of decorators. Be sure to use"
                             " only 1 decorator per property!")
        for tp, prop in zip(decorator_types, decorated_properties):
            if prop not in allowed_decorators:
                raise ValueError(f"Property {prop} does not have any implemented"
                                 f" decorator!")
            if tp not in allowed_decorators[prop]:
                raise ValueError(f"Decorator {tp} is not implemented for"
                                 f" property {prop}!")
        if (len(decorator_args) > 0 and
                len(decorator_args) != len(decorated_properties)):
            raise ValueError("Number of provided kwargs must be the same as"
                             " the number of decorated properties!")
        if len(decorator_args) == 0:
            decorator_args = [{} for _ in decorated_properties]

        if (len(decorator_types) > 0 and
                len(decorator_types) != len(decorated_properties)):
            raise ValueError("Number of provided decorators must be the same as"
                             " the number of decorated properties!")
        if len(decorator_types) == 0:
            warnings.warn(f"No decorator specified for properties"
                          f" {decorated_properties}. The first allowed"
                          f" decorator for each property will be selected"
                          f" based on dictionary order. Use this at your"
                          f" own risk!")
            decorator_types = [sorted(allowed_decorators[prop])[0]
                               for prop in decorated_properties]

        return {'decorated_properties': decorated_properties,
                'decorator_types': decorator_types,
                'decorator_kwargs': decorator_args
                }

    # TODO: communicate with sparse-lm team to make estimators easier to import.
    @property
    def fitting_options(self):
        """Get fitting options.

        Returns:
            Dict.

        Included keys:
        estimator_type(str):
            The name of an estimator class from sparce-lm. Default to
            'L2L0'.
        weighting_scheme(str):
            Weighting scheme to use. All available weighting schemes include
            "e_above_composition", "e_above_hull" and "unweighted" (default).
            See documentation for smol.cofe.wrangling.tool.
        use_hierarchy(str):
            Whether to use hierarchy in regularization fitting, when
            regression type is mixedL0. Default to True.
        estimator_kwargs(dict):
            Other keyword arguments to pass in when constructing an
            estimator. See sparselm.models
        optimizer_type(str):
            The name of optimizer class used to optimize model hyper parameters
            over cross validation. Default is None. Supports "grid-search" and
            "line-search" optimizers. See sparselm.optimizer.
        optimizer_kwargs(dict):
            Keyword arguments when constructing GridSearch or LineSearch class.
            See sparselm.optimizer.
        fit_kwargs(dict):
            Keyword arguments when calling GridSearch/LineSearch/Estimator.fit.
            See sparselm.
        """
        return {'estimator_type':
                    self._options.get('estimator_type', 'L2L0'),
                'weighting_scheme':
                    self._options.get('weighting_scheme', 'unweighted'),
                'use_hierarchy': self._options.get('use_hierarchy', True),
                "estimator_kwargs":
                    self._options.get("estimator_kwargs", {}),
                'optimizer_type':
                    self._options.get('optimizer_type', None),
                'optimizer_kwargs':
                    self._options.get('optimizer_kwargs', {}),
                'fit_kwargs':
                    self._options.get('fit_kwargs', {}),
                }

    @property
    def convergence_options(self):
        """Get convergence criterion.

        Returns:
            dict.

        Included keys:
        cv_atol(float): optional
            Maximum allowed CV value. Unit in meV per site.
            (not eV per atom because some CE may contain Vacancies.)
            Default to 5 meV/site, but setting it manually is highly
            recommended!
        cv_std_rtol(float): optional
            Maximum allowed standard deviation of CV from parallel validations,
            divided by mean CV value.
            This is another measure of model variance.
            Dimensionless, default to 1/2.
        delta_cv_rtol(float): optional
            Maximum allowed  absolute difference of CV between the latest
            2 iterations, divided by the standard deviation of CV.
            Dimensionless, default to 1. (CV can not change more than 1
            standard deviation).
        delta_eci_rtol(float): optional
            Maximum allowed mangnitude of change in ECIs, measured by:
                ||J_1 - J_0||_1 | / || J_1 ||_1. (L1-norms)
            Dimensionless, default to 0.3.
        delta_min_e_rtol(float): optional
            Maximum allowed change to the predicted minimum CE and DFT energy
            under every composition between the last 2 iterations, divided by
            the value of CV. Dimensionless, default set to 1.
        continue_on_new_gs_structure(bool): optional
            If true, whenever a new ground-state structure is detected (
            can not be matched by StructureMatcher), the CE iteration will
            continue even if all other criterion are satisfied.
            Default to False because this may significantly increase the
            number of required iterations.
        """
        return {'cv_atol':
                    self._options.get('cv_atol', 5),
                'cv_std_rtol':
                    self._options.get('cv_var_rtol', 1 / 2),
                'delta_cv_rtol':
                    self._options.get('delta_cv_rtol', 1),
                'delta_min_e_rtol':
                    self._options.get('delta_min_e_rtol', 1),
                "delta_eci_rtol":
                    self._options.get('delta_eci_rtol', 1),
                "continue_on_new_gs_structure":
                    self._options.get('continue_on_new_gs_structure', False)
                }

    def _process_options(self):
        """Pre-process options at initialization."""
        all_options = [{"project_name": self.project_name},
                       self.space_group_options,
                       self.cluster_space_options,
                       self.supercell_enumerator_options,
                       self.composition_enumerator_options,
                       self.structure_enumerator_options,
                       self.calculation_options,
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
        socket = InputsHandler(prim=self.prim.copy(),
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
        if os.path.isfile(INPUTS_FILE):
            with open(INPUTS_FILE) as fin:
                d = json.load(fin, cls=MontyDecoder)
                return cls.from_dict(d)
        elif os.path.isfile(PRIM_FILE):
            prim = Structure.from_file(PRIM_FILE)
            if os.path.isfile(OPTIONS_FILE):
                with open(OPTIONS_FILE) as fin:
                    options = json.load(fin, cls=MontyDecoder)
            else:
                options = {}
            return cls(prim, **options)
        else:
            raise ValueError("No initial setting file provided! "
                             "Please provide at least one of: "
                             f"{INPUTS_FILE} or {PRIM_FILE}.")

    def to_file(self):
        """Automatically save data into a json file."""
        with open(INPUTS_FILE, 'w') as fout:
            json.dump(self.as_dict(), fout, cls=MontyEncoder)


class CeHistoryHandler(MSONable):
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
              each iteration. Dict must contain at least the key "coefs".
              Currently, only stores the expansion on energy.
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
        return set(key for record in self.history for key in record)

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

    def get_attribute(self, name, iter_id=-1):
        """Get any other historical attributes.

        If none exist, will return np.inf to block convergence.
        Args:
            name(str):
                Name of attribute to get. For example, can be "cv".
            iter_id(int):
                Iteration index to get CE with. Default to -1.
        Returns:
            float.
        """
        if len(self.history) == 0:
            return np.inf
        elif name not in self.existing_attributes:
            raise ValueError(f"Attribute {name} not inserted yet!")
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
        for attr in record:
            if attr not in self.existing_attributes and len(self.history) > 0:
                warnings.warn(f"Record contains a new key: {attr} "
                              f"not found in previous records!")
            record[attr] = np.array(record[attr]).tolist()  # Must store list.
        self._history.append(record)

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
