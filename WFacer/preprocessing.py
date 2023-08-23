"""All preprocessing needed before initializing Makers."""

__author__ = "Fengyu Xie"

import itertools

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.domain import Vacancy, get_allowed_species, get_site_spaces


# Parse and process primitive cell.
def reduce_prim(prim, **kwargs):
    """Reduce given cell to make it real primitive.

    Args:
        prim(Structure):
            A primitive cell with partial occupancy to be expanded.
        kwargs:
            Keyword arguments for SpacegroupAnalyzer.
    Returns:
        Structure
    """
    sa = SpacegroupAnalyzer(prim, **kwargs)
    # TODO: maybe we can re-define site_properties transformation
    #  in the future.
    return sa.find_primitive(keep_site_properties=True)


def construct_prim(bits, sublattice_sites, lattice, frac_coords, **kwargs):
    """Construct a primitive cell based on lattice info.

    Provides a helper method to initialize a primitive cell. Of
    course, a prim cell can also be parsed directly from a given
    Structure object or file.
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
        kwargs:
            Keyword arguments for SpacegroupAnalyzer.

    Returns:
        a reduced primitive cell (not necessarily charge neutral):
            Structure
    """
    n_sites = len(frac_coords)
    if not np.allclose(
        np.arange(n_sites), sorted(list(itertools.chain(*sublattice_sites)))
    ):
        raise ValueError(
            f"Provided site indices: {sublattice_sites} "
            f"does not include all {n_sites} sites!"
        )

    site_comps = [{} for _ in range(n_sites)]
    for sublatt_id, sublatt in enumerate(sublattice_sites):
        comp = Composition(
            {
                sp: 1 / len(bits[sublatt_id])
                for sp in bits[sublatt_id]
                if not isinstance(sp, Vacancy)
            }
        )
        for i in sublatt:
            site_comps[i] = comp

    return reduce_prim(Structure(lattice, site_comps, frac_coords), **kwargs)


def get_prim_specs(prim):
    """Get specs of a reduced primitive cell.

    Args:
        prim(Structure):
            Reduced primitive cell with partial occupancy to expand.
            If a prim only contains charge neutral atoms, it is your
            responsibility to make sure that all species in prim are
            either Element or Vacancy. No Species or DummySpecies
            with 0+ charge is allowed, otherwise computed structures
            will not map into prim!
            Also notice that only the sites with the same allowed species
            and species concentrations are considered the same sub-lattice!
    Returns:
        dict:
           a spec dict containing bits, sub-lattice sites,
           sub-lattice sizes, and more.
    """
    unique_spaces = sorted(set(get_site_spaces(prim)))

    # Same order as smol.moca.Sublattice in processor.
    # Ordering between species in a sub-lattice is fixed.
    bits = [list(space.keys()) for space in unique_spaces]
    allowed_species = get_allowed_species(prim)
    sublattice_sites = [
        [i for i, sp in enumerate(allowed_species) if sp == list(space.keys())]
        for space in unique_spaces
    ]

    charge_decorated = False
    for sp in itertools.chain(*bits):
        if not isinstance(sp, (Vacancy, Element)) and sp.oxi_state != 0:
            charge_decorated = True
            break

    d_nns = []
    for i, site1 in enumerate(prim):
        d_ij = []
        for j, site2 in enumerate(prim):
            if j > i:
                d_ij.append(site1.distance(site2))
        d_nns.append(min(d_ij + [prim.lattice.a, prim.lattice.b, prim.lattice.c]))
    d_nn = min(d_nns)

    return {
        "bits": bits,
        "sublattice_sites": sublattice_sites,
        "charge_decorated": charge_decorated,
        "nn_distance": d_nn,
    }


# Get cluster subspace.
def get_cluster_subspace(
    prim,
    charge_decorated,
    nn_distance,
    cutoffs=None,
    use_ewald=True,
    ewald_kwargs=None,
    other_terms=None,
    **kwargs,
):
    """Get cluster subspace from primitive structure and cutoffs.

    Args:
        prim(Structure):
            Reduced primitive cell.
        charge_decorated(bool):
            Whether to use a charge deocration in CE.
        nn_distance(float):
            Nearest neighbor distance in structure, used to guess cluster cutoffs
            if argument "cutoffs" is not given.
        cutoffs(dict): optional
            Cluster cutoff diameters in Angstrom.
            If cutoff values not given, will use a guessing from the nearest neighbor
            distance d:
                pair=3.5d, triplet=2d, quad=2d.
            This guessing is formed based on empirical cutoffs in DRX, but not always
            good for your system. Setting your own cutoffs is highly recommended.
        use_ewald(bool): optional
            Whether to use the EwaldTerm when CE is charge decorated. Default to True.
        ewald_kwargs(dict): optional
            Keyword arguments to initialize EwaldTerm. See docs in smol.cofe.extern.
        other_terms(list[ExternalTerm]): optional
            List of other external terms to be added besides the EwaldTerm. (Reserved
            for extensibility.)
        kwargs:
            Other keyword arguments for ClusterSubspace.from_cutoffs.
    Returns:
        ClusterSubspace:
            A cluster subspace generated from cutoffs.
    """
    if cutoffs is None:
        d_nn = nn_distance
        cutoffs = {2: 3.5 * d_nn, 3: 2 * d_nn, 4: 2 * d_nn}
    space = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs, **kwargs)
    externals = []
    other_terms = other_terms or []
    if use_ewald and charge_decorated:
        ewald_kwargs = ewald_kwargs or {}
        externals.append(EwaldTerm(**ewald_kwargs))
    externals = externals + other_terms
    for e in externals:
        space.add_external_term(e)
    return space


# Parse options.
def process_supercell_options(d):
    """Get options to enumerate supercell matrices.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing supercell matrix options, including the following keys:
        supercell_from_conventional(bool):
            Whether to find out primitive cell to conventional
            standard structure transformation matrix T, and enumerate
            super-cell matrices in the form of: M = M'T.
            Default to true. If not, will set T to eye(3).
        objective_num_sites(int):
            The Supercel sizes (in number of sites, both active and inactive)
            to approach.
            Default to 64. Enumerated super-cell size will be
            a multiple of det(T) but the closest one to this objective
            size.
            Note: since super-cell matrices with too high a conditional
            number will be dropped, do not use a super-cell size whose
            decompose to 3 integer factors are different in scale.
            For example, 17 = 1 * 1 * 17 is the only possible factor
            decomposition for 17, whose matrix conditional number will
            always be larger than the cut-off (8).
            Currently, we only support enumerating super-cells with the
            same size.
        spacegroup_kwargs(dict):
            Keyword arguments used to initialize a SpaceGroupAnalyzer.
            Will also be used in reducing the primitive cell.
        max_sc_condition_number(float):
            Maximum conditional number of the supercell lattice matrix.
            Default to 8, prevent overly slender super-cells.
        min_sc_angle(float):
            Minimum allowed angle of the supercell lattice.
            Default to 30, prevent overly skewed super-cells.
        sc_matrices(List[3*3 ArrayLike[int]]):
            Supercell matrices. Will not enumerate super-cells if this
            is given. Default to None. Note: if given, all supercell matrices
            must be of the same size!
    """
    return {
        "supercell_from_conventional": d.get("supercell_from_conventional", True),
        "objective_num_sites": d.get("objective_num_sites", 64),
        "spacegroup_kwargs": d.get("spacegroup_kwargs", {}),
        "max_sc_condition_number": d.get("max_sc_condition_number", 8),
        "min_sc_angle": d.get("min_sc_angle", 30),
        "sc_matrices": d.get("sc_matrices"),
    }


def process_composition_options(d):
    """Get options to enumerate compositions with CompositionSpace.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing composition options, including the following keys:
            charge_neutral (bool): optional
                Whether to add charge balance constraint. Default to true.
            other_constraints:
            (list of tuples of (1D arrayLike[float], float, str) or str): optional
                Other composition constraints to be applied to restrict the
                enumerated compositions.
                Allows two formats for each constraint in the list:
                    1, A string that encodes the constraint equation.
                    For example: "2 Ag+(0) + Cl-(1) +3 H+(2) <= 3 Mn2+ +4".
                       A string representation of constraint must satisfy the following
                       rules,
                       a, Contains a relation symbol ("==", "<=", ">=" or "=") are
                       allowed.
                       The relation symbol must have exactly one space before and one
                       space after to separate the left and the right sides.
                       b, Species strings must be readable by get_species in smol.cofe
                       .space.domain. No space is allowed within a species string.
                       For the format of a legal species string, refer to
                       pymatgen.core.species and smol.cofe.
                       c, You can add a number in brackets following a species string
                       to specify constraining the amount of species in a particular
                       sub-lattice. If not given, will apply the constraint to this
                       species on all sub-lattices.
                       This sub-lattice index label must not be separated from
                       the species string with space or any other character.
                       d, Species strings along with any sub-lattice index label must
                       be separated from other parts (such as operators and numbers)
                       with at least one space.
                       e, The intercept terms (a number with no species that follows)
                       must always be written at the end on both side of the equation.
                    2, The equation expression, which is a tuple containing a list of
                    floats of length self.n_dims to give the left-hand side coefficients
                    of each component in the composition "counts" format, a float to
                    give the right-hand side, and a string to specify the comparative
                    relationship between the left- and right-hand sides. Constrained in
                    the form of a_left @ n = (or <= or >=) b_right.
                    The components in the left-hand side are in the same order as in
                    itertools.chain(*self.bits).
                Note that all numerical values in the constraints must be set as they are
                to be satisfied per primitive cell given the sublattice_sizes!
                For example, if each primitive cell contains 1 site in 1 sub-lattice
                specified as sublattice_sizes=[1], with the requirement that species
                A, B and C sum up to occupy less than 0.6 sites per sub-lattice, then
                you must write: "A + B + C <= 0.6".
                While if you specify sublattice_sizes=[2] in the same system per
                primitive cell, to specify the same constraint, write
                "A + B + C <= 1.2" or "0.5 A + 0.5 B + 0.5 C <= 0.6", etc.
            See documentation of smol.moca.composition.space.
        comp_enumeration_step (int): optional
            Skip step in returning the enumerated compositions.
            If step > 1, on each dimension of the composition space,
            we will only yield one composition in every N compositions.
            Default to 1.
        compositions (2D arrayLike[int]): optional
            Fixed compositions with which to enumerate the structures. If
            given, will not enumerate other compositions.
            Should be provided as the "species count"-format of CompositionSpace.
    """
    return {
        "comp_enumeration_step": d.get("comp_enumeration_step", 1),
        "charge_neutral": d.get("charge_neutral", True),
        "other_constraints": d.get("other_constraints", None),
        "compositions": d.get("compositions", []),
    }


def process_structure_options(d):
    """Get options to enumerate structures.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing structure options, including the following keys:
        num_structs_per_iter_init (int):
            Number of new structures to enumerate in the first iteration.
            It is recommended that in each iteration, at least 2~3
            structures are added for each composition.
            Default is 60.
        num_structs_per_iter_add (int):
            Number of new structures to enumerate in each followed iteration.
            Default is 40.
        sample_generator_kwargs(Dict):
            kwargs of CanonicalSampleGenerator.
        init_method(str):
            Structure selection method in the first iteration.
            Default is "leverage". Allowed options include: "leverage" and
            "random".
        add_method(str):
            Structure selection method in subsequent iterations.
            Default is 'leverage'. Allowed options are: 'leverage'
            and 'random'.
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
        n_parallel(int): optional
            Number of generators to run in parallel. Default is to use
            a quarter of cpu count.
        keep_ground_states(bool):
            Whether always to add new ground states to the training set.
            Default to True.
    """
    return {
        "num_structs_per_iter_init": d.get("num_structs_per_iter_init", 60),
        "num_structs_per_iter_add": d.get("num_structs_per_iter_add", 40),
        "sample_generator_kwargs": d.get("sample_generator_kwargs", {}),
        "init_method": d.get("init_method", "leverage"),
        "add_method": d.get("add_method", "leverage"),
        "duplicacy_criteria": d.get("duplicacy_criteria", "correlations"),
        "n_parallel": d.get("n_parallel"),
        "keep_ground_states": d.get("keep_ground_states", True),
    }


def process_calculation_options(d):
    """Get options to do vasp calculations in atomate2.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing calculation options, including the following keys:
        apply_strain(3*3 ArrayLike or 1D ArrayLike[float] of 3):
            Strain matrix to apply to the structure before relaxation,
            in order to break structural symmetry of forces.
            Default is [1.03, 1.02, 1.01], which means to
            stretch the structure by 3%, 2% and 1% along a, b, and c
            directions, respectively.
        relax_generator_kwargs(dict):
            Additional arguments to pass into an atomate2
            VaspInputGenerator that is used to initialize RelaxMaker.
            This is where the pymatgen vaspset arguments should go.
        relax_maker_kwargs(dict):
            Additional arguments to initialize an atomate2 RelaxMaker.
            Not frequently used.
        add_tight_relax(bool):
            Whether to add a tight relaxation job after a coarse
            relaxation. Default to True.
            You may want to disable this if your system has
            difficulty converging forces or energies.
        tight_generator_kwargs(dict):
            Additional arguments to pass into an atomate2 VaspInputGenerator
            that is used to initialize TightRelaxMaker.
            This is where the pymatgen vaspset arguments should go.
        tight_maker_kwargs(dict):
            Additional arguments to pass into an atomate2
            TightRelaxMaker. A tight relax is performed after
            relaxation, if add_tight_relax is True.
            Not frequently used.
       static_generator_kwargs(dict):
            Additional arguments to pass into an atomate2
            VaspInputGenerator that is used to initialize StaticMaker.
            This is where the pymatgen vaspset arguments should go.
        static_maker_kwargs(dict):
            Additional arguments to pass into an atomate2
            StaticMaker.
            Not frequently used.
        other_properties(list[(str, str)| str]): optional
            Other property names beyond "energy" and "uncorrected_energy"
            to be retrieved from taskdoc and recorded into the wrangler,
             and the query string to retrieve them, paired in tuples.
            If only strings are given, will also query with the given
            string.
            For the rules in writing the query string, refer to utils.query.
            By default, will not record any other property.
        Refer to the atomate2 documentation for more information.
        Note: the default vasp sets in atomate 2 are not specifically
        chosen for specific systems. Using your own vasp set input
        settings is highly recommended!
    """
    strain_before_relax = d.get("apply_strain", [1.03, 1.02, 1.01])
    strain_before_relax = np.array(strain_before_relax)
    if len(strain_before_relax.shape) == 1:
        strain_before_relax = np.diag(strain_before_relax)

    if strain_before_relax.shape != (3, 3):
        raise ValueError(
            "Provided strain format "
            "must be either 3*3 arraylike or "
            "an 1d arraylike of length 3!"
        )

    return {
        "apply_strain": strain_before_relax.tolist(),
        "relax_generator_kwargs": d.get("relax_generator_kwargs", {}),
        "relax_maker_kwargs": d.get("relax_maker_kwargs", {}),
        "add_tight_relax": d.get("add_tight_relax", True),
        "tight_generator_kwargs": d.get("tight_generator_kwargs", {}),
        "tight_maker_kwargs": d.get("tight_maker_kwargs", {}),
        "static_generator_kwargs": d.get("static_generator_kwargs", {}),
        "static_maker_kwargs": d.get("static_maker_kwargs", {}),
        "other_properties": d.get("other_properties"),
    }


def process_decorator_options(d):
    """Get options to decorate species with charge or other properties.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing calculation options, including the following keys:
        decorator_types(list(str)): optional
            Name of decorators to use for each property. If None, will
            choose the first one in all implemented decorators
            (see specie_decorators module).
        decorator_kwargs(list[dict]): optional
            Arguments to pass into each decorator. See the doc of each specific
            decorator.
        decorator_train_kwargs(list[dict]): optional
            Arguments to pass into each decorator when calling decorator.train.
    """
    # Update these pre-processing rules when necessary,
    # if you have new decorators implemented.
    decorator_types = d.get("decorator_types", [])
    decorator_kwargs = d.get("decorator_kwargs", [])
    decorator_train_kwargs = d.get("decorator_train_kwargs", [])

    if len(decorator_kwargs) > 0 and len(decorator_kwargs) != len(decorator_types):
        raise ValueError(
            "If provided any, number of kwargs must match"
            " the number of decorators exactly!"
        )
    if len(decorator_kwargs) == 0:
        decorator_kwargs = [{} for _ in decorator_types]

    if len(decorator_train_kwargs) > 0 and len(decorator_train_kwargs) != len(
        decorator_types
    ):
        raise ValueError(
            "If provided any, number of train kwargs must match"
            " the number of decorators exactly!"
        )
    if len(decorator_train_kwargs) == 0:
        decorator_train_kwargs = [{} for _ in decorator_types]

    return {
        "decorator_types": decorator_types,
        "decorator_kwargs": decorator_kwargs,
        "decorator_train_kwargs": decorator_train_kwargs,
    }


def process_subspace_options(d):
    """Get options to create cluster subspace.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing fit options, including the following keys:
        cutoffs(dict{int: float}):
            Cluster cutoff diameters of each type of clusters. If not given,
            will guess with nearest neighbor distance in the structure.
            Setting your own is highly recommended.
        use_ewald(bool):
            Whether to use the EwaldTerm as an ExternalTerm in the cluster
            space. Only available when the expansion is charge decorated.
            Default to True.
        ewald_kwargs(dict):
            Keyword arguments used to initialize EwaldTerm.
            Note: Other external terms than ewald term not supported yet.
        from_cutoffs_kwargs(dict):
            Other keyword arguments to be used in ClusterSubspace.from_cutoffs,
            for example, the cluster basis type. Check smol.cofe for detail.
    """
    return {
        "cutoffs": d.get("cutoffs", None),
        "use_ewald": d.get("use_ewald", True),
        "ewald_kwargs": d.get("ewald_kwargs", {}),
        "from_cutoffs_kwargs": d.get("from_cutoffs_kwargs", {}),
    }


def process_fit_options(d):
    """Get options to fit ECIs with sparse-lm.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing fit options, including the following keys:
        estimator_type(str):
            The name of an estimator class in sparce-lm. Default to
            'Lasso'.
        use_hierarchy(str):
            Whether to use hierarchy in regularization fitting, when
            estimator type is mixedL0. Default to True.
        center_point_external(bool): optional
            Whether to fit the point and external terms with linear regression
            first, then fit the residue with regressor. Default to None, which means
            when the feature matrix is full rank, will not use centering, otherwise
            centers. If set to True, will force centering, but use at your own risk
            because this may cause very large CV. If set to False, will never use
            centering.
        filter_unique_correlations(bool):
            If the wrangler have structures with duplicated correlation vectors,
            whether to fit with only the one with the lowest energy.
            Default to True.
        estimator_kwargs(dict):
            Other keyword arguments to pass in when constructing an
            estimator. See sparselm.models
        optimizer_type(str):
            The name of optimizer class used to optimize model hyperparameters
            over cross validation. Default is None. Supports "grid-search-CV" and
            "line-search-CV" optimizers. See sparselm.model_selection.
        param_grid(dict|list(tuple)):
            Parameters grid to search for estimator hyperparameters.
            See sparselm.optimizer.
        optimizer_kwargs(dict):
            Keyword arguments when constructing GridSearch or LineSearch class.
            See sparselm.optimizer.
        fit_kwargs(dict):
            Keyword arguments when calling GridSearch/LineSearch/Estimator.fit.
            See docs of the specific estimator.
    """
    return {
        "estimator_type": d.get("estimator_type", "lasso"),
        # Under Seko's iterative procedure, there is not much sense in weighting
        # over energy, because low energy samples are always preferred.
        # We will not include sample weighting scheme here. You can play with the
        # final CeDataWangler if you want.
        "use_hierarchy": d.get("use_hierarchy", True),
        "center_point_external": d.get("center_point_external"),
        "filter_unique_correlations": d.get("filter_unique_correlations", True),
        "estimator_kwargs": d.get("estimator_kwargs", {}),
        "optimizer_type": d.get("optimizer_type", "grid-search"),
        "param_grid":
        # Use lasso as default as mixed l0 might get too aggressive.
        d.get("param_grid", {"alpha": (2 ** np.linspace(-20, 4, 25)).tolist()}),
        "optimizer_kwargs": d.get("optimizer_kwargs", {}),
        "fit_kwargs": d.get("fit_kwargs", {}),
    }


def process_convergence_options(d):
    """Get convergence check options.

    Args:
        d(dict):
            An input dictionary containing various options in the input file.
    Returns:
        dict:
            A dict containing convergence options, including the following keys:
        cv_tol(float): optional
            Maximum allowed CV value in meV per site (including vacancies).
            (not eV per atom because some CE may contain Vacancies.)
            Default to None, but better set it manually!
        std_cv_rtol(float): optional
            Maximum standard deviation of CV allowed in cross validations,
            normalized by mean CV value.
            Dimensionless, default to None, which means this standard deviation
            of cv will not be checked.
        delta_cv_rtol(float): optional
            Maximum difference of CV allowed between the last 2 iterations,
            divided by the standard deviation of CV in cross validation.
            Dimensionless, default to 0.5.
        delta_eci_rtol(float): optional
            Maximum allowed mangnitude of change in ECIs, measured by:
                ||J' - J||_1 | / || J' ||_1. (L1-norms)
            Dimensionless. If not given, will not check ECI values for
            convergence, because this may significantly increase the
            number of iterations.
        delta_min_e_rtol(float): optional
            Maximum difference allowed to the predicted minimum CE and DFT energy
            at every composition between the last 2 iterations. Dimensionless,
            divided by the value of CV.
            Default set to 2.
        continue_on_finding_new_gs(bool): optional
            If true, whenever a new ground-state structure is detected (
            symmetrically distinct), the CE iteration will
            continue even if all other criterion are satisfied.
            Default to False because this may also increase the
            number of iterations.
        max_iter(int): optional
            Maximum number of iterations allowed. Will not limit number
            of iterations if set to None, but setting one limit is still
            recommended. Default to 10.
    """
    return {
        "cv_tol": d.get("cv_tol"),
        "std_cv_rtol": d.get("std_cv_rtol"),
        "delta_cv_rtol": d.get("delta_cv_rtol", 0.5),
        "delta_eci_rtol": d.get("delta_eci_rtol"),
        "delta_min_e_rtol": d.get("delta_min_e_rtol", 2),
        "continue_on_finding_new_gs": d.get("continue_on_finding_new_gs", False),
        "max_iter": d.get("max_iter", 10),
    }


def get_initial_ce_coefficients(cluster_subspace):
    """Initialize null ce coefficients.

    Any coefficient, except those for external terms, will be initialized to 0.
    This guarantees that for ionic systems, structures with lower ewald energy
    are always selected first.
    Args:
        cluster_subspace(ClusterSubspace):
            The initial cluster subspace.
    Returns:
        np.ndarray[float].
    """
    return np.array(
        [0 for _ in range(cluster_subspace.num_corr_functions)]
        + [1 for _ in range(len(cluster_subspace.external_terms))]
    ).astype(float)
