"""Wrappers to load options and history files."""

__author__ = "Fengyu Xie"

import itertools
import numpy as np
import json
import yaml
from copy import deepcopy
import os

from monty.json import MSONable
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.core.periodic_table import Element

from smol.cofe import ClusterSubspace, ClusterExpansion
from smol.cofe.space.domain import (get_site_spaces, Vacancy,
                                    get_allowed_species)

from smol.cofe.extern import EwaldTerm  # Currently only allow EwaldTerm

from .utils.serial_utils import decode_from_dict, serialize_any
from .utils.format_utils import merge_dicts

from .config_paths import (CE_HISTORY_FILE,
                           WRAPPER_FILE,
                           OPTIONS_FILE,
                           PRIM_FILE)

import logging

log = logging.getLogger(__name__)


def parse_comp_restrictions(comp_restrictions):


def construct_prim(bits, sublattice_sites, lattice, frac_coords):
    """Construct a primitive cell based on lattice info.

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

        self._prim = prim

        # These will always be reconstructed.
        self._bits = None
        self._sublattice_sites = None

    @property
    def prim(self):
        """Primitive cell for expansion.

        Returns:
            Structure
        """
        return self._prim

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
                                 self.space_options['cutoffs'],
                                 basis=self.space_options['basis_type'],
                                 **self.space_options['matcher_kwargs'])
                    )

        if self.is_charged_ce:
            subspace.add_external_term(EwaldTerm(**self.space_options["ewald_kwargs"]))

        return subspace

    @property
    def space_options(self):
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
        # Auto-generate cutoffs.
        if self._options.get('cutoffs') is None:
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
            self._options['cutoffs'] = cutoffs

        return {'cutoffs': self._options.get('cutoffs'),
                'basis_type': self._options.get('basis_type', 'indicator'),
                'matcher_kwargs': self._options.get('matcher_kwargs', {}),
                'ewald_kwargs': self._options.get('ewald_kwargs', {})
                }

    # TODO: More generic comp_restriction?
    @property
    def enumerator_options(self):
        """Get enumerator options.

        Return:
            Dict.

        Included keys:
            pre_transmat(3*3 List of int):
                Transformation matrix T applied to primitive cell before
                enumerating supercell matrix, so that each enumerated
                super-cell matrix M = M0 @ T. For example, when doing
                CE on FCC, we recommend using:
                 T = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
                Default to np.eye(3).
            sc_size(int | List[int]):
                Supercel sizes (by number of prims) to enumerate.
                 This must be a multiple of abs(det(transmat)).
                Default to 32 // abs(det(transmat)) * abs(det(transmat)).
                to make the super-cell size close to 32.
            max_sc_cond(float):
                Maximum conditional number of the supercell lattice matrix.
                Default to 8, prevent overly slender super-cells.
            max_sc_angle(float):
                Minimum allowed angle of the supercell lattice.
                Default to 30, prevent overly skewed super-cells.
            sc_mats(List of 3*3 int list):
                Supercell matrices. Will not enumerate super-cells if this
                is given.
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
        transmat = self._options.get('transmat',
                                     np.eye(3, dtype=int))
        transmat = np.array(np.round(transmat), dtype=int).tolist()
        det = int(round(abs(np.linalg.det(transmat))))
        return {'pre_transmat': transmat,
                'sc_size': self._options.get('sc_size', 32 // det * det),
                'max_sc_cond': self._options.get('max_sc_cond', 8),
                'min_sc_angle': self._options.get('min_sc_angle', 30),
                'sc_mats': self._options.get('sc_mats'),
                # If sc_mats is given, will overwrite matrices enumeration.
                'comp_restrictions': self._options.get('comp_restrictions'),
                'comp_enumeration_step': self._options.get('comp_enumstep', 1),
                'structs_to_comp_ratio': self._options.get('structs_to_comp_ratio',
                                                           {"init": 4, "add": 2}),
                'handler_args': self._options.get('handler_args', {}),
                'select_method': self._options.get('select_method', 'leverage')
                }

    @property
    def io_preset_name(self):
        """Name of interface preset.

        Default is arch_sge, if not specified.
        Return:
            str.
        """
        return self._options.get("io_preset_name", 'arch_sge')

    @property
    def calc_writer_options(self):
        """Get calculation writer options.

        Returns:
            Dict.

        path(str):
            Path to write VASP output files locally (only used by ArchQueue)
            Default to 'vasp_run/'
        lp_file(str):
            Fireworks launchpad file path.
        writer_strain(3*3 ArrayLike or length 3 ArrayLike of float):
            Strain to apply in vasp POSCAR. Default is [1.05, 1.03, 1.01],
            meaning 5%, 3% and 1% strain in a, b, and c axis. This is to
            break symmetry and help more accurate relaxation.
        is_metal(bool):
            Whether this system is a metal or not. Default to False.
            Determines what set of INCAR settings will be used.
        ab_setting(dict): optional
            Settings to pass into calc writer. Can have 2 keys:
            'relax', 'static', each is a MITSet setting dictionary.
            Refer to documentation of calc_writer and
            pymatgen.io.sets.
        writer_type(str):
            Name of CalcWriter to be used.
        """
        return {'path': self._options.get('path', 'vasp_run'),
                'lp_file': self._options.get('lp_file'),
                'writer_strain': self._options.get('writer_strain',
                                                   [1.05, 1.03, 1.01]),
                'is_metal': self._options.get('is_metal', False),
                'ab_setting': self._options.get('ab_setting', {}),
                'writer_type': self.io_presets[self.io_preset_name][0]
                }

    @property
    def calc_manager_options(self):
        """Get calculation manager options.

        Returns:
            Dict.

        manager_path(str):
            Path to write VASP output files locally (only used by ArchQueue)
            Default to 'vasp_run/'
        lp_file(str):
            Fireworks launchpad file path.
        fw_file(str):
            Fireworks fireworker file path.
        qa_file(str):
            Fireworks queue adapter file.
        I would recommend you to set up your Fireworks properly before using,
        rather than specifying here.
        kill_command(str):
            queued job kill command of the current Queue system.
        ab_command(str):
            Command to run ab initio package, such as vasp.
        ncores(int):
            Number of cores to use for each DFT calculation. Default to 16.
        time_limit(float):
            Time limit to run a single DFT (in seconds). Defualt to 72 hours.
        check_interval(float):
            Time interval to check queue status (in seconds).
            Default to 300 seconds.
        manager_type(str):
            Name of the CalcManager class to use.
        """
        return {'path': self._options.get('path', 'vasp_run'),
                'lp_file': self._options.get('lp_file'),
                'fw_file': self._options.get('fw_file'),
                'qa_file': self._options.get('qa_file'),
                'kill_command': self._options.get('kill_command'),
                'ab_command': self._options.get('ab_command', 'vasp'),
                'ncores': self._options.get('ncores', 16),
                'time_limit': self._options.get('time_limit', 259200),
                'check_interval': self._options.get('check_interval', 300),
                'manager_type': self.io_presets[self.io_preset_name][1]
                }

    @property
    def calc_reader_options(self):
        """Get calculation reader options.

        Returns:
            Dict.

        reader_path(str):
            Path to write VASP output files locally (only used by ArchQueue)
            Default to 'vasp_run/'
        md_file(str):
            Path to mongodb setting file. Only used by MongoFWReader.
        reader_type(str):
            Name of the CalcReader to use.
        """
        return {'path': self._options.get('path', 'vasp_run'),
                'md_file': self._options.get('md_file'),
                'reader_type': self.io_presets[self.io_preset_name][2]
                }

    @property
    def featurizer_options(self):
        """Get featurizer options.

        Returns:
            Dict.

        other_props(list[str]):
            Name of other properties to expand except energy.
        max_charge(int > 0):
            Maximum charge abs value allowed in decorated structure.
            Structures having more charge than this will be considered
            fail assignment. Default to 0, namely charge balanced.
        decorator_types(str):
            Name of decorator class to use before mapping structure.
            For example, expansion with charge needs MagChargeDecorator.
        decorator_args(str):
            Arguments to pass into decorator. See documentation of 
            each in species_decorator.
        """
        # Since we can not automatically generate labels_table, currently
        # we don't have automatically specify species decorator. If your
        # species need decoration, you have to provide the decorator types
        # and arguments in your options file.
        decorators_types = self._options.get('decorators_types', [])

        for b in itertools.chain(*self.bits):
            if (not isinstance(b, (Vacancy, Element))
                    and len(decorators_types) == 0):
                raise ValueError("Cluster expasion requires decoration, " +
                                 "but no decorator to {} is given!".format(b))

        return {'other_props': self._options.get('other_props', []),
                'max_charge': self._options.get('max_charge', 0),
                'decorators_types': decorators_types,
                'decorators_args': self._options.get('decorators_args',
                                                     [])}

    @property
    def fitter_options(self):
        """Get fitter options.

        Returns:
            Dict.

        regression_flavor(str):
            Choosen theorytoolkit regularization method. Default to
            'lasso'.
        weights_flavor(str):
            Weights to use in theorytoolkit estimators. Default to
            'unweighted'.
        use_hierarchy(str):
            Whether or not to use hierarchy in regularization fitting.
            Default to True.
        regression_params(dict):
            Argument to pass into estimator.
            See theorytoolkit.regression documentations.
        weighter_params(dict):
            Argument to pass into weights generator.
            See smol.cofe.wangling.tool.
        """
        return {'regression_flavor': self._options.get('regression_flavor',
                                                       'lasso'),
                'weights_flavor': self._options.get('weights_flavor',
                                                    'unweighted'),
                'use_hierarchy': self._options.get('use_hierarchy', True),
                'regression_params': self._options.get('regression_params',
                                                       {}),
                'weighter_params': self._options.get('weighter_params', {})
                }

    @property
    def gs_checker_options(self):
        """Get ground state checker options.

        Note: currently only checks canonical GS convergence!
        Returns:
            Dict.

        e_tol_in_cv(float):
            Energy tolerance in the unit of CV value. Default is 3*CV.
            When new GS is within 3*CV of the previous one under each
            composition, we will see CE as converged.
        cv_change_tol(float):
            Relative tolerance to decrease in CV. Default is 20%(0.2).
            If decrease of CE is smaller than this precentage, we see
            CE as converged.
        """
        return {'e_tol_in_cv': self._options.get('e_tol_in_cv', 3),
                'cv_change_tol': self._options.get('cv_change_tol', 0.2)
                }

    @property
    def calc_writer(self):
        """Calculation writer object initialized from the options.

        Return:
            BaseCalcWriter
        """
        kwargs = self.calc_writer_options.copy()
        name = kwargs.pop('writer_type')
        return writer_factory(name, **kwargs)

    @property
    def calc_manager(self):
        """Calculation manager object initialized from the options.

        Return:
            BaseCalcManager
        """
        kwargs = self.calc_manager_options.copy()
        name = kwargs.pop('manager_type')
        return manager_factory(name, **kwargs)

    # Used in featurizer. Calc reader usually will not be explicitly called.
    @property
    def calc_reader(self):
        """Calculation reader object initialized from the options.

        This class does not directly interact with the datamanager.
        It is only an attachment to Featurizer.
        Return:
            BaseCalcReader
        """
        kwargs = self.calc_reader_options.copy()
        name = kwargs.pop('reader_type')
        return reader_factory(name, **kwargs)

    @property
    def options(self):
        """Returns completed options dictionary for the user's reference.

        Returns:
            Dict.
        """
        all_options = [self.enumerator_options,
                       self.calc_writer_options,
                       self.calc_reader_options,
                       self.calc_manager_options,
                       self.featurizer_options,
                       self.fitter_options,
                       self.gs_checker_options,
                       self.space_options]

        # If conflicting keys appear, will only take the first value.
        # It is your responsibility to avoid conflicting keys!
        self._options = merge_dicts(all_options, keep_all=False)
        return self._options

    def copy(self):
        """Deepcopy InputsWrapper object."""
        socket = InputsWrapper(bits=deepcopy(self.bits),
                               sublat_list=deepcopy(self.sublat_list),
                               lattice=self.lattice.copy(),
                               frac_coords=self.frac_coords.copy(),
                               prim=self.prim.copy(),
                               prim_file=deepcopy(self.prim_file),
                               **self.options)
        socket._subspace = self.subspace.copy()
        socket._compspace = deepcopy(self.compspace)

        return socket

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize from a dictionary.
        Args:
            d(dict):
                Dictionary containing all lattice information, options.
                Notice: History information can not be loaded by this method!
                        It must be loaded separately!
        Returns:
            InputsWrapper object.
        """
        lat_keys = ['prim_file', 'prim',
                    'lattice', 'frac_coords',
                    'bits', 'sublat_list']

        attr_keys = ['_subspace', '_compspace']

        mson_keys = ["@module", "@class"]

        prim_file = d.get('prim_file')
        prim = d.get('prim')
        if isinstance(prim, dict):
            prim = Structure.from_dict(prim)

        lattice = d.get('lattice')
        if isinstance(lattice, dict):
            lattice = Lattice.from_dict(lattice)

        frac_coords = d.get('frac_coords')

        bits = d.get('bits')
        bits_deser = []
        if bits is not None:
            sl_bits_deser = []
            for sl_bits in bits:
                for b in sl_bits:
                    if isinstance(b, dict):
                        sl_bits_deser.append(decode_from_dict(b))
                    else:
                        sl_bits_deser.append(b)
                bits_deser.append(sl_bits_deser)

        sublat_list = d.get('sublat_list')

        options = {k: v for k, v in d.items()
                   if k not in lat_keys + attr_keys + mson_keys}

        # Radius keys may be loaded as strings rather than ints.
        if options.get('radius') is not None:
            options['radius'] = {int(k): float(v)
                                 for k, v in options['radius'].items()}

        socket = cls(bits=bits, sublat_list=sublat_list,
                     lattice=lattice, frac_coords=frac_coords,
                     prim=prim, prim_file=prim_file,
                     **options)
        # Deserialize attributes. Must all be MSONable.
        for attr_k in attr_keys:
            attr = d.get(attr_k)
            if isinstance(attr, dict):
                attr = decode_from_dict(attr)
            setattr(socket, attr_k, attr)

        return socket

    def as_dict(self):
        """Serialize all class data into a dictionary.

        This dictionary will be saved into json by auto_save.
        History data will not be saved. It is handled separately!

        Returns:
            dict. Containing all serialized lattice data, options,
            and important attributes 
        """
        lat_keys = ['prim_file', 'prim', 'lattice', 'frac_coords',
                    'bits', 'sublat_list']

        attr_keys = ['_subspace', '_compspace']

        ds = [{k: serialize_any(getattr(self, k)) for k in lat_keys},
              self.options,
              {k: serialize_any(getattr(self, k[1:])) for k in attr_keys},
              {'@module': self.__class__.__module__,
               '@class': self.__class__.__name__}
              ]

        return merge_dicts(ds)

    @classmethod
    def auto_load(cls,
                  wrapper_file=WRAPPER_FILE,
                  options_file=OPTIONS_FILE):
        """Automatically load object from files.

        If the wrapper_file is already present, will load from the wrapper
        file first.

        All paths can be changed, but I don't recommend you to do so.

        Returns:
            InputsWrapper.
        """
        if os.path.isfile(wrapper_file):
            with open(wrapper_file) as ops:
                d = json.load(ops)
        elif os.path.isfile(options_file):
            with open(options_file) as ops:
                d = yaml.load(ops, Loader=yaml.FullLoader)
        else:
            raise ValueError("No calculation setting specified!")

        socket = cls.from_dict(d)

        return socket

    def auto_save(self, wrapper_file=WRAPPER_FILE):
        """Automatically save this object data into a json file.

        Default save path can be changed, but I won't recommend.
        """
        with open(wrapper_file, 'w') as fout:
            json.dump(self.as_dict(), fout)


class HistoryWrapper(MSONable):
    """History reader class."""

    def __init__(self, cluster_subspace, history=[]):
        """Initialize HistoryWrapper.

        Args:
            cluster_subspace(smol.cofe.ClusterSubspace):
              Cluster subspace of CE.
            history(List[dict]): optional
              A list storing all past history CE models.
        """
        self._history = history
        self._subspace = cluster_subspace

    @property
    def history(self):
        """History Dict."""
        return self._history

    @property
    def subspace(self):
        """Cluster subspace."""
        return self._subspace

    def get_ce_n_iters_ago(self, n_ago=1):
        """Get the cluster expansion object n iterations ago.

        Does not store past feature matrices, and only handles
        energy cluster expansion.

        If none exist, will initialize a dummy CE, with all
        external terms with coef 1, and cluster terms with coef 0.

        Args:
            n_ago(int):
                Specifies which history ce step to read. Default is 1,
                will read the latest ce available.
        Returns:
            ClusterExpansion.
        """
        if len(self._history) < n_ago:
            log.warning("Cluster expansion history can not be " +
                        "dated back to {} iteration(s) ago. ".format(n_ago) +
                        "Making dummy cluster expasnion.")

            coefs = np.zeros(self.subspace.num_corr_functions +
                             len(self.subspace.external_terms))
            if len(self.subspace.external_terms) > 0:
                coefs[-len(self.subspace.external_terms):] = 1.0
        else:
            coefs = np.array(self._history[-n_ago]['coefs'])

        return ClusterExpansion(self.subspace, coefs, None)

    @property
    def last_ce(self):
        """Get the last energy CE in history.

        If none exist, will initialize a cluster expansion with
        external ECIs=1, and cluster ECIs=0.

        Returns:
            ClusterExpansion
        """
        return self.get_ce_n_iters_ago(n_ago=1)

    def update(self, new_entry):
        """Insert the last fitted ce record into history.

        Args:
           new_entry(dict):
               Dict containing past energy CE fit information.
               must have the following keys:
                 "cv", "rmse", "coefs".
        """
        self._history.append(new_entry)

    def copy(self):
        """Copy HistoryWrapper."""
        return HistoryWrapper(self.subspace.copy(),
                              deepcopy(self.history))

    def as_dict(self):
        """Serialize HistoryWrapper into dict.

        Returns:
           Dict.
        """
        return {'cluster_subspace': self.subspace.as_dict(),
                'history': self.history,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d):
        """Initialize HistoryWrapper from dict.

        Args:
           d(Dict):
               Dict containing serialized HistoryWrapper.

        Returns:
           HistoryWrapper.
        """
        return cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                   d['history'])

    def auto_save(self, history_path=CE_HISTORY_FILE):
        """Automatically save object to file.

        Args:
            history_path(str):
              Path to history file. Default set in config_paths.py.
              Not recommend to change.
        """
        with open(history_path, 'w') as fout:
            json.dump(self.as_dict(), fout)

    @classmethod
    def auto_load(cls, history_file=CE_HISTORY_FILE):
        """Automatically load object from file.

        Args:
            history_file(str):
              Path to history file. Default set in config_paths.py.
              Not recommend to change.

        Returns:
            HistoryWrapper.
        """
        with open(history_file, 'r') as fin:
            d = json.load(fin)

        return cls.from_dict(d)
