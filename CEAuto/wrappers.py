"""Wrappers to load options and history files."""

__author__ = "Fengyu Xie"

import itertools
import numpy as np
import warnings
import json
import yaml
from copy import deepcopy

from monty.json import MSONable
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.core.periodic_table import Element

from smol.cofe import ClusterSubspace, ClusterExpansion
from smol.cofe.space.domain import (get_site_spaces, Vacancy,
                                    get_allowed_species)
from smol.cofe.extern import *
from smol.moca import CompSpace
from smol.moca.ensemble.sublattice import Sublattice

from .utils.serial_utils import decode_from_dict, serialize_any
from .utils.format_utils import merge_dicts

from .calc_reader import *
from .calc_writer import *
from .calc_manager import *

from .config_paths import *


class InputsWrapper(MSONable):
    """Wrapps options into formats required to init other modules.
    Can be saved and re-initialized from the save.
    Direct initialization is not recommended. You are supposed to 
    initialize it with auto_load().
    """
    # Presets of vasp and database interfaces.
    # If you have implemented new combos, add them here.
    io_presets = {'arch_sge': ('Archvasp',
                                'Archsge',
                                'Archvasp'),
                  'mongo_fw': ('Mongovasp',
                               'Mongofw',
                               'Mongovasp')
                 }

    def __init__(self,
                 bits=None, sublat_list=None,
                 lattice=None, frac_coords=None,
                 prim=None, prim_file=None,
                 **options):
        """Initialize.

        Args: 
            bits(List[List[Specie|Vacancy]]):
                Species occupying each sublattice.
            sublat_list(List[List[int]]):
                Indices of PRIMITIVE CELL sites in the same sublattices.
            lattice(pymatgen.Lattice):
                Lattice of primitive cell.
            frac_coords(n*3 ArrayLike):
                Fractional coordinates of sites in a primitive cell.
            prim(pymatgen.Structure):
                primitive cell containing all comp space info and lattice
                info. Make sure it is the same with the one used to initialize
                CE.
                Does not check whether the previous 4 options and prim match,
                it is your responsibility.
            prim_file(str):
                Path to the primitive cell file.
            **options:
                Other options used in modules, in the form of keyword
                arguments.

        Concentration dependent CE not supported.
        """
        self._options = options

        # Always reconstruct prim, because bits
        # need to be sorted in the order of SiteSpace.
        _bits = bits
        _lattice = lattice
        _frac_coords = frac_coords
        _sublat_list = sublat_list
        self._prim = prim
        self._prim_file = prim_file

        if (self._prim is None and self._prim_file is not None and
            os.path.isfile(self._prim_file):
            self._prim = Structure.from_file(self._prim_file)

        if self._prim is None:
            if (_bits is None or _lattice is None or
                _frac_coords is None or _sublat_list is None):
                raise ValueError("Structure information not sufficient! "+
                                 "Can't start CE.")
            else:
                self._prim = InputsWrapper.construct_prim(_bits,
                                                          _sublat_list,
                                                          _lattice,
                                                          _frac_coords)

        # These will always be reconstructed.
        self._bits = None
        self._sublat_list = None
        self._lattice = None
        self._frac_coords = None

        if self.prim.charge != 0:
            self._prim = construct_prim(self.bits, self.sublat_list,
                                        self.lattice, self.frac_coords)

        # Compspace can be saved and re-used.
        self._compspace = None
        self._is_charged_ce = None

        self._radius = options.get('radius')
        self._subspace = None

    @classmethod
    def construct_prim(bits, sublat_list, lattice, frac_coords):
        """Construct a primitive cell based on lattice info.

        Args:
            bits(List[List[Specie]]):
                Allowed species on each sublattice. No sorting
                required.
            sublat_list(List[List[int]]):
                Site indices of each sublattice.
            lattice(Lattice):
                Lattice of the primitive cell.
            frac_coords(ArrayLike):
                Fractional coordinates of sites.
        Returns:
            Structure, a primitive cell structure including all
            expansion species (includes vacancy if allowed).
            Charge neutral guaranteed.
        """
        # Modify prim to a charge neutral composition that includes
        # all species.
        sl_sizes = [len(s) for s in sublat_list]
        compspace = CompSpace(bits, sl_sizes)
        typical_comp = (compspace.
                        get_random_point_in_unit_space(form='composition'))

        N_sites = sum(sl_sizes)
        prim_comps = [{} for i in range(N_sites)]
        for sl_id, sl in enumerate(sublat_list):
            processed_comp = Composition({sp: n for sp, n in
                                         typical_comp[sl_id].items()
                                         if not isinstance(sp, Vacancy)})
            for i in sl:
                prim_comps[i] = processed_comp

        return Structure(lattice, prim_comps, frac_coords)

    @property
    def prim_file(self):
        return self._prim_file

    @property
    def bits(self):
        """
        List of species on each sublattice.
        Returns:
            List[List[Specie]]
        """
        if self._bits is None:
            # By default, does NOT include measure!
            # Different sites with same species but different numbers
            # are considered the same sublattice!
            unique_spaces = tuple(set(get_site_spaces(self.prim)))
            allowed_species = get_allowed_species(self.prim)

            # Automatic sublattices, same rule as smol.moca.Sublattice
            self._bits = [list(space.keys()) for space in unique_spaces]

        # No sort of bits on sublattices! Order based on dict key order.
        return self._bits

    @property
    def sublat_list(self):
        """
        List of site indices in sublattices.
        Returns:
            List[List[int]]
        """
        if self._sublat_list is None:
            unique_spaces = tuple(set(get_site_spaces(self.prim)))
            allowed_species = get_allowed_species(self.prim)

            # Automatic sublattices, same rule as smol.moca.Sublattice
            self._sublat_list = [[i for i, sp in enumerate(allowed_species)
                                 if sp == list(space.keys())]
                                 for space in unique_spaces]
                  
        # No sort of bits on sublattices! Order based on dict keys.
        return self._sublat_list

    @property
    def sl_sizes(self):
        """
        Sizes of each sublattice.
        Returns:
            List[int]
        """
        return [len(s) for s in self.sublat_list]

    def get_all_sublattices(self, scmatrix=np.identity(3, dtype=int)):
        """Get all smol.moca.sublattices in a super-cell.

        Args:
            scmatrix(Arraylike of int):
                Supercell matrix.
        """
        unique_site_spaces = tuple(set(get_site_spaces(self.prim)))
        supercell = self.prim.copy()
        supercell.make_supercell(scmatrix)
        allowed_species = get_allowed_species(supercell)

        return [Sublattice(site_space,
                np.array([i for i, sp in enumerate(allowed_species)
                         if sp == list(site_space.keys())]))
                for site_space in unique_site_spaces]

    @property
    def lattice(self):
        """
        Lattice of primitive cell.
        Returns:
            pymatgen.Lattice
        """
        if self._lattice is None:
            self._lattice = self.prim.lattice
        return self._lattice

    @property
    def frac_coords(self):
        """
        Fractional coordinates of sites in a primitive cell.
        Returns:
            np.ndarray, shape (n,3)
        """
        if self._frac_coords is None:
            self._frac_coords = self.prim.frac_coords
        return self._frac_coords

    @property
    def compspace(self):
        """Compositional space object corresponding to this system.

        Will be saved and re-utilized!
        Returns:
            CompSpace
        """
        if self._compspace is None:
            self._compspace = CompSpace(self.bits, self.sl_sizes)
        return self._compspace

    @property
    def is_charged_ce(self):
        """Charged CE or not.

        If true, system species have charge, and requires charged 
        cluster expansion.
        Returns:
            Boolean.
        """
        if self._is_charged_ce is None:
            self._is_charged_ce = False
            for sp in itertools.chain(*self.bits):
                if (not isinstance(sp, (Vacancy, Element)) and
                    sp.oxi_state != 0):
                    self._is_charged_ce = True
                    break
        return self._is_charged_ce

    @property
    def prim(self):
        """Primitive cell.

        Will obey:
           1, Charge neutrality.
           2, Have all species in self.bits. 
             (Except vacancies, they shouldn't be explicitly included)
        Returns: 
            pymatgen.Structure
        """
        return self._prim

    @property
    def radius(self):
        """Max cluster radii of different cluster types.

        Returns:
           Dict{size(int):radius(float)}
        """
        if self._radius is None or len(self._radius) == 0:
            d_nns = []
            for i, site1 in enumerate(self.prim):
                d_ij = []
                for j, site2 in enumerate(self.prim):
                    if j > i:
                        d_ij.append(site1.distance(site2))
                    if j == i:
                        d_ij.append(min([self.lattice.a,
                                    self.lattice.b, self.lattice.c]))
                d_nns.append(min(d_ij))
            d_nn = min(d_nns)
    
            self._radius = {}
            # Default cluster radius
            self._radius[2] = d_nn * 4.0
            self._radius[3] = d_nn * 2.0
            self._radius[4] = d_nn * 2.0

        return self._radius

    @property
    def subspace(self):
        """
        Cluster subspace of this system. Will be saved and re-utilized!
        Returns:
            ClusterSubspace
        """
        if self._subspace is None:
            self._subspace = (ClusterSubspace.
                              from_cutoffs(self.prim, self.radius,
                                           basis=
                                           self.enumerator_options['basis_type']))

            for ex_name,args in zip(self.extern_types,self.extern_args):
                self._subspace.add_external_term(globals()[ex_name](**args))

        return self._subspace
        
    @property
    def extern_types(self):
        """Types of external terms in clustersubspace.
 
        If given system is charge cluster expansion,
        will add ewald term by default. If you don't want it,
        You have to specify empty list in options file.
        """
        return self.featurizer_options.get('extern_types')
  
    @property
    def extern_args(self):
        """Args of external terms.

        If given system is charge cluster expansion,
        will add ewald term by default. If you don't want it,
        You have to specify empty list in options file.
        """
        return self.featurizer_options.get('extern_args')

    @property
    def enumerator_options(self):
        """
        Get enumerator options.
        """
        return {'transmat': self._options.get('transmat',
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                'sc_size': self._options.get('sc_size', 32),
                'max_sc_cond': self._options.get('max_sc_cond', 8),
                'min_sc_angle': self._options.get('min_sc_angle', 30),
                'sc_mats': self._options.get('sc_mats'),
                # If sc_mats is given, will overwrite enumerated sc matrices.
                'comp_restrictions': self._options.get('comp_restrictions'),
                'comp_enumstep': self._options.get('comp_enumstep', 1),
                'n_strs_enum': self._options.get('n_strs_enum', 100),
                'unfreeze_series': self._options.get('unfreeze_series',
                                                     [500, 1500, 5000]),
                'basis_type': self._options.get('basis_type','indicator'),
                'select_method': self._options.get('select_method','CUR')
               }

    @property
    def io_preset_name(self):
        """Name of interface preset.

        Default is arch_sge, if not specified.
        """
        return self._options.get("io_preset_name", 'arch_sge')

    @property
    def calc_writer_options(self):
        """Get calculation writer options.

        Returns:
            Dict.
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
        """
        # Since we can not automatically generate labels_table, currently
        # we don't have automatically specify species decorator. If your
        # species need decoration, you have to provide the decorator types
        # and arguments in your options file.
        decorators_types = self._options.get('decorators_types',[])

        for b in itertools.chain(*self.bits):
            if (not isinstance(b, (Vacancy, Element))
                and len(decorators_types)==0):
                raise ValueError('Cluster expasion requires decoration, "+
                                 "but no decorator to {} is given!'.format(b))

        default_extern_types = ['EwaldTerm'] if self.is_charged_ce else []
        default_extern_args = [{}] if self.is_charged_ce else []

        return {'other_props': self._options.get('other_props', []),
                'decorators_types': decorators_types,
                'decorators_args': self._options.get('decorators_args', []),
                'extern_types': self._options.get('extern_types',
                                                  default_extern_types),
                'extern_args': self._options.get('extern_args',
                                                 default_extern_args)
               }

    @property
    def fitter_options(self):
        """Get fitter options.

        Returns:
            Dict.
        """
        return {'estimator_flavor': self._options.get('estimator_flavor',
                                                      'L2L0Estimator'),
                'weights_flavor': self._options.get('weights_flavor',
                                                    'unweighted'),
                'use_hierarchy': self._options.get('use_hierarchy', True),
                'estimator_params': self._options.get('estimator_params', {}),
                'weighter_params': self._options.get('weighter_params', {})
               }

    @property
    def gs_checker_options(self):
        """Get ground state checker options.

        Returns:
            Dict.
        """
        return {'e_tol_in_cv': self._options.get('e_tol_in_cv', 3),
                'comp_tol': self._options.get('comp_tol', 0.05),
                'cv_change_tol': self._options.get('cv_change_tol', 0.2)
               }
    
    @property
    def gs_generator_options(self):
        """Get ground state generator options.

        Returns:
            Dict.
        """
        return {'handler_flavor': self._options.get('handler_flavor',
                                                    'CanonicalMCHandler'),
                'handler_args': self._options.get('handler_args', {})
               }

    @property
    def calc_writer(self):
        """Calculation writer object initialized from the options.

        Return:
            BaseCalcWriter
        """
        kwargs = self.calc_writer_options.copy()
        name = kwargs.pop('writer_type')
        return globals()[name](**kwargs)  #TODO: this is not very safe, better write a generic class factory like Luis did.

    @property
    def calc_manager(self):
        """Calculation manager object initialized from the options.

        Return:
            BaseCalcManager
        """
        kwargs = self.calc_manager_options.copy()
        name = kwargs.pop('manager_type')
        return globals()[name](**kwargs)

    #Used in featurizer. Calc reader usually will not be explicitly called.
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
        return globals()[name](**kwargs)

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
                       self.gs_generator_options]

        # If conflicting keys appear, will only take the first value.
        # It is your responsibility to avoid conflicting keys!
        self._options = merge_dicts(all_options,keep_all=False)
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
    def from_dict(cls,d):
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

        attr_keys = ['_subspace','_compspace']

        prim_file = d.get('prim_file')
        prim = d.get('prim')
        if isinstance(prim, dict):
            prim = Structure.from_dict(prim)

        lattice = d.get('lattice')
        if isinstance(lattice,dict):
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
                   if k not in lat_keys + attr_keys}

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
        else:
            with open(options_file) as ops:
                d = yaml.load(ops, Loader=yaml.FullLoader)

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

    def __init__(self, cluster_subspace, history):
        """Initialize HistoryWrapper.

        Args:
            cluster_subspace(smol.cofe.ClusterSubspace):
              Cluster subspace of CE.
            history(List[dict]):
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

        Does not store past feature matrices!
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
            warnings.warn("Cluster expansion history can not be " +
                          "dated back to {} iteration(s) ago. ".format(n_ago) +
                          "Making dummy cluster expasnion")

            coefs = np.zeros(self.subspace.num_corr_functions +
                             len(self.subspace.external_terms))
            if len(self.subspace.external_terms) > 0:
                coefs[-len(self.subspace.external_terms): ] = 1.0
        else:
            coefs = np.array(self._history[-n_ago]['coefs'])

        return ClusterExpansion(self.subspace, coefs, None)

    @property
    def last_ce(self):
        """Get the last cluster expansion in history.

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
               Dict containing latest CE fit information.
               Must have the following keys:
                 "cv", "rmse", "coefs".
        """
        self._history.append(new_entry)

    def copy(self):
        """Copy HistoryWrapper."""
        return HistoryWrapper(self.subspace,
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

    def auto_save(self, history_path=CE_HISTORY_PATH):
        """Automatically save object to file.

        Args:
            history_path(str):
              Path to history file. Default set in config_paths.py.
              Not recommend to change.
        """
        with open(history_path, 'w') as fout:
            json.dump(self.as_dict(), fout)

    @classmethod
    def auto_load(cls, history_path=CE_HISTORY_PATH):
        """Automatically load object from file.

        Args:
            history_path(str):
              Path to history file. Default set in config_paths.py.
              Not recommend to change.

        Returns:
            HistoryWrapper.
        """
        with open(history_path, 'r') as fin:
            d = json.load(fin)

        return cls.from_dict(d)