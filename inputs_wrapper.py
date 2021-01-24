"""
A wrapper to convert options and potential history files into
a bunch of elements that other CEAuto modules will need.
"""
__author__ = "Fengyu Xie"

import itertools
import numpy as np
import warnings

from monty.json import MSONable
from pymatgen import Structure,Element

from smol.cofe import ClusterSubspace,ClusterExpansion
from smol.cofe.space.domain import get_allowed_species,Vacancy
from smol.cofe.space.extern import *

from .comp_space import CompSpace
from .calc_reader import *
from .calc_writer import *
from .calc_manager import *

class InputsWrapper(MSONable):

    """
    This class wrapps options and history into objects required by 
    other modules. Can be saved and re-initialized from the save.
    Direct initialization is not recommended. You are supposed to 
    initialize it with auto_load().

    Args: 
        lat_data(Dict):
            Dictionary of deserialized objects, storing everthing
            about the expansion lattice, including 'bits', 'lattice',
            'frac_coords','prim', etc.
        options(Dict):
            other options used in modules.
        history(List[Dict]):
            history cluster expansions information. Same as history 
            option in other modules.
    """
    
    def __init__(self, lat_data, options={}, history=[]):

        self._lat_data = lat_data
        self._options = options
        self._history = history

        self._bits = lat_data.get('bits')
        self._lattice = lat_data.get('lattice')
        self._frac_coords = lat_data.get('frac_coords')
        self._sublat_list = lat_data.get('sublat_list')
        self._prim = lat_data.get('prim')

        self._compspace = None     
        self._is_charged_ce = None

        self._radius = options.get('radius')
        self._subspace = None

    @property
    def bits(self):
        """
        List of species on each sublattice.
        Returns:
            List[List[Specie]]
        """
        if self._bits is None:
            if self._prim is None:
                raise ValueError("Lattice information not sufficient!") 
            prim_bits = get_allowed_species(self._prim)
            if self._sublat_list is not None:
                #User define sublattices, which you may not need very often.
                self._bits = [prim_bits[sl[0]] for sl in self._sublat_list]
            else:
                #Automatic sublattices, same rule as smol.moca.Sublattice:
                #sites with the same compositioon are considered same sublattice.
                self._sublat_list = []
                self._bits = []
                for s_id,s_bits in enumerate(prim_bits):
                    if s_bits in self._bits:
                        sl_id = self._bits.index(s_bits)
                        self._sublat_list[sl_id].append(s_id)
                    else:
                        self._sublat_list.append([s_id])
                        self._bits.append(s_bits)                  

        return self._bits

    @property
    def lattice(self):
        """
        Lattice of primitive cell.
        Returns:
            pymatgen.Lattice
        """
        if self._lattice is None:
            if self._prim is None:
                raise ValueError("Lattice information not sufficient!")        
            self._lattice = self._prim.lattice
        return self._lattice

    @property
    def frac_coords(self):
        """
        Fractional coordinates of sites in a primitive cell.
        Returns:
            np.ndarray, shape (n,3)
        """
        if self._frac_coords is None:
            if self._prim is None:
                raise ValueError("Lattice information not sufficient!")        
            self._frac_coords = self._prim.frac_coords
        return self._frac_coords

    @property
    def sublat_list(self):
        """
        List of site indices in sublattices.
        Returns:
            List[List[int]]
        """
        if self._sublat_list is None:
            bits = self.bits
        return self._sublat_list

    @property
    def sl_sizes(self):
        """
        Sizes of each sublattice.
        Returns:
            List[int]
        """
        return [len(sl) for sl in self.sublat_list]

    @property
    def compspace(self):
        """
        Compositional space object corresponding to this system.
        Returns:
            CompSpace
        """
        if self._compspace is None:
            self._compspace = Compspace(self.bits,self.sl_sizes)
        return self._compspace

    @property
    def is_charged_ce(self):
        """
        If true, system species have charge, and requires charged 
        cluster expansion.
        Returns:
            Boolean.
        """
        if self._is_charged_ce is None:
            self._is_charged_ce = False
            for sp in itertools.chain(*self.bits):
                if not isinstance(sp,(Vacancy,Element)) and \
                   sp.oxi_state!=0:
                    self._is_charged_ce = True
                    break
        return self._is_charged_ce

    @property
    def prim(self):
        """
        Modified primitive cell, should obey:
           1, Charge neutrality.
           2, Have all species in self.bits.
        Returns: 
            pymatgen.Structure
        """
        #Modify prim to a charge neutral composition that has all the species
        #included. This will make a lot of things easier.
        if self._prim is None or self._prim.charge!=0:
            typical_comp = self.compspace.\
                      get_random_point_in_unit_spc(form='composition')

            N_sites = sum(self.sl_sizes)
            prim_comps = [{} for i in range(N_sites)]
            for i in range(N_sites):
                for sl_id,sl in self.sublat_list:
                    if i in sl:
                        prim_comps[i] = typical_comp[sl_id]
                        break
                
            self._prim = Structure(self.lattice,prim_comps,self.frac_coords)

        return self._prim

    @property
    def radius(self):
        """
        Cluster radii of different cluster sizes.
        Returns:
           Dict{size(int):radius(float)}
        """
        if self._radius is None or len(self._radius)==0:
            d_nns = []
            for i,site1 in enumerate(self.prim):
                d_ij = []
                for j,site2 in enumerate(self.prim):
                    if j<i: continue;
                    if j>i:
                        d_ij.append(site1.distance(site2))
                    if j==i:
                        d_ij.append(min([self.lattice.a,\
                                    self.lattice.b,self.lattice.c]))
                d_nns.append(min(d_ij))
            d_nn = min(d_nns)
    
            self._radius= {}
            # Default cluster radius
            self._radius[2]=d_nn*4.0
            self._radius[3]=d_nn*2.0
            self._radius[4]=d_nn*2.0

        return self._radius

    @property
    def subspace(self):
        """
        Cluster subspace of this system.
        Returns:
            ClusterSubspace
        """
        if self._subspace is None:
            self._subspace = ClusterSubspace.from_cutoffs(self.prim,self.radius,\
                                                        basis = self.basis_type)
            for ex_name,args in zip(self.other_extern_types,self.other_extern_args):
                self._subspace.add_external_term(globals()[ex_name](**args))

        return self._subspace

    def get_ce_n_iters_ago(self,n_ago=1):
        """
        Get the cluster expansion object n iterations ago from history.
        Does not store past feature matrices!

        If none exist, will initialize a cluster expansion with only 
        external terms, and no cluster terms.

        Args:
            n_ago(int):
                Specifies which history ce step to read. Default is 1,
                will read the latest ce available.
        Returns:
            ClusterExpansion.
        """
        if len(self.history)<n_ago:
            warnings.warn("Cluster expansion history can not be dated back to {} \
                           iteration(s) ago. Making dummy cluster expasnion"\
                           .format(n_ago))

            coefs = np.zeros(self.subspace.num_corr_functions+\
                             len(self.subspace.external_terms))
            coefs[-len(self.subspace.external_terms)] = 1.0
        else:
            coefs = np.array(self.history[-n_ago]['coefs'])
        return ClusterExpansion(self.subspace,coefs,[])

    @property
    def last_ce(self):
        """
        Get the last cluster expansion in history. If none exist, will 
        initialize a cluster expansion with only external terms, and no
        cluster terms.
        Returns:
            ClusterExpansion
        """
        return self.get_ce_n_iters_ago(n_ago=1)

    @property
    def enumerator_options(self):
        """
        Get enumerator options.
        """
        return {'transmat':self._options.get('transmat',\
                             [[1,0,0],[0,1,0],[0,0,1]]),\
                'sc_size':self._options.get('sc_size',32),\
                'max_sc_cond':self._options.get('max_sc_cond',8),\
                'min_sc_angle':self._options.get('min_sc_angle',30),\
                'comp_restrictions':self._options.get('comp_restrictions'),\
                'comp_enumstep':self._options.get('comp_enumstep',1),\
                'basis_type':self._options.get('basis_type','indicator'),\
                'select_method':self._options.get('select_method','CUR')
               }

    #I added dummy **kwargs to calculation related objects to simply their
    #initialization. Although they have different pamameters, we can just
    #pass options as dict 

    @property
    def calc_writer_options(self):
        """
        Get calculation writer options.
        """
        return {'path':self._options.get('path','vasp_run'),\
                'lp_file':self._options.get('lp_file'),\
                'writer_strain':self._options.get('writer_strain',[1.05,1.03,1.01]),\
                'is_metal':self._options.get('is_metal',False),\
                'ab_setting':self._options.get('ab_setting',{}),\
                'writer_type':self._options.get('writer_type','ArchVaspWriter')
               }
        #TODO: need to change calc_writer to allow vasp options.

    @property
    def calc_manager_options(self):
        """
        Get calculation manager options.
        """
        return {'path':self._options.get('path','vasp_run'),\
                'lp_file':self._options.get('lp_file'),\
                'fw_file':self._options.get('fw_file'),\
                'qa_file':self._options.get('qa_file'),\
                'kill_command': self._options.get('kill_command'),\
                'ab_command':self._options.get('ab_command','vasp'),\
                'ncores':self._options.get('ncores',16),\
                'time_limit':self._options.get('time_limit',259200),\
                'check_interval':self._options.get('check_interval',300),\
                'manager_type':self._options.get('writer_type','ArchSGEManager')
               }

    @property
    def calc_reader_options(self):
        """
        Get calculation reader options.
        """
        return {'path':self._options.get('path','vasp_run'),\
                'md_file':self._options.get('md_file'),\
                'reader_type':self._options.get('reader_type','ArchVaspReader')
               }

    @property
    def featurizer_options(self):
        """
        Get featurizer options.
        """
        #Since we can not automatically generate labels_table, currently we don't have
        #automatic species_decorator detection. If your system needs decoration, you have
        #to provide the decorator types and arguments in your options file.
        decorators_types = self._options.get('decorators_types',[])

        for b in itertools.chain(*self.bits):
            if not isinstance(b,(Vacancy,Element)) and len(decorators_types)==0:
                raise ValueError('Cluster expasion requires decorated {}, \
                                  but no decoration is given!'.format(b))

        return {'other_props':self._options.get('other_props',[]),\
                'decorators_types':decorators_types,\
                'decorators_args':self._options.get('decorators_args',[])
               }

    @property
    def fitter_options(self):
        """
        Get fitter options.
        """
        return {'estimator_flavor':self._options.get('estimator_flavor','L2L0Estimator'),\
                'weights_flavor':self._options.get('weights_flavor','unweighted'),\
                'use_hierarchy':self._options.get('use_hierarchy',True),\
                'estimator_params':self._options.get('estimator_params',{}),\
                'weighter_params':self._options.get('weighter_params',{})
               }

    @property
    def gs_checker_options(self):
        """
        Get ground state checker options.
        """
        return {'e_tol_in_cv':self._options.get('e_tol_in_cv',3),\
                'comp_tol':self._options.get('comp_tol',0.05)
               }
    
    def gs_generator_options(self):
        """
        Get ground state generator options.
        """
        return {'handler_flavor':self._options.get('handler_flavor','CanonicalHandler'),\
                'handler_args':self._options.get('handler_args',{})
               }

    #Not frequently used
    @property
    def calc_writer(self):
        name = self.calc_writer_options['writer_type']
        kwargs = self.calc_writer_options.copy()
        kwargs.pop('writer_type')
        return globals()[name](**kwargs)

    #Not frequently used
    @property
    def calc_manager(self):
        name = self.calc_manager_options['manager_type']
        kwargs = self.calc_manager_options.copy()
        kwargs.pop('manager_type')
        return globals()[name](**kwargs)

    #Used in featurizer
    @property
    def calc_reader(self):
        name = self.calc_reader_options['reader_type']
        kwargs = self.calc_reader_options.copy()
        kwargs.pop('reader_type')
        return globals()[name](**kwargs)

    @property
    def options(self):
        """
        Returns completed options dictionary for the user's reference.
        """
        all_options = [self.enumerator_options,\
                       self.calc_writer_options,\
                       self.calc_reader_options,\
                       self.calc_manager_options,\
                       self.featurizer_options,\
                       self.fitter_options,\
                       self.gs_checker_options,\
                       self.gs_generator_options\
                      ]

        merged_d = {}

        #If conflicting keys appear, will only take the first value.
        #It is your responsibility to avoid conflicting keys!
        for sub_d in all_options:
            for key in sub_d:
                if key not in merged_d:
                    merged_d[key]=sub_d[key]

        self._options = merged_d
        return self._options

    #TODO
    #auto_load

    #auto_save
