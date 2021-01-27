__author__="Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.
Ground state structures will also be added to the structure pool, but 
they are not added here. They will be added in the convergence checker
module.
"""
import warnings
import random
from copy import deepcopy
import numpy as np
import pandas as pd
import os
import json

from monty.json import MSONable

from pymatgen import Structure,Element
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe.extern.ewald import EwaldTerm
from smol.cofe.space.clusterspace import ClusterSubspace
from smol.cofe.space.domain import get_allowed_species,get_species, Vacancy
from smol.cofe.expansion import ClusterExpansion
from smol.moca import CanonicalEnsemble,Sampler

from .utils.math_utils import enumerate_matrices,select_rows,combinatorial_number
from .utils.format_utils import flatten_2d,deflat_2d,serialize_comp,deser_comp,\
                                structure_from_occu
from .utils.calc_utils import get_ewald_from_occu
from .utils.comp_utils import check_comp_restriction

from .comp_space import CompSpace
from .data_manager import DataManager 
from .inputs_wrapper import InputsWrapper

class StructureEnumerator(MSONable):
    """
    Dircet initialization no longer recommended!
    Attributes:
        prim(Structure):
            primitive cell of the structure to do cluster expansion on.

        bits(List[List[Specie]]):
            Occupying species on each sublattice. Vacancy() should be included.

        sublat_list(List of lists):
            Stores primitive cell indices of sites in the same sublattices
            If none, sublattices will be automatically generated.

        is_charged(Boolean):
            If true, will do charged cluster expansion.

        previous_ce(ClusterExpansion):
            A cluster expansion containing information of cluster expansion
            in previously enumerated structures. Used when doing mc sampling.

        transmat(3*3 arraylike):
            A transformation matrix to apply to the primitive cell before
            enumerating supercell shapes. This can help to increase the 
            symmetry of the enumerated supercell. For example, for a rocksalt
            primitive cell, you can use [[1,-1,-1],[-1,1,-1],[-1,-1,1]] as a
            transmat to modify the primitive cell as cubic.

        sc_size(Int): 
            Supercell matrix deternimant of each enumerated structure.
            By default, set to 32, to restrict DFT computation cost.
            Currently, better not to make supercell have over 200 atoms!
            We recommend up to 64 atoms in a supercell.

        max_sc_cond(float):
            Maximum allowed lattice matrix conditional number of the enumerated 
            supercells.

        min_sc_angle(float):
            Minumum allowed lattice angle of the enumerated supercells.
 
        max_sc_cond and min_sc_angle controls the skewness of a supercell, so you
        can avoid potential structural instability during DFT structural relaxation.

        comp_restrictions(Dict or List of Dict or None):
            Restriction on certain species.
            If this is a dictionary, this dictionary provide constraint of the 
            atomic fraction of specified species in the whole structure;
            If this is a List of dictionary, then each dictionary provides
            atomic fraction constraints on each sublattice.

            For each dictionary, the keys should be Specie/Vacancy object
            or String repr of a specie (anything readable by get_specie() in
            smol.cofe.configspace.domain). And the values shall be tuples 
            consisting of 2 float numbers, in the form of (lb,ub). 
            lb constrains the lower bound of atomic fraction, while ub constrains
            the upperbound. lb <= x <= ub.

            You may need to specify this in phases with high concentration of 
            vancancy, so you structure does not collapse.
            
            By default, is None (no constraint is applied.)
   
        comp_enumstep(int):
            Enumeration step for compositions. If otherwise specified, will thin
            enumerated compositions by this value. For example, if we have BCC 
            Ag-Li alloy, and a supercell of totally 256 sites. If step = 1, we 
            can have 257 compositions. But if we don't want to enumerate so many,
            we can simply write step = 4, so 4 sites are replaced each time, we 
            get totally 65 compositions.

        basis_type(string):
            Type of basis used in cluster expansion. Needs to be specified if you 
            initalize enumeration from an existing CE, and its basis is different 
            from 'indicator'!
            If you used custom basis, just type 'custom' for this term. But hopefully
            this will not happen too often.
    
        select_method(str): 
            Method used in structure selection from enumerated pool.
            'CUR'(default):
                Select by highest CUR scores
            'random':
                Select randomly
            Both methods guarantee inclusion of the ground states at initialization.

        data_manager(DataManager):
            The datamanager to use for this instance.
        All generated structure entree will be saved in a star-schema:
        The dimension tables will contain serialized supercell matrices and compositions,
        and their id's as primary keys.(sc_id, comp_id). These dimension tables will mostly be fixed
        as they were initalized, unless new ground state compostions has been detected.

        The fact table will contain entrees of enumerated occupations(encoded), structures, their original
        feature vectors, computation convergence(Boolean), properties used for charge assignment if ever, 
        mapped occupation(encoded,None if can't map), mapped feature vector, sc_comp_id as foreign key, 
        and entry_id as primary key. The primary keys will be sorted by their order of adding into this
        table. 
 
        Both the dimension tables and the fact table can be changed by all modules in CEAuto. They are 
        main repositories of CEdata.

        Computation results will be stored under 'vasp_run/{}'.format(entry_id). CEAuto will parse these 
        files for charge assignment and featurization only.
    """
    def __init__(self,prim,
                 bits=None,\
                 sublat_list=None,\
                 is_charged=False,\
                 previous_ce = None,\
                 transmat=[[1,0,0],[0,1,0],[0,0,1]],\
                 sc_size=32,\
                 max_sc_cond = 8,\
                 min_sc_angle = 30,\
                 comp_restrictions=None,\
                 comp_enumstep=1,\
                 basis_type = 'indicator',\
                 select_method = 'CUR',
                 data_manager=DataManager.auto_load()):

        self.prim = prim

        #Calculation of bits, sublat_list, is_charges_ce
        #previous_ce are no longer computed here.
        #I have moved them to InputsWrapper.
        self.bits = bits
        self.sublat_list = sublat_list
        self.sl_sizes = [len(sl) for sl in sublat_list]

        #Check if this cluster expansion should be charged ce
        self.is_charged_ce = is_charged

        self.basis_type = basis_type
        self.ce = previous_ce
            
        self.transmat = transmat
        self.sc_size = sc_size
        self.max_sc_cond = max_sc_cond
        self.min_sc_angle = min_sc_angle

        self._compspace = compspace
        self.comp_restrictions = comp_restrictions
        self.comp_enumstep = comp_enumstep

        if self.sc_size%self.comp_enumstep!=0:
            raise ValueError("Composition enumeration step can't divide supercell size.")

        self.select_method = select_method

        #data manager
        self._dm = data_manager
    
    @property
    def n_strs(self):
        """
        Number of enumerated structures.
        """
        return len(self.fact_df)

    @property
    def n_iter(self):
        """
        Current iteration number. This index starts from 0.
        """
        return self.dm.cur_iter_id

    @property
    def compspace(self):
        """
        Constained compositional space of this system.
        """
        if self._compspace is None:
            self._compspace = CompSpace(self.bits,self.sl_sizes)
        return self._compspace

    @property
    def sc_df(self):
        """
        Supercell matrices used for structural enumeration. If none yet, shall be 
        enumerated.
        Return:
            Supercell matrix dimension table. Is a pd.Dataframe.
        """
        return self._dm.sc_df

    @property
    def comp_df(self):
        """
        Enumerate proper compositions under each supercell matrix, and gives a dimension table with
        sc_id, comp_id, and the exact composition.
        Return:
            Dimension table containing all enumerated compositions, and their correponding supercell
            matrices. Is a pandas.DataFrame.
        """
        return self._dm.comp_df

    @proeprty
    def fact_df(self):
        """
        Fact dataframe, storing all generated structures and their caluclated properties.
        """
        return self._dm.fact_df

    def set_sc_matrices(self,matrices=[],add_new=False):
        """
        Interface method to preset supercell matrices before enumeration. If no preset
        is given, supercell matrices will be automatically enumerated. Otherwise will
        skip.

        Not frequently called.
        Args:
            matrices(List[3*3 ArrayLike]): 
                Supercell matrices to insert
            add_new(Boolean):
                If true, will add matices to the supercell dataframe, even if the
                current dataframe is not empty.
        """
        if len(self.sc_df)>0 and not add_new:
            warnings.warn("Attempt to set matrices after matrix enumeration. Skipping")
            return

        for m in matrices:
            if isinstance(m,np.ndarray):
                m = m.tolist()
            self._dm.insert_one_supercell(m)

    def enumerate_sc_matrices(self,add_new=False):
        """
        Enumerate supercell matrices if nothing is present.
        Args:
            add_new(Boolean):
                If true, will add matices to the supercell dataframe, even if the
                current dataframe is not empty.
        """
        print("**Start supercell enumeration.")
        mats = enumerate_matrices(self.sc_size,self.prim.lattice,\
                                  transmat=self.transmat,\
                                  max_sc_cond = self.max_sc_cond,\
                                  min_sc_angle = self.min_sc_angle)
        print("**Enumerated Supercell matrices: \n")
        for m in mat:
            print("  {}".format(m))
        if len(self.sc_df)==0:
            self.set_sc_matrices(matrices=mats,add_new = add_new)

    def enumerate_comps(self,add_new=False):
        """
        Enumerate Compositions under supercells.
        Args:
            add_new(Boolean):
                If true, will add composition to the comp dataframe, even if the
                current dataframe is not empty.
        """
        if len(self.comp_df)>0 and not add_new:
            warnings.warn("Attempt to set composition after compositions enumeration. Skipping")
            return

        if len(self.sc_df) == 0:
            self.enumerate_sc_matrices()

        print("**Start compositions enumeration")
        for sc_id,m in zip(self.sc_df.sc_id,self.sc_df.matrix):
            scs = int(round(abs(np.linalg.det(m))))
            ucoords = self.compspace.frac_grids(sc_size=scs//self.comp_enumstep)
            comps = self.compspace.frac_grids(sc_size=scs//self.comp_enumstep)
            print("****Enumerated {} compositions under matrix {}."\
                  .format(len(ucoords),m))
            for ucoord,comp in zip(ucoords,comps):
                if check_comp_restriction(comp,self.sl_sizes,self.comp_restrictions):
                    _,comp_id = self._dm.insert_one_comp(ucoord,sc_id=sc_id)

    #TODO: add simulator support, and parallelize.
    def generate_structures(self, n_per_key = 3, keep_gs = True,weight_by_comp=True):
        """
        Enumerate structures under all different key = (sc_matrix, composition). The eneumerated structures
        will be deduplicated and selected based on CUR decomposition score to maximize sampling efficiency.
        Will run on initalization mode if no previous enumearation present, and on addition mode if previous
        enum present.
        Inputs:
            old_femat(2D arraylike):
                Feature matrices of previously generated structures. If not specified, will do initialization,
                rather than structure addition.
            n_per_key(int):
                will select n_per_key*N_keys number of new structures. If pool not large enough, select all of
                pool. Default is 3.
            keep_gs(Bool):
                If true, will always select current ground states. Default is True.
            Weight_by_comp(Boolean):
                If true, will preprocess sample pool before structure selection, to make sure that the number of 
                structures selected with each composition is proportional to the total number of structures with
                that composition in the whole configurational space. 
                Default is True.
    
        No outputs. Updates in self._fact_df
        """
        #TODO: use schecker to check iteration step

        N_keys = len(self.comp_df)

        #Check for dumplicated strcture enumerator call
        filt_ = (self._fact_df.iter_id==self.n_iter) & (self._fact_df.module=='enum')
        if self.n_iter>=0 and np.all(self._fact_df[filt_].calc_status=='NC'):
            warnings.warn("Structure enumerator called twice in one iteration, skipping.")
            return

        #Build the table
        if self._fact_df is None or len(self._fact_df)==0:
            self._fact_df = pd.DataFrame(columns=['entry_id','sc_id','comp_id','iter_id',\
                                                  'module',\
                                                  'ori_occu','ori_corr',\
                                                  'calc_status',\
                                                  'map_occu','map_corr',\
                                                  'e_prim',\
                                                  'other_props'])           

        # Results of this enumerator run
        eq_occus_update = []
        enum_strs = []
        enum_occus = []
        enum_corrs = []
        comp_weights = []
        keep_first = []


        old_femat = self._fact_df.ori_corr.tolist()

        for sc_id,comp_id,comp,eq_occu in zip(self.comp_df.sc_id, self.comp_df.comp_id,\
                                              self.comp_df.comp, self.comp_df.eq_occu):
            
            #Query dimension tables
            sc_mat = self.sc_df[self.sc_df.sc_id == sc_id].iloc[0]

            #Query previous fact table
            old_strs = self._fact_df[self._fact_df.comp_id == comp_id]\
                       .ori_occu.map(lambda o: structure_from_occu(self.prim,sc_mat,o))

            str_pool,occu_pool,comp_weight = self._enum_configs_under_sccomp(sc_mat,comp,eq_occu)
            corr_pool = [list(self.ce.cluster_subspace.corr_from_structure(s,sc_matrix=sc_mat)) 
                         for s in str_pool]

            #Update GS
            gs_occu = deepcopy(occu_pool[0])
            eq_occus_update.append(gs_occu)

            dedup_ids = []


            for s1_id, s1 in enumerate(str_pool):
                dupe = False
                for s2_id, s2 in enumerate(old_strs):
                    if sm.fit(s1,s2):
                        dupe = True
                        break
                if not dupe:
                    dedup_ids.append(s1_id)
 
            #gs(id_0 in occu_pool) will always be selected if unique
            enum_strs.append([str_pool[d_id] for d_id in dedup_ids])
            enum_occus.append([occu_pool[d_id] for d_id in dedup_ids])
            enum_corrs.append([corr_pool[d_id] for d_id in dedup_ids])
            comp_weights.append(comp_weight)
            n_enum += len(enum_strs[-1])
            
            if 0 in dedup_ids: #New GS detected, should be keeped
                keep_first.append(True)
            else:
                keep_first.append(False)

        #Preprocess to weight over compositions, to determine how many structures to select under 
        #each composition.
        W_max = max(1,max(comp_weights))
        N_max = max(1,max([len(key_str) for key_str in enum_strs]))
        r = N_max/W_max
        if weight_by_comp:
            n_selects = [min(max(1,int(round(w*r))),len(k)) for w,k in zip(comp_weights,enum_strs)]
        else:
            n_selects = [len(k) for w,k in zip(comp_weights,enum_strs)]

        #Marking canonical ground states as 'must-keep'.
        keep_fids = [] #Indices to keep in the flatten structure pool
        n_enum = 0
        sec_ids = [] #Indices to select in the unflatten structure pool
        for n_sel,kf,key_str in zip(n_selects,keep_first,enum_strs):
            if not kf:
                sec_ids.append(random.sample(list(range(len(key_str))),n_sel))
            else:
                if len(key_str)>0:
                    sec_ids.append([0]+random.sample(list(range(1,len(key_str))),n_sel-1))
                    keep_fids.append(n_enum)
                else:
                    raise ValueError("GS scan suggested to keep first, but structure pool under key is empty!")
            n_enum += n_sel

        enum_strs = [[enum_strs[k_id][s_id] for s_id in key_s_ids] for k_id,key_s_id in enumerate(sec_ids)]
        enum_occus = [[enum_occus[k_id][s_id] for s_id in key_s_ids] for k_id,key_s_id in enumerate(sec_ids)]
        enum_corrs = [[enum_corrs[k_id][s_id] for s_id in key_s_ids] for k_id,key_s_id in enumerate(sec_ids)]
      
        #If don't keep GS mode selected, or is initalizing pool and GS can't be obtained.
        if not keep_gs or self.n_strs==0:
            keep_fids = [] #Flatten ids of ground states that must be keeped

        print('*Enumerated {} unique structures. Selecting.'.format(n_enum))

        #Flatten data for processing. deflat_rules will be the same.
        str_pool_flat, deflat_rule = flatten_2d(enum_strs)
        occu_pool_flat, deflat_rule = flatten_2d(enum_occus)
        corr_pool_flat, deflat_rule = flatten_2d(enum_corrs)

        #Selecting structures that contributes to most feature variance.
        selected_fids = select_rows(corr_pool_flat,n_select=n_per_key*N_keys,
                                     old_femat = old_femat,
                                     method=self.select_method,
                                     keep=keep_fids) 
            
        #Muting unselected structure, prepare for deflattening
        str_pool_flat = [i for i_id,i in enumerate(str_pool_flat) if i_id in selected_fids else None]
        occu_pool_flat = [i for i_id,i in enumerate(occu_pool_flat) if i_id in selected_fids else None]
        corr_pool_flat = [i for i_id,i in enumerate(corr_pool_flat) if i_id in selected_fids else None]

        #Deflatten
        enum_strs = deflat_2d(str_pool_flat,deflat_rule)
        enum_occus = deflat_2d(occu_pool_flat,deflat_rule)
        enum_corrs = deflat_2d(corr_pool_flat,deflat_rule)

        #Adding new structures into the fact table. All fact entry ids starts from 0
        cur_id = deepcopy(self.n_strs)
        n_strs_init = deepcopy(self.n_strs)
        cur_iter_id = self.n_iter+1

        for sc_id,comp_id,key_occus,key_corrs in zip(self.comp_df.sc_id, self.comp_df.comp_id,\
                                                     enum_occus,enum_corrs):
            for occu,corr in zip(key_occus,key_corrs):
                self._fact_df = self._fact_df.append({'entry_id':cur_id,
                                      'sc_id':sc_id,
                                      'comp_id':comp_id,
                                      'iter_id':cur_iter_id,
                                      'module':'enum',
                                      'ori_occu':occu,
                                      'ori_corr':corr,
                                      'calc_status':'NC',
                                      'map_occu':None,
                                      'map_corr':None,
                                      'e_prim':None,
                                      'other_props':None
                                     }, ignore_index = True)
                cur_id += 1

        #Updating eq_occus in comp_df
        self._comp_df.eq_occus = eq_occus_update

        print("*Added with {} new unique structures.".format(self.n_strs-n_strs_init))
            
    def clear_structures(self):
        """
        Clear enumerated structures.
        """
        warnings.warn("All previous enumerations cleared. Do this at your own risk!")
        self._dm.reset()

    def _enum_configs_under_sccomp(self,sc_mat,comp,eq_occu=None):
        """
        Built in method to generate occupations under a supercell matrix and a fixed composition.
        Assuming that atoms in the supercells are generated by pymatgen.structure.make_supercell
        from the primitive cell, which simply replicates and stacks atom in their initial order.
        For example: [Ag, Cu] -> [Ag, Ag, Cu, Cu]

        Inputs:
            sc_mat(3*3 ArrayLike):
                Supercell matrix
            comp(Union([List[pymatgen.Composition],List[SiteSpace], List[dict] ])):
                Compositions on each sublattice. Fractional.           
            eq_occu(List of ints):
                Occupation array of ground state under that composition. If None, will anneal to
                calculate
        Return:
            rand_strs_dedup:
                List of deduped pymatgen.Structures
            rand_occus_dedup:
                List of deduped occupation arrays. All in list of ints.
            comp_weight:
                Total number of all possible structures with the current composition.
                Integer.
        """

        print("\nEnumerating under supercell: {}, composition: {}.".format(sc_mat,comp))
 
        is_indicator = (self.basis_type == 'indicator')
        scs = int(round(abs(np.linalg.det(sc_mat))))

        #Anneal n_atoms*100 per temp, Sample n_atoms*500, give 100 samples for practical ccomputation
        n_steps_anneal = scs*len(self.prim)*100
        n_steps_sample = scs*len(self.prim)*500
        thin = max(1,n_steps_sample//100)

        anneal_series = [2000,1340,1020,700,440,280,200,120,80,20]
        sample_series = [500,1500,10000]

        ensemble = CanonicalEnsemble.from_cluster_expansion(self.ce, sc_mat, 
                                                            optimize_inidicator=is_indicator)
        sampler = Sampler.from_ensemble(ensemble,temperature=1000)
        processor = ensemble.processor
        sm = StructureMatcher()
 
        print("**Initializing occupation.")
        init_occu, comp_weight = self._initialize_occu_under_sccomp(sc,comp)
 
        if eq_occu is None:
        #If not annealed before, will anneal and save GS
            print("****Annealing to the ground state.")
            sampler.anneal(anneal_series,n_steps_anneal,
                           initial_occupancies=np.array([init_occu]))
 
            print('*****Equilibrium GS found!')
            gs_occu = list(sampler.samples.get_minimum_energy_occupancy())
        else:
        #If annealed before, will use old GS
            gs_occu = eq_occu
 
        #Will always contain GS structure at the first position in list
        rand_occus = [gs_occu]
        #Sampling temperatures
        
        for T in sample_series:
            print('**Getting samples under {} K.'.format(T))
            sampler.samples.clear()
            sampler._kernel.temperature = T
            #Equilibriate
            print("****Equilibration run.")
            sampler.run(n_steps_sample,
                        initial_occupancies=np.array([gs_occu]),
                        thin_by=thin,
                        progress=True)
            sa_occu = sampler.samples.get_occupancies()[-1]
            sampler.samples.clear()
            #Sampling
            print("****Generative run.")
            sampler.run(n_steps_sample,
                        initial_occupancies=np.array([sa_occu]),
                        thin_by=thin,
                        progress=True)
            #default flat=True will remove n_walkers dimension. See moca docs.
            rand_occus.extend(np.array(sampler.samples.get_occupancies()).tolist())          

        rand_strs = [processor.structure_from_occupancy(occu) for occu in rand_occus]
        #Internal deduplication
        rand_dedup = []
        for s1_id,s1 in enumerate(rand_strs):
            duped = False
            for s2_id,s2 in enumerate(rand_dedup):
                if sm.fit(s1,s2):
                    duped = True
                    break
            if not duped:
                rand_dedup.append(s1_id)

        print('{} unique structures generated.'.format(len(rand_dedup)))
        rand_strs_dedup = [rand_strs[s_id] for s_id in rand_dedup]
        rand_occus_dedup = [rand_occus[s_id] for s_id in rand_dedup]

        return rand_strs_dedup, rand_occus_dedup, comp_weight

    def _initialize_occu_under_sccomp(self,sc_mat,comp):
        """
        Get an initial occupation under certain supercell matrix and composition.
        Composition must be pre_nomalized into fractional form.
        If n_atoms is not 1, then the rest will be filled with Vacancy().
        Inputs:
            sc_mat(3*3 ArrayLike):
                Supercell matrix
            comp(Union([List[pymatgen.Composition],List[SiteSpace], List[dict] ])):
                Compositions on each sublattice. Fractional.
        Output:
            init_occu:
                Arraylike of integers. Encoded occupation array.
            comp_weight:
                Total number of all possible structures with the current composition.
                Integer.
        """
        scs = int(round(abs(np.linalg.det(sc_mat))))
        
        sc_sublat_list = []
        #Generating sublattice list for a supercell, to be used in later radomization code.
        for sl in self.sublat_list:
            sl_sites = []
            for s in sl:
                sl_sites.extend(list(range(s*scs,(s+1)*scs)))
            sc_sublat_list.append(sl_sites)

        #randomly initalize 50 occus, pick 10 with lowest ewald energy (if is_charged_ce),
        #Then choose one final as initalization randomly
        int_comp = []
        for sl_frac_comp, sl_sites in zip(comp,sc_sublat_list):

            if sum(sl_frac_comp.values())<1 and sum(sl_frac_comp.values())>=0:
                x_vac_add = 1-sum(sl_frac_comp.values())
            elif sum(sl_frac_comp.values())==1:
                pass
            else:
                raise ValueError('Composition {} not a proper normalized composition!'.format(sl_frac_comp))

            sl_int_comp = {}
            vac_key = None
            for k,v in sl_frac_comp.items():
            #Tolerance of irrational compositions
                if abs(v*len(sl_sites)-round(v*len(sl_sites)))>0.1:
                    raise ValueError("Sublattice compostion {} can not be achieved with sublattice size {}."\
                                     .format(sl_frac_comp,len(sl_sites)))
                sl_int_comp[k] = int(round(v*len(sl_sites)))
                if isinstance(k,Vacancy):
                    vac_key = k
            #Fraction <1 filled with vacancies
            if vac_key is not None:
                sl_int_comp[vac_key]+= int(round(x_vac_add*len(sl_sites)))
            else:
                sl_int_comp[Vacancy()] = int(round(x_vac_add*len(sl_sites)))
            
            int_comp.append(sl_int_comp)

        rand_occus = []
        for i in range(50):
            #Occupancy is coded
            occu = [None for i in range(len(self.prim)*scs)]
            for sl_id, (sl_int_comp, sl_sites) in enumerate(zip(int_comp,sc_sublat_list)):
                sl_sites_shuffled = random.shuffle(deepcopy(sl_sites))

                n_assigned = 0
                for sp,n_sp in sl_int_comp.items():
                    for s_id in sl_sites_shuffled[n_assigned:n_assigned+n_sp]:
                        sp_name = get_species(sp)
                        sp_id = self.bits[sl_id].index(sp_name)
                        occu[s_id] = sp_id
                    n_assigned += n_sp

            for sp in occu:
                if sp is None:
                    raise ValueError("Unassigned site in occupation: {}, composition is: {}!".format(occu,comp))    

            rand_occus.append(occu)

        if self.is_charged_ce:
            rand_occus = sorted(rand_occus,key=lambda occu:\
                                get_ewald_from_occu(occu,sc_sublat_list,self.bits,self.prim,sc_mat))

        comp_weight = 1
        for sl_int_comp,sl_sites in zip(int_comp,sc_sublat_list):
            N_sl = len(sc_sublat_list)
            for n_sp in sl_int_comp.values():
                comp_weight = comp_weight*combinatorial_number(N_sl,n_sp)
                N_sl = N_sl - n_sp

        return random.choice(rand_occus[:10]), comp_weight

    def auto_save(self,sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Automatically save dataframes into specified files.
        Args:
            sc_file(str):
                supercell matrix file path
            comp_file(str):
                composition file path
            fact_file(str):
                fact table file path
        All optional, but I don't recommend you to change the paths.
        """
        self._dm.auto_save(sc_file=sc_file,comp_file=comp_file,fact_file=fact_file)

    @classmethod
    def auto_load(cls,options_file='options.yaml',\
                  sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv',\
                  ce_history_file='ce_history.json'):
        """
        This method is the recommended way to initialize this object.
        It automatically reads all setting files with FIXED NAMES.
        YOU ARE NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!
        Args:
            options_file(str):
                path to options file. Options must be stored as yaml
                format. Default: 'options.yaml'
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'             
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'             
            ce_history_file(str):
                path to cluster expansion history file.
                Default: 'ce_history.json'

        Returns:
             StructureEnumerator object.
        """
        options = InputsWrapper.auto_load(options_file=options_file,\
                                          ce_history_file=ce_history_file)

        dm = DataManager.auto_load(options_file=options_file,\
                                   sc_file=sc_file,\
                                   comp_file=comp_file,\
                                   fact_file=fact_file,\
                                   ce_history_file=ce_history_file)

        return cls(options.prim,
                   bits=options.bits,\
                   sublat_list=options.sublat_list,\
                   is_charged=options.is_charged_ce,\
                   previous_ce=options.last_ce,\
                   data_manager=dm,\
                   **options.enumerator_options
                  )
