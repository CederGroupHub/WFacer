"""
Canonical monte-carlo handler.
"""
__author__="Fengyu Xie"

import numpy as np
from copy import deepcopy
from abc import ABC,abstractmethod
import random

from smol.cofe.space.domain import get_allowed_species
from smol.moca import CanonicalEnsemble, CNSemiGrandDiscEnsemble,\
                      CNSemiGrandEnsemble,Sampler

from .base import BaseHandler
from ..comp_space import CompSpace
from ..utils.comp_utils import scale_compstat
from ..utils.occu_utils import get_sc_sllist_from_prim
from ..utils.calc_utils import get_ewald_from_occu


class MCHandler(BaseHandler,ABC):
    """
    Base monte-carlo handler class. Provides ground states, de-freeze
    sampling.
    Note: In the future, will support auto-equilibration.
    """
    def __init__(self,ce,sc_mat,\
                      gs_occu=None,\
                      anneal_series=[3200,1600,800,400,200,100,50],\
                      unfreeze_series=[500,1500,5000],\
                      n_runs_sa = 100,\
                      n_runs_unfreeze = 500,\
                      **kwargs):
        """
        Args:
            ce(ClusterExpansion):
                A cluster expansion object to solve on.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
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
            gs_occu(List[int]):
                Encoded occupation array of previous ground states.
                 Optional, but if you have it, you can save the 
                 ground state solution time when sampling.
        """
        super().__init__(ce,sc_mat,**kwargs)

        self.anneal_series = anneal_series
        self.unfreeze_series = unfreeze_series
        self.n_runs_sa = n_runs_sa
        self.n_runs_unfreeze = n_runs_unfreeze

        self.sc_size = int(round(abs(np.linalg.det(sc_mat))))

        self.is_indicator = (self.ce.cluster_subspace.orbits[0].basis_type == 'indicator')

        self.prim = self.ce.cluster_subspace.structure
        prim_bits = get_allowed_species(self.prim)

        #Automatic sublattices, same rule as smol.moca.Sublattice:
        #sites with the same compositioon are considered same sublattice.
        self.sublat_list = []
        self.bits = []
        for s_id,s_bits in enumerate(prim_bits):
            if s_bits in self.bits:
                sl_id = self.bits.index(s_bits)
                self.sublat_list[sl_id].append(s_id)
            else:
                self.sublat_list.append([s_id])
                self.bits.append(s_bits)                  

        self.sc_sublat_list = get_sc_sllist_from_prim(self.sublat_list,sc_size=self.sc_size)
 
        self._skip_anneal = (gs_occu is not None)
        self._gs_occu = gs_occu
 
        self._ensemble = None
        self._sampler = None
        self._processor = None

     def _get_min_occu_enthalpy(self):
        """
        Get minimum thermo function from the current ensemble's sampler.
        Different ensemble types have different thermo potentials. For 
        example, E for canonical ensemble, E-mu*x for semi-grand canonical 
        ensemble.
        In smol.moca, this quantity is called 'enthalpy'.
        """
        gs_occu = self._sampler.samples.get_minimum_enthalpy_occupancy()
        gs_enth = self._sampler.samples.get_minimum_enthalpy()
        return gs_occu, gs_enth

    def _initialize_occu_from_int_comp(self,int_comp):
        """
        Get an initial occupation under certain supercell matrix and composition.

        Args:
            int_comp(List[List[int]]):
                integer composition, in compstat form.

        Output:
            init_occu:
                Arraylike of integers. Encoded occupation array.
        """
        rand_occus = []

        for i in range(50):
            #Occupancy is coded
            occu = [None for i in range(len(self.prim)*self.sc_size)]
            for sl_id, (sl_int_comp, sl_sites) in enumerate(zip(int_comp,self.sc_sublat_list)):
                if sum(sl_int_comp)!=len(sl_sites):
                    raise ValueError("Site number mismatches composition on sublattice {}.".format(sl_id))

                sl_sites_shuffled = random.shuffle(deepcopy(sl_sites))

                for sp_id,n_sp in enumerate(sl_int_comp): #compstat already ensures correct species ordering.
                    for s_id in sl_sites_shuffled[n_assigned:n_assigned+n_sp]:
                        occu[s_id] = sp_id
                    n_assigned += n_sp

            rand_occus.append(occu)

        rand_occus = sorted(rand_occus,key=lambda occu:get_ewald_from_occu(occu,self.prim,self.sc_mat))

        return random.choice(rand_occus[:5])     

    def solve(self):
        """
        Use simulated annealing to solve a ground state under current condition.
        
        Returns:
            gs_occu, gs_e
        """
        n_steps_anneal = self.sc_size*len(self.prim)*self.n_runs_sa

        print("****Annealing to the ground state. T series: {}.".format(anneal_series))
        self._sampler.anneal(self.anneal_series,n_steps_anneal,\
                             initial_occupancies=np.array([self._gs_occu]))
        print("****GS annealing finished!")
        gs_occu, gs_e = self._get_min_occu_enthalpy()
  
        #Updates
        self._gs_occu = gs_occu
        return gs_occu, gs_e


    def get_unfreeze_sample(self):
        """
        Built in method to generate occupations under a supercell matrix and a fixed composition.
 
        Return:
            sample_occus(List[List[int]]):
                A list of sampled encoded occupation arrays. The first
                one will always be the one with lowest energy!
        
        """

        #Anneal n_atoms*100 per temp, Sample n_atoms*500, give 100 samples for practical ccomputation
        n_steps_sample = self.sc_sizes*len(self.prim)*self.n_runs_unfreeze
        thin = max(1,n_steps_sample//200)

        sm = StructureMatcher()
 
        #In structure enumerator, will always not skip anneal.
        if not self._skip_anneal:
            sa_occu, sa_e = self.solve()
        else:
            sa_occu = self._gs_occu
 
        #Will always contain GS structure at the first position in list
        rand_occus = [deepcopy(sa_occu)]

        #Sampling temperatures        
        for T in self.unfreeze_series:
            print('****Getting samples under {} K.'.format(T))
            self._sampler.samples.clear()
            self._sampler._kernel.temperature = T
            #Equilibriate
            print("******Equilibration run.")
            self._sampler.run(n_steps_sample,
                              initial_occupancies=np.array([sa_occu]),
                              thin_by=thin,
                              progress=True)
            sa_occu = deepcopy(self._sampler.samples.get_occupancies()[-1])
            self._sampler.samples.clear()
            #Sampling
            print("******Generative run.")
            self._sampler.run(n_steps_sample,
                              initial_occupancies=np.array([sa_occu]),
                              thin_by=thin,
                              progress=True)
            #default flat=True will remove n_walkers dimension. See moca docs.
            rand_occus.extend(np.array(self._sampler.samples.get_occupancies()).tolist())          

        rand_strs = [self._processor.structure_from_occupancy(occu) for occu in rand_occus]
        #Internal deduplication
        rand_dedup = []
        for s1_id,s1 in enumerate(rand_strs):
            duped = False
            for s2_id in rand_dedup:
                if sm.fit(s1,rand_strs[s2_id]):
                    duped = True
                    break
            if not duped:
                rand_dedup.append(s1_id)

        print('****{} unique structures generated.'.format(len(rand_dedup)))
        rand_strs_dedup = [rand_strs[s_id] for s_id in rand_dedup]
        rand_occus_dedup = [rand_occus[s_id] for s_id in rand_dedup]

        return rand_occus_dedup


class CanonicalMCHandler(MCHandler):
    """
    MC handler that solves and samples the canonical ensemble.
    """
    def __init__(self,ce,sc_mat,compstat,\
                      gs_occu=None,\
                      anneal_series=[3200,1600,800,400,200,100,50],\
                      unfreeze_series=[500,1500,5000],\
                      n_runs_sa = 100,\
                      n_runs_unfreeze = 500,\
                      **kwargs):
        """
        Args:
            ce(ClusterExpansion):
                A cluster expansion object to solve on.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
            compstat(2D List):
                Compositional statistics table, normalized to primitive cell.
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
            gs_occu(List[int]):
                Encoded occupation array of previous ground states.
                 Optional, but if you have it, you can save the 
                 ground state solution time when sampling.

        """
        super().__init__(ce,sc_mat,gs_occu=gs_occu,\
                                   anneal_series = anneal_series,\
                                   unfreeze_series = unfreeze_series,\
                                   n_runs_sa = n_runs_sa,\
                                   n_runs_unfreeze = n_runs_unfreeze,\
                                   **kwargs)

        self.compstat = compstat
        self.int_comp = scale_compstat(compstat,scale_by=self.sc_size)

        self._gs_occu = gs_occu or self._initialize_occu_from_int_comp(self.int_comp)
        self._ensemble = CanonicalEnsemble.from_cluster_expansion(self.ce, self.sc_mat, 
                                                            optimize_inidicator=self.is_indicator)
        self._sampler = Sampler.from_ensemble(ensemble,temperature=1000)
        self._processor = self._ensemble.processor


class SemigrandDiscMCHandler(MCHandler):
    """
    Sublattice DISCRIMINATIVE Charge neutral semigrand canonical ensemble!
    Used for grand canonical ground states only.
    """
    def __init__(self,ce,sc_mat,mu,\
                      gs_occu=None,\
                      anneal_series=[3200,1600,800,400,200,100,50],\
                      unfreeze_series=[500,1500,5000],\
                      n_runs_sa = 100,\
                      n_runs_unfreeze = 500,\
                      **kwargs):

        """
        Args:
            ce(ClusterExpansion):
                A cluster expansion object to solve on.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
            mu(1D ArrayLike):
                mu in charge neutral compostional space.
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
            gs_occu(List[int]):
                Encoded occupation array of previous ground states.
                 Optional, but if you have it, you can save the 
                 ground state solution time when sampling.

        """
        super().__init__(ce,sc_mat,gs_occu=gs_occu,\
                                   anneal_series = anneal_series,\
                                   unfreeze_series = unfreeze_series,\
                                   n_runs_sa = n_runs_sa,\
                                   n_runs_unfreeze = n_runs_unfreeze,\
                                   **kwargs)
        self.sl_sizes = [len(sl) for sl in self.sublat_list]

        self.mu = mu

        compspace = CompSpace(self.bits,self.sl_sizes)

        int_comp = random.choice(compspace.int_grids(sc_size=self.sc_size,form='compstat'))
        self._gs_occu = gs_occu or self._initialize_occu_from_int_comp(int_comp)

        self._ensemble = CNSemiGrandDiscEnsemble.from_cluster_expansion(self.ce, self.sc_mat,\
                                                   optimize_inidicator=self.is_indicator,\
                                                   mu=self.mu)

        self._sampler = Sampler.from_ensemble(ensemble,temperature=1000)
        self._processor = self._ensemble.processor

class SemigrandMCHandler(MCHandler):
    """
    Sublattice NON-DISCRIMINATIVE, Charge neutral semigrand canonical ensemble!
    Used for other physical applications!
    """
    def __init__(self,ce,sc_mat,chemical_potentials,\
                      gs_occu=None,\
                      anneal_series=[3200,1600,800,400,200,100,50],\
                      unfreeze_series=[500,1500,5000],\
                      n_runs_sa = 100,\
                      n_runs_unfreeze = 500,\
                      **kwargs):

        """
        Args:
            ce(ClusterExpansion):
                A cluster expansion object to solve on.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
            chemical_potentials(Dict{Specie|Vacancy|DummySpecie:float}):
                Chemical potentials of all species, regardless of their
                sublattices.
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
            gs_occu(List[int]):
                Encoded occupation array of previous ground states.
                 Optional, but if you have it, you can save the 
                 ground state solution time when sampling.

        """
        super().__init__(ce,sc_mat,gs_occu=gs_occu,\
                                   anneal_series = anneal_series,\
                                   unfreeze_series = unfreeze_series,\
                                   n_runs_sa = n_runs_sa,\
                                   n_runs_unfreeze = n_runs_unfreeze,\
                                   **kwargs)

        self.sl_sizes = [len(sl) for sl in self.sublat_list]

        self.mu = mu

        compspace = CompSpace(self.bits,self.sl_sizes)

        int_comp = random.choice(compspace.int_grids(sc_size=self.sc_size,form='compstat'))
        self._gs_occu = gs_occu or self._initialize_occu_from_int_comp(int_comp)

        self._ensemble = CNSemiGrandEnsemble.from_cluster_expansion(self.ce, self.sc_mat,\
                                                   optimize_inidicator=self.is_indicator,\
                                                   chemical_potentials=chemical_potentials)

        self._sampler = Sampler.from_ensemble(ensemble,temperature=1000)
        self._processor = self._ensemble.processor
