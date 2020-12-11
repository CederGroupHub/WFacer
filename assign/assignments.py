__author__='Fengyu Xie'

"""
This file defines a generic propertie assigner class to assign properties 
to undecorated species, and returns their decorated forms.

Possible decorations includes charge (most commonly used), spin polarization.
If the user wishes to define other properties assignment methods, just derive
a new class Assignment class, and write assignment methods accordingly.

For charge assignment, charges will be assigned by magnitudes of magnetization vectors.
"""

from pymatgen import Specie,Structure,Element,DummySpecie
from pymatgen.core.periodic_table import get_el_sp
from abc import ABC, abstractmethod

import numpy as np
from sklearn.mixture import GaussianMixture
from copy import deepcopy

class Assignment(ABC):
    """
    Abstract assignment class.
    Attributes:
        labels_av(Dict{Element: List[int|float]...}):
            A dictionary, specifying the elements, and the labels
            that we should assign to this specific element.
            By default, Vacancy will not appear in structure pool when
            directly read from vasp output, so no need to consider.
            For example:
            OrderedDict({Element.from_string('Li'):[1]})
            when assigning charge +1 to Li in a structure.
            When there are multiple assigned property values possible, 
            the values in the list[int|float] should be sorted by the
            order of their cluster centers in the properties axis.
            For example, If I have Mn2+, Mn3+ and Mn4+ (all on high spin), 
            and we assign charges to Mn atoms by magnetization, then we expect 
            a order of [4,3,2], because this is the order of the magnetic moment
            in these three types of oxidation states.
    """
    def __init__(self):
        

    @abstractmethod
    def train(self,str_pool,properties):
        """
        Train a properties assignment model. Model or model parameters
        should be stored in a property of the object.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            properties(2D ArrayLike):
                Numerical properties used to classify sites.
                Shape should be N_strs*N_sites
        """
        return

    @abstractmethod
    def assign(self,str_pool,properties):
        """
        Give assignment to structures. If an assigned structure is not valid,
        for example, in charge assignment, if an assigned structure is not
        charge neutral, then this structure will be returned as None.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            properties(2D ArrayLike):
                Numerical properties used to classify sites.
                Shape should be N_strs*N_sites       
        Return:
            List[Structure|Nonetype], a list of assigned structures, consisting of
        Species|Element, or None. Vacancies will be handled by structure matcher
        in smol.ClusterSubspace, so there is no need to explicitly add them.
        """
        return

    @abstractmethod
    def as_dict(self):
        """
        Serialization method. Please save the trained property partition or clustering here.
        """
        return

class MagChargeAssignment(Assignment):
    """
    Assign charges from magnitudes of magentic moments. Partition dividers will be given by
    a mixture of gaussians model. Takes in a pool of structures, gives assigned strutures.

    Attributes:
         labels_table(Dict{STRING of element: List[int|float]...}):
            A dictionary, specifying the elements, and the labels
            that we should assign to this specific element.
            By default, Vacancy will not appear in structure pool when
            directly read from vasp output, so no need to consider.
            For example:
            {'Li':[1]}
            when assigning charge +1 to Li in a structure.
            When there are multiple assigned property values possible, 
            the values in the list[int|float] should be sorted by the
            order of their cluster centers in the properties axis.
            For example, If I have Mn2+, Mn3+ and Mn4+ (all on high spin), 
            and we assign charges to Mn atoms by magnetization, then we expect 
            a order of [4,3,2], because this is the order of the magnetic moment
            in these three types of oxidation states.       

         Note: 
            1, All elements in structure pool must be present in this table!
            2, If oxidation state is 0, will assign to Element, not an ox=0 specie!
         check_balance:
            whether the assigner check charge balance, and return None balanced
            structures as None. Default is True.
    """
    def __init__(self,labels_table):
         self.labels_table = labels_table
         self._models_by_elements = {e:None for e in self.labels_table.keys()}

    def train(self,str_pool,mags):
        """
        Train a properties assignment model. Model or model parameters
        should be stored in a property of the object.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            mags(2D ArrayLike):
                Magnetic moments of each site in each structure.
                Shape should be N_strs*N_sites
        No return value.
        """
        #flatten all structures, and group by elements
        sites_by_elements = {e:[] for e in self.labels_table.keys()}
        
        for s_id,s in enumerate(str_pool):
            for st_id,st in enumerate(s):
                #Save site's magetic moment, and location in structure pool
                entry = (mags[s_id][st_id],s_id,st_id)
                sites_by_elements[st.specie.symbol].append(entry)

        for e in sites_by_elements:
            e_props = np.array(sites_by_elements[e])[:,0].reshape((-1,1))
            gm = GaussianMixture(n_components=len(self.labels_table[e])).fit(e_props)
            self._models_by_elements[e] = deepcopy(gm)
         
    @property
    def params_by_elements(self):
        """
        Stores MoG models into a dictionary.
        """
        ps_by_e = {}
        for e,gm_e in self._models_by_elements.items():
            if gm_e is None:
                ps_by_e[e] = None
            else:
                ps_by_e[e]  = {'weights':gm_e.weights_.tolist(),
                               'means':gm_e.means_.tolist(),
                               'covariances':gm_e.covariances_.tolist(),
                               'precisions':gm_e.precisions_.tolist(),
                               'precisions_cho':gm_e.precisions_cholesky_.tolist(),
                               'converged':gm_e.converged_,
                               'n_iter':gm_e.n_iter_,
                               'lower_bound':gm_e.lower_bound_
                              }
        return ps_by_e

    def set_models_with_params(self,params_by_elements):
        """
        Recover MoG models from dictioanries. All elements in self.labels_table 
        should appear in params_by_elements.
        """
        for e in self._models_by_elements:
            gm_e = GaussianMixture(n_components=len(self.labels_table[e]))
            params = params_by_elements[e]
            if params is None:
                self._models_by_elements[e] = None
                continue
            gm_e.weights_ = np.array(params['weights'])
            gm_e.means_ = np.array(params['means'])
            gm_e.covariances_ = np.array(params['covariances'])
            gm_e.precisions_ = np.array(params['precisions'])
            gm_e.precisions_cholesky_ = np.array(params['precisions_cho'])
            gm_e.converged_ = params['converged']
            gm_e.n_iter_ = params['n_iter']
            gm_e.lower_bound_ = params['lower_bound']
            self._models_by_elements[e] = deepcopy(gm_e)

    def assign(self,str_pool,mags,check_neutral=True):
        """
        Assign charges to all sites in a structure pool.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            mags(2D ArrayLike):
                Magnetic moments of each site in each structure.
                Shape should be N_strs*N_sites
            check_neutral(Bool):
                Check whether the assigned structures are charge neutral.
                Non-neutral structures will be returned as None.
        Returns:
            A list of Structures or Nones.
        """
        #Establish a mapping list of MoG clusters and oxidation states. means sorted by ascending order.
        clusorders_by_element = {e:np.argsort(self._models_by_elements[e].means_.reshape((-1,))).tolist()
                                 for e in self._models_by_elements}
        
         #flatten all structures, and group by elements
        sites_by_elements = {e:[] for e in self.labels_table.keys()}
        
        for s_id,s in enumerate(str_pool):
            for st_id,st in enumerate(s):
                #Save site's magetic moment, and location in structure pool
                entry = (mags[s_id][st_id],s_id,st_id)
                sites_by_elements[st.specie.symbol].append(entry)

        #Assign for each site
        sites_by_elements_assigned = {e:[] for e in self.labels_table.keys()}
        assignments = [[None for st in str_mags] for str_mags in mags]

        for e in sites_by_elements:
            clusters_e = self._models_by_elements[e].predict(sites_by_elements[e])
            assignments_e = [ self.labels_table[e][clusorders_by_element[e].index(c_id)] for \
                              c_id in clusters_e ]
            for a_id,a in enumerate(assignments_e):
                s_id, st_id = (sites_by_elements[e][a_id][1:]
                assignments[s_id][st_id]=a

        str_assigned = []
        n_fails = 0
        for s_id,s in enumerate(str_pool):
            species_of_s = []
            for st_id,st in enumerate(s):
                ox = assignments[s_id][st_id]
                e_str = st.specie.symbol
                #If oxidation state is 0, will assign to Element!
                if ox == 0:
                    try:
                        sp = Element(e_str)
                    except:
                        sp = DummySpecie(e_str)
                else:
                    try:
                        sp = Specie(e_str,oxidation_state=ox)
                    except:
                        sp = DummySpecie(e_str,oxidation_state=ox)
                species_of_s.append(sp)
            if (not check_neutral) or (check_neutral and np.sum(assignments[s_id])==0):
                str_assigned.append(Structure(s.lattice, species_of_s, s.frac_coords))
            else:
                str_assigned.append(None)
                n_fails += 1

        n_all = len(str_assigned)
        n_success = n_all-n_fails
        print("****{}/{} Structures Assigned. Success percentage: {:.3f}.".format(n_success,n_all,float(n_success)/n_all)) 
                
        return str_assigned    

    def as_dict(self):
        """
        Serialize into dictionary.
        """
        return {'labels_table':self.labels_table,
                'model_params':self.params_by_elements
               }
    
    @classmethod
    def from_dict(cls,d):
        """
        Recover from dict.
        """
        socket = cls(d['labels_table'])
        socket.set_models_with_params(d['model_params'])
        return socket
