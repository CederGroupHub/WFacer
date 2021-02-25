__author__='Fengyu Xie'

"""
For charge assignment, charges will be assigned by magnitudes of magnetization vectors.
"""

from pymatgen import Specie,Structure,Element,DummySpecies

from smol.cofe.space.domain import get_species

from .base import BaseDecorator

import numpy as np
from sklearn.mixture import GaussianMixture
from copy import deepcopy

class MagChargeDecorator(BaseDecorator):
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
         
    @property
    def trained(self):
        """
        Gives whether the model is trained or not.
        """
        for val in self._models_by_elements.values():
            if val is None:
                return False
        return True

    def train(self,str_pool,mags_3d,reset=False):
        """
        Train a properties assignment model. Model or model parameters
        should be stored in a property of the object.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            mags_3d(3D ArrayLike):
                Magnetic moments of each site in each structure.
                Shape should be 1*N_strs*N_sites
                This must be a 3D array because some other decorators may
                take multiple proerties as classifier features. We must keep
                a consistency of input formats between all classifiers.
            reset(Boolean):
                If you want to re-train the decorator model, set this value
                to true. Otherwise we will skip training if self.trained is 
                true.
        No return value.
        """
        if self.trained and (not reset):
            print("Decorator model trained! Skip training.")
            return

        mags = mags_3d[0]
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
            A dictionary, specifying name of assigned properties and their
            values by structure and by site. If assignment failed for a
            structure, will give None for it.
            For example: 
            {'charge':[[1,4,2,...],None,[...],...]}
            Currently, in pymatgen.Specie's
            other_properties, only 'spin' is allowed. If you want to add more, do
            your own study!
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
            mags_e = np.array(sites_by_elements[e])[:,0].reshape((-1,1))
            clusters_e = self._models_by_elements[e].predict(mags_e)
            assignments_e = [ self.labels_table[e][clusorders_by_element[e].index(c_id)] for \
                              c_id in clusters_e ]
            for a_id,a in enumerate(assignments_e):
                s_id, st_id = (sites_by_elements[e][a_id][1:])
                assignments[s_id][st_id]=a

        oxi_assigned = []
        n_fails = 0
        for s_id,s in enumerate(str_pool):
            if (not check_neutral) or (check_neutral and np.sum(assignments[s_id])==0):
                oxi_assigned.append(assignments[s_id])
            else:
                oxi_assigned.append(None)
                n_fails += 1

        n_all = len(oxi_assigned)
        n_success = n_all-n_fails
        print("****{}/{} Structures Assigned. Success percentage: {:.3f}.".format(n_success,n_all,float(n_success)/n_all)) 
                
        return {'charge':oxi_assigned}    

    def as_dict(self):
        """
        Serialize into dictionary.
        """
        return {'labels_table':self.labels_table,
                'model_params':self.params_by_elements,
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__
               }
    
    @classmethod
    def from_dict(cls,d):
        """
        Recover from dict.
        """
        socket = cls(d['labels_table'])
        socket.set_models_with_params(d['model_params'])
        return socket
