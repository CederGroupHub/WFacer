"""
Defines a fitter class that fits and saves CE models from the fact table.
THIS MODULE DOES NOT CHANGE THE FACT TABLE!
"""
__author__ = 'Fengyu Xie'

import numpy as np
import pandas as pd
import json
from monty.json import MSONable
import warnings

from smol.cofe import ClusterExpansion

from .regression import *
from .utils.weight_utils import weights_from_fact

class CEFitter(MSONable):
    """
    Attributes:
        estimator_flavor(str):
            Name of estimator to be used from .regression module. 
            By default, we will always use L2L0Estimator.
            Available regressions are listed in .regression
        weights_flavor(str):
            Method of adding weights to the rows before fitting.
            Currently supports three methods:
                'unweighted': equal weight for each row (default)
                'e_above_comp': weight by energy above composition
                'e_above_hull': weight by energy above hull
        hierarchy(2D arraylike or None):
            A list of integers storing hirarchy relation between clusters.
            Each sublist contains indices of higher order correlation functions 
            that contains this correlation function.
            If none given, will not add hierarchy constraints.
            (TODO: add a hierarchy proerty to ClusterSubspace)
    """
    supported_weights = ('unweighted','e_above_hull','e_above_comp')
    def __init__(self,estimator_flavor='L2L0Estimator',weights_flavor='unweighted',hierarchy=None):

        self.estimator_flavor = estimator_flavor
        if weights_flavor not in supported_weights:
            raise ValueError("Weighting method {} currently not supoortede!".format(weights_flavor))
        self.weights_flavor = weights_flavor

        #check if possible to use hierarchy.
        self._estimator = globals()[estimator_flavor]()
        self.use_hierarchy = ('hierarchy' in self._estimator.fit.__code__.co_varnames)
        self.hierarchy = hierarchy
        if self.hierarchy is None and self.use_hierarchy:
            warnings.warn("Hierarchy constraints possible on {}, but no hierarchy supplied.".\
                          format(estimator_flavor))

        #Will be refreshed after every fit, so it is your respoinsibility not to 
        #double-fit the same dataset!!
        self._coefs = {}
        self._femat = []

    def pack_cluster_expansions(self,cluster_subspace):
        """
        Pack up fitted data into ClusterExpansion objects.
        Inputs:
            cluster_subspace(smol.ClusterSubspace):
                The cluster subspace you used to featurize structures.
                It is your responsibility to check whether it is the exact
                subspace you used before.
        Output:
            A dictionary: {property_name:cluster expansion of that property}
        """
        if len(self._coefs) == 0 or len(self._femat) == 0:
            print("No cluster expansion model fitted yet.")
            return None
        return {pname:ClusterExpansion(cluster_subspace,
                                       np.array(self._coefs[pname]),
                                       np.array(self._femat))
                for pname in self._coefs
               }
 
    def pack_energy_expansion(self,cluster_subspace):
        """
        Pack up cluster expansion for normalized energies (eV/prim) only.
        Inputs:
            cluster_subspace(smol.ClusterSubspace):
                The cluster subspace you used to featurize structures.
                It is your responsibility to check whether it is the exact
                subspace you used before.
        Output:
            Cluster expansion object of normalized energies. 
        """
        if len(self._coefs) == 0 or len(self._femat) == 0:
            print("No cluster expansion model fitted yet.")
            return None
        return ClusterExpansion(cluster_subspace,np.array(self._coefs['e_prim']),
                                np.array(self._femat))

    def fit_from_fact(self,fact_df,estimator_params={},weighter_params={}):
        """
        Inputs:
            fact_df(pd.DataFrame):
                The fact table dataframe containing information of all calculations.
                Must contain at least one 'e_prim' row for energy CE fit!
            estimator_params(Dict of dicts or None):
                used to set estimator parameters. For example, mu, log_mu_ranges,
                log_mu_steps, M, tlimit, etc.
                This can write like:
                    self.fit_from_fact(df,estimator_params={'e_prim':{'log_mu_ranges':[(-5,5)]}})
                to pass log_mu_ranges into your estimator's fit method, when fitting 'e_prim'.
                If empty dict given, will use default setting for all estimator runs.
                See documentation of your Estimator of choice for detail.
            weighter_params(Dict):
                used to pass parameters of the weights_from_occu function. keywords may
                usually include 'prim','sc_mats', and may include 'temperature', etc, 
                depending on the weighting method.
                See doc for util.weight_utils.weights_from_occu for more detail.

         Class attibutes will be refreshed after every fit, so it is your respoinsibility not to 
         double-fit the same dataset and to save your own time!
        """
        fact_avail = fact_df[fact_df.calc_status=='SC']

        #Given a fact table (pd.DataFrame), computes weights.
        weights = weights_from_fact(fact_avail, flavor = self.weights_flavor, **weighter_params)

        self._femat = fact_avail.map_corr.tolist()
        feature_matrix = np.array(self._femat)
        #Fit energy coefficients
        target_vector = np.array(fact_avail.e_prim)

        e_prim_params = estimator_params['e_prim'] if 'e_prim' in estimator_params else \
                        {}
        if self.use_hierarchy:
            self._estimator.fit(feature_matrix,target_vector,hierarchy=self.hierarchy,\
                                sample_weight = weights,\
                                **e_prim_params)
            self._coefs['e_prim']=self._estimator.coef_.copy().tolist()
        else:
            self._estimator.fit(feature_matrix,target_vector,sample_weight = weights,\
                                **e_prim_params)
            self._coefs['e_prim']=self._estimator.coef_.copy().tolist()

        #Fit other properties coefficients. All other properties must be scalars
        other_props_asdf = pd.DataFrame(fact_avail.other_props.tolist())

        for prop_name in other_props_asdf:
            target_vector = np.array(other_props_asdf[prop_name].tolist())
            prop_params = estimator_params[prop_name] if prop_name in estimator_params else \
                          {}

            if self.use_hierarchy:
                self._estimator.fit(feature_matrix,target_vector,hierarchy=self.hierarchy,\
                                    sample_weight = weights,\
                                    **prop_params)
                self._coefs[prop_name]=self._estimator.coef_.copy().tolist()
            else:
                self._estimator.fit(feature_matrix,target_vector,sample_weight = weights,\
                                    **prop_params)
                self._coefs[prop_name]=self._estimator.coef_.copy().tolist()
 
    def as_dict(self):
        return {"weights_flavor": self.weights_flavor,
                "estimator_flavor": self.estimator_flavor,
                "hierarchy": self.hierarchy,
                "coefs": self._coefs,
                "femat": self._femat,
                "@module":self.__class__.__module__
                "@class":self.__class__.__name__
               }          

    @classmethod
    def from_dict(cls,d):
        socket = cls(weights_flavor = d['weights_flavor'],
                     estimator_flavor = d['estimator_flavor'],
                     hierarchy = d['hierarchy'])
        socket._coefs = d['coefs']
        socket._femat = d['femat']
        return socket
