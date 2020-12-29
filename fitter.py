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
from smol.cofe.space import ClusterSubspace

from .regression import *

class CEFitter(MSONable):
    """
    Attributes:
        cluster_subspace(smol.ClusterSubspace):
            The cluster subspace object you used to featurize structures.
            NOTE: it is your responsibility to check that, this subspace
                  is the same one used in Featurizer!

        estimator_flavor(str):
            Name of estimator to be used from .regression module. 
            By default, we will always use L2L0Estimator.
            Available regressions are listed in .regression
        hierarchy(2D arraylike or None):
            A list of integers storing hirarchy relation between clusters.
            Each sublist contains indices of higher order correlation functions 
            that contains this correlation function.
            If none given, will not add hierarchy constraints.
            (TODO: add a hierarchy proerty to ClusterSubspace so we can deprecate
                   this attribute)

    Will always use equal weights for all structures. May add more weighting methods
    in the future update.
    """

    def __init__(self,cluster_subspace,estimator_flavor='L2L0Estimator',hierarchy=None):

        self.cluster_subspace = cluster_subspace
        self.estimator_flavor = estimator_flavor

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

    @property
    def cluster_expansions(self):
        if len(self._coefs) == 0 or len(self._femat) == 0:
            print("No cluster expansion model fitted yet.")
            return None
        return {pname:ClusterExpansion(cluster_subspace,
                                       np.array(self._coefs[pname]),
                                       np.array(self._femat))
                for pname in self._coefs
               }
 
    @property
    def norm_energy_expansion(self):
        if len(self._coefs) == 0 or len(self._femat) == 0:
            print("No cluster expansion model fitted yet.")
            return None
        return ClusterExpansion(cluster_subspace,np.array(self._coefs['e_prim']),
                                np.array(self._femat))

    def fit_from_fact(self,fact_df,estimator_params={}):
        """
        Inputs:
            fact_df(pd.DataFrame):
                The fact table dataframe containing information of all calculations.
                Must contain at least one 'e_prim' row for energy CE fit!
            params(Dict of dicts or None):
                used to set estimator parameters. For example, mu, log_mu_ranges,
                log_mu_steps, M, tlimit, etc.
                This can write like:
                    self.fit_from_fact(df,estimator_params={'e_prim':{'log_mu_ranges':[(-5,5)]}})
                to pass log_mu_ranges into your estimator's fit method, when fitting 'e_prim'.
                If empty dict given, will use default setting for all estimator runs.
                See documentation of your Estimator of choice for detail.

         Class attibutes will be refreshed after every fit, so it is your respoinsibility not to 
         double-fit the same dataset and to save your own time!
        """
        fact_avail = fact_df[fact_df.calc_status=='SC']
        self._femat = fact_avail.map_corr.tolist()
        feature_matrix = np.array(self._femat)
        #Fit energy coefficients
        target_vector = np.array(fact_avail.e_prim)

        e_prim_params = estimator_params['e_prim'] if 'e_prim' in estimator_params else \
                        {}
        if self.use_hierarchy:
            self._estimator.fit(feature_matrix,target_vector,hierarchy=self.hierarchy,\
                                **e_prim_params)
            self._coefs['e_prim']=self._estimator.coef_.copy().tolist()
        else:
            self._estimator.fit(feature_matrix,target_vector,**e_prim_params)
            self._coefs['e_prim']=self._estimator.coef_.copy().tolist()

        #Fit other properties coefficients. All other properties must be scalars
        other_props_asdf = pd.DataFrame(fact_avail.other_props.tolist())

        for prop_name in other_props_asdf:
            target_vector = np.array(other_props_asdf[prop_name].tolist())
            prop_params = estimator_params[prop_name] if prop_name in estimator_params else \
                          {}

            if self.use_hierarchy:
                self._estimator.fit(feature_matrix,target_vector,hierarchy=self.hierarchy,\
                                    **prop_params)
                self._coefs[prop_name]=self._estimator.coef_.copy().tolist()
            else:
                self._estimator.fit(feature_matrix,target_vector,**prop_params)
                self._coefs[prop_name]=self._estimator.coef_.copy().tolist()
 
    def as_dict(self):
        return {"cluster_subspace": self.cluster_subspace,
                "estimator_flavor": self.estimator_flavor,
                "hierarchy": self.hierarchy,
                "coefs": self._coefs,
                "femat": self._femat,
                "@module":self.__class__.__module__
                "@class":self.__class__.__name__
               }          

    @classmethod
    def from_dict(cls,d):
        socket = cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                     estimator_flavor = d['estimator_flavor'],
                     hierarchy = d['hierarchy'])
        socket._coefs = d['coefs']
        socket._femat = d['femat']
        return socket
