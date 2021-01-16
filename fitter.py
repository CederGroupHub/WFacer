"""
Defines a fitter class that fits and saves CE models from the fact table.
THIS MODULE WILL NOT WRITE FACT TABLE!
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
    You may not want to initialize CEFitter directly.
    Since this class does not check which iteration number it is in, it is possible
    that you initialize and fit it twice in one iteration.
    If you fail to recognize this and pass it down as update_his=False, your history
    will be dupe appended in one iteration, which may cause error in the termination 
    criteria codes.
    Make sure to init it only ONCE for each CE iteration. And if you have to init
    it multiple times, make sure not to call update_history in your later inits!

    Attributes:
        cluster_subspace(smol.ClusterSubspace):
            The cluster subspace used in featurization.
            Compulsory.
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
        use_hierarchy(Boolean):
            Whether or not to add hierarchy constraints.
    """
    supported_weights = ('unweighted','e_above_hull','e_above_comp')

    def __init__(self,cluster_subspace,\
                      estimator_flavor='L2L0Estimator',weights_flavor='unweighted',\
                      use_hierarchy=True):
        self.cspc = cluster_subspace
        self.estimator_flavor = estimator_flavor
        if weights_flavor not in supported_weights:
            raise ValueError("Weighting method {} currently not supoortede!".format(weights_flavor))
        self.weights_flavor = weights_flavor

        #check if possible to use hierarchy.
        self._estimator = globals()[estimator_flavor]()
        self.use_hierarchy = use_hierarchy
        self.hierarchy = self.cspc.bit_combo_hierarchy()

        #Will be refreshed after every fit, so it is your respoinsibility not to 
        #double-fit the same dataset!!
        self._history = []
        self._updated = False
        self._coefs = {}
        self._cv = {}
        self._rmse = {}
        self._femat = []

    def pack_cluster_expansions(self):
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
        return {pname:ClusterExpansion(self.cspc,
                                       np.array(self._coefs[pname]),
                                       np.array(self._femat))
                for pname in self._coefs
               }
 
    def pack_energy_expansion(self):
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
        return ClusterExpansion(self.cspc,np.array(self._coefs['e_prim']),
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

         No return value.

         Class attibutes will be refreshed after every fit, so it is your responsibility not to 
         double-fit the same dataset!
        """
        fact_avail = fact_df[fact_df.calc_status=='SC']

        #Given a fact table (pd.DataFrame), computes weights.
        weights = weights_from_fact(fact_avail, flavor = self.weights_flavor, **weighter_params)

        self._femat = fact_avail.map_corr.tolist()
        X = np.array(self._femat)
        #Fit energy coefficients
        y = np.array(fact_avail.e_prim)

        e_prim_params = estimator_params['e_prim'] if 'e_prim' in estimator_params else \
                        {}
        mu = self._estimator.fit(X,y,sample_weight = weights,\
                                 hierarchy=self.hierarchy,\
                                 **e_prim_params)
        preds = self._estimator.predict(X)
        self._coefs['e_prim'] = self._estimator.coef_.copy().tolist()
        cvs = self._estimator.calc_cv_score(X,y,sample_weight=weights,\
                                            hierarchy=self.hierarchy,\
                                            **e_prim_params)
        self._cv['e_prim'] = np.sqrt((1-cvs)*np.sum((y-np.average(y))**2)/len(y))
        self._rmse['e_prim'] = np.sqrt(np.sum((np.array(preds)-y)**2)/len(y))
        

        #Fit other properties coefficients. All other properties must be scalars
        other_props_asdf = pd.DataFrame(fact_avail.other_props.tolist())

        for prop_name in other_props_asdf:
            y = np.array(other_props_asdf[prop_name].tolist())
            prop_params = estimator_params[prop_name] if prop_name in estimator_params else \
                          {}

            self._estimator.fit(X,y,sample_weight = weights,\
                                hierarchy=self.hierarchy,\
                                **prop_params)
            preds = self._estimator.predict(X)
            self._coefs[prop_name]=self._estimator.coef_.copy().tolist()
            cvs = self._estimator.calc_cv_score(X,y,sample_weight=weights,\
                                            hierarchy=self.hierarchy,\
                                            **e_prim_params)
            self._cv[prop_name] = np.sqrt((1-cvs)*np.sum((y-np.average(y))**2)/len(y))
            self._rmse[prop_name] = np.sqrt(np.sum((np.array(preds)-y)**2)/len(y))

    def update_history(self):
        """
        Append current fit to the history log.
        """
        if len(self._cv) and len(self._rmse) and len(self._coefs) and not self._updated:
            self._history.append({"cv":self._cv,
                                  "rmse":self._rmse,
                                  "coefs":self._coefs})
            self._updated = True

    def as_dict(self,update_his=True):        
        """
        Serialization. 
        Since this class does not check which iteration number it is in, it is possible
        that you initialize and fit it twice in one iteration.
        If you fail to recognize this and pass it down as update_his=False, your history
        will be dupe appended in one iteration, which may cause error in the termination 
        criteria codes.
        Make sure to init it only ONCE for each CE iteration. And if you have to init
        it multiple times, make sure not to call update_history in your later inits!
        Args:
            update_his(Bool):
                whether to update history log or not. Default to True.
        """
        if update_his:
            self.update_history()
        return {"cluster_subspace":self.cspc.as_dict(),
                "weights_flavor": self.weights_flavor,
                "estimator_flavor": self.estimator_flavor,
                "use_hierarchy": self.use_hierarchy,
                "history": self._history,
                "femat": self._femat,
                "@module":self.__class__.__module__
                "@class":self.__class__.__name__
               }          

    def as_file(self,update_his= True,fname = 'ce_fitter.json'):
        """
        Serialization. 
        Make sure to call it only ONCE for each CE iteration!
        Args:
            fname(str):
                path to object saving file. Can be changed, but
                not recommended!
            update_his(Bool):
                whether to update history log or not. Default to True.
        """
        with open(fname,'w') as fout:
            json.dump(self.as_dict(update_his=update_his),fout)

    @classmethod
    def from_dict(cls,d):
        """
        Deserialization.
        Make sure to call it only ONCE for each CE iteration!
        If you have to call it multiple times, make sure not to call as_dict, as_file
        again, or pass update_his = False to as_dict and as_file.
        """
        socket = cls(cluster_subspace = d.get('cluster_subspace',None),
                     weights_flavor = d.get('weights_flavor','unweighted'),
                     estimator_flavor = d.get('estimator_flavor','L2L0Estimator'),
                     use_hierarchy = d.get('use_hierarchy',True))
        socket._history = d.get('history',[])
        socket._femat = d.get('femat',[])
        return socket

    def from_file(cls,fname= 'ce_fitter.json'):
        """
        De-Serialization. 
        Make sure to call it only ONCE for each CE iteration!
        If you have to call it multiple times, make sure not to call as_dict, as_file
        again, or pass update_his = False to as_dict and as_file.
        Args:
            fname(str):
                path to object saving file. Can be changed, but
                not recommended!
        """
        with open(fname) as fin:
            return cls.from_dict(json.load(fin))
