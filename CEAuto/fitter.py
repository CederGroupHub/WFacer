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
import matplotlib.pyplot as plt

from smol.cofe import ClusterExpansion

from .regression import *
from .data_manager import DataManager

from .utils.weight_utils import weights_from_fact

from .config_paths import *

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
    """

    supported_weights = ('unweighted','e_above_hull','e_above_comp')

    def __init__(self,cluster_subspace,data_manager,\
                      estimator_flavor='L2L0Estimator',\
                      weights_flavor='unweighted',\
                      use_hierarchy=True,\
                      estimator_params={},\
                      weighter_params={}
                ):

        """
        Args:
            cluster_subspace(smol.ClusterSubspace):
                The cluster subspace used in featurization.
                Compulsory.
            data_manager(DataManager):
               A data manager object to read and write dataframes.
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
            estimator_params(Dict{Dict}|Dict):
                Parameters to pass into the estimators. When is a dictionary, will only
                use it for fitting 'e_prim'. When is a dictionary
                of dictionaries, the keys will specify which property are the parameters 
                being applied to.
                See regression module.
            weighter_params(Dict):
                Parameters to pass into the weighting module. See utils.weight_utils
        """

        self.cspc = cluster_subspace
        self.estimator_flavor = estimator_flavor
        if weights_flavor not in supported_weights:
            raise ValueError("Weighting method {} currently not supoortede!".format(weights_flavor))
        self.weights_flavor = weights_flavor

        #check if possible to use hierarchy.
        self._estimator = globals()[estimator_flavor]()
        self.use_hierarchy = use_hierarchy
        #Use low-to-up hierarchy
        self.hierarchy = self.cspc.bit_combo_hierarchy(invert=True)

        #Will be refreshed after every fit, so it is your respoinsibility not to 
        #double-fit the same dataset!!
        self._history = []
        self._updated = False
        self._coefs = {}
        self._cv = {}
        self._rmse = {}
        self._femat = []

        if len(estimator_params) == 0:
            self.estimator_params = estimator_params

        elif not isinstance(list(estimator_params.values())[0],dict):
            self.estimator_params = {'e_prim':estimator_params}
        else:
            self.estimator_parmas = estimator_params
 
        self.weighter_params = weighter_params

        self._dm = data_manager
        self._schecker = self._dm._schecker

        self._history_load_path = CE_HISTORY_FILE

    @property
    def sc_df(self):
        """
        Supercell dataframe.
        """
        return self._dm.sc_df

    @property
    def comp_df(self):
        """
        Composition dataframe.
        """
        return self._dm.comp_df

    @property
    def fact_df(self):
        """
        Fact dataframe.
        """
        return self._dm.fact_df


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
            warnings.warn("No cluster expansion model fitted yet. Call fit first.")
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
            warnings.warn("No cluster expansion model fitted yet. Call fit first.")
            return None
        return ClusterExpansion(self.cspc,np.array(self._coefs['e_prim']),
                                np.array(self._femat))

    def fit(self):
        """
         Fits cluster expansion from the featurized fact dataframe. No return value.

         Class attibutes will be refreshed after every fit, so it is your responsibility not to 
         double-fit the same dataset!

        """
        estimator_params = self.estimator_params
        weighter_params = self.weighter_params       

        if self._schecker.after('fit'): #Fitter already finished in current cycle.
            print("**ECIs aleady fitted for current iteration {}. Loading from history."\
                  .format(self._schecker.cur_iter_id))
            self._coefs = self._history[-1]['coefs']
            self._cv = self._history[-1]['cv']
            self._rmse = self._history[-1]['rmse']
            self._updated = True #mute history update!
            return

        fact_df = self.fact_df.copy()

        fact_avail = fact_df[fact_df.calc_status=='SC']

        #Given a fact table (pd.DataFrame), computes weights.
        weights = weights_from_fact(fact_avail, flavor = self.weights_flavor, **weighter_params)

        self._femat = fact_avail.map_corr.tolist()
        X = np.array(self._femat)
        #Fit energy coefficients
        y = np.array(fact_avail.e_prim)

        e_prim_params = estimator_params.get('e_prim',{})

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
            prop_params = estimator_params.get(prop_name,{})

            self._estimator.fit(X,y,sample_weight = weights,\
                                hierarchy=self.hierarchy,\
                                **prop_params)
            preds = self._estimator.predict(X)
            self._coefs[prop_name]=self._estimator.coef_.copy().tolist()
            cvs = self._estimator.calc_cv_score(X,y,sample_weight=weights,\
                                            hierarchy=self.hierarchy,\
                                            **prop_params)
            self._cv[prop_name] = np.sqrt((1-cvs)*np.sum((y-np.average(y))**2)/len(y))
            self._rmse[prop_name] = np.sqrt(np.sum((np.array(preds)-y)**2)/len(y))


    def _update_history(self):
        """
        Append current fit to the history log.
        Status Checker will prevent double appending in a same iteration, therefore 
        lenth of history should always be the same as the current iteration number.
        """
        if len(self._cv) and len(self._rmse) and len(self._coefs) and not self._updated:
            self._history.append({"cv":self._cv,
                                  "rmse":self._rmse,
                                  "coefs":self._coefs})
            self._updated = True


    def get_eci_plot(self,prop_name='e_prim'):
        """
        Plot eci of the latest cluster expansion.
        Args:
           prop_name(str):
               Name of the property to plot eci for.
               Default is e_prim.
        Return:
           plt.figure, plt.axes. Remember to close the figure after save.
        """
        if len(self._cv)==0 or len(self._rmse)==0 or len(self._coefs)==0:
            warnings.warn("ECIs not fitted yet. Trying with history CE.")
            if len(self._history)==0:
                raise ValueError("No history CE to read from!")
            _cv = self._history[-1]['cv'][prop_name]
            _rmse = self._history[-1]['rmse'][prop_name]
            _coefs = self._hitory[-1]['coefs'][prop_name]

        else:
            _cv = self._cv[prop_name]
            _rmse = self._rmse[prop_name]
            _coefs = self._rmse[prop_name]

        ce = ClusterExpansion(self.cspc,_coefs,np.array(self._femat))
        ecis = ce.eci

        fs = 16
        fst = 12
        fig,ax = plt.subplots(figsize=(8.0,6.0))
        xs = np.arange(0,len(ecis)-1)+0.5
        ax.bar(xs,ecis[1:],width=1.0)

        #Number of each type of clusters, marking lines.
        n1 = sum([len(o) for o in self.cspc.orbits_by_size.get(1,[[]])])
        n2 = sum([len(o) for o in self.cspc.orbits_by_size.get(2,[[]])])
        n3 = sum([len(o) for o in self.cspc.orbits_by_size.get(3,[[]])])
        n4 = sum([len(o) for o in self.cspc.orbits_by_size.get(4,[[]])])

        ax.axvline(x=n1,color='k')
        ax.axvline(x=n1+n2,color='k')
        ax.axvline(x=n1+n2+n3,color='k')
        ax.axvline(x=n1+n2+n3+n4,color='k')

        #marking external terms
        n_ext = self.cspc.external_terms
        if n_ext!=0:
            ext_names = [et.__class__.__name__ for et in self.cspc.external_terms]
            ext_names = ','.join(ext_names)
            ax.bar(xs[-n_ext:],ecis[-n_ext:],width=1.0,color='r',label=ext_names)

        ax.tick_params(labelsize=fst)
        ax.set_xlabel('Cluster indices (w.o. zero-term)',fontsize=fs)
        ax.set_ylabel('ECIs',fontsize=fs)
        ax.set_title('ECI plot of {} (zero={:.3f})'.format(prop_name,ecis[0]),fontsize=fs)
        if n_ext!=0:
            ax.legend(fontsize=fst)

        return fig,ax


    def get_scatter_plot(self,prop_name='e_prim'):
        """
        Plot scatter plot of the latest cluster expansion.
        Args:
           prop_name(str):
               Name of the property to plot eci for.
               Default is e_prim.
        Return:
           plt.figure, plt.axes. Remember to close the figure after save.
        """ 
        fact_df = self.fact_df.copy()

        fact_avail = fact_df[fact_df.calc_status=='SC']     
        X = np.array(fact_avail.map_corr.tolist())

        if len(self._cv)==0 or len(self._rmse)==0 or len(self._coefs)==0:
            warnings.warn("ECIs not fitted yet. Trying with history CE.")
            if len(self._history)==0:
                raise ValueError("No history CE to read from!")
            _cv = self._history[-1]['cv'][prop_name]
            _rmse = self._history[-1]['rmse'][prop_name]
            _coefs = self._hitory[-1]['coefs'][prop_name]
        else:
            _cv = self._cv[prop_name]
            _rmse = self._rmse[prop_name]
            _coefs = self._rmse[prop_name]       

        y_pred = X@np.array(_coefs)
        if prop_name == 'e_prim':
            y = fact_avail.e_prim
        else:
            other_props_asdf = pd.DataFrame(fact_avail.other_props.tolist())
            y = np.array(other_props_asdf[prop_name].to_list())

        textstr = '\n'.join((
        r'$RMSE = %.3f$'%(_rmse,),
        r'$CV value = %.3f$'%(_cv,),
        r'$N_{strs} = %d$'%len(y)
        ))

        fig,ax = plt.subplots(figsize=(8,6))
        ax.scatter(y,y_pred,label=text_str)
        fs = 16
        fst = 12
        ax.tick_params(labelsize=fst)
        ax.set_title("CE scatter plot of {}".format(prop_name))
        ax.set_xlabel(prop_name+'_input',fontsize=fs)
        ax.set_ylabel(prop_name+'_ce',fontsize=fs)
        ax.legend(fontsize=fst)

        return fig,ax
            

    def auto_save(self,update_his= True,ce_history_file = CE_HISTORY_FILE,\
                       to_load_paths=True):
        """
        Serialization to file.
        Make sure to call it only ONCE for each CE iteration!
        Args:
            update_his(Bool):
                whether to update history log or not. Default to True.
            ce_history_file(str):
                path to cluster expansion history file.
            to_load_paths(Boolean):
                If true, will save history to the path from which this object is loaded.
                Default is true.
        """
        if to_load paths:
            ce_history_file = self._history_load_path

        with open(ce_history_file,'w') as fout:
            if update_his:
                self._update_history()
            json.dump(self._history,fout)


    @classmethod
    def auto_load(cls,options_file=OPTIONS_FILE,\
                      sc_file=SC_FILE,\
                      comp_file=COMP_FILE,\
                      fact_file=FACT_FILE,\
                      ce_history_file=CE_HISTORY_FILE):
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
            comp_file(str):
                path to compositions file, in csv format.
                Default: 'comps.csv'             
            fact_file(str):
                path to enumerated structures dataframe file, in csv format.
                Default: 'data.csv'             
            ce_history_file(str):
                path to cluster expansion history file.
                Default: 'ce_history.json'
        Returns:
             Fitter object.
        """
        options = InputsWrapper.auto_load(options_file=options_file,\
                                          ce_history_file=ce_history_file)

        dm = DataManager.auto_load(options_file=options_file,\
                                   sc_file=sc_file,\
                                   comp_file=comp_file,\
                                   fact_file=fact_file,\
                                   ce_history_file=ce_history_file)

        socket = cls(options.subspace,\
                     data_manager=dm,\
                     **options.fitter_options
                    )

        socket._history = options.history
        socket._history_load_path = ce_history_file

        return socket
