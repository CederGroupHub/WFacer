"""
A module to check the current iteration number and the last finished module
from the dataframes.

This class only appears as an attachment to DataManager.

No save.
"""
import json
import pandas as pd
import os

from .config_paths import *

__author__ = 'Fengyu Xie'

def load_dataframes(sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
    """
    Loading dimension tables and the fact table. 
    comp_df needs a little bit de-serialization.
    File names can be changed, but not recommended!
    Notice: pandas loads lists as strings. You have to serialize them!
    """
    list_conv = lambda x: json.loads(x) if x is not None else None
    if os.path.isfile(sc_file):
        sc_df = pd.read_csv(sc_file,converters={'matrix':list_conv})
    else:
        sc_df = pd.DataFrame(columns=['sc_id','matrix'])

    if os.path.isfile(comp_file):
        #De-serialize compositions and list values
        comp_df = pd.read_csv(comp_file,
                                    converters={'ucoord':list_conv,
                                                'ccoord':list_conv,
                                                'cstat':list_conv,
                                                'eq_occu':list_conv,
                                                'comp':deser_comp
                                               })
    else:
        comp_df = pd.DataFrame(columns=['comp_id','sc_id',\
                                        'ucoord','ccoord',\
                                        'comp','cstat',\
                                        'eq_occu'])

    if os.path.isfile(fact_file):
        fact_df = pd.read_csv(fact_file,
                                    converters={'ori_occu':list_conv,
                                                'ori_corr':list_conv,
                                                'map_occu':list_conv,
                                                'map_corr':list_conv,
                                                'other_props':list_conv
                                               })
    else:
         fact_df = pd.DataFrame(columns=['entry_id','sc_id','comp_id',\
                                         'iter_id','module',\
                                         'ori_occu','ori_corr',\
                                         'calc_status',\
                                         'map_occu','map_corr',\
                                         'e_prim','other_props'])

    return sc_df, comp_df, fact_df

class StatusChecker:
    """
    A class that checks current iteration number and the last completed
    module from the dataframes, and the history document. Readonly, will
    not be saved. 
    Mostly this will serve as an attachment to DataManager, and you may 
    not need to directly initialize this.

    Direct __init__ not recommended.
    """
    #CEAuto modules in their cycle order
    #calc includes calc_reader, calc_writer, calc_manager;
    #gs includes gs_check, gs_solve
    #GS considered as the first module in an iteration(except for
    #iteration 0, the first one).

    modules = ['gs','enum','write','calc','feat','fit']

    def __init__(self,sc_df,comp_df,fact_df,history=[]):
        """
        Args:
            sc_df(DataFrame):
                supercells dataframe
            comp_df(DataFrame):
                compositons dataframe
            fact_df(DataFrame):
                fact dataframe, containing all computation entree.
            history(List[Dict]):
                Historical cluster expansion record.
        """
        self.sc_df = sc_df
        self.comp_df = comp_df        
        self.fact_df = fact_df

        self.history = history

    def _get_iter_id_last_module(self):
        """
        Get the current iteration index and the last completed module name
        Returns:
            iter_id, last_module(str|Nonetype)
        """
        if len(self.fact_df)==0:
            return 0,None

        max_iter_id = self.fact_df.iter_id.max()

        filt_ = (self.fact_df.iter_id==max_iter_id)

        last_df = self.fact_df[filt_]
        
        if 'NC' in last_df.calc_status:
            last_nc_df = last_df[last_df.calc_status=='NC']
            if 'enum' in last_nc_df.module:
                return max_iter_id, 'enum'
            elif 'gs' in last_nc_df.module:
                return max_iter_id, 'gs'
            else:
                raise ValueError("Module other than enumerator or gs solver appeared.")

        if 'CC' in last_df.calc_status:
            return max_iter_id, 'write'

        if 'CL' in last_df.calc_status: 
            return max_iter_id, 'calc'  

        if len(history)<max_iter_id or len(history)>max_iter_id+1:
            raise ValueError("History record broken! Currently at iteration {}, but only {} history steps found!".format(max_iter_id,len(history)))

        if len(history)==max_iter_id:
            return max_iter_id, 'feat'

        if len(history)==max_iter_id+1:
            return max_iter_id+1, 'fit'

    @property
    def cur_iter_id(self):
        """
        current iteration id.
        """
        return self._get_iter_id_last_module()[0]

    @property
    def last_completed_module(self):
        """
        last completed module.
        """
        return self._get_iter_id_last_module()[1]

    def before(self,module_name):
        """
        Check whether the specified module HAS NOT been finished 
        in the current iteration.
        Args:
            module_name(str):
                name of the module to check.
        """
        if self.last_completed_module is None:
            return True
        return modules.index(self.last_completed_module)<modules.index(module_name)

    def after(self,module_name):
        """
        Check whether the specified module HAS been finished 
        in the current iteration.
        Args:
            module_name(str):
                name of the module to check.
        """
        return (not self.before(module_name))

    @classmethod
    def auto_load(cls,sc_file=SC_FILE,comp_file=COMP_FILE,fact_file=FACT_FILE,\
                      ce_history_file = CE_HISTORY_FILE):
        """
        This method is the recommended way to initialize this object.
        It automatically reads all setting files with FIXED NAMES.
        YOU ARE NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!

        Returns:
             StatusChecker object.
        """
        sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,\
                                                  comp_file=comp_file,\
                                                  fact_file=fact_file)

        if os.path.isfile(ce_history_file) as fin:
            history = json.load(fin)
        else:
            history = []

        sock = cls(sc_df,comp_df,fact_df,history)
        sock._sc_load_path = sc_file
        sock._comp_load_path = comp_file
        sock._fact_load_path = fact_file
        sock._history_load_path = ce_history_file

        return sock

    def re_load(self,from_load_paths=True,\
                sc_file=SC_FILE,comp_file=COMP_FILE,fact_file=FACT_FILE,\
                ce_history_file = CE_HISTORY_FILE):
        """
        During a cluster expansion flow, the database might be frequently modified,
        therefore an easy renew of statuschecker is required.
        Args: 
            from_load_paths(Boolean):
                If true, will re-load status checker from where it is initialized.
        No return value.
        """
        if from_load_paths:
            sc_file = self._sc_load_path
            comp_file = self._comp_load_path
            fact_file = self._fact_load_path
            ce_history_file = self._history_load_path

        sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,\
                                                  comp_file=comp_file,\
                                                  fact_file=fact_file)

        if os.path.isfile(ce_history_file) as fin:
            history = json.load(fin)
        else:
            history = []

        self.sc_df = sc_df
        self.comp_df = comp_df
        self.fact_df = fact_df
        self.history = history
