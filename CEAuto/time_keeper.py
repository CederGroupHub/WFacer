"""Module to check the current iter number and module from dataframes.

This class only appears as an attachment to DataManager.
No save or load.
"""

__author__ = 'Fengyu Xie'

import json
import pandas as pd
import os
from copy import deepcopy
import warnings

from monty.serialization import MSONable

from .config_paths import *
from .utils.frame_utils import load_dataframes


class TimeKeeper(MSONable):
    """A class that checks status of CE workflow. 

    Returns current iteration number and the last completed module
    from the dataframes and the history.

    You may not need to directly initialize this.
    """
    # CEAuto modules in their cycle order
    # calc includes calc_reader, calc_writer, calc_manager;
    # gs includes gs_check, gs_solve
    # GS considered as the first module in an iteration(except for
    # iteration 0, the first one).

    modules = ['gs', 'enum', 'write', 'calc', 'feat', 'fit']

    def __init__(self, cur_iter_id=0, last_completed_module='gs'):
        """Initialize TimeKeeper.

        Args:
           cur_iter_id(int):
             Index of the current CE iteration. Default to 0.
           last_completed_module(str):
             Name of the last completed module. Default to 'gs'.
        """
        if last_completed_module not in self.modules:
            raise ValueError("{} is not a valid CEAuto module!"
                             .format(last_completed_module))

        self._cur_iter_id = cur_iter_id
        self._last_completed_module = last_completed_module

    @property
    def cur_iter_id(self):
        """Current iteration id."""
        return self._cur_iter_id

    @property
    def last_completed_module(self):
        """Last completed module."""
        return self._last_completed_module

    @staticmethod
    def check_data_status(sc_df=None, comp_df=None, fact_df=None,
                          history=[]):
        """Get the current iteration index and the last completed module.

        Args:
            sc_df(pd.DataFrame):
                Super-cell table. Refer to DataManager. Default None.
            comp_df(pd.DataFrame):
                Composition table. Refer to DataManager. Default None.
            fact_df(pd.DataFrame):
                Fact table. Refer to DataManager. Default None.
            history(List[Dict]):
                History ce written in dict. Default None.

        Returns:
            int, str
        """
        # 0, gs is the beginning status of a time keeper.
        if (len(sc_df) == 0 or len(comp_df) == 0 or len(fact_df) == 0
            or sc_df is None or comp_df is None or fact_df is None):
            return 0, 'gs'
        # No previous sample, or any data frame is damaged.

        max_iter_id = fact_df.iter_id.max()

        filt_ = (fact_df.iter_id == max_iter_id)

        last_df = fact_df[filt_]
        
        # pd.Series must be converted to list before checking 'in'.
        if 'NC' in last_df.calc_status.tolist():
            last_nc_df = last_df[last_df.calc_status == 'NC']
            if 'enum' in last_nc_df.module:
                return max_iter_id, 'enum'
            elif 'gs' in last_nc_df.module:
                return max_iter_id, 'gs'
            else:
                raise ValueError("Module other than enumerator or gs "+
                                 "solver appeared.")

        if 'CC' in last_df.calc_status.tolist():
            return max_iter_id, 'write'

        if 'CL' in last_df.calc_status.tolist():
            return max_iter_id, 'calc'

        if (len(history) < max_iter_id or
            len(history) > max_iter_id + 1):
            raise ValueError("History record broken! "+
                             "Currently at iteration {}, "
                             .format(max_iter_id)+
                             "but only {} history steps found!"
                             .format(len(history)))

        if len(history) == max_iter_id:
            return max_iter_id, 'feat'

        if len(history) == max_iter_id + 1:
            return max_iter_id + 1, 'fit'

    def set_to_data_status(self, sc_df=None, comp_df=None, fact_df=None,
                           history=[]):
        """Set to status indicated by dataframes.

        Args:
            sc_df(pd.DataFrame):
                Super-cell table. Refer to DataManager. Default None.
            comp_df(pd.DataFrame):
                Composition table. Refer to DataManager. Default None.
            fact_df(pd.DataFrame):
                Fact table. Refer to DataManager. Default None.
            history(List[Dict]):
                History ce written in dict. Default [].

        Note: In a normal CEAuto workflow, this function should not be
              called frequently. You should only use this when DataManager
              clears all data from some iterations.
        """
        iter_id, module = self.check_data_status(sc_df=sc_df, comp_df=comp_df,
                                                 fact_df=fact_df,
                                                 history=history)
        if (iter_id != self.cur_iter_id or
            module != self.last_completed_module):
            warnings.warn("Status inicated by dataframes: {}, {};"
                          .format(iter_id, module) +
                          " Status of time keeper: {}, {};"
                          .format(self.cur_iter_id,
                                  self.last_completed_module) +
                          " Resetting time keeper.")
            self._cur_iter_id = iter_id
            self._last_completed_module = _last_completed_module

    def set_to_file_status(self,
                           sc_file=SC_FILE,
                           comp_file=COMP_FILE,
                           fact_file=FACT_FILE,
                           ce_history_file=CE_HISTORY_FILE):
        """Set to status indicated by data files.

        Args:
           sc_file(str):
             Path to sc table. Default set in config_paths.json.
           comp_file(str):
             Path to comp table. Default set in config_paths.json.
           fact_file(str):
             Path to fact table. Default set in config_paths.json.
           ce_history_file(str):
             Path to ce history. Default set in config_paths.json.
        """
        sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                                  comp_file=comp_file,
                                                  fact_file=fact_file)
        with open(ce_history_file, 'r') as fin:
            d = json.load(fin)

        history = d['history']

        return self.get_iter_id_last_module(sc_df, comp_df, fact_df,
                                            history)

    def before(self, module_name):
        """Check if the specified module has NOT completed in current iter.

        Args:
            module_name(str):
                name of the module to check.
        """
        if self.cur_iter_id == 0 and self.last_completed_module == 'gs':
            return True

        return (self.modules.index(self.last_completed_module) < 
                self.modules.index(module_name))

    def after(self, module_name):
        """Check if the specified module has completed in current iter.

        Args:
            module_name(str):
                name of the module to check.
        """
        return (not self.before(module_name))

    def advance(self, n_modules=1):
        """Advance number of modules for the time keeper.

        Args:
            n_modules(int):
              Number of modules to advance. Default is 1 module.
        """
        last_module_id = self.modules.index(self.last_completed_module)
        next_module_id = (last_module_id + n_modules) % len(self.modules)
        iter_advance = (last_module_id + n_modules) // len(self.modules)

        self._last_completed_module = self.modules[next_module_id]
        self._cur_iter_id += iter_advance

    def as_dict(self):
        """Serialize into dict.

        Returns:
            Dict.
        """
        return {'cur_iter_id': self.cur_iter_id,
                'last_completed_module': self.last_completed_module,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d):
        """Deserialize to object.

        Args:
            d(Dict):
              Serialized dictionary.

        Return:
            TimeKeeper.
        """
        return cls(d.get('cur_iter_id', 0),
                   d.get('last_completed_module', 'gs'))

    def auto_save(self, time_keeper_file=TIME_KEEPER_FILE):
        """Automatically save to time keeper file.

        Args:
            time_keeper_file(str):
              Time keeper file. Default set in config_paths.json. Not
              recommend to change.
        """
        with open(time_keeper_file, 'w') as fout:
            json.dump(self.as_dict(), fout)

    @classmethod
    def auto_load(cls, time_keeper_file=TIME_KEEPER_FILE):
        """Automatically load time keeper file.

        Args:
            time_keeper_file(str):
              Time keeper file. Default set in config_paths.json. Not
              recommend to change.

        Return:
            TimeKeeper.
        """
        with open(time_keeper_file, 'r') as fin:
            d = json.load(fin)

        return cls.from_dict(d)
