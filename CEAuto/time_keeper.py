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
    """

    modules = ['enum', 'write', 'calc', 'feat', 'fit', 'check']
    # Check module may include ground state generation
    # (canonical or grand canonical) in the future.

    def __init__(self, cursor=0):
        """Initialize TimeKeeper.

        Args:
           cursor(int):
               Cursor pointing to the NEXT tast to do in 
               workflow. Default is 0, which it starts.
        """
        self._cursor = cursor

    @property
    def cursor(self):
        """Interger cursor state.

        Note: last status of the cursor is the NEXT task TO DO!
        which means if you resume broken workflow, you need to start
        from the exact task which this cursor is pointing at!
        """
        return self._cursor

    @cursor.setter
    def cursor(self, c):
        """Setter method for cursor."""
        if c < 0:
            raise ValueError("Cursor value can't be negative!")

        self._cursor = c

    @property
    def iter_id(self):
        """Iteration id at cursor state."""
        return self.cursor // len(self.modules)

    @property
    def next_module_todo(self):
        """Name of the next module to do."""
        return self.modules[self.cursor % len(self.modules)]

    @staticmethod
    def check_data_status(sc_df=None, comp_df=None, fact_df=None,
                          history=[]):
        """Get the cursor state from dataframes and history.

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
            int, the next cursor state to compute.
        """
        if (len(sc_df) == 0 or len(comp_df) == 0 or len(fact_df) == 0
            or sc_df is None or comp_df is None or fact_df is None):
            return 0
        # No previous sample, or any data frame is damaged.

        max_iter_id = fact_df.iter_id.max()

        filt_ = (fact_df.iter_id == max_iter_id)

        last_df = fact_df[filt_]
        
        # pd.Series must be converted to list before checking 'in'.
        # enum and check are the only 2 modules that may generate
        # new entree.
        if 'NC' in last_df.calc_status.tolist():
            last_nc_df = last_df[last_df.calc_status == 'NC']
            if 'enum' in last_nc_df.module.tolist():
                return (max_iter_id * len(self.modules) +
                        self.modules.index('enum') + 1)
            elif 'check' in last_nc_df.module.tolist():
                return (max_iter_id * len(self.modules) +
                        self.modules.index('check') + 1)
            else:
                raise ValueError("Module other than enumerator or gs " +
                                 "solver inserted entree.")

        if 'CC' in last_df.calc_status.tolist():
            return (max_iter_id * len(self.modules) +
                    self.modules.index('write') + 1)

        if 'CL' in last_df.calc_status.tolist():
            return (max_iter_id * len(self.modules) +
                    self.modules.index('calc') + 1)

        if (len(history) < max_iter_id or
            len(history) > max_iter_id + 1):
            raise ValueError("History record broken! "+
                             "Currently at iteration {}, "
                             .format(max_iter_id)+
                             "but {} history steps found!"
                             .format(len(history)))

        if len(history) == max_iter_id:
            return (max_iter_id * len(self.modules) +
                    self.modules.index('feat') + 1)

        if len(history) == max_iter_id + 1:
            return (max_iter_id * len(self.modules) +
                    self.modules.index('fit') + 1)

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
        c = self.check_data_status(sc_df=sc_df, comp_df=comp_df,
                                   fact_df=fact_df,
                                   history=history)

        if c != self.cursor:
            iter_id = c // len(self.modules)
            module = self.modules[c % len(self.modules)]
            warnings.warn("Next indicated by dataframes: {}, {};"
                          .format(iter_id, module) +
                          " Next by this time keeper: {}, {};"
                          .format(self.iter_id,
                                  self.next_module_todo) +
                          " Resetting time keeper.")

        self.cursor = c

    def set_to_file_status(self,
                           sc_file=SC_FILE,
                           comp_file=COMP_FILE,
                           fact_file=FACT_FILE,
                           ce_history_file=CE_HISTORY_FILE):
        """Set to status indicated by data files.

        Recommend not to use this unless timekeeper is broken.
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

        self.set_to_data_status(sc_df, comp_df, fact_df, history)

    def todo(self, module_name):
        """Check if the specified module has not been done in current iter.

        If it is not done yet, we can still run it in the current iteration.

        Args:
            module_name(str):
                name of the module to check.
        """
        mid = self.modules.index(module_name)

        return self.cursor % len(self.modules) <= mid

    def done(self, module_name):
        """Check if the specified module has been done in current iter.

        Args:
            module_name(str):
                name of the module to check.
        """
        return (not self.todo(module_name))

    def advance(self, n_modules=1):
        """Advance number of modules for the time keeper.

        It is recommended you auto_save immediately after an
        advance.

        Args:
            n_modules(int):
              Number of modules to advance. Default is 1 module.
        """
        self.cursor = self.cursor + n_modules

    def as_dict(self):
        """Serialize into dict.

        Returns:
            Dict.
        """
        return {'cursor': self.cursor,
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
        return cls(d.get('cursor', 0))

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
        if os.path.isfile(time_keeper_file):
            with open(time_keeper_file, 'r') as fin:
                d = json.load(fin)
        else:
            d = {}  # Set a 0 cursor.

        return cls.from_dict(d)
