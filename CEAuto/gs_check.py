"""Ground state scanner that checks grounds states for convergence.

THIS CLASS DOES NOT CHANGE DATA TABLES!
"""
__author__ = "Fengyu Xie"


import json
import numpy as np
import pandas as pd

from smol.cofe import ClusterSubspace

from .utils.hull_utils import hulls_match
from .data_manager import DataManager


class GSChecker:
    """A ground state checker class. 

    Check for canonical ground state energies convergence.
    """
    def __init__(self, data_manager, history_wrapper):
        """Initialize.

        Args:
            data_manager(DataManager):
                The datamanager object to socket enumerated data.
            history_wrapper(HistoryWrapper):
                Wrapper containing previous CE fits.
        """
        self._dm = data_manager
        self._hw = history_wrapper
 
        self.e_tol_in_cv = (self.inputs_wrapper.gs_checker_options
                            .get('e_tol_in_cv'))
        self.cv_change_tol = (self.inputs_wrapper.gs_checker_options
                              .get('cv_change_tol'))

        self._sc_table = self._dm.sc_df
        self._comp_table = self._dm.comp_df
        self._fact_table = self._dm.fact_df
        self._dft_hulls_ahead = {}  # Will not be saved
        self._ce_hulls_ahead = {}

    @property
    def inputs_wrapper(self):
        """InputsWrapper."""
        return self._dm._iw

    @property
    def iter_id(self):
        """Current iteration id."""
        return len(self._hw.history) - 1

    def get_hull_ahead(self, n_it_ahead=0, mode='dft'):
        """Gets a minimum energy hull.

        Args:
            n_it_ahead(int):
                Number of iterations ahead.
                Default is 0, meaning current hull.
            mode(str):
                Type of hull to compute. Can be either 'dft' or 'ce'.
                Default is 'dft'.
        Returns:
            pd.DataFrame:
                Containing composition indices and minimum energies of a
                composition.
        """
        if mode == 'dft':
            if n_it_ahead in self._dft_hulls_ahead:
                return self._dft_hulls_ahead[n_it_ahead]
        if mode == 'ce':
            if n_it_ahead in self._ce_hulls_ahead:
                return self._ce_hulls_ahead[n_it_ahead]

        filt_ = ((self._fact_table.iter_id <= self.iter_id - n_it_ahead) &
                 (self._fact_table.calc_status == 'SC') &
                 (~self._fact_table.e_prim.isna()))

        if filt_.sum() == 0:
            # Might be the first iteration, or fact_table is empty
            return None

        fact_prev = self._fact_table[filt_]

        if mode == 'ce':
            coef_prev = self._hw.history[- (n_it_ahead + 1)]['coefs']
            fact_prev['e_ce'] = (np.array(fact_prev.map_corr.tolist()) @
                                 np.array(coef_prev))

            # If multiple GSs have the same energy, only one will be taken.
            _prev_hull = (fact_prev.groupby('comp_id')
                          .agg(lambda df: df.loc[df['e_ce'].idxmin()])
                          .reset_index())
            _prev_hull = _prev_hull.loc[:, ['comp_id', 'e_ce']]
            _prev_hull = _prev_hull.merge(self._comp_table, how='left',
                                          on='comp_id')
            _prev_hull = _prev_hull.rename(columns={'e_ce': 'e_prim'})
            self._ce_hulls_ahead[n_it_ahead] = _prev_hull.copy()
            return self._ce_hulls_ahead[n_it_ahead]

        if mode == 'dft':
            _prev_hull = (fact_prev.groupby('comp_id')
                          .agg(lambda df: df.loc[df['e_prim'].idxmin()])
                          .reset_index())
            _prev_hull = _prev_hull.loc[:,['comp_id', 'e_prim']]
            _prev_hull = _prev_hull.merge(self._comp_table, how='left',
                                          on='comp_id')
            self._dft_hulls_ahead[n_it_ahead] = _prev_hull.copy()
            return self._dft_hulls_ahead[n_it_ahead]

    @property
    def prev_ce_hull(self):
        """Gets previous minimum CE energy hull.

        Returns:
            A pd.DataFrame, containing composition indices and minimum
            energies of a composition.
        """
        return self.get_hull_ahead(n_it_ahead=1, mode='ce')

    @property
    def curr_ce_hull(self):
        """Gets current minimum CE energy hull.

        Returns:
            A pd.DataFrame, containing composition indices and minimum
            energies of a composition.
        """   
        return self.get_hull_ahead(n_it_ahead=0, mode='ce')
 
    @property
    def prev_dft_hull(self):
        """Gets previous minimum DFT energy hull.

        Returns:
            A pd.DataFrame, containing composition indices and minimum
            energies of a composition.
        """
        return self.get_hull_ahead(n_it_ahead=1, mode='dft')


    @property
    def curr_dft_hull(self):
        """Gets current minimum DFT energy hull.

        Returns:
            A pd.DataFrame, containing composition indices and minimum
            energies of a composition.
        """
        return self.get_dft_hull_ahead(n_it_ahead=0, mode='dft')

    def check_convergence(self):
        """Checks convergence with last iteration.

        Both CE and DFT energies will be checked.
        Returns:
            bool.
        """
        if self.prev_ce_hull is None or self.curr_ce_hull is None or
           self.prev_dft_hull is None or self.curr_dft_hull is None:
            return False

        cv = self._hw.history[-1].get('cv',  0.001)
        cv_1 = self._hw.history[-2].get('cv', 0.001)

        return (hulls_match(self.prev_ce_hull, self.curr_ce_hull,
                            e_tol=self.e_tol_in_cv * cv) and
                hulls_match(self.prev_dft_hull, self.curr_dft_hull,
                            e_tol=self.e_tol_in_cv * cv) and
                abs(cv_1 - cv) / cv_1 < self.cv_change_tol)
        # change of cv < 20 % in last 2 iterations.
