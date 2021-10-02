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
from smol.cofe.wrangling.tools import (weights_energy_above_composition,
                                       weights_energy_above_hull)

# Will have more regressions when they are published.
from .regression import OLSEstimator, LassoEstimator
from .data_manager import DataManager


class CEFitter(MSONable):
    """Cluster expansion fitting class."""

    supported_weights = ('unweighted', 'e_above_hull', 'e_above_comp')
    supported_regressions = {'ols': OLSEstimator,
                             'lasso': LassoEstimator}

    def __init__(self, data_manager, history_wrapper):
        """Initialization.

        Args:
            data_manager(DataManager):
                The datamanager object to socket enumerated data.
            history_wrapper(HistoryWrapper):
                Wrapper containing previous CE fits.
        """
        self._dm = data_manager
        self.cspc = self._dm.subspace

        regression_flavor = (self.inputs_wrapper.fitter_options
                             .get('regression_flavor'))
        if regression_flavor not in self.supported_regressions:
            raise ValueError("Regression type {} not supported!"
                             .format(regression_flavor))

        if weights_flavor not in self.supported_weights:
            raise ValueError("Weighting method {} not supported!"
                             .format(weights_flavor))
        self.weights_flavor = weights_flavor

        reg_name = self.supported_regressions[regression_flavor]
        self._estimator = reg_name()
        self.estimator_params = (self.inputs_wrapper.fitter_options
                                 .get('regression_params'))

        # TODO: may remove hierarchy for release branch.
        self.use_hierarchy = (self.inputs_wrapper.fitter_options
                              .get('use_hierarchy'))
        if self.use_hierarchy:
            self.hierarchy = self.cspc.bit_combo_hierarchy(invert=True)
        else:
            self.hierarchy = None
 
        self.weighter_params = (self.inputs_wrapper.fitter_options
                                .get('weighter_params'))

        self._hw = history_wrapper

    @property
    def inputs_wrapper(self):
        """InputsWrapper object."""
        return self._dm._iw

    @property
    def history_wrapper(self):
        """HistoryWrapper."""
        return self._hw

    @property
    def sc_df(self):
        """Supercell dataframe."""
        return self._dm.sc_df

    @property
    def comp_df(self):
        """Composition dataframe."""
        return self._dm.comp_df

    @property
    def fact_df(self):
        """Fact dataframe."""
        return self._dm.fact_df

    def fit(self):
        """Fits cluster expansion from the featurized fact dataframe.

        No return value. Only fits and stores energy cluster expansion.
        Check your TimeKeeper, and don't dupe run modules in one cycle!
        """
        fact_df = self.fact_df.copy()

        fact_avail = fact_df[fact_df.calc_status == 'SC']

        # Computes weights.
        if self.weights_flavor == 'unweighted':
            weights = None
        else:
            fact_w_strs = self._dm.fact_df_with_structures.copy()
            fact_w_strs_avail = fact_w_strs[fact_w_strs.calc_status == 'SC']
            structs = fact_w_strs.map_str
            sc_sizes = fact_w_strs.matrix.map(lambda m:
                                              round(abs(np.linalg.det(m))))
            # Requires un-normalized energies.
            energies = fact_w_strs.e_prim * sc_sizes
            if self.weights_flavor == 'e_above_comp':
                weights = (weights_energy_above_composition(structs, energies,
                           **self.weighter_params))
            elif self.weights_flavor=='e_above_hull':
                weights = weights_energy_above_hull(structs, energies,
                                                    self.cspc.structure,
                                                    **self.weighter_params)

        femat = fact_avail.map_corr.tolist()
        X = np.array(femat)
        #Fit energy coefficients
        y = np.array(fact_avail.e_prim)


        _ = self._estimator.fit(X, y, sample_weight = weights,
                                hierarchy=self.hierarchy,
                                **self.estimator_params)
            
        preds = self._estimator.predict(X)
        _coefs = self._estimator.coef_.copy().tolist()
        cvs = self._estimator.calc_cv_score(X, y, sample_weight=weights,
                                            hierarchy=self.hierarchy,
                                            **self.estimator_params)
        _cv = np.sqrt((1 - cvs) * np.sum((y - np.average(y)) ** 2) /
                      len(y))
        _rmse = np.sqrt(np.sum((np.array(preds) - y) ** 2) / len(y))

        self._hw.update({"coefs": _coefs, "cv": _cv, "rmse": _rmse})

# Will not load and save these.
