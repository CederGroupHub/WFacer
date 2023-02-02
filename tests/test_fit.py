"""Test the fitting functions."""
import pytest
import numpy as np
import numpy.testing as npt

from CEAuto.fit import (estimator_factory,
                        is_hierarchy_estimator,
                        fit_ecis_from_wrangler)

from sparselm.model import __all__ as all_estimator_names


@pytest.mark.parametrize(params=all_estimator_names)
def test_estimator_factory(request):
    if is_hierarchy_estimator(request.param):
        groups = [[i] for i in range(10)]
        estimator = estimator_factory(request.param,
                                      groups=groups)
    else:
        estimator = estimator_factory(request.param)
    assert estimator.__class__.__name__ == request.param


def test_fit_ecis_indicator(data_wrangler):
    # Centering, L0L2.
    grid = [("alpha", [0] + (2 ** np.linspace(-30, 0, 16)).tolist()),
            ("eta", (2 ** np.linspace(-20, 4, 25)).tolist())]
    best_coef, best_cv, best_cv_std, rmse, best_params\
        = fit_ecis_from_wrangler(data_wrangler,
                                 "l2-l0",
                                 "line-search",
                                 grid,
                                 optimizer_kwargs
                                 ={"n_iter": 3})
    assert len(best_coef) == data_wrangler.feature_matrix.shape[1]
    e_predict = np.dot(data_wrangler.feature_matrix,
                       best_coef)
    e = data_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.8

    # No centering, L0L2.
    grid = [("alpha", [0] + (2 ** np.linspace(-30, 0, 16)).tolist()),
            ("eta", (2 ** np.linspace(-20, 4, 25)).tolist())]
    best_coef, best_cv, best_cv_std, rmse, best_params\
        = fit_ecis_from_wrangler(data_wrangler,
                                 "l2-l0",
                                 "line-search",
                                 grid,
                                 center_point_external=False,
                                 optimizer_kwargs
                                 ={"n_iter": 3})
    e_predict = np.dot(data_wrangler.feature_matrix,
                       best_coef)
    e = data_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.8

    # No filtering, L0L2.
    grid = [("alpha", [0] + (2 ** np.linspace(-30, 0, 16)).tolist()),
            ("eta", (2 ** np.linspace(-20, 4, 25)).tolist())]
    best_coef, best_cv, best_cv_std, rmse, best_params\
        = fit_ecis_from_wrangler(data_wrangler,
                                 "l2-l0",
                                 "line-search",
                                 grid,
                                 filter_unique_correlations=False,
                                 optimizer_kwargs
                                 ={"n_iter": 3})
    e_predict = np.dot(data_wrangler.feature_matrix,
                       best_coef)
    e = data_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.8

    # Filtered, OLS.
    best_coef1, best_cv1, best_cv_std1, rmse1, best_params1\
        = fit_ecis_from_wrangler(data_wrangler,
                                 "ordinary-least-squares",
                                 "what-ever",
                                 {})
    e_predict = np.dot(data_wrangler.feature_matrix,
                       best_coef1)
    e = data_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.6
    assert best_params1 is None

    # Not filtered, OLS. Should have larger fitting error than filtered ones.
    best_coef2, best_cv2, best_cv_std2, rmse2, best_params2\
        = fit_ecis_from_wrangler(data_wrangler,
                                 "ordinary-least-squares",
                                 "what-ever",
                                 {},
                                 filter_unique_correlations=False,
                                 )
    e_predict = np.dot(data_wrangler.feature_matrix,
                       best_coef2)
    e = data_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.6
    assert rmse2 >= rmse1

    # Center or not should not affect OLS at all.
    best_coef3, best_cv3, best_cv_std3, rmse3, best_params3\
        = fit_ecis_from_wrangler(data_wrangler,
                                 "ordinary-least-squares",
                                 "what-ever",
                                 {},
                                 center_point_external=False,
                                 )
    npt.assert_array_almost_equal(best_coef3, best_coef1)
    assert np.isclose(rmse3, rmse1)
