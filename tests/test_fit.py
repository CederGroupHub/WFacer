"""Test the fitting functions."""
import cvxpy as cp
import numpy as np

from AceCE.fit import fit_ecis_from_wrangler


# Only test a single small-sized wrangler, because fitting can take long.
def test_fit_ecis_indicator(single_wrangler):
    # e = single_wrangler.get_property_vector("energy")

    # TODO: May need to mute L2L0 tests or change to Lasso-only
    #  because gurobi license is not available in testing suites.
    #  Do this if github auto test cannot work.

    # print("feature matrix:", single_wrangler.feature_matrix)
    # print("energies:", e)
    # Centering, L0L2. Not very good as all the other coefficients
    # can be suppressed to 0.
    if "GUROBI" in cp.installed_solvers():
        solver = "GUROBI"  # gurobi might require license to test.
    else:
        solver = "ECOS_BB"

    grid = [
        ("eta", (2 ** np.linspace(-20, 4, 25)).tolist()),
        ("alpha", [0] + (2 ** np.linspace(-30, 0, 16)).tolist()),
    ]
    best_coef, best_cv, best_cv_std, rmse, best_params = fit_ecis_from_wrangler(
        single_wrangler,
        "l2-l0",
        "line-search",
        grid,
        optimizer_kwargs={
            "n_iter": 3,
            "opt_selection_method": ["max_score", "max_score"],
        },
        estimator_kwargs={"solver": solver},
    )
    assert len(best_coef) == single_wrangler.feature_matrix.shape[1]
    assert best_cv >= -1e-8
    assert best_cv_std >= -1e-8
    assert rmse >= 0
    assert best_params is not None
    for param in best_params:
        assert "__" not in param  # parameters are clean.
    assert np.any(np.isclose(best_params["eta"], grid[0][1]))
    assert np.any(np.isclose(best_params["alpha"], grid[1][1]))

    e_predict = np.dot(single_wrangler.feature_matrix, best_coef)
    e = single_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.9
    # print("\nTest centered L0L2:")
    # print("fitted coefficients:", best_coef)
    # print("fitted cv:", best_cv)
    # print("fitted rmse:", rmse)
    # print("fitted parameters:", best_params)
    # print("predicted energy:", e_predict)
    # print("r2 score of prediction:", r2)

    # No centering, L0L2. L0L2 without centering does really awful
    # as it aggressively suppresses all params to 0.
    grid = [
        ("eta", (2 ** np.linspace(-20, 4, 25)).tolist()),
        ("alpha", [0] + (2 ** np.linspace(-30, 0, 16)).tolist()),
    ]
    best_coef, best_cv, best_cv_std, rmse, best_params = fit_ecis_from_wrangler(
        single_wrangler,
        "l2-l0",
        "line-search",
        grid,
        center_point_external=False,
        optimizer_kwargs={
            "n_iter": 3,
            "opt_selection_method": ["max_score", "max_score"],
        },
        estimator_kwargs={"solver": solver},
    )
    assert len(best_coef) == single_wrangler.feature_matrix.shape[1]
    e_predict = np.dot(single_wrangler.feature_matrix, best_coef)
    e = single_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    # print("\nTest non-centered L0L2:")
    # print("fitted coefficients:", best_coef)
    # print("fitted cv:", best_cv)
    # print("fitted rmse:", rmse)
    # print("fitted parameters:", best_params)
    # print("predicted energy:", e_predict)
    # print("r2 score of prediction:", r2)
    assert r2 >= 0.9

    # No filtering, L0L2.
    grid = [
        ("eta", (2 ** np.linspace(-20, 4, 25)).tolist()),
        ("alpha", [0] + (2 ** np.linspace(-30, 0, 16)).tolist()),
    ]
    best_coef, best_cv, best_cv_std, rmse, best_params = fit_ecis_from_wrangler(
        single_wrangler,
        "l2-l0",
        "line-search",
        grid,
        filter_unique_correlations=False,
        optimizer_kwargs={
            "n_iter": 5,
            "opt_selection_method": ["max_score", "max_score"],
        },
        estimator_kwargs={"solver": solver},
    )
    e_predict = np.dot(single_wrangler.feature_matrix, best_coef)
    e = single_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.8
    # print("\nTest non-filtered L0L2:")
    # print("fitted coefficients:", best_coef)
    # print("fitted cv:", best_cv)
    # print("fitted rmse:", rmse)
    # print("fitted parameters:", best_params)
    # print("predicted energy:", e_predict)
    # print("r2 score of prediction:", r2)

    # Filtered, OLS. (overfit domain so cv >= rmse is correct.)
    best_coef1, best_cv1, best_cv_std1, rmse1, best_params1 = fit_ecis_from_wrangler(
        single_wrangler, "ordinary-least-squares", "what-ever", {}
    )
    assert best_cv >= -1e-8
    assert best_cv_std >= -1e-8
    assert (best_cv - rmse1) / rmse1 > -0.1  # Slack style tolerance.
    e_predict = np.dot(single_wrangler.feature_matrix, best_coef1)
    e = single_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.9
    assert best_params1 is None
    # print("\nTest OLS:")
    # print("fitted coefficients:", best_coef1)
    # print("fitted cv:", best_cv1)
    # print("fitted rmse:", rmse1)
    # print("fitted parameters:", best_params1)
    # print("predicted energy:", e_predict)
    # print("r2 score of prediction:", r2)

    # Not filtered, OLS. Should have larger fitting error than filtered ones.
    best_coef2, best_cv2, best_cv_std2, rmse2, best_params2 = fit_ecis_from_wrangler(
        single_wrangler,
        "ordinary-least-squares",
        "what-ever",
        {},
        filter_unique_correlations=False,
    )
    e_predict = np.dot(single_wrangler.feature_matrix, best_coef2)
    e = single_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.9
    assert rmse2 >= rmse1 - 1e-8
    # print("\nTest unfiltered OLS:")
    # print("fitted coefficients:", best_coef2)
    # print("fitted cv:", best_cv2)
    # print("fitted rmse:", rmse2)
    # print("fitted parameters:", best_params2)
    # print("predicted energy:", e_predict)
    # print("r2 score of prediction:", r2)

    # Center or not should not affect OLS greatly?
    best_coef3, best_cv3, best_cv_std3, rmse3, best_params3 = fit_ecis_from_wrangler(
        single_wrangler,
        "ordinary-least-squares",
        "what-ever",
        {},
        center_point_external=False,
    )
    # npt.assert_array_almost_equal(best_coef3, best_coef1)
    assert abs(rmse3 - rmse1) / rmse1 <= 0.2
    # print("\nTest uncentered OLS:")
    # print("fitted coefficients:", best_coef3)
    # print("fitted cv:", best_cv3)
    # print("fitted rmse:", rmse3)
    # print("fitted parameters:", best_params3)
    # print("predicted energy:", e_predict)

    # Centering, lasso.
    grid = {"alpha": (2 ** np.linspace(-20, 4, 25)).tolist()}
    best_coef, best_cv, best_cv_std, rmse, best_params = fit_ecis_from_wrangler(
        single_wrangler,
        "lasso",
        "grid-search",
        grid,
        optimizer_kwargs={"opt_selection_method": "max_score"},
    )
    assert len(best_coef) == single_wrangler.feature_matrix.shape[1]
    e_predict = np.dot(single_wrangler.feature_matrix, best_coef)
    e = single_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.9
    # print("\nTest LASSO:")
    # print("fitted coefficients:", best_coef)
    # print("fitted cv:", best_cv)
    # print("fitted rmse:", rmse)
    # print("fitted parameters:", best_params)
    # print("predicted energy:", e_predict)
    # print("r2 score of prediction:", r2)
    # assert False


def test_fit_ecis_sinusoid(single_wrangler_sin):
    # Centering, lasso.
    grid = {"alpha": (2 ** np.linspace(-20, 4, 25)).tolist()}
    best_coef, best_cv, best_cv_std, rmse, best_params = fit_ecis_from_wrangler(
        single_wrangler_sin,
        "lasso",
        "grid-search",
        grid,
        optimizer_kwargs={"opt_selection_method": "one_std_score"},
    )
    assert len(best_coef) == single_wrangler_sin.feature_matrix.shape[1]
    e_predict = np.dot(single_wrangler_sin.feature_matrix, best_coef)
    e = single_wrangler_sin.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    # prediction error per atom should not be too large.
    assert r2 >= 0.9
