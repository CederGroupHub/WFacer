"""Fit ECIs from Wrangler."""
from warnings import warn

import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score
from smol.cofe.wrangling.tools import unique_corr_vector_indices
from sparselm.model import OrdinaryLeastSquares
from sparselm.stepwise import StepwiseEstimator

from .utils.sparselm_estimators import prepare_estimator


# As mentioned in CeDataWrangler, weights does not make much
# sense and will not be used. Also, only fitting with energy is
# supported.
def fit_ecis_from_wrangler(
    wrangler,
    estimator_name,
    optimizer_name,
    param_grid,
    use_hierarchy=True,
    center_point_external=None,
    filter_unique_correlations=True,
    estimator_kwargs=None,
    optimizer_kwargs=None,
    **kwargs,
):
    """Fit ECIs from a fully processed wrangler.

    No weights will be used.
    Args:
        wrangler(CeDataWrangler):
            A CeDataWrangler storing all training structures.
        estimator_name(str):
            The name of estimator, following the rules in
            smol.utils.class_name_from_str.
        optimizer_name(str):
            Name of hyperparameter optimizer. Currently, only supports GridSearch and
            LineSearch.
        param_grid(dict|list[tuple]):
            Parameter grid to initialize the optimizer. See docs of
            sparselm.model_selection.
        use_hierarchy(bool): optional
            Whether to use cluster hierarchy constraints when available. Default to
            true.
        center_point_external(bool): optional
            Whether to fit the point and external terms with linear regression
            first, then fit the residue with regressor. Default to None, which means
            when the feature matrix is full rank, will not use centering, otherwise
            centers. If set to True, will force centering, but use at your own risk
            because this may cause very large CV. If set to False, will never use
            centering.
        filter_unique_correlations(bool):
            If the wrangler have structures with duplicated correlation vectors,
            whether to fit with only the one with the lowest energy.
            Default to True.
        estimator_kwargs(dict): optional
            Other keyword arguments to initialize an estimator.
        optimizer_kwargs(dict): optional
            Other keyword arguments to initialize an optimizer.
        kwargs:
            Keyword arguments used by estimator._fit. For example, solver arguments.
    Returns:
        Estimator, 1D np.ndarray, float, float, float, 1D np.ndarray:
            Fitted estimator, coefficients (not ECIs), cross validation error (meV/site),
            standard deviation of CV (meV/site) , RMSE(meV/site)
            and corresponding best parameters.
    """
    space = wrangler.cluster_subspace
    feature_matrix = wrangler.feature_matrix.copy()
    # Corrected and normalized DFT energy in eV/prim.
    normalized_energy = wrangler.get_property_vector("energy", normalize=True)
    if filter_unique_correlations:
        unique_inds = unique_corr_vector_indices(wrangler, "energy")
        feature_matrix = feature_matrix[unique_inds, :]
        normalized_energy = normalized_energy[unique_inds]

    # Prepare the estimator. If do centering, will return a stepwise estimator.
    estimator_kwargs = estimator_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {}

    # Set default cv splitter to shuffle rows.
    cv = optimizer_kwargs.get("cv")
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=3)
    elif isinstance(cv, int):
        cv = RepeatedKFold(n_splits=cv)

    optimizer_kwargs["cv"] = cv

    # Check if the matrix is full rank. Do not apply centering when doing full-rank.
    n_features = feature_matrix.shape[1]
    if wrangler.get_feature_matrix_rank() >= n_features:
        if center_point_external:
            warn(
                "The handled feature matrix is full rank, but center_point_external"
                " is forced to be true! Use at your own risk as this might result in"
                " very large fitting error!"
            )
        if center_point_external is None:
            center_point_external = False
    elif center_point_external is None:
        center_point_external = True

    estimator = prepare_estimator(
        space,
        estimator_name,
        optimizer_name,
        param_grid,
        use_hierarchy=use_hierarchy,
        center_point_external=center_point_external,
        estimator_kwargs=estimator_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )
    # Prepare the optimizer.
    is_stepwise = isinstance(estimator, StepwiseEstimator)
    is_ols = isinstance(estimator, OrdinaryLeastSquares)

    # Perform the optimization and fit.
    if not is_ols:
        estimator = estimator.fit(X=feature_matrix, y=normalized_energy, **kwargs)
        # StepwiseEstimator
        if is_stepwise:
            best_coef = estimator.coef_
            # Add intercept to the first coefficient.
            best_coef[0] += estimator.intercept_
            # Default sparse-lm scoring has changed to "neg_root_mean_square"
            best_cv = -estimator.steps[-1][1].best_score_
            best_cv_std = estimator.steps[-1][1].best_score_std_
            best_params = estimator.steps[-1][1].best_params_
        # Searcher.
        else:
            best_coef = estimator.best_estimator_.coef_
            # Add intercept to the first coefficient.
            best_coef[0] += estimator.best_estimator_.intercept_
            # Default sparse-lm scoring has changed to "neg_root_mean_square"
            best_cv = -estimator.best_score_
            best_cv_std = estimator.best_score_std_
            best_params = estimator.best_params_
    else:
        # Set default CV splitter.
        cvs = cross_val_score(
            estimator,
            X=feature_matrix,
            y=normalized_energy,
            scoring="neg_root_mean_squared_error",
            **optimizer_kwargs,
        )
        estimator = estimator.fit(X=feature_matrix, y=normalized_energy, **kwargs)
        best_coef = estimator.coef_
        best_coef[0] += estimator.intercept_
        best_cv = -np.average(cvs)  # negative rmse.
        best_cv_std = np.std(cvs)
        best_params = None

    predicted_energy = np.dot(feature_matrix, best_coef)
    rmse = np.sqrt(
        np.sum((predicted_energy - normalized_energy) ** 2) / len(normalized_energy)
    )

    # Convert from eV/prim to meV/site.
    n_sites = len(wrangler.cluster_subspace.structure)
    return (
        estimator,
        best_coef,
        best_cv * 1000 / n_sites,
        best_cv_std * 1000 / n_sites,
        rmse * 1000 / n_sites,
        best_params,
    )
