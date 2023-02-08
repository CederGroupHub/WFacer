"""Fit ECIs from Wrangler."""

import numpy as np
from sklearn.model_selection import cross_val_score

from smol.utils import (class_name_from_str,
                        derived_class_factory,
                        get_subclasses)
from smol.cofe.wrangling.tools import unique_corr_vector_indices

from sparselm.model._base import CVXEstimator
from sparselm.model import OrdinaryLeastSquares, Lasso
from sparselm.model import BestSubsetSelection
from sparselm.model import RegularizedL0
from sparselm.model_selection import GridSearchCV, LineSearchCV

all_optimizers = {"GridSearch": GridSearchCV,
                  "LineSearch": LineSearchCV}
hierarchy_classes = get_subclasses(RegularizedL0)
hierarchy_classes.update({"BestSubsetSelection": BestSubsetSelection})
hierarchy_classes.update(get_subclasses(BestSubsetSelection))


# Model factories for sparse-lm.
def estimator_factory(estimator_name, **kwargs):
    """Get an estimator object from class name.

    Args:
        estimator_name (str):
            Name of the estimator.
        kwargs:
            Other keyword arguments to initialize an estimator.
            Depends on the specific class
    Returns:
        Estimator
    """
    class_name = class_name_from_str(estimator_name)
    return derived_class_factory(class_name, CVXEstimator, **kwargs)


def is_hierarchy_estimator(class_name):
    """Find whether an estimator needs hierarchy.

    Args:
        class_name(str):
            Name of the estimator.
    Returns:
        bool.
    """
    class_name = class_name_from_str(class_name)
    return class_name in hierarchy_classes


# As mentioned in CeDataWrangler, weights does not make much sense and will not be used.
# Also only energy fitting is supported.
def fit_ecis_from_wrangler(wrangler,
                           estimator_name,
                           optimizer_name,
                           param_grid,
                           use_hierarchy=True,
                           center_point_external=True,
                           filter_unique_correlations=True,
                           estimator_kwargs=None,
                           optimizer_kwargs=None,
                           **kwargs):
    """Fit ECIs from a fully processed wrangler.

    Args:
        wrangler(CeDataWrangler):
            A CeDataWrangler storing all training structures.
        estimator_name(str):
            The name of estimator, following the rules in smol.utils.class_name_from_str.
        optimizer_name(str):
            Name of hyperparameter optimizer. Currently, only supports GridSearch and
            LineSearch.
        param_grid(dict|list[tuple]):
            Parameter grid to initialize the optimizer. See docs of sparselm.optimizer.
        use_hierarchy(bool): optional
            Whether to use cluster hierarchy constraints when available. Default to
            true.
        center_point_external(bool): optional
            Whether to fit the point and external terms with linear regression
            first, then fit the residue with regressor. Default to true,
            because this usually greatly improves the decrease of ECI over
            cluster radius.
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
        1D np.ndarray, float, float, float, 1D np.ndarray:
            Fitted coefficients (not ECIs), cross validation error (meV/site),
            standard deviation of CV (meV/site) , RMSE(meV/site)
            and corresponding best parameters.
    """
    space = wrangler.cluster_subspace
    feature_matrix = wrangler.feature_matrix.copy()
    # Corrected and normalized DFT energy in eV/prim.
    normalized_energy = wrangler.get_property_vector("energy", normalize=True)
    point_func_inds = space.function_inds_by_size[1]
    num_point_funcs = len(point_func_inds)
    num_point_orbs = len(space.orbits_by_size[1])
    external_inds = list(range(space.num_corr_functions,
                               space.num_corr_functions +
                               len(space.external_terms)))
    centered_inds = point_func_inds + external_inds
    other_inds = np.setdiff1d(np.arange(1,
                                        space.num_corr_functions +
                                        len(space.external_terms)))
    if filter_unique_correlations:
        unique_inds = unique_corr_vector_indices(wrangler, "energy")
        feature_matrix = feature_matrix[unique_inds, :]
        normalized_energy = normalized_energy[unique_inds]

    # Prepare the estimator.
    # Using function hierarchy instead of orbits hierarchy might not be correct
    # for basis other than indicator can be wrong!
    est_class_name = class_name_from_str(estimator_name)
    estimator_kwargs = estimator_kwargs or {}
    if center_point_external:
        # Here we must set fit_intercept to False because
        # the CE intercept is already subtracted.
        estimator_kwargs["fit_intercept"] = False
    if is_hierarchy_estimator(est_class_name) and use_hierarchy:
        if space.basis_type == "indicator":
            # Use function hierarchy for indicator.
            hierarchy = space.function_hierarchy()
            if center_point_external:
                # Get hierarchy without point terms.
                # minus 1 more to exclude the intercept term.
                hierarchy = [[func_id - num_point_funcs - 1 for func_id in sub]
                             for sub in hierarchy[num_point_funcs + 1:]]
            groups = list(range(space.num_corr_functions +
                                len(space.external_terms)))
            if center_point_external:
                groups = list(range(space.num_corr_functions
                                    - num_point_funcs - 1))
        else:
            # Use orbit hierarchy for other bases.
            hierarchy = space.orbit_hierarchy()
            if center_point_external:
                hierarchy = [[orb_id - num_point_orbs - 1 for orb_id in sub]
                             for sub in hierarchy[num_point_orbs + 1:]]
            groups = np.append(space.function_orbit_ids,
                               np.arange(len(space.external_terms), dtype=int)
                               + space.num_orbits)
            if center_point_external:
                groups = [oid - num_point_funcs - 1
                          for oid in
                          space.function_orbit_ids[num_point_funcs + 1:]]

        estimator = estimator_factory(estimator_name,
                                      groups=groups,
                                      hierarchy=hierarchy,
                                      **estimator_kwargs)
    else:
        estimator = estimator_factory(estimator_name, **estimator_kwargs)

    # Prepare the optimizer.
    if not isinstance(estimator, OrdinaryLeastSquares):
        if ("-cv" not in optimizer_name
                or "-CV" not in optimizer_name
                or "-Cv" not in optimizer_name):
            optimizer_name += "-CV"
        opt_class_name = class_name_from_str(optimizer_name)
        optimizer_kwargs = optimizer_kwargs or {}
        if opt_class_name not in all_optimizers:
            raise ValueError(f"Hyperparameters optimization method"
                             f" {opt_class_name} not implemented!")

        optimizer = all_optimizers[opt_class_name](estimator,
                                                   param_grid,
                                                   **optimizer_kwargs)

        # Perform the optimization and fit.
        if center_point_external:
            # Fit point and external terms first with a lasso.
            lasso = Lasso(alpha=1e-6, fit_intercept=True)
            lasso.fit(feature_matrix[:, centered_inds], normalized_energy)
            residuals = (normalized_energy -
                         lasso.predict(feature_matrix[:, centered_inds]))
            optimizer = optimizer.fit(X=feature_matrix[:, other_inds],
                                      y=residuals,
                                      **kwargs)
            best_coef = np.concatenate([[lasso.intercept_],
                                        lasso.coef_[:-len(space.external_terms)],
                                        optimizer.best_estimator_.coef_,
                                        lasso.coef_[-len(space.external_terms):]])
            # Default sparse-lm scoring has changed to "neg_root_mean_square"
            best_cv = -optimizer.best_score_
            best_cv_std = optimizer.best_score_std_
            best_params = optimizer.best_params_
        else:
            optimizer = optimizer.fit(X=feature_matrix,
                                      y=normalized_energy,
                                      **kwargs)
            best_coef = optimizer.best_estimator_.coef_,
            # Default sparse-lm scoring has changed to "neg_root_mean_square"
            best_cv = -optimizer.best_score_
            best_cv_std = optimizer.best_score_std_
            best_params = optimizer.best_params_

    else:
        cvs = cross_val_score(estimator,
                              X=feature_matrix,
                              y=normalized_energy,
                              scoring="neg_root_mean_squared_error",
                              **optimizer_kwargs)
        estimator = estimator.fit(X=feature_matrix,
                                  y=normalized_energy,
                                  **kwargs)
        best_coef = estimator.coef_
        best_cv = np.average(cvs)
        best_cv_std = np.std(cvs)
        best_params = None

    predicted_energy = np.dot(feature_matrix,
                              best_coef)
    rmse = np.sqrt(np.sum((predicted_energy - normalized_energy) ** 2)
                   / len(normalized_energy))

    # Convert from eV/prim to meV/site.
    n_sites = len(wrangler.cluster_subspace.structure)
    return (best_coef,
            best_cv * 1000 / n_sites,
            best_cv_std * 1000 / n_sites,
            rmse * 1000 / n_sites,
            best_params)
