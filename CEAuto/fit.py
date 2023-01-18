"""Fit ECIs from Wrangler."""

import numpy as np
from sklearn.model_selection import cross_val_score

from smol.utils import class_name_from_str, derived_class_factory, get_subclasses

from sparselm.model._base import CVXEstimator
from sparselm.model import OrdinaryLeastSquares
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
        estimator_kwargs(dict): optional
            Other keyword arguments to initialize an estimator.
        optimizer_kwargs(dict): optional
            Other keyword arguments to initialize an optimizer.
        kwargs:
            Keyword arguments used by estimator._fit. For example, solver arguments.
    Returns:
        1D np.ndarray, float, float, 1D np.ndarray:
            Fitted coefficients (not ECIs), cross validation error (eV/site),
            standard deviation of CV (eV/site) ,and corresponding best parameters.
    """
    space = wrangler.cluster_subspace
    feature_matrix = wrangler.feature_matrix
    # Corrected and normalized DFT energy in eV/prim.
    normalized_energy = wrangler.get_property_vector("energy", normalize=True)

    # Prepare the estimator.
    # Using function hierarchy instead of orbits hierarchy might not be correct
    # for basis other than indicator can be wrong!
    est_class_name = class_name_from_str(estimator_name)
    estimator_kwargs = estimator_kwargs or {}
    if is_hierarchy_estimator(est_class_name) and use_hierarchy:
        if space.basis_type == "indicator":
            # Use function hierarchy for indicator.
            hierarchy = space.function_hierarchy()
            groups = list(range(space.num_corr_functions +
                                len(space.external_terms)))
        else:
            # Use orbit hierarchy for other.
            hierarchy = space.orbit_hierarchy()
            groups = np.append(space.function_orbit_ids,
                               np.arange(len(space.external_terms), dtype=int)
                               + space.num_corr_functions)

        estimator = estimator_factory(estimator_name,
                                      groups=groups,
                                      hierarchy=hierarchy,
                                      **estimator_kwargs)
    else:
        estimator = estimator_factory(estimator_name, **estimator_kwargs)

    # Prepare the optimizer.
    if not isinstance(estimator, OrdinaryLeastSquares):
        opt_class_name = class_name_from_str(optimizer_name)
        optimizer_kwargs = optimizer_kwargs or {}
        if opt_class_name not in all_optimizers:
            raise ValueError(f"Hyperparameters optimization method"
                             f" {opt_class_name} not implemented!")
        optimizer = all_optimizers[opt_class_name](estimator,
                                                   param_grid,
                                                   **optimizer_kwargs)

        # Perform the optimization and fit.
        optimizer = optimizer.fit(X=feature_matrix,
                                  y=normalized_energy,
                                  **kwargs)
        best_coef = optimizer.best_estimator_.coef_
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

    return best_coef, best_cv, best_cv_std, best_params
