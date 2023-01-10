"""Fit ECIs from Wrangler."""

import numpy as np
from sklearn.model_selection import cross_val_score

from smol.utils import class_name_from_str, derived_class_factory, get_subclasses

from sparselm.model.base import Estimator
from sparselm.model.ols import OrdinaryLeastSquares
from sparselm.model.miqp.best_subset import BestSubsetSelection
from sparselm.model.miqp.regularized_l0 import MixedL0
from sparselm.optimizer import GridSearch, LineSearch


all_optimizers = {"GridSearch": GridSearch,
                  "LineSearch": LineSearch}
hierarchy_classes = get_subclasses(MixedL0)
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
    return derived_class_factory(class_name, Estimator, **kwargs)


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
    # TODO: using function hierarchy instead of orbits hierarchy might not be correct
    #  for basis other than indicator can be wrong! Currently, sparse-lm can only use
    #  cluster hierarchy!!!
    est_class_name = class_name_from_str(estimator_name)
    estimator_kwargs = estimator_kwargs or {}
    if est_class_name in hierarchy_classes and use_hierarchy:
        hierarchy = space.function_hierarchy()  # Need a better case-study in the future!
        estimator = estimator_factory(estimator_name,
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
