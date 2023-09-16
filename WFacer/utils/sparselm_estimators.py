"""Utility functions to prepare sparse-lm estimators."""
from warnings import warn

import numpy as np
import sparselm
from sklearn.model_selection import RepeatedKFold
from smol.utils.class_utils import class_name_from_str
from sparselm.model import Lasso, OverlapGroupLasso
from sparselm.model import __all__ as all_estimator_names
from sparselm.model_selection import GridSearchCV, LineSearchCV
from sparselm.stepwise import StepwiseEstimator


def is_subclass(classname, parent_classname):
    """Check whether the estimator is a subclass of some parent.

    Args:
        classname(str):
            Name of the sparselm estimator class.
        parent_classname(str):
            Name of the parent class. Also in sparselm.model.

    Returns:
        bool
    """
    cls = getattr(sparselm.model, classname)
    if hasattr(sparselm.model, parent_classname):
        pcls = getattr(sparselm.model, parent_classname)
    else:
        import sparselm.model._base as base

        if hasattr(base, parent_classname):
            pcls = getattr(base, parent_classname)
        else:
            import sparselm.model._miqp._base as base

            if hasattr(base, parent_classname):
                pcls = getattr(base, parent_classname)
            else:
                raise ValueError(
                    f"sparse-lm does not have parent" f" class {parent_classname}"
                )
    return issubclass(cls, pcls)


# For now, Overlapped group lasso is not supported!
unsupported_parents = [OverlapGroupLasso]
unsupported_estimator_names = []
for name in all_estimator_names:
    for parent_class in unsupported_parents:
        if issubclass(getattr(getattr(sparselm, "model"), name), parent_class):
            unsupported_estimator_names.append(name)
supported_estimator_names = [
    name for name in all_estimator_names if name not in unsupported_estimator_names
]


# smol 0.3.1 cannot correctly identify subclasses in sparse-lm.
# Temporarily writing as import __all__.
def estimator_factory(estimator_name, **kwargs):
    """Get an estimator object from its class name.

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

    if class_name not in supported_estimator_names:
        raise ValueError(
            f"Estimator {class_name} is not supported" f" by sparse-lm automation!"
        )
    cls = getattr(getattr(sparselm, "model"), class_name)
    return cls(**kwargs)


def optimizer_factory(optimizer_name, estimator, param_grid=None, **kwargs):
    """Get an optimizer object from its class name.

    Args:
        optimizer_name (str):
            Name of the optimizer.
        estimator(CVXRegressor):
            An estimator used to initialize the optimizer.
        param_grid(dict|list[tuple]):
            Parameters grid used to initialize the optimizer. Format
            depends on the type of optimizer. See sparselm.model_selection.
        kwargs:
            Other keyword arguments to initialize an optimizer.
            Depends on the specific class
    Returns:
        GridSearchCV or LineSearchCV.
    """
    all_optimizers = {"GridSearchCV": GridSearchCV, "LineSearchCV": LineSearchCV}
    if (
        "-cv" not in optimizer_name
        or "-CV" not in optimizer_name
        or "-Cv" not in optimizer_name
    ):
        optimizer_name += "-CV"
    opt_class_name = class_name_from_str(optimizer_name)
    if opt_class_name not in all_optimizers:
        raise ValueError(
            f"Hyperparameters optimization method" f" {opt_class_name} not implemented!"
        )
    return all_optimizers[opt_class_name](estimator, param_grid, **kwargs)


def prepare_estimator(
    cluster_subspace,
    estimator_name,
    optimizer_name,
    param_grid=None,
    use_hierarchy=True,
    center_point_external=True,
    estimator_kwargs=None,
    optimizer_kwargs=None,
):
    """Prepare an estimator for fitting.

    .. note:: Sample weights are not supported yet.

    Args:
        cluster_subspace(ClusterSubspace):
            A :class:`ClusterSubspace` to expand with.
        estimator_name(str):
            The name of estimator, following the rules in
            :mod:`smol.utils`.
        optimizer_name(str):
            Name of hyperparameter optimizer.
            Currently, only supports :class:`GridSearch` and :class:`LineSearch`.
        param_grid(dict|list[tuple]):
            Parameter grid to initialize the optimizer. See docs of the
            :mod:`sparselm.model_selection` module.
            **Not needed when using** :class:`OrdinaryLeastSquares`.
        use_hierarchy(bool): optional
            Whether to use cluster hierarchy constraints when available. Default to
            true.
        center_point_external(bool): optional
            Whether to fit the point and external terms with linear regression
            first, then fit the residue with regressor. Default to true,
            because this usually greatly improves the decrease of ECI over
            cluster radius.
        estimator_kwargs(dict): optional
            Other keyword arguments to initialize an estimator.
        optimizer_kwargs(dict): optional
            Other keyword arguments to initialize an optimizer.

    Returns:
        Estimator:
            The estimator wrapped up for fitting.
    """
    # Corrected and normalized DFT energy in eV/prim.
    point_func_inds = cluster_subspace.function_inds_by_size[1]
    num_point_funcs = len(point_func_inds)
    num_point_orbs = len(cluster_subspace.orbits_by_size[1])

    # Prepare the estimator.
    # Using function hierarchy instead of orbits hierarchy might not be correct
    # for basis other than indicator can be wrong!
    est_class_name = class_name_from_str(estimator_name)
    estimator_kwargs = estimator_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {}

    # Groups are required, and hierarchy might be as well.
    is_l0 = is_subclass(est_class_name, "MIQPl0")
    # Only groups are required.
    is_group = is_subclass(est_class_name, "GroupLasso")
    # sparse_bound would also be needed.
    is_subset = is_subclass(est_class_name, "BestSubsetSelection")
    is_ols = est_class_name == "OrdinaryLeastSquares"

    # Need argument groups.
    if is_l0 or is_group:
        if cluster_subspace.basis_type == "indicator":
            # Use function hierarchy for indicator.
            hierarchy = cluster_subspace.function_hierarchy()
            if center_point_external:
                # Points and empty are not included in hierarchy.
                hierarchy = [
                    [
                        func_id - num_point_funcs - 1
                        for func_id in sub
                        if func_id - num_point_funcs - 1 >= 0
                    ]
                    for sub in hierarchy[num_point_funcs + 1 :]
                ]
            # groups argument should be a 1d array.
            groups = list(
                range(
                    cluster_subspace.num_corr_functions
                    + len(cluster_subspace.external_terms)
                )
            )
            if center_point_external:
                groups = list(
                    range(cluster_subspace.num_corr_functions - num_point_funcs - 1)
                )
        else:
            # Use orbit hierarchy for other bases.
            hierarchy = cluster_subspace.orbit_hierarchy()
            if center_point_external:
                # Points and empty are not included in hierarchy.
                hierarchy = [
                    [
                        orb_id - num_point_orbs - 1
                        for orb_id in sub
                        if orb_id - num_point_orbs - 1 >= 0
                    ]
                    for sub in hierarchy[num_point_orbs + 1 :]
                ]
            groups = np.append(
                cluster_subspace.function_orbit_ids,
                np.arange(len(cluster_subspace.external_terms), dtype=int)
                + cluster_subspace.num_orbits,
            ).tolist()
            if center_point_external:
                groups = [
                    oid - num_point_orbs - 1
                    for oid in cluster_subspace.function_orbit_ids[
                        num_point_funcs + 1 :
                    ]
                ]

        estimator_kwargs["groups"] = groups
        # Mute hierarchy when not needed.
        if is_l0 and use_hierarchy:
            estimator_kwargs["hierarchy"] = hierarchy

        if is_subset and "sparse_bound" not in estimator_kwargs:
            default_sparse_bound = int(round(0.6 * len(groups)))
            warn(
                f"Estimator class {est_class_name} is a subclass of"
                f" BestSubsetSelection, but argument sparse_bound is"
                f" not specified. Setting to 60% of all available"
                f" correlation function groups by default. In this case,"
                f" will be: {default_sparse_bound}"
            )
            estimator_kwargs["sparse_bound"] = default_sparse_bound

    if center_point_external and not is_ols:
        external_inds = list(
            range(
                cluster_subspace.num_corr_functions,
                cluster_subspace.num_corr_functions
                + len(cluster_subspace.external_terms),
            )
        )
        centered_inds = [0] + point_func_inds + external_inds
        other_inds = np.setdiff1d(
            np.arange(
                cluster_subspace.num_corr_functions
                + len(cluster_subspace.external_terms)
            ),
            centered_inds,
        ).tolist()

        # Only the first center estimator is allowed to fit intercept.
        center_estimator = Lasso(
            alpha=1e-6, fit_intercept=estimator_kwargs.get("fit_intercept", False)
        )
        center_cv = optimizer_kwargs.get("cv", RepeatedKFold(n_splits=5, n_repeats=3))
        center_optimizer = optimizer_factory(
            "grid-search",
            center_estimator,
            {"alpha": 2 ** np.linspace(-30, 2, 17)},
            cv=center_cv,  # Make sure to shuffle before fit.
        )
        estimator_kwargs["fit_intercept"] = False
        main_estimator = estimator_factory(estimator_name, **estimator_kwargs)
        main_optimizer = optimizer_factory(
            optimizer_name, main_estimator, param_grid, **optimizer_kwargs
        )

        stepwise = StepwiseEstimator(
            [("center", center_optimizer), ("main", main_optimizer)],
            (tuple(centered_inds), tuple(other_inds)),
        )
        return stepwise
    elif not is_ols:
        estimator = estimator_factory(estimator_name, **estimator_kwargs)
        return optimizer_factory(
            optimizer_name, estimator, param_grid, **optimizer_kwargs
        )
    # OLS does not require optimization and centering.
    else:
        return estimator_factory(estimator_name, **estimator_kwargs)
