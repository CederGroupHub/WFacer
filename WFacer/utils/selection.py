"""Structure selection methods from feature matrices."""

from warnings import warn

import numpy as np


def select_initial_rows(
    femat, n_select=10, method="leverage", num_external_terms=0, keep_indices=None
):
    """Select structures to initialize an empty CE project.

    Args:
        femat(2D arrayLike):
            Correlation vectors of each structure.
        n_select(int): optional
            Number of structures to select. Default is 10.
        method(str): optional
            The method used to select structures. Default is
            base on leverage score, which minimizes the Frobenius
            norm difference between the covariance matrix of all
            structures and the covariance matrix of a selection.
            "random" is also supported.
        num_external_terms(int): optional
            Number of external terms in cluster subspace. These
            terms should not be compared in a structure selection.
        keep_indices(list of int): optional
            Indices of structures that must be selected. Usually
            those of important ground state structures.

    Returns:
        list of int:
            Indices of selected rows in the feature matrix,
            corresponding to the selected structures.
    """
    # Leave out external terms.
    a = np.array(femat)[:, : len(femat[0]) - num_external_terms]
    n, d = a.shape

    if keep_indices is None:
        keep_indices = []
    n_keep = len(keep_indices)
    if n_keep > n:
        raise ValueError("Can not keep more structures than provided!")
    if n_keep > n_select:
        warn(
            "Keeping more structures than to be selected!"
            " Cannot select new structures."
        )
        return keep_indices
    if n_select > n:
        warn(
            "Structures to select more than provided,"
            " will select all provided structures."
        )
        return list(range(n))
    dn = n_select - n_keep

    selected_indices = np.array(keep_indices, dtype=int)
    available_indices = np.setdiff1d(np.arange(n, dtype=int), keep_indices)

    cov = a.T @ a  # Covariance matrix of features.

    for _ in range(dn):
        if method == "leverage":
            errs = []
            for trial_index in available_indices:
                trial_indices = np.append(selected_indices, trial_index)
                a_trial = a[trial_indices, :]
                cov_trial = a_trial.T @ a_trial
                errs.append(np.sum((cov - cov_trial) ** 2))

            select_index = available_indices[np.argmin(errs)]

        elif method == "random":
            select_index = np.random.choice(available_indices)

        else:
            raise NotImplementedError

        selected_indices = np.append(selected_indices, select_index)
        available_indices = np.setdiff1d(available_indices, [select_index])

    return selected_indices.tolist()


# TODO: implement composition dependent domain matrices. (not urgent)
def select_added_rows(
    femat,
    old_femat,
    n_select=10,
    method="leverage",
    keep_indices=None,
    num_external_terms=0,
    domain_matrix=None,
):
    """Select structures to add to an existing CE project.

    We select structures by minimizing the leverage score under a
    certain domain matrix, or fully at random.
    Refer to `T. Mueller et al. <https://doi.org/10.1103/PhysRevB.82.184107>`_

    Args:
        femat(2D arraylike):
            Correlation vectors of new structures.
        old_femat(2D arraylike):
            Existing old feature matrix.
        n_select(int): optional
            Number of structures to select. Default is 10.
        method(str): optional
            The method used to select structures. Default is
            by maximizing leverage score reduction ("leverage").
            "random" is also supported.
        keep_indices(list of int): optional
            Indices of structures that must be selected. Usually
            those of important ground state structures.
        num_external_terms(int): optional
            Number of external terms in cluster subspace. These
            terms should not be compared in a structure selection.
        domain_matrix(2D arraylike): optional
            The domain matrix used to compute leverage score. By
            default, we use an identity matrix.

    Returns:
        list of int:
            Indices of selected rows in the feature matrix,
            corresponding to the selected structures.
    """
    # Leave out external terms.
    a = np.array(femat)[:, : len(femat[0]) - num_external_terms]
    old_a = np.array(old_femat)[:, : len(old_femat[0]) - num_external_terms]
    n, d = a.shape
    if domain_matrix is None:
        domain_matrix = np.eye(d)
    domain_matrix = np.array(domain_matrix)
    # Using an identity matrix as the domain matrix by default.

    if keep_indices is None:
        keep_indices = []
    n_keep = len(keep_indices)
    if n_keep > n:
        raise ValueError("Can not keep more structures than provided!")
    if n_keep > n_select:
        warn(
            "Keeping more structures than to be selected!"
            " Cannot select new structures."
        )
        return keep_indices
    if n_select > n:
        warn(
            "Structures to select more than provided,"
            " will select all provided structures."
        )
        return list(range(n))
    dn = n_select - n_keep

    selected_indices = np.array(keep_indices, dtype=int)
    available_indices = np.setdiff1d(np.arange(n, dtype=int), keep_indices)

    # Used Penrose-Moore inverse
    for _ in range(dn):
        if method == "leverage":
            # Update feature matrix.
            prev_a = np.concatenate((old_a, a[selected_indices, :]), axis=0)
            prev_cov = prev_a.T @ prev_a
            prev_inv = np.linalg.pinv(prev_cov)
            reductions = []
            for trial_index in available_indices:
                trial_indices = np.append(selected_indices, trial_index)
                trial_a = np.concatenate((old_a, a[trial_indices, :]), axis=0)
                trial_cov = trial_a.T @ trial_a
                trial_inv = np.linalg.pinv(trial_cov)
                # By assertion, should all be <= 0.
                reductions.append(
                    np.sum(np.multiply((trial_inv - prev_inv), domain_matrix))
                )

            select_index = available_indices[np.argmin(reductions)]

        elif method == "random":
            select_index = np.random.choice(available_indices)

        else:
            raise NotImplementedError

        selected_indices = np.append(selected_indices, select_index)
        available_indices = np.setdiff1d(available_indices, [select_index])

    return selected_indices.tolist()
