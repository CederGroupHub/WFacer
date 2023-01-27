"""Provide structure selection methods."""
import numpy as np
import logging


def _cur_decompose(g, c, r):
    """Calculate u that g = cur.

    Args:
        g(np.ndarray):
            g to compute u with.
        c(np.ndarray):
            c to compute u with.
        r(np.ndarray):
            r to compute u with.
    Returns:
        np.ndarray: u with g=cur.
    """
    return np.dot(np.dot(np.linalg.pinv(c), g), np.linalg.pinv(r))


def select_initial_rows(femat, n_select=10,
                        method="CUR",
                        num_external_terms=0,
                        keep_indices=None):
    """Select structures to initialize an empty CE project.

    Args:
        femat(2D arrayLike):
            Correlation vectors of each structure.
        n_select(int): optional
            Number of structures to select. Default is 10.
        method(str): optional
            The method used to select structures. Default is
            CUR decomposition ("CUR"). "random" is also supported.
        num_external_terms(int): optional
            Number of external terms in cluster subspace. These
            terms should not be compared in a structure selection.
        keep_indices(list[int]): optional
            Indices of structures that must be selected. Usually
            those of important ground state structures.
    Returns:
        List[int]: indices of selected structures.
    """
    # Leave out external terms.
    a = np.array(femat)[:, :len(femat[0]) - num_external_terms]
    n, d = a.shape

    if keep_indices is None:
        keep_indices = []
    n_keep = len(keep_indices)
    if n_keep > n:
        raise ValueError("Can not keep more structures than provided!")
    if n_keep > n_select:
        logging.warning("Keeping more structures than to be selected!"
                        " Cannot select new structures.")
        return keep_indices
    if n_select > n:
        logging.warning("Structures to select more than provided,"
                        " will select all provided structures.")
        return list(range(n))
    dn = n_select - n_keep

    selected_indices = np.array(keep_indices, dtype=int)
    available_indices = np.setdiff1d(np.arange(n, dtype=int),
                                     keep_indices)

    g = a @ a.T  # Gram matrix of features.

    for _ in range(dn):
        if method == 'CUR':
            errs = []
            for trial_index in available_indices:
                trial_indices = np.append(selected_indices, trial_index)
                c = g[:, trial_indices]
                r = g[trial_indices, :]

                u = _cur_decompose(g, c, r)

                errs.append(np.linalg.norm(g - np.dot(np.dot(c, u), r)))
            select_index = available_indices[np.argmin(errs)]

        elif method == 'random':
            select_index = np.random.choice(available_indices)

        else:
            raise NotImplementedError

        selected_indices = np.append(selected_indices, select_index)
        available_indices = np.setdiff1d(available_indices, [select_index])

    return selected_indices.tolist()


# TODO: implement composition dependent domain matrices. (not urgent)
def select_added_rows(femat, old_femat,
                      n_select=10,
                      method="leverage",
                      keep_indices=None,
                      num_external_terms=0,
                      domain_matrix=None):

    """Select structures to add to an existing CE project.

    We select structures by minimizing the leverage score under a
    certain domain matrix, or fully at random.
    Refer to: Phys. Rev. B 82, 184107 (2010).
    Inputs:
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
        keep_indices(List[int]): optional
            Indices of structures that must be selected. Usually
            those of important ground state structures.
        num_external_terms(int): optional
            Number of external terms in cluster subspace. These
            terms should not be compared in a structure selection.
        domain_matrix(2D arraylike): optional
            The domain matrix used to compute leverage score. By
            default, we use an identity matrix.

    Outputs:
        List of ints. Indices of selected rows in femat.
    """
    # Leave out external terms.
    a = np.array(femat)[:, :len(femat[0]) - num_external_terms]
    old_a = np.array(old_femat)[:, :len(old_femat[0]) - num_external_terms]
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
        logging.warning("Keeping more structures than to be selected!"
                        " Cannot select new structures.")
        return keep_indices
    if n_select > n:
        logging.warning("Structures to select more than provided,"
                        " will select all provided structures.")
        return list(range(n))
    dn = n_select - n_keep

    selected_indices = np.array(keep_indices, dtype=int)
    available_indices = np.setdiff1d(np.arange(n, dtype=int),
                                     keep_indices)

    # Used Penrose-Moore inverse
    for _ in range(dn):
        if method == 'leverage':
            # Update feature matrix.
            old_a = np.concatenate(old_a, a[selected_indices, :], axis=0)
            old_cov = old_a.T @ old_a
            old_inv = np.linalg.pinv(old_cov)
            reductions = []
            for trial_index in enumerate(available_indices):
                trial_a = np.concatenate((old_a, a[trial_index].reshape(1, d)),
                                         axis=0)
                trial_cov = trial_a.T @ trial_a
                trial_inv = np.linalg.pinv(trial_cov)
                # By assertion, should all be <= 0.
                reductions.append(np.sum(np.multiply((trial_inv - old_inv),
                                                     domain_matrix)))

            select_index = available_indices[np.argmin(reductions)]

        elif method == 'random':
            select_index = np.random.choice(available_indices)

        else:
            raise NotImplementedError

        selected_indices = np.append(selected_indices, select_index)
        available_indices = np.setdiff1d(available_indices, [select_index])

    return selected_indices.tolist()
