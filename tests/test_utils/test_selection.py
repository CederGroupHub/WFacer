"""Test structure selection utility methods."""

from CEAuto.utils.selection import (select_added_rows,
                                    select_initial_rows,
                                    _cur_decompose)
import numpy as np
import numpy.testing as npt
import pytest


def test_intial_selection(data_wrangler):
    femat = data_wrangler.feature_matrix
    n, d = femat.shape
    num_external_terms = len(data_wrangler.cluster_subspace.external_terms)
    keep = np.random.choice(len(femat),
                            replace=False,
                            size=(len(femat) // 10,))

    # Handle bad test cases.
    selected_ids = select_initial_rows(femat,
                                       n_select=len(keep) // 2,
                                       method="random",
                                       num_external_terms=num_external_terms,
                                       keep_indices=keep)
    assert npt.assert_array_equal(selected_ids, keep)
    selected_ids = select_initial_rows(femat,
                                       n_select=len(femat) + 1,
                                       method="random",
                                       num_external_terms=num_external_terms,
                                       keep_indices=keep)
    assert npt.assert_array_equal(selected_ids,
                                  np.arange(len(femat)))
    with pytest.raises(ValueError):
        _ = select_initial_rows(femat,
                                n_select=len(femat) // 2,
                                method="random",
                                num_external_terms=num_external_terms,
                                keep_indices=
                                list(np.arange(len(femat) + 1)))

    selected_ids = select_initial_rows(femat,
                                       n_select=len(femat) // 2,
                                       method="CUR",
                                       num_external_terms=num_external_terms,
                                       keep_indices=keep)
    a = np.array(femat)[:, :len(femat[0]) - num_external_terms]
    g = a @ a.T
    assert npt.assert_array_equal(selected_ids[: len(keep)],
                                  keep)
    assert len(selected_ids) == len(femat) // 2
    available_indices = np.setdiff1d(np.arange(n, dtype=int),
                                     keep)
    for iid, ii in enumerate(selected_ids[len(keep):]):
        best_indices = np.array(selected_ids[: len(keep) + iid + 1])
        c_best = g[:, best_indices]
        r_best = g[best_indices, :]

        u_best = _cur_decompose(g, c_best, r_best)
        best_err = np.linalg.norm(g - np.dot(np.dot(c_best, u_best),
                                             r_best))
        for _ in range(5):
            trial_index = np.random.choice(available_indices)
            trial_indices = np.append(selected_ids[: len(keep) + iid],
                                      trial_index)

            c = g[:, trial_indices]
            r = g[trial_indices, :]

            u = _cur_decompose(g, c, r)

            err = np.linalg.norm(g - np.dot(np.dot(c, u), r))
            assert err >= best_err
        available_indices = np.setdiff1d(available_indices,
                                         ii)

    selected_ids = select_initial_rows(femat,
                                       n_select=len(femat) // 2,
                                       method="random",
                                       num_external_terms=num_external_terms,
                                       keep_indices=keep)
    assert npt.assert_array_equal(selected_ids[: len(keep)],
                                  keep)
    assert len(selected_ids) == len(femat) // 2


def test_addition(data_wrangler):
    total_femat = data_wrangler.feature_matrix[:]
    n, d = total_femat.shape
    old_femat = total_femat[: n // 4]
    femat = total_femat[n // 4:]
    num_external_terms = len(data_wrangler.cluster_subspace.external_terms)
    keep = np.random.choice(len(femat),
                            replace=False,
                            size=(len(femat) // 10,))
    # Handle bad test cases.
    selected_ids = select_added_rows(femat,
                                     old_femat,
                                     n_select=len(keep) // 2,
                                     method="random",
                                     num_external_terms=num_external_terms,
                                     keep_indices=keep)
    assert npt.assert_array_equal(selected_ids, keep)
    selected_ids = select_added_rows(femat,
                                     old_femat,
                                     n_select=len(femat) + 1,
                                     method="random",
                                     num_external_terms=num_external_terms,
                                     keep_indices=keep)
    assert npt.assert_array_equal(selected_ids,
                                  np.arange(len(femat)))
    with pytest.raises(ValueError):
        _ = select_added_rows(femat,
                              old_femat,
                              n_select=len(femat) // 2,
                              method="random",
                              num_external_terms=num_external_terms,
                              keep_indices=
                              list(np.arange(len(femat) + 1)))

    selected_ids = select_added_rows(femat,
                                     old_femat,
                                     n_select=len(femat) // 2,
                                     method="leverage",
                                     num_external_terms=num_external_terms,
                                     keep_indices=keep)
    a = np.array(femat)[:, :len(femat[0]) - num_external_terms]
    domain = np.eye(d)
    assert npt.assert_array_equal(selected_ids[: len(keep)],
                                  keep)
    assert len(selected_ids) == len(femat) // 2
    available_indices = np.setdiff1d(np.arange(len(femat), dtype=int),
                                     keep)
    for iid, ii in enumerate(selected_ids[len(keep):]):
        best_indices = np.array(selected_ids[: len(keep) + iid + 1])
        a_old = a[selected_ids[: len(keep) + iid]]
        a_best = a[best_indices, :]
        score_best = np.sum(np.multiply(np.linalg.pinv(a_best.T @ a_best)
                                        - np.linalg.pinv(a_old.T @ a_old),
                                        domain))
        for _ in range(5):
            trial_index = np.random.choice(available_indices)
            trial_indices = np.append(selected_ids[: len(keep) + iid],
                                      trial_index)

            a_try = a[trial_indices, :]
            score_try = np.sum(np.multiply(np.linalg.pinv(a_try.T @ a_try)
                                           - np.linalg.pinv(a_old.T @ a_old),
                                           domain))
            assert score_try >= score_best
        available_indices = np.setdiff1d(available_indices,
                                         ii)

    selected_ids = select_added_rows(femat,
                                     old_femat,
                                     n_select=len(femat) // 2,
                                     method="random",
                                     num_external_terms=num_external_terms,
                                     keep_indices=keep)
    assert npt.assert_array_equal(selected_ids[: len(keep)],
                                  keep)
    assert len(selected_ids) == len(femat) // 2
