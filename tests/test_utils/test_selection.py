"""Test structure selection utility methods."""

import numpy as np
import numpy.testing as npt
import pytest

from WFacer.utils.selection import select_added_rows, select_initial_rows


def test_initial_selection(subspace):
    n = 400
    d = subspace.num_corr_functions + len(subspace.external_terms)
    femat = np.random.random(size=(n, d))
    femat[:, 0] = 1

    num_external_terms = len(subspace.external_terms)
    keep = np.random.choice(len(femat), replace=False, size=(len(femat) // 10,))

    # Handle bad test cases.
    selected_ids = select_initial_rows(
        femat,
        n_select=len(keep) // 2,
        method="random",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )
    npt.assert_array_equal(selected_ids, keep)
    selected_ids = select_initial_rows(
        femat,
        n_select=len(femat) + 1,
        method="random",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )
    npt.assert_array_equal(selected_ids, np.arange(len(femat)))

    with pytest.raises(ValueError):
        _ = select_initial_rows(
            femat,
            n_select=len(femat) // 2,
            method="random",
            num_external_terms=num_external_terms,
            keep_indices=list(np.arange(len(femat) + 1)),
        )

    # Test random selection.
    selected_ids = select_initial_rows(
        femat,
        n_select=len(femat) // 2,
        method="random",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )
    npt.assert_array_equal(selected_ids[: len(keep)], keep)
    assert len(selected_ids) == len(femat) // 2

    # Test leverage selection.
    selected_ids = select_initial_rows(
        femat,
        n_select=len(femat) // 2,
        method="leverage",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )

    a = np.array(femat)[:, : len(femat[0]) - num_external_terms]
    cov = a.T @ a
    npt.assert_array_equal(selected_ids[: len(keep)], keep)
    assert len(selected_ids) == len(femat) // 2
    available_indices = np.setdiff1d(np.arange(n, dtype=int), keep)

    for iid, ii in enumerate(selected_ids[len(keep) :]):
        best_indices = np.array(selected_ids[: len(keep) + iid + 1])

        a_best = a[best_indices, :]
        cov_best = a_best.T @ a_best
        best_err = np.sum((cov_best - cov) ** 2)
        for _ in range(5):
            trial_index = np.random.choice(available_indices)
            trial_indices = np.append(selected_ids[: len(keep) + iid], trial_index)

            a_trial = a[trial_indices]
            cov_trial = a_trial.T @ a_trial

            err = np.sum((cov_trial - cov) ** 2)
            assert err >= best_err
        available_indices = np.setdiff1d(available_indices, ii)


def test_addition(subspace):
    n = 400
    d = subspace.num_corr_functions + len(subspace.external_terms)
    total_femat = np.random.random(size=(n, d))
    total_femat[:, 0] = 1

    old_femat = total_femat[: n // 4]
    femat = total_femat[n // 4 :]
    num_external_terms = len(subspace.external_terms)
    keep = np.random.choice(len(femat), replace=False, size=(len(femat) // 10,))
    # Handle bad test cases.
    selected_ids = select_added_rows(
        femat,
        old_femat,
        n_select=len(keep) // 2,
        method="random",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )
    npt.assert_array_equal(selected_ids, keep)
    selected_ids = select_added_rows(
        femat,
        old_femat,
        n_select=len(femat) + 1,
        method="random",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )
    npt.assert_array_equal(selected_ids, np.arange(len(femat)))
    with pytest.raises(ValueError):
        _ = select_added_rows(
            femat,
            old_femat,
            n_select=len(femat) // 2,
            method="random",
            num_external_terms=num_external_terms,
            keep_indices=list(np.arange(len(femat) + 1)),
        )

    # Test random selection.
    selected_ids = select_added_rows(
        femat,
        old_femat,
        n_select=len(femat) // 2,
        method="random",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )
    npt.assert_array_equal(selected_ids[: len(keep)], keep)
    assert len(selected_ids) == len(femat) // 2

    # Test leverage selection.
    selected_ids = select_added_rows(
        femat,
        old_femat,
        n_select=len(femat) // 2,
        method="leverage",
        num_external_terms=num_external_terms,
        keep_indices=keep,
    )
    npt.assert_array_equal(selected_ids[: len(keep)], keep)

    a = np.array(femat)[:, : len(femat[0]) - num_external_terms]
    domain = np.eye(subspace.num_corr_functions)
    npt.assert_array_equal(selected_ids[: len(keep)], keep)
    assert len(selected_ids) == len(femat) // 2
    available_indices = np.setdiff1d(np.arange(len(femat), dtype=int), keep)
    for iid, ii in enumerate(selected_ids[len(keep) :]):
        best_indices = np.array(selected_ids[: len(keep) + iid + 1])
        a_old = np.concatenate(
            (
                old_femat[:, : len(femat[0]) - num_external_terms],
                a[selected_ids[: len(keep) + iid]],
            ),
            axis=0,
        )
        a_best = np.concatenate(
            (old_femat[:, : len(femat[0]) - num_external_terms], a[best_indices, :]),
            axis=0,
        )
        score_best = np.sum(
            np.multiply(
                np.linalg.pinv(a_best.T @ a_best) - np.linalg.pinv(a_old.T @ a_old),
                domain,
            )
        )

        for _ in range(5):
            trial_index = np.random.choice(available_indices)

            trial_indices = np.append(selected_ids[: len(keep) + iid], trial_index)

            a_try = np.concatenate(
                (
                    old_femat[:, : len(femat[0]) - num_external_terms],
                    a[trial_indices, :],
                ),
                axis=0,
            )
            score_try = np.sum(
                np.multiply(
                    np.linalg.pinv(a_try.T @ a_try) - np.linalg.pinv(a_old.T @ a_old),
                    domain,
                )
            )

            assert score_try >= score_best
        available_indices = np.setdiff1d(available_indices, ii)
