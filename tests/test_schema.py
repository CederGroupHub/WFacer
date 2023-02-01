"""Test initialization of the CeOutputsDocument schema."""
import pytest

from CEAuto.schema import CeOutputsDocument


def test_ce_outputs(subspace, data_wrangler):
    empty_document = CeOutputsDocument(cluster_subspace=subspace)
    assert empty_document.last_iter_id == -1

    # history lengths are not consistent.
    bad_document1 = CeOutputsDocument(cluster_subspace=subspace,
                                      cv_history=[1.0])
    with pytest.raises(ValueError):
        _ = bad_document1.last_iter_id

    # History length does not match with document.
    bad_document2 = CeOutputsDocument(cluster_subspace=subspace,
                                      coefs_history=[[1.0]],
                                      cv_history=[1.0],
                                      cv_std_history=[1.0],
                                      params_history=[1.0])
    with pytest.raises(ValueError):
        _ = bad_document2.last_iter_id

    # History length does not match with wrangler.
    bad_document3 = CeOutputsDocument(cluster_subspace=subspace,
                                      data_wrangler=data_wrangler,
                                      coefs_history=[[1.0]],
                                      cv_history=[1.0],
                                      cv_std_history=[1.0],
                                      params_history=[1.0])
    with pytest.raises(ValueError):
        _ = bad_document3.last_iter_id

