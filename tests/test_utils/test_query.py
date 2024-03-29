"""Test query rules."""
import numpy as np
import numpy.testing as npt
import pytest
from emmet.core.tasks import TaskDoc

from WFacer.utils.query import (
    get_property_from_object,
    query_keypath,
    query_name_iteratively,
)


def test_query(single_taskdoc):
    d_test = {
        "students": [
            {"name": None, "age": 114514},
            {"name": "luis", "age": 27},
            {"name": "whatever", "age": 1919810, "test_set": {"a", "major", "failure"}},
        ]
    }
    assert d_test["students"] == query_keypath(d_test, ["students"])
    assert query_keypath(d_test, "students.name".split(".")) is None
    assert query_keypath(d_test, "students.age".split(".")) == 114514
    assert query_keypath(d_test, "students.1-name".split(".")) == "luis"
    assert query_keypath(d_test, "students.1-age".split(".")) == 27
    assert query_keypath(d_test, "students.2-name".split(".")) == "whatever"
    assert query_keypath(d_test, "students.2-age".split(".")) == 1919810
    with pytest.raises(ValueError):
        _ = query_keypath(d_test, "students.2-test_set.a".split("."))
    with pytest.raises(ValueError):
        _ = query_keypath(d_test, ["crazy_stuff"])
    with pytest.raises(ValueError):
        _ = query_keypath(d_test, "students.test_set".split("."))
    with pytest.raises(ValueError):
        _ = query_keypath(single_taskdoc, ["whatever"])
    assert query_name_iteratively(single_taskdoc, "structure") is not None
    if isinstance(single_taskdoc, TaskDoc):
        assert isinstance(
            query_keypath(single_taskdoc, "calcs_reversed.0-output.outcar".split(".")),
            dict,
        )
        d_mags = query_keypath(
            single_taskdoc, "calcs_reversed.0-output.outcar.magnetization".split(".")
        )
        assert d_mags is not None
        assert isinstance(d_mags, list)
        assert isinstance(d_mags[0], dict)
        assert "tot" in d_mags[0]
        mags = query_keypath(
            single_taskdoc,
            "calcs_reversed.0-output.outcar.magnetization.^tot".split("."),
        )
        assert isinstance(mags, list)
        assert not isinstance(mags[0], dict)
        npt.assert_array_almost_equal([d["tot"] for d in d_mags], mags)
        assert query_name_iteratively(single_taskdoc, "volume") is not None

    assert query_name_iteratively(d_test, "age") == 114514
    assert query_name_iteratively(d_test, "name") == "luis"
    assert query_name_iteratively(d_test, "whatever") is None


def test_get_property(single_taskdoc):
    # Currently only testing "energy" and "magmom".
    energy = get_property_from_object(single_taskdoc, "energy")
    assert np.isclose(energy, single_taskdoc.output.energy)
    if isinstance(single_taskdoc, TaskDoc):
        magmom = get_property_from_object(single_taskdoc, "magnetization")
        bandgap = get_property_from_object(single_taskdoc, "bandgap")
        assert len(magmom) == len(single_taskdoc.structure)
        assert bandgap > 0
        assert np.isclose(energy, single_taskdoc.entry.energy)  # Check this.
        assert np.isclose(energy, single_taskdoc.calcs_reversed[0].output.energy)
        entry = get_property_from_object(single_taskdoc, "entry")
        assert entry is not None
        assert entry.data["aspherical"]
        assert get_property_from_object(single_taskdoc, "aspherical")
    with pytest.raises(ValueError):
        _ = get_property_from_object(single_taskdoc, "whatever")
    # Should be able to look into entry.data as well. Should give true.
