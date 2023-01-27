"""Test taskdocument utilities."""
import numpy as np
import pytest

from CEAuto.utils.task_document import (get_property_from_taskdoc,
                                        get_entry_from_taskdoc)


# TODO: generate a taskdoc for testing.
def test_get_property(taskdoc):
    # Currently only testing "energy" and "magmom".
    magmom = get_property_from_taskdoc(taskdoc, "magnetization")
    volume = get_property_from_taskdoc(taskdoc, "bandgap")
    energy = get_property_from_taskdoc(taskdoc, "energy")
    assert len(magmom) == len(taskdoc.structure)
    assert volume > 0
    assert np.isclose(energy, taskdoc.energy)  # Check this.
    with pytest.raises(ValueError):
        _ = get_property_from_taskdoc(taskdoc, "whatever")


def test_get_entry(taskdoc):
    entry, props = get_entry_from_taskdoc(taskdoc,
                                          properties=["volume"],
                                          decorator_names=["magnetic-charge"])
    assert "volume" in props
    assert "magmom" in entry.structure[0].properties

