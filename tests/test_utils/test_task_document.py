"""Test single_taskdocument utilities."""
import numpy as np
import pytest

from CEAuto.utils.task_document import get_entry_from_taskdoc


def test_get_entry(single_taskdoc):
    entry, props = get_entry_from_taskdoc(
        single_taskdoc,
        property_and_queries=["volume"],
        decorator_names=["magnetic-charge"],
    )
    assert "volume" in props
    assert "magmom" in entry.structure[0].properties
    assert np.isclose(entry.energy, single_taskdoc.entry.energy)
    assert np.isclose(entry.uncorrected_energy, single_taskdoc.entry.uncorrected_energy)

    with pytest.raises(ValueError):
        _, _ = get_entry_from_taskdoc(
            single_taskdoc,
            property_and_queries=["whocares"],
            decorator_names=["magnetic-charge"],
        )
