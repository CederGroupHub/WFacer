"""Test single_taskdocument utilities."""
import numpy as np
import pytest
from emmet.core.tasks import TaskDoc

from WFacer.utils.task_document import get_entry_from_taskdoc


def test_get_entry(single_taskdoc):
    if isinstance(single_taskdoc, TaskDoc):
        property_and_queries = ["volume"]
    else:
        property_and_queries = []
    entry, props = get_entry_from_taskdoc(
        single_taskdoc,
        property_and_queries=property_and_queries,
        decorator_names=["magnetic-charge"],
    )
    if len(property_and_queries) > 0:
        assert "volume" in props
        assert np.isclose(entry.energy, single_taskdoc.entry.energy)
        assert np.isclose(
            entry.uncorrected_energy, single_taskdoc.entry.uncorrected_energy
        )
    assert np.isclose(entry.energy, single_taskdoc.output.energy)
    assert "magmom" in entry.structure[0].properties

    with pytest.raises(ValueError):
        _, _ = get_entry_from_taskdoc(
            single_taskdoc,
            property_and_queries=["whocares"],
            decorator_names=["magnetic-charge"],
        )
