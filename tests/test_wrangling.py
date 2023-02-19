"""Test CeDataWrangler."""

from pymatgen.analysis.structure_matcher import StructureMatcher


def test_data_wrangler(data_wrangler):
    # No duplication can occur.
    # Every entry must have "iter_id" and enum_id.
    n = data_wrangler.num_structures
    sm = StructureMatcher()
    for i in range(n):
        assert "spec" in data_wrangler.entries[i].data["properties"]
        spec = data_wrangler.entries[i].data["properties"]["spec"]
        assert "iter_id" in spec
        assert spec["iter_id"] <= 7
        assert "enum_id" in spec
        assert spec["enum_id"] < (spec["iter_id"] + 1) * 50  # 50 structs per iter.
        for j in range(i + 1, n):
            assert not sm.fit(
                data_wrangler.entries[i].structure, data_wrangler.entries[j].structure
            )
    # Must give the correct iteration index.
    # See conftest.py
    assert data_wrangler.max_iter_id == 7
