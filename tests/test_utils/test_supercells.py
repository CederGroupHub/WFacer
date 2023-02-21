"""Test super-cell matrix utilities."""

import numpy as np
import numpy.testing as npt
from pymatgen.core import Lattice, Structure

from AceCE.utils.supercells import get_three_factors, is_duplicate_sc, is_proper_sc


def test_three_factors():
    for _ in range(10):
        rand = np.random.randint(low=1, high=100)
        factors = get_three_factors(rand)
        factors = np.array(factors, dtype=int)
        n, d = factors.shape
        assert n > 0
        assert d == 3
        npt.assert_array_equal(np.product(factors, axis=-1), rand)
        npt.assert_array_equal(sorted(factors.tolist()), factors)

    factors_12 = [[3, 2, 2], [4, 3, 1], [6, 2, 1], [12, 1, 1]]
    npt.assert_array_equal(factors_12, get_three_factors(12))


def test_proper_sc():
    lat = Lattice.cubic(1.0)
    sc_mat = np.diag([1, 2, 10])
    assert not is_proper_sc(sc_mat, lat)
    sc_mat = np.diag([1, 1, 1])
    assert is_proper_sc(sc_mat, lat)
    sc_mat = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert not is_proper_sc(sc_mat, lat, min_angle=60)
    assert is_proper_sc(sc_mat, lat, min_angle=30)


def test_duplicate_sc(prim):
    sc_mat1 = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    sc_mat2 = [[0, 1, 0], [0, 1, 1], [1, 0, 0]]
    assert is_duplicate_sc(sc_mat1, sc_mat2, prim)
    sc_mat2 = [[0, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert not is_duplicate_sc(sc_mat1, sc_mat2, prim)
    sc_mat2 = [[0, 1, 0], [0, 1, 1], [1, 0, 0]]
    s = Structure(
        Lattice.from_parameters(1, 10, 40, 90, 90, 90),
        ["H", "H"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    assert not is_duplicate_sc(sc_mat1, sc_mat2, s)
