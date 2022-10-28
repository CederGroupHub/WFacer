"""Utility functions to enumerate supercell matrices."""

__author__ = "Fengyu Xie"

import numpy as np
from sympy import factorint
from itertools import permutations, product, chain
import warnings

from pymatgen.core import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher, \
    OrderDisorderElementComparator


def get_three_factors(n):
    """Enumerate all 3 factor decompositions of an integer.

    Note:
        Do not use this to factorize an integer with many
        possible factors.
    Args:
        n(int):
            The integer to factorize.

    Returns:
        All 3 factor decompositions:
            List[tuple(int)]
    """

    def enumerate_three_summations(c):
        # Yield all (x, y, z) that x + y + z = c.
        three_nums = set([])
        for x in range(c + 1):
            for y in range(c + 1 - x):
                z = c - x - y
                for perm in set(permutations([x, y, z])):
                    three_nums.add(perm)
        return sorted(list(three_nums), reverse=True)

    if n == 0:
        return []
    if n == 1:
        return [(1, 1, 1)]
    prime_factor_counts = factorint(n)
    prime_factors = sorted(prime_factor_counts.keys(), reverse=True)
    prime_factors = np.array(prime_factors, dtype=int)
    all_three_nums = [enumerate_three_summations(prime_factor_counts[p])
                      for p in prime_factors]
    all_factors = []
    for sol in product(*all_three_nums):
        ns = np.array(sol, dtype=int)
        factors = sorted(np.product(prime_factors[:, None] ** ns, axis=0).tolist(),
                         reverse=True)
        if factors not in all_factors:
            all_factors.append(factors)
    return sorted(all_factors)


def is_proper_sc(sc_matrix, lat, max_cond=8, min_angle=30):
    """Assess the quality of a given supercell matrix.

    If too skewed or too slender, this matrix will be dropped
    because it does not fit for DFT calculation.
    Args:
        sc_matrix(3 * 3 ArrayLike):
            Supercell matrix
        lat(pymatgen.Lattice):
            Lattice of the primitive cell
        max_cond(float): optional
            Maximum conditional number allowed in the supercell lattice
            matrix. This is to avoid overly imbalance in the lengths of
            three lattice vectors. By default, set to 8.
        min_angle(float): optional
            Minimum allowed angle of the supercell lattice. By default, set
            to 30, to prevent over-skewing.
    Returns:
       Boolean.
    """
    new_mat = np.dot(sc_matrix, lat.matrix)
    new_lat = Lattice(new_mat)
    angles = [new_lat.alpha, new_lat.beta, new_lat.gamma,
              180 - new_lat.alpha, 180 - new_lat.beta,
              180 - new_lat.gamma]

    return (abs(np.linalg.cond(new_mat)) <= max_cond and
            min(angles) >= min_angle)


def is_duplicate_sc(m1, m2, prim, matcher=None):
    """Gives whether two super-cell matrices give identical super-cell.

    Args:
        m1(3*3 ArrayLike[int]):
            Supercell matrices to compare.
        m2(3*3 ArrayLike[int]):
            Supercell matrices to compare.
        prim(pymatgen.Structure):
            Primitive cell object.
        matcher(pymatgen.StructureMatcher): optional
            A StructureMatcher.
    Returns:
        bool.
    """
    m1 = np.round(m1).astype(int)
    m2 = np.round(m2).astype(int)
    matcher = matcher or StructureMatcher(
        primitive_cell=False,
        attempt_supercell=True,
        allow_subset=True,
        comparator=OrderDisorderElementComparator(),
        scale=True
    )
    s1 = prim.copy()
    s2 = prim.copy()
    s1.make_supercell(m1)
    s2.make_supercell(m2)
    return matcher.fit(s1, s2)

