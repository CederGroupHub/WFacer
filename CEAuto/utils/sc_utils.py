__author__ = "Fengyu Xie"

"""
Utility functions to enumerate supercell matrices.
"""
import numpy as np
import random
from sympy import factorint
from itertools import permutations, product

from pymatgen.core import Lattice


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


def get_aliases(sc_matrix, cluster_subspace):
    """Aliased correlation functions in a super-cell.

    Given a cluster space, the more alias a super-cell produce, the less
    favorable the super-cell matrix is. In real cases, since we can only
    compute limited size super-cell, we can not always guarantee that
    a super-cell matrix does not create any alias, but we would always
    like to minimize the number of aliases.
    This along with the proper-ness check, are two criterion of choosing
    the optimal super-cell matrices.
    Note: alias check requires smol >= 0.0.5.

    Args:
        sc_matrix(3 * 3 ArrayLike):
            The super-cell matrix.
        cluster_subspace(smol.cofe.ClusterSubSpace):
            The cluster subspace object, containing all clusters to expand.
    Returns:
        list[(int, int)]:
            Indices of aliased correlation function pairs. The first key is
            always given as smaller than the second.
    """
    # TODO: write get_aliases.


def enumerate_matrices(det, cluster_subspace,
                       conv_mat=np.eye(3, dtype=int),
                       max_sc_cond=8,
                       min_sc_angle=30):
    """Enumerate proper matrices with det size.

    Will give 1 unskewed matrix and up to 3 skewed matrices.
    We add skewed matrices to avoid symmtric duplicacy of clusters.

    Args:
        det(int):
            Required determinant size of enumerated supercell
            matrices. Must be multiple of det(transmat).
        lat(pymatgen.Lattice):
            Lattice vectors of a primitive cell
        transmat(2D arraylike):
            pre-transformation matrix before returning result,
            such that we return MT instead of enumerated M.
        max_sc_cond(float):
            Maximum conditional number allowed of the skewed supercell
            matrices. By default set to 8, to prevent overstretching in one
            direction
        min_sc_angle(float):
            Minmum allowed angle of the supercell lattice. By default set
            to 30, to prevent over-skewing.
    Returns:
        List of 2D lists.
    """
    trans_size = int(round(abs(np.linalg.det(transmat))))
    if det % trans_size != 0:
        raise ValueError("Supercell size must be divisible by " +
                         "transformation matrix determinant!")
    scs_diagonal = [np.diag(m) for m in get_three_factors(det // trans_size)]

    scs_diagonal = [sc for sc in scs_diagonal
                    if is_proper_sc(np.dot(sc, transmat),
                                    lat,
                                    max_cond=max_sc_cond,
                                    min_angle=min_sc_angle)]

    # Take the diagonal matrix with minimal conditional number.
    sc_diagonal = sorted(scs_diagonal, key=lambda x: np.linalg.cond(x))[0]
    n1 = sc_diagonal[0][0]
    n2 = sc_diagonal[1][1]
    n3 = sc_diagonal[2][2]
    # n1 > n2 > n3, already sorted in get_diag_matrices.
    sc_off = sc_diagonal.copy()

    # TODO: change this to check number of aliased clusters in
    #  ClusterSubspace and sort to find the one with least aliases.
    sc_off[0][1] = random.choice(np.arange(1, n1 + 1, dtype=int))

    # Select 1 diagonal, 1 off diagonal.
    return [sc_diagonal @ transmat, sc_off @ transmat]
