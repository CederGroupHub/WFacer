__author__ = "Fengyu Xie"

"""
Utility functions to enumerate supercell matrices.
"""
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


# TODO: in the future, may generate with mcsqs type algos.
def enumerate_matrices(sc_size, cluster_subspace,
                       conv_mat=None,
                       max_sc_cond=8,
                       min_sc_angle=30):
    """Enumerate proper matrices with det size.

    Will give 1 unskewed matrix and 1 skewed matrix.
    Skewed matrix usually helps to avoid aliasing of clusters.

    Args:
        sc_size(int):
            Required supercell size in number of primitive cells.
            Better be a multiple of det(conv_mat).
        cluster_subspace(smol.ClusterSubspace):
            The cluster subspace. cluster_subspace.structure must
            be pre-processed such that it is the true primitive cell
            in under its space group symmetry.
        conv_mat(2D arraylike):
            pre-transformation matrix to convert the primitive cell to
            the conventional. The returned results will always be divisible
            by conv_mat. (conv_mat @ some_integer_matrix = result.)
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
    if conv_mat is None:
        conv_mat = np.eye(3, dtype=int)
    conv_mat = np.round(conv_mat).astype(int)
    conv_size = cluster_subspace.num_prims_from_matrix(conv_mat)
    if sc_size % conv_size != 0:
        warnings.warn(f"Supercell size: {sc_size} to enumerate is not "
                      "divisible by primitive to conventional matrix"
                      f" size {conv_size}."
                      f" Will be rounded to {sc_size // conv_size * conv_size}!")
        sc_size = sc_size // conv_size * conv_size

    scs_diagonal = [np.diag(sorted(m, reverse=True))
                    for m in get_three_factors(sc_size // conv_size)]

    def get_skews(m, conv, space):
        # Get skews of a matrix. only upper-triangular used.
        skews = []
        margin = m[0, 0]
        n_range = sorted({0, 1, margin // 2, margin})
        for i in n_range:
            for j in n_range:
                for k in n_range:
                    if i * j * k == 0:
                        continue
                    skewed = m.copy()
                    skewed[0, 1] = i
                    skewed[0, 2] = k
                    skewed[1, 2] = j
                    dupe = False
                    for m_old in skews:
                        if is_duplicate_sc(m_old @ conv,
                                           skewed @ conv,
                                           space.structure,
                                           space._sc_matcher):
                            dupe = True
                            break
                    if not dupe:
                        skews.append(skewed)
        return skews

    scs_skew = list(chain(*[get_skews(sc, conv_mat, cluster_subspace)
                            for sc in scs_diagonal]))

    # filter out bad matrices.
    lat = cluster_subspace.structure.lattice

    def cond_and_angle(sc):
        new_mat = np.dot(sc, lat.matrix)
        new_lat = Lattice(new_mat)
        return (np.linalg.cond(sc),
                min([new_lat.alpha, new_lat.beta, new_lat.gamma,
                     180 - new_lat.alpha, 180 - new_lat.beta,
                     180 - new_lat.gamma]))

    def filt_func_(sc):
        cond, angle = cond_and_angle(sc @ conv_mat)
        return cond <= max_sc_cond and angle >= min_sc_angle

    scs_diagonal = list(filter(filt_func_, scs_diagonal))
    scs_skew = list(filter(filt_func_, scs_skew))

    def alias_level(sc):
        return len(list(chain(*cluster_subspace.get_aliasd_orbits(sc))))

    # Sort diagonal by low stretch, then low alias level.
    def diagonal_sort_key(sc):
        cond, angle = cond_and_angle(sc)
        return cond, alias_level(sc)

    # Sort diagonal by low stretch, then low alias level.
    def skew_sort_key(sc):
        cond, angle = cond_and_angle(sc)
        return alias_level(sc), -angle, cond

    scs_diagonal = sorted(scs_diagonal, key=diagonal_sort_key)
    scs_skew = sorted(scs_skew, key=skew_sort_key)

    # Select 1 diagonal, 1 off diagonal.
    return scs_diagonal[0], scs_skew[0]
