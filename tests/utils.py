from monty.json import MSONable, MontyDecoder
import json
import numpy as np
import numpy.testing as npt

from pymatgen.core import Element
from smol.cofe.space.domain import Vacancy
from smol.moca.utils.occu import (get_dim_ids_table,
                                  occu_to_counts)


def assert_msonable(obj, test_if_subclass=True):
    """
    Tests if obj is MSONable and tries to verify whether the contract is
    fulfilled.
    By default, the method tests whether obj is an instance of MSONable.
    This check can be deactivated by setting test_if_subclass to False.
    """
    if test_if_subclass:
        assert isinstance(obj, MSONable)
    assert obj.as_dict() == obj.__class__.from_dict(obj.as_dict()).as_dict()
    _ = json.loads(obj.to_json(), cls=MontyDecoder)


def assert_dict_equal(d1, d2):
    assert sorted(list(d1.keys())) == sorted(list(d2.keys()))
    for k in d1.keys():
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            assert_dict_equal(d1[k], d2[k])
        else:
            if d1[k] != d2[k]:
                print("Difference in key: {}, d1: {}, d2: {}"
                      .format(k, d1[k], d2[k]))
            assert d1[k] == d2[k]


def assert_array_permuted_equal(a1, a2):
    """Assert two arrays are equal after some row permutation.

    Args:
        a1, a2 (np.ndarray):
            Arrays to be compared.
    """
    a1 = np.array(a1)
    a2 = np.array(a2)
    assert a1.shape == a2.shape
    a1 = sorted(a1.tolist())
    a2 = sorted(a2.tolist())
    return npt.assert_array_almost_equal(a1, a2)


def gen_random_occupancy(sublattices, rng=None):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """

    num_sites = sum(len(sl.sites) for sl in sublattices)
    rand_occu = np.zeros(num_sites, dtype=int)
    rng = np.random.default_rng(rng)
    for sublatt in sublattices:
        rand_occu[sublatt.sites] = rng.choice(
            sublatt.encoding, size=len(sublatt.sites), replace=True
        )
    return rand_occu


def gen_random_neutral_occupancy(sublattices, lam=10, rng=None):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator},
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """
    rng = np.random.default_rng(rng)

    def get_charge(sp):
        if isinstance(sp, (Element, Vacancy)):
            return 0
        else:
            return sp.oxi_state

    def charge(occu, sublattices):
        charge = 0
        for sl in sublattices:
            for site in sl.sites:
                sp_id = sl.encoding.tolist().index(occu[site])
                charge += get_charge(sl.species[sp_id])
        return charge

    def flip(occu, sublattices, lam=10):
        actives = [s for s in sublattices if s.is_active]
        sl = rng.choice(actives)
        site = rng.choice(sl.sites)
        code = rng.choice(list(set(sl.encoding) - {occu[site]}))
        occu_next = occu.copy()
        occu_next[site] = code
        C = charge(occu, sublattices)
        C_next = charge(occu_next, sublattices)
        accept = np.log(rng.random()) < -lam * (C_next**2 - C**2)
        if accept and C != 0:
            return occu_next.copy(), C_next
        else:
            return occu.copy(), C

    occu = gen_random_occupancy(sublattices)
    for _ in range(10000):
        occu, C = flip(occu, sublattices, lam=lam)
        if C == 0:
            return occu.copy()

    raise TimeoutError("Can not generate a neutral occupancy in 10000 flips!")


def get_counts_from_occu(occu, sublattices):
    dim_ids_table = get_dim_ids_table(sublattices)
    n_dims = sum([len(sl.species) for sl in sublattices])
    return occu_to_counts(occu, n_dims, dim_ids_table)


def gen_random_neutral_counts(sublattices, lam=10, rng=None):
    """Generate a random composition in species counts format.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator},
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray:
            a charge balanced composition.
    """
    occu = gen_random_neutral_occupancy(sublattices, lam=lam, rng=rng)

    return get_counts_from_occu(occu, sublattices)
