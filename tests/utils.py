import builtins
import json
import types

import numpy as np
import numpy.testing as npt
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Element
from pymatgen.entries.computed_entries import ComputedStructureEntry
from smol.cofe.space.domain import Vacancy
from smol.moca.utils.occu import get_dim_ids_table, occu_to_counts

from WFacer.wrangling import CeDataWrangler


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


def execute_job_function(job):
    # if Job was created using the job decorator, then access the original function
    function = getattr(job.function, "original", job.function)

    # if function is bound method we need to do some magic to bind the unwrapped
    # function to the class/instance
    bound = getattr(job.function, "__job__", None)
    if bound is not None and bound is not builtins:
        function = types.MethodType(function, bound)

    return function(*job.function_args, **job.function_kwargs)


def assert_dict_equal(d1, d2):
    assert sorted(list(d1.keys())) == sorted(list(d2.keys()))
    for k in d1.keys():
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            assert_dict_equal(d1[k], d2[k])
        else:
            if d1[k] != d2[k]:
                print(f"Difference in key: {k}, d1: {d1[k]}, d2: {d2[k]}")
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


def gen_random_occu_from_counts(ensemble, counts):
    n_species = 0
    occu = np.zeros(ensemble.num_sites, dtype=int) - 1
    for sublatt in ensemble.sublattices:
        n_sublatt = counts[n_species : n_species + len(sublatt.encoding)]
        occu_sublatt = [
            code for code, n in zip(sublatt.encoding, n_sublatt) for _ in range(n)
        ]
        np.random.shuffle(occu_sublatt)
        occu[sublatt.sites] = occu_sublatt
        n_species += len(sublatt.encoding)
    return occu


def gen_random_wrangler(ensemble, n_entries_per_iter=50, n_iters=8):
    """Generate a random wrangler from ensemble object.

    Args:
        ensemble(Ensemble):
            An ensemble object.
        n_entries_per_iter(int):
            Number of entries per iteration.
        n_iters(int):
            Number of iterations.
    Returns:
        CeDataWrangler.
    """
    n_enum = 0
    structures = []
    specs = []
    energies = []
    for iter_id in range(n_iters):
        for s_id in range(n_entries_per_iter):
            occu = gen_random_neutral_occupancy(sublattices=ensemble.sublattices)
            structures.append(ensemble.processor.structure_from_occupancy(occu))
            specs.append({"iter_id": iter_id, "enum_id": n_enum + s_id})
            energies.append(
                ensemble.natural_parameters @ ensemble.compute_feature_vector(occu)
            )
            # print("len:", len(energies))
        n_enum += n_entries_per_iter
    noise = np.random.normal(
        loc=0, scale=np.sqrt(np.var(energies)) * 0.0001, size=(len(energies),)
    )
    energies = np.array(energies) + noise
    entries = [ComputedStructureEntry(s, e) for s, e in zip(structures, energies)]
    wrangler = CeDataWrangler(ensemble.processor.cluster_subspace)
    # print("Inserting entries.")
    for ent, spec in zip(entries, specs):
        wrangler.add_entry(
            ent,
            properties={"spec": spec},
            supercell_matrix=ensemble.processor.supercell_matrix,
        )
    return wrangler
