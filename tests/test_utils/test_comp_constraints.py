"""Test composition constraints parsing functions."""
import random
from itertools import chain

import numpy as np
import numpy.testing as npt
from smol.moca.composition import get_dim_ids_by_sublattice

from CEAuto.utils.comp_constraints import (
    parse_generic_constraint,
    parse_species_constraints,
)
from tests.utils import assert_array_permuted_equal


def _get_dim_sl_ids_from_specie(bits, sp):
    i = 0
    dim_ids = []
    sl_ids = []
    for sl_id, sl_bits in enumerate(bits):
        for bit in sl_bits:
            if bit == sp:
                dim_ids.append(i)
                sl_ids.append(sl_id)
            i += 1
    return dim_ids, sl_ids


def _assert_single_species_d(d, bits, sl_sizes):
    leqs_expected = []
    geqs_expected = []
    n_dims = len(list(chain(*bits)))
    for sp in d:
        dim_ids_sp, sl_ids_sp = _get_dim_sl_ids_from_specie(bits, sp)
        leq_l = np.zeros(n_dims)
        geq_l = np.zeros(n_dims)
        leq_l[dim_ids_sp] = 1
        geq_l[dim_ids_sp] = 1
        if isinstance(d[sp], tuple):
            leq_r = d[sp][1] * np.sum(sl_sizes[sl_ids_sp])
            geq_r = d[sp][0] * np.sum(sl_sizes[sl_ids_sp])
        else:
            leq_r = d[sp] * np.sum(sl_sizes[sl_ids_sp])
            geq_r = 0
        leqs_expected.append(np.append(leq_l, leq_r))
        geqs_expected.append(np.append(geq_l, geq_r))

    leqs_gen, geqs_gen = parse_species_constraints(d, bits, sl_sizes)
    leqs = [np.append(leq_gl, leq_gr) for leq_gl, leq_gr in leqs_gen]
    geqs = [np.append(geq_gl, geq_gr) for geq_gl, geq_gr in geqs_gen]

    assert_array_permuted_equal(leqs, leqs_expected)
    assert_array_permuted_equal(geqs, geqs_expected)


def _assert_list_species_d(ds, bits, sl_sizes):
    leqs_expected = []
    geqs_expected = []
    n_dims = len(list(chain(*bits)))
    dim_ids = get_dim_ids_by_sublattice(bits)
    for d, sl_bits, sl_dim_ids, sl_size in zip(ds, bits, dim_ids, sl_sizes):
        for bit in d:
            leq_l = np.zeros(n_dims)
            geq_l = np.zeros(n_dims)
            dim_id = sl_dim_ids[sl_bits.index(bit)]
            leq_l[dim_id] = 1
            geq_l[dim_id] = 1
            if isinstance(d[bit], tuple):
                geq_r = sl_size * d[bit][0]
                leq_r = sl_size * d[bit][1]
            else:
                geq_r = 0
                leq_r = sl_size * d[bit]
            leqs_expected.append(np.append(leq_l, leq_r))
            geqs_expected.append(np.append(geq_l, geq_r))
    leqs_gen, geqs_gen = parse_species_constraints(ds, bits, sl_sizes)
    leqs = [np.append(leq_gl, leq_gr) for leq_gl, leq_gr in leqs_gen]
    geqs = [np.append(geq_gl, geq_gr) for geq_gl, geq_gr in geqs_gen]

    assert_array_permuted_equal(leqs, leqs_expected)
    assert_array_permuted_equal(geqs, geqs_expected)


def test_species_constraints(specs):
    bits = specs["bits"]
    species = set(list(chain(*bits)))
    sl_sizes = np.array([len(s) for s in specs["sublattice_sites"]])
    for _ in range(5):
        d = {sp: (random.uniform(0, 0.2), random.uniform(0.6, 0.8)) for sp in species}
        _assert_single_species_d(d, bits, sl_sizes)
    for _ in range(5):
        d = {sp: random.uniform(0.3, 0.6) for sp in species}
        _assert_single_species_d(d, bits, sl_sizes)
    for _ in range(5):
        # Half of random species are not constrained.
        random_missing_sps = random.sample(species, len(species) // 2)
        ds = [
            {
                bit: (random.uniform(0, 0.2), random.uniform(0.6, 0.8))
                for bit in sl_bits
                if bit not in random_missing_sps
            }
            for sl_bits in bits
        ]
        _assert_list_species_d(ds, bits, sl_sizes)
    for _ in range(5):
        # Half of random species are not constrained.
        random_missing_sps = random.sample(species, len(species) // 2)
        ds = [
            {
                bit: random.uniform(0.3, 0.8)
                for bit in sl_bits
                if bit not in random_missing_sps
            }
            for sl_bits in bits
        ]
        _assert_list_species_d(ds, bits, sl_sizes)


def _assert_single_generic_d(d, bits, right):
    con, _ = parse_generic_constraint(d, right, bits)
    n_dims = len(list(chain(*bits)))
    con_expected = np.zeros(n_dims)
    for sp in d:
        dim_ids_sp, sl_ids_sp = _get_dim_sl_ids_from_specie(bits, sp)
        con_expected[dim_ids_sp] = d[sp]
    npt.assert_array_almost_equal(con, con_expected)


def _assert_list_generic_d(ds, bits, right):
    con, _ = parse_generic_constraint(ds, right, bits)
    n_dims = len(list(chain(*bits)))
    con_expected = np.zeros(n_dims)
    dim_ids = get_dim_ids_by_sublattice(bits)
    for d, sl_bits, sl_dim_ids in zip(ds, bits, dim_ids):
        for sp in d:
            dim_id = sl_dim_ids[sl_bits.index(sp)]
            con_expected[dim_id] = d[sp]
    npt.assert_array_almost_equal(con, con_expected)


def test_generic_constraints(specs):
    bits = specs["bits"]
    species = set(list(chain(*bits)))
    sl_sizes = np.array([len(s) for s in specs["sublattice_sites"]])
    for _ in range(5):
        d = {sp: random.randint(0, 10) for sp in species}
        _assert_single_generic_d(d, bits, sl_sizes)
    for _ in range(5):
        # Half of random species are not constrained.
        random_missing_sps = random.sample(species, len(species) // 2)
        ds = [
            {
                bit: random.randint(0, 10)
                for bit in sl_bits
                if bit not in random_missing_sps
            }
            for sl_bits in bits
        ]
        _assert_list_generic_d(ds, bits, sl_sizes)
