import numpy as np
import numpy.testing as npt
import random
import pytest
from itertools import product

from sklearn.mixture import GaussianMixture

from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core import Lattice, Structure

from CEAuto.specie_decorators import (decorator_factory,
                                      MagneticChargeDecorator,
                                      PmgGuessChargeDecorator,
                                      FixedChargeDecorator)
from CEAuto.specie_decorators.base import (MixtureGaussianDecorator,
                                           NoTrainDecorator)

from .utils import assert_msonable

all_decorator_names = ["magnetic-charge", "pmg-guess-charge", "fixed-charge"]
all_decorators = [MagneticChargeDecorator, PmgGuessChargeDecorator, FixedChargeDecorator]


@pytest.fixture
def undecorated_entries_standards():
    # Set up a series of undecorated structures.
    standard_mag_decors = []
    entries = []
    for _ in range(500):
        species = ["O", "O", "O", "O", "Ca", "Ca", "Ca", "Ca"]
        case = np.random.choice(3)
        dists = [(1.0, 0.1), (2.0, 0.2), (3.0, 0.15)]
        magmoms = [0, 0, 0, 0]
        standard_decor = []
        if case == 0:
            # +1, +3, +1, +3.
            for _ in range(2):
                magmoms.append(np.random.normal(dists[0][0], dists[0][1]))
            for _ in range(2):
                magmoms.append(np.random.normal(dists[2][0], dists[2][1]))
            standard_decor.append(["O2-", "O2-", "O2-", "O2-",
                                   "Ca+", "Ca+", "Ca3+", "Ca3+"])
        if case == 1:
            # +2, +2, +1, +3.
            for _ in range(2):
                magmoms.append(np.random.normal(dists[1][0], dists[1][1]))
            magmoms.append(np.random.normal(dists[0][0], dists[0][1]))
            magmoms.append(np.random.normal(dists[2][0], dists[2][1]))
            standard_decor.append(["O2-", "O2-", "O2-", "O2-",
                                   "Ca2+", "Ca2+", "Ca+", "Ca3+"])

        if case == 2:
            # +2, +2, +2, +2.
            for _ in range(4):
                magmoms.append(np.random.normal(dists[1][0], dists[1][1]))
            standard_decor.append(["O2-", "O2-", "O2-", "O2-",
                                   "Ca2+", "Ca2+", "Ca2+", "Ca2+"])

        # shuffle structure.
        shuff = list(range(8))
        random.shuffle(shuff)
        species_shuff = [species[i] for i in shuff]
        magmoms_shuff = [magmoms[i] for i in shuff]
        standard_decor_shuff = [standard_decor[i] for i in shuff]
        species = species_shuff.copy()
        magmoms = magmoms_shuff.copy()
        standard_decor = standard_decor_shuff.copy()

        points = np.array([[0, 0, 0],
                           [0.5, 0.5, 0],
                           [0, 0.5, 0.5],
                           [0.5, 0, 0.5]])
        s = Structure(Lattice.cubic(4.0),
                      species,
                      np.concatenate([points, points + 0.25],
                                     axis=0))
        s.add_site_property("magmom", magmoms)

        entries.append(ComputedStructureEntry(s, np.random.random()))

    return entries, standard_mag_decors


@pytest.fixture(params=all_decorators)
def decorator(request):
    if "Magnetic" in request.param.__name__:
        labels = {"Ca": [1, 2, 3], "O": [-2]}
    elif "Guess" in request.param.__name__:
        labels = None
    else:
        labels = {"Ca": 2, "O": -2}
    return request.param(labels=labels)  # Maximum abs charge always 0.


@pytest.mark.parametrize("decorator_name, decorator_class",
                         zip(all_decorator_names, all_decorators))
def test_decorator_factory(decorator_name, decorator_class):
    labels = {"A": [1, 2, 3, 4]}
    decorator = decorator_factory(decorator_name, labels=labels)
    assert isinstance(decorator, decorator_class)


def test_msonable(decorator):
    assert_msonable(decorator)


def test_group_sites(decorator, undecorated_entries_standards):
    undecorated_entries, _ = undecorated_entries_standards
    groups = decorator.group_site_by_species(undecorated_entries)
    all_eid_sid = []
    n_structs = len(undecorated_entries)
    n_sites = len(undecorated_entries[0].structure)
    for sp in groups:
        for eid, sid in groups[sp]:
            assert (undecorated_entries[eid].structure[sid].species
                    == sp)
            all_eid_sid.append((eid, sid))
    # All sites in all structures must be included.
    npt.assert_array_equal(sorted(all_eid_sid),
                           sorted(list(product(range(n_structs),
                                               range(n_sites)))))


def test_train_decorate(decorator, undecorated_entries_standards):
    undecorated_entries, standards = undecorated_entries_standards
    if issubclass(decorator.__class__, NoTrainDecorator):
        assert decorator.is_trained

    decorator.train(undecorated_entries)
    entries = decorator.decorate(undecorated_entries)

    # There should not be too many fails given these conditions.
    num_fails = len([ent for ent in entries if ent is None])
    if issubclass(NoTrainDecorator):
        assert num_fails == 0
    else:
        assert num_fails / len(undecorated_entries) <= 0.8

    n_match = 0
    for ent_decor, ent_undecor ,standard in zip(entries,
                                                undecorated_entries,
                                                standards):
        if ent_decor is not None:
            assert np.isclose(ent_decor.structure.charge, 0)
            assert np.isclose(ent_decor.energy, ent_undecor.energy)
            assert ent_decor.data == ent_undecor.data
            match_standard = True
            for site1, site2, s_str in zip(ent_decor.structure,
                                           ent_undecor.structure,
                                           standard):
                assert site1.species.symbol == site2.species_symbol
                assert site1.properties == site2.properties
                if issubclass(decorator, NoTrainDecorator):
                    if site1.species.symbol == "O":
                        s_str = "O2-"
                    else:
                        s_str = "Ca2+"
                if s_str != str(site1.species):
                    match_standard = False
            if match_standard:
                n_match += 1

    if issubclass(decorator, NoTrainDecorator):
        assert n_match == len(undecorated_entries)
    else:
        assert n_match / len(undecorated_entries) >= 0.7


def test_serialize_gaussian():
    model = GaussianMixture(n_components=3)
    assert not MixtureGaussianDecorator.is_trained_gaussian_model(model)
    data = np.concatenate([np.random.normal(1.0, 0.2, size=(200, 1)),
                           np.random.normal(2.0, 0.1, size=(150, 1)),
                           np.random.normal(3.0, 0.15, size=(300, 1))])

    model.fit(data)
    labels = model.predict(data)
    assert MixtureGaussianDecorator.is_trained_gaussian_model(model)

    model_dict = MixtureGaussianDecorator.serialize_gaussian_model(model)
    model_reload = MixtureGaussianDecorator.deserialize_gaussian_model(model_dict)
    assert MixtureGaussianDecorator.is_trained_gaussian_model(model_reload)
    labels_reload = model_reload.predict(data)

    npt.assert_array_equal(labels, labels_reload)








