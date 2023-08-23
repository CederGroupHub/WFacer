import random
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from sklearn.mixture import GaussianMixture

from WFacer.specie_decorators import (
    FixedChargeDecorator,
    MagneticChargeDecorator,
    PmgGuessChargeDecorator,
    decorator_factory,
)
from WFacer.specie_decorators.base import (
    BaseDecorator,
    MixtureGaussianDecorator,
    NoTrainDecorator,
)

from .utils import assert_msonable

all_decorator_names = ["magnetic-charge", "pmg-guess-charge", "fixed-charge"]
all_decorators = [
    MagneticChargeDecorator,
    PmgGuessChargeDecorator,
    FixedChargeDecorator,
]


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
        if case == 0:
            # +1, +3, +1, +3.
            for _ in range(2):
                magmoms.append(np.random.normal(dists[0][0], dists[0][1]))
            for _ in range(2):
                magmoms.append(np.random.normal(dists[2][0], dists[2][1]))
            standard_decor = ["O2-", "O2-", "O2-", "O2-", "Ca+", "Ca+", "Ca3+", "Ca3+"]
        if case == 1:
            # +2, +2, +1, +3.
            for _ in range(2):
                magmoms.append(np.random.normal(dists[1][0], dists[1][1]))
            magmoms.append(np.random.normal(dists[0][0], dists[0][1]))
            magmoms.append(np.random.normal(dists[2][0], dists[2][1]))
            standard_decor = ["O2-", "O2-", "O2-", "O2-", "Ca2+", "Ca2+", "Ca+", "Ca3+"]

        if case == 2:
            # +2, +2, +2, +2.
            for _ in range(4):
                magmoms.append(np.random.normal(dists[1][0], dists[1][1]))
            standard_decor = [
                "O2-",
                "O2-",
                "O2-",
                "O2-",
                "Ca2+",
                "Ca2+",
                "Ca2+",
                "Ca2+",
            ]

        # shuffle structure.
        shuff = list(range(8))
        random.shuffle(shuff)
        species_shuff = [species[i] for i in shuff]
        magmoms_shuff = [magmoms[i] for i in shuff]
        standard_decor_shuff = [standard_decor[i] for i in shuff]
        species = species_shuff.copy()
        magmoms = magmoms_shuff.copy()
        standard_decor = standard_decor_shuff.copy()

        points = np.array([[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
        s = Structure(
            Lattice.cubic(4.0), species, np.concatenate([points, points + 0.25], axis=0)
        )
        s.add_site_property("magmom", magmoms)

        entries.append(ComputedStructureEntry(s, np.random.random()))
        standard_mag_decors.append(standard_decor)

        assert s.composition.element_composition["Ca"] == 4
        assert s.composition.element_composition["O"] == 4

    have_disproportionate = False
    for std in standard_mag_decors:
        if "Ca+" in std or "Ca3+" in std:
            have_disproportionate = True
            break
    assert have_disproportionate

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


@pytest.mark.parametrize(
    "decorator_name, decorator_class", zip(all_decorator_names, all_decorators)
)
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
            # site.species will give a composition, not a species!
            assert undecorated_entries[eid].structure[sid].specie == sp
            all_eid_sid.append((eid, sid))
    # All sites in all structures must be included.
    npt.assert_array_equal(
        sorted(all_eid_sid), sorted(list(product(range(n_structs), range(n_sites))))
    )


def test_train_decorate(decorator, undecorated_entries_standards):
    undecorated_entries, standards = undecorated_entries_standards
    if issubclass(decorator.__class__, NoTrainDecorator):
        assert decorator.is_trained

    decorator.train(undecorated_entries)
    entries = decorator.decorate(undecorated_entries)

    # There should not be too many fails given these conditions.
    num_fails = len([ent for ent in entries if ent is None])
    if issubclass(decorator.__class__, NoTrainDecorator):
        assert num_fails == 0
    else:
        assert num_fails / len(undecorated_entries) <= 0.95

    n_match = 0
    for ent_decor, ent_undecor, standard in zip(
        entries, undecorated_entries, standards
    ):
        if ent_decor is not None:
            assert np.isclose(ent_decor.structure.charge, 0)
            assert np.isclose(ent_decor.energy, ent_undecor.energy)
            assert ent_decor.data == ent_undecor.data
            match_standard = True
            for site1, site2, s_str in zip(
                ent_decor.structure, ent_undecor.structure, standard
            ):
                assert site1.specie.symbol == site2.specie.symbol
                assert site1.properties == site2.properties
                if issubclass(decorator.__class__, NoTrainDecorator):
                    if site1.specie.symbol == "O":
                        s_str = "O2-"
                    else:
                        s_str = "Ca2+"
                if s_str != str(site1.specie):
                    match_standard = False
            if match_standard:
                n_match += 1

    if issubclass(decorator.__class__, NoTrainDecorator):
        assert n_match == len(undecorated_entries)
    else:
        assert n_match / len(undecorated_entries) >= 0.9

    # if isinstance(decorator, MagneticChargeDecorator):
    #     for ent, undecor, std in zip(entries, undecorated_entries, standards):
    #         print("Undecorated:", undecor.structure.species)
    #         print("Decorated:", ent.structure.species)
    #         print("Charge:", ent.structure.charge)
    #         print("Standard:", std)
    #         print("\n")
    #     assert False


def test_serialize_gaussian():
    model = GaussianMixture(n_components=3)
    assert not MixtureGaussianDecorator.is_trained_gaussian_model(model)
    data = np.concatenate(
        [
            np.random.normal(1.0, 0.2, size=(200, 1)),
            np.random.normal(2.0, 0.1, size=(150, 1)),
            np.random.normal(3.0, 0.15, size=(300, 1)),
        ],
        axis=0,
    )

    model.fit(data)
    labels = model.predict(data)
    assert MixtureGaussianDecorator.is_trained_gaussian_model(model)

    model_dict = MixtureGaussianDecorator.serialize_gaussian_model(model)
    model_reload = MixtureGaussianDecorator.deserialize_gaussian_model(model_dict)
    assert MixtureGaussianDecorator.is_trained_gaussian_model(model_reload)
    labels_reload = model_reload.predict(data)

    npt.assert_array_equal(labels, labels_reload)


def test_gaussian_label_markings():
    # Test methods used to reference gaussian center indices in MoG and GpMinimizer.
    labels = np.array([20, 5, 9])
    std_clusters = np.array([1, 2, 0])
    std_values = np.array([-1, 0, 1])
    for _ in range(100):
        real_label_inds = np.random.choice(3, size=(100,), p=[0.2, 0.5, 0.3]).astype(
            int
        )
        real_labels = labels[real_label_inds]
        real_values = std_values[real_label_inds]
        cluster_inds = std_clusters[real_label_inds]

        refered_label_inds = [std_clusters.tolist().index(c) for c in cluster_inds]
        refered_labels = labels[refered_label_inds]
        refered_values = std_values[refered_label_inds]
        npt.assert_array_equal(refered_labels, real_labels)
        npt.assert_array_equal(refered_values, real_values)


def test_bad_decorator(undecorated_entries_standards):
    # Test behavior when a species property is not acceptable.
    class BadDecorator(BaseDecorator):
        decorated_prop_name = "whatever"
        required_prop_names = []

        def train(self, entries, reset=False):
            return None

        def is_trained(self):
            return True

        def decorate(self, entries):
            return [0] * len(entries)

        def _filter(self, entries):
            return entries

        def from_dict(cls, d):
            return None

    with pytest.raises(ValueError):
        _ = BadDecorator()._process(undecorated_entries_standards[0], {0: {0: 1}})
