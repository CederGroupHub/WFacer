from CEAuto.specie_decorators import (decorator_factory,
                                      MagneticChargeDecorator,
                                      PmgGuessChargeDecorator,
                                      FixedChargeDecorator)
from pymatgen.core import Lattice, Structure
import numpy as np
import random
from pymatgen.core import Species, DummySpecies

from .utils import assert_msonable

all_decorator_names = ["magnetic-charge", "pmg-guess-charge", "fixed-charge"]
all_decorators = [MagneticChargeDecorator, PmgGuessChargeDecorator, FixedChargeDecorator]

def test_decorator_factory(prim):
    decorator_names = ["Magcharge"]
    labels_table = {"Li": [1], "Ca": [1], "Br": [-1]}
    decorator_args = [{}]
    classes = [MagneticChargeDecorator]
    for name, args, cls in zip(decorator_names, decorator_args, classes):
        assert isinstance(decorator_factory(name, labels_table, **args),
                          cls)


def test_decorate_single():
    lat = Lattice.cubic(1)
    s = Structure(lat,['B','B','C','C'],
                  [[0,0,0],[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]])
    decor_keys = ["charge", "spin"]
    decor_values = [[1, 1, -1, -1], [0, 1, 0, 1]]
    s_dec = decorate_single_structure(s, decor_keys, decor_values)
    assert s_dec.charge == 0
    for site in s_dec:
        assert isinstance(site.specie, (Species, DummySpecies))

    decor_values = [[-1,-3,1,1],[1,1,1,1]]
    s_dec = decorate_single_structure(s, decor_keys, decor_values)
    assert s_dec is None
    s_dec = decorate_single_structure(s, decor_keys, decor_values, max_charge=2)
    assert isinstance(s_dec, Structure)

def test_mag_charge_decorator():
    lat = Lattice.cubic(1)
    s = Structure(lat,['B','B','C','C'],
                  [[0,0,0],[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]])
    s_pool = [s.copy() for sid in range(300)]

    def get_random_neutral_labels():
        ca_labels = random.choice([[1,3], [3, 1], [2, 2]])
        an_labels = [-2, -2]
        return ca_labels + an_labels

    label_true = [get_random_neutral_labels()
                  for j in range(300)]
    labels_table = {'B':[1,2,3],
                    'C':[-2]}

    mags = []

    for i in range(300):
        mags_i = []
        for j in range(4):
            if label_true[i][j] == 1:
                mags_i.append(np.random.normal(0.1,0.3))
    
            if label_true[i][j] == 2:
                mags_i.append(np.random.normal(1.0,0.25))
    
            if label_true[i][j] == 3:
                mags_i.append(np.random.normal(2.1,0.3))
   
            if label_true[i][j] == -2:
                mags_i.append(np.random.normal(0.0, 0.5))
    
        mags.append(mags_i)
    
    properties = {'magnetization': mags}
    
    decor = MagchargeDecorator(labels_table)
    decor.train(s_pool, properties)
    label_assign = decor.assign(s_pool, properties)['charge']
    fails = (np.sum(label_assign, axis=-1) != 0)
    mismatches = ~np.all(np.array(label_assign) == np.array(label_true), axis=-1)

    print("Unbalanced assignments:")
    for l1, l2 in zip(np.array(label_true)[fails], np.array(label_assign)[fails]):
        print("True label:", l1)
        print("Assignment:", l2)

    print("Mismatch assignments:")
    for l1, l2 in zip(np.array(label_true)[mismatches], np.array(label_assign)[mismatches]):
        print("True label:", l1)
        print("Assignment:", l2)

    assert np.average(fails) <= 0.2
    assert np.average(mismatches) <= 0.2

    label_true_ca = np.array(label_true)[:, :2].flatten()
    label_assign_ca = np.array(label_assign)[:, :2].flatten()

    # Success rates for each specie
    assert np.average(label_assign_ca[label_true_ca==1] == 1) >= 0.8
    assert np.average(label_assign_ca[label_true_ca==2] == 2) >= 0.8
    assert np.average(label_assign_ca[label_true_ca==3] == 3) >= 0.8
    assert np.average(label_assign_ca[label_true_ca==1] == 3) == 0 # Non-crossing.
    assert np.average(label_assign_ca[label_true_ca==3] == 1) == 0

    decor2 = MagchargeDecorator.from_dict(decor.as_dict())
    label_assign2 = decor2.assign(s_pool, properties)['charge']
    assert label_assign2 == label_assign

    assert_msonable(decor)
