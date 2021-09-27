from CEAuto.specie_decorator import *
from pymatgen.core import Lattice,Structure
import numpy as np
import random

def test_mag_charge_decorator():
    lat = Lattice.cubic(1)
    s = Structure(lat,['B','B','C','C'],[[0,0,0],[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]])
    s_pool = [s.copy() for sid in range(100)]
    
    def get_random_neutral_labels():
        ca_labels = random.choice([[1,3], [3, 1], [2, 2]])
        an_labels = [-2, -2]
        return ca_labels + an_labels

    label_true = [get_random_neutral_labels() for j in range(100)]
    labels_table = {'B':[1,2,3], 'C':[-2]}
    
    mags = []
    
    for i in range(100):
        mags_i = []
        for j in range(4):
            if label_true[i][j] == 1:
                mags_i.append(np.random.normal(0.1,0.3))
    
            if label_true[i][j] == 2:
                mags_i.append(np.random.normal(1.0,0.25))
    
            if label_true[i][j] == 3:
                mags_i.append(np.random.normal(2.1,0.3))
   
            if label_true[i][j] == -2:
                mags_i.append(np.random.normal(0.0, 0.1))
    
        mags.append(mags_i)
    
    properties = {'magnetization':mags}
    
    decor = MagChargeDecorator(labels_table)
    decor.train(s_pool, properties)
    label_assign = decor.assign(s_pool, properties)['charge']
    n_fails = 0

    for i in range(100):
        if label_assign[i] is None:
            n_fails += 4
        else:
            for j in range(4):
                if label_assign[i][j]!=label_true[i][j]:
                    n_fails+=1
    
    n_str_fails = 0
    for i in range(100):
        if label_assign[i] is None:
            n_str_fails += 1
        else:
            for j in range(4):
                if label_assign[i][j]!=label_true[i][j]:
                    n_str_fails+=1
                    break
    
    assert n_fails<=80
    assert n_str_fails<=20
    
    decor2 = MagChargeDecorator.from_dict(decor.as_dict())
    label_assign2 = decor2.assign(s_pool, properties)['charge']
    assert label_assign2 == label_assign