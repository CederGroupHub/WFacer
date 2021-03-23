from CEAuto.specie_decorator import *
from pymatgen import Lattice,Structure
import numpy as np
import random

def test_mag_charge_decorator():
    lat = Lattice.cubic(1)
    s = Structure(lat,['B','B','B','B'],[[0,0,0],[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]])
    s_pool = [s.copy() for sid in range(100)]
    
    label_true = [[random.choice([1,2,3]) for i in range(4)] for j in range(100)]
    labels_table = {'B':[1,2,3]}
    
    mags = []
    
    for i in range(100):
        mags_i = []
        for j in range(4):
            if label_true[i][j] == 1:
                mags_i.append(np.random.normal(0.1,0.2))
    
            if label_true[i][j] == 2:
                mags_i.append(np.random.normal(1.0,0.15))
    
            if label_true[i][j] == 3:
                mags_i.append(np.random.normal(2.1,0.1))
    
        mags.append(mags_i)
    
    mags_3d = [mags]
    
    decor = MagChargeDecorator(labels_table)
    decor.train(s_pool,mags_3d)
    label_assign = decor.assign(s_pool,mags,check_neutral=False)['charge']
    n_fails = 0
    
    for i in range(100):
        for j in range(4):
            if label_assign[i][j]!=label_true[i][j]:
                n_fails+=1
    
    n_str_fails = 0
    for i in range(100):
        for j in range(4):
            if label_assign[i][j]!=label_true[i][j]:
                n_str_fails+=1
                break
    
    assert n_fails<40
    assert n_str_fails<10
    
    decor2 = MagChargeDecorator.from_dict(decor.as_dict())
    label_assign2 = decor2.assign(s_pool,mags,check_neutral=False)['charge']
    for i in range(100):
        for j in range(4):
            assert label_assign[i][j]==label_assign2[i][j]
