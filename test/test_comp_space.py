import os,sys
this_file_path = os.path.abspath(__file__)
this_file_dir = os.path.dirname(this_file_path)
parent_dir = os.path.dirname(this_file_dir)
sys.path.append(parent_dir)
sys.path.append(this_file_dir)

from ce_components.comp_space import *
from ce_components.specie import *

li = CESpecie.from_string('Li+')
mn = CESpecie.from_string('Mn3+')
ti = CESpecie.from_string('Ti4+')
o = CESpecie.from_string('O2-')
p = CESpecie.from_string('P3-')
bits = [[li,mn,ti],[p,o]]
nbits = [[0,1,2],[0,1]]

unit_n_swps_true = [(0,2,0),(1,2,0),(0,1,1)]
chg_of_swps_true = [-3,-1,-1]
swp_ids_in_sublat_true = [[0,1],[2]]

prim_lat_vecs_test = [[-1,2,1],[0,-1,1]]
op1 = {'from':{0:{0:1, 2:1}, 1:{1:1}}, \
       'to':{0:{1:2}, 1:{0:1}}}
op2 = {'from':{0:{1:1}, 1:{1:1}}, \
       'to':{0:{2:1}, 1:{0:1}}}
operations_true = [op1,op2]

visualized_operations_1 = \
'1 Li+(0) + 1 Ti4+(0) + 1 O2-(1) -> 2 Mn3+(0) + 1 P3-(1)'
visualized_operations_2 = \
'1 Mn3+(0) + 1 O2-(1) -> 1 Ti4+(0) + 1 P3-(1)'
visualized_operations_true = visualized_operations_1 + \
                        '\n'+ \
                        visualized_operations_2

bkgrnd_chg_true = 2
unconstr_dim_true = 3
is_charge_constred_true = True
dim_true = 2

def test_func1():
    unit_n_swps, chg_of_swps, swp_ids_in_sublat = get_unit_swps(bits)
    assert (unit_n_swps == unit_n_swps_true)\
       and (chg_of_swps == chg_of_swps_true)\
       and (swp_ids_in_sublat == swp_ids_in_sublat_true)

def test_func2():
    unit_n_swps, chg_of_swps, swp_ids_in_sublat = get_unit_swps(bits)
    operations = flipvec_to_operations(unit_n_swps,nbits,prim_lat_vecs_test)

    assert operations == operations_true

def test_func3():
    unit_n_swps, chg_of_swps, swp_ids_in_sublat = get_unit_swps(bits)
    operations = flipvec_to_operations(unit_n_swps,nbits,prim_lat_vecs_test)
    vis_ops = visualize_operations(operations,bits)
    assert vis_ops == visualized_operations_true

def test_clsfunc_init():
    cs = CompSpace(bits)
