import unittest
import numpy as np

from ..comp_space import CompSpace
from pymatgen import Specie,Element
from smol.cofe.space.domain import Vacancy

#Charged test case, without vacancy
class TestCompSpace1(unittest.TestCase):
    def setUp(self) -> None:
        li = Specie.from_string('Li+')
        mn = Specie.from_string('Mn3+')
        ti = Specie.from_string('Ti4+')
        o = Specie.from_string('O2-')
        p = Specie.from_string('P3-')
        
        self.bits = [sorted([li,mn,ti]),sorted([p,o])]
        self.nbits = [[0,1,2],[0,1]]
        self.unit_n_swps = [(0,2,0),(1,2,0),(0,1,1)]
        self.chg_of_swps = [-3,-1,-1]
        self.swp_ids_in_sublat = [[0,1],[2]]
        self.flip_vecs = [[-1,2,1],[0,-1,1]]
        
        op1 = {'from':{0:{0:1, 2:1}, 1:{1:1}}, \
               'to':{0:{1:2}, 1:{0:1}}}
        op2 = {'from':{0:{1:1}, 1:{1:1}}, \
               'to':{0:{2:1}, 1:{0:1}}}
        self.flip_table = [op1,op2]
        
        visualized_operation_1 = \
        '1 Li+(0) + 1 Ti4+(0) + 1 O2-(1) -> 2 Mn3+(0) + 1 P3-(1)'
        visualized_operation_2 = \
        '1 Mn3+(0) + 1 O2-(1) -> 1 Ti4+(0) + 1 P3-(1)'
        self.visualized_operations = visualized_operation_1 + \
                                    '\n'+ \
                                    visualized_operation_2

        self.vertices = np.array()

        self.comp_space = CompSpace(self.bits)

    def test_swps(self):
        self.assertEqual(self.comp_space.unit_n_swps,self.unit_n_swps)
        self.assertEqual(self.comp_space.chg_of_swps,self.chg_of_swps)
        self.assertEqual(self.comp_space.swp_ids_in_sublat,self.swp_ids_in_sublat)

    def test_space_specs(self):
        self.assertEqual(self.comp_space.bkgrnd_chg,2)
        self.assertEqual(self.comp_space.unconstr_dim,3)
        self.assertEqual(self.comp_space.is_charge_constred,True)
        self.assertEqual(self.comp_space.dim,2)
        self.assertEqual(self.comp_space.dim_nondisc,5)

    def test_flip_table(self):
        self.assertEqual(self.comp_space.unit_spc_basis.tolist(),self.flip_vecs)
        self.assertEqual(self.comp_space.min_flips,self.flip_table)
        self.assertEqual(self.comp_space.min_flip_strings,self.visualized_operations)

    def test_vertices(self):
        
