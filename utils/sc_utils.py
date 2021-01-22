__author__ = "Fengyu Xie"

"""
Utility functions to enumerate supercell matrices.
"""
import numpy as np
from .math_utils import get_diag_matrices

from pymatgen import Lattice

def is_proper_sc(sc_matrix,lat,max_cond=8,min_angle=30):
    """
    Assess the skewness of a given supercell matrix. If the skewness is 
    too high, then this matrix will be dropped.
    Inputs:
        sc_matrix(Arraylike):
            Supercell matrix
        lat(pymatgen.Lattice):
            Lattice vectors of a primitive cell
        max_cond(float):
            Maximum conditional number allowed of the supercell lattice
            matrix. By default set to 8, to prevent overstretching in one
            direction
        min_angle(float):
            Minmum allowed angle of the supercell lattice. By default set
            to 30, to prevent over-skewing.
    Output:
       Boolean
    """
    newmat = np.matmul(lat.matrix, sc_matrix)
    newlat = Lattice(newmat)
    angles = [newlat.alpha,newlat.beta,newlat.gamma,\
              180-newlat.alpha,180-newlat.beta,180-newlat.gamma]

    return abs(np.linalg.cond(newmat))<=max_cond and \
           min(angles)>min_angle

def enumerate_matrices(det, lat,\
                           transmat=[[1,0,0],[0,1,0],[0,0,1]],\
                           max_sc_cond = 8,\
                           min_sc_angle = 30):
    """
    Enumerate proper matrices with maximum det up to a number.
    4 steps are used in the size enumeration.
    Will give 1 unskewed matrix and up to 3 skewed matrices.
    We add skewed matrices to avoid symmtric duplicacy of clusters.
    Inputs:
        det(int):
            Required determinant size of enumerated supercell
            matrices
        lat(pymatgen.Lattice):
            Lattice vectors of a primitive cell
        transmat(2D arraylike):
            Symmetrizaiton matrix to apply on the primitive cell, in 
            order to pre-define an 'unskewed supercell'.
            For example, in FCC rhombohydral primitive cell, apply
            [[-1,1,1],[1,-1,1],[1,1,-1]] to convert into conventional 
            FCC cubic cell.
        max_cond(float):
            Maximum conditional number allowed of the skewed supercell
            matrices. By default set to 8, to prevent overstretching in one
            direction
        min_angle(float):
            Minmum allowed angle of the supercell lattice. By default set
            to 30, to prevent over-skewing.
    Outputs:
        List of 2D lists.
    """
    trans_size = int(round(abs(np.linalg.det(self.transmat))))
    if det%trans_size!=0:
        raise ValueError("Supercell size must be divisible by transformation matrix determinant!")
    scs_unsk=get_diag_matrices(det//trans_size)

    scs_unsk = [np.matmul(sc,transmat,dtype=np.int64).tolist() for sc in scs_unsk \
           if is_proper_sc(np.matmul(sc,transmat), lat,\
                            max_sc_cond = max_sc_cond,\
                            min_sc_angle = min_sc_angle)]

    #Take the unskewed matrix with minimal conditional number
    sc_unsk = sorted(scs_unsk,key=lambda x:np.linalg.cond(x))[0]
    n1 = sc_unsk[0][0]
    n2 = sc_unsk[1][1]
    n3 = sc_unsk[2][2]
    #n1>n2>n3, already sorted in get_diag_matrices
    sc_sk1 = deepcopy(sc_unsk)
    sc_sk2 = deepcopy(sc_unsk)
    sc_sk3 = deepcopy(sc_unsk)
    
    sc_sk1[0][1] = np.random.choice(np.arange(n1))
    sc_sk2[0][2] = np.random.choice(np.arange(n1))
    sc_sk3[1][2] = np.random.choice(np.arange(n2))

    selected_scs = [sc_unsk,sc_sk1,sc_sk2,sc_sk3]

    return selected_scs