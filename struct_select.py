from __future__ import division
from __future__ import unicode_literals

__author__ = 'Peichen Zhong & Fengyu Xie'
__version__ = 'Dev'

"""
2019-09-28 Notes from Fengyu Xie:
I edited this file to fit it into the CEAuto project
"""

import os
import sys
import argparse
import json
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from cluster_expansion.ce import ClusterExpansion
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import *
from pymatgen.io.vasp.inputs import *
from pymatgen.core.structure import Structure
from itertools import permutations
from monty.serialization import loadfn, dumpfn
from cluster_expansion.mc import *
from math import gcd
from functools import reduce
import random
from itertools import permutations
from operator import mul
from functools import partial, reduce
import multiprocessing as mp
import pickle
import scipy.io as sio
import random

#from cluster_expansion.ce_opt import *
#from cluster_expansion.ce_select import *

def find_indices(A, init_A):
    indices = []
    for i in range(init_A.shape[0]):
        index = np.argwhere(np.all(A == init_A[i], axis=1))
        indices.append(index[0][0])
    indices = np.array(indices)
    return indices

def calc_rmse(A, f, ecis):
    rmse = np.sqrt(np.average((np.dot(A, ecis) - f)**2))
    return rmse *500

def leverage_score(A, k=0):
    m, d = A.shape
    U, _, VT =np.linalg.svd(A)
    
    if k==0:
        k = d
    
    U = U[:,:k]
    
    L = np.linalg.norm(U, axis = 1)
   
    return L**2

def mkdir(path): 
    folder = os.path.exists(path)
    if not folder:         
        os.makedirs(path)           
    else:
        print("Folder exists")
        

def CX_decompose(A, C):
    """C is a matrix of selected rows of A"""
    
    X = np.dot(np.linalg.pinv(C), A)
    
    return X

def CUR_decompose(G, C, R):
    """ calcualate U s.t. G = CUR """
    
    C_inv = np.linalg.pinv(C)
    R_inv = np.linalg.pinv(R)
    U = np.dot(np.dot(C_inv, G), R_inv)
    
    return U

def Nystrom_k(A ,indices, k = 5, full_rank = True):
    
    G = np.dot(np.transpose(A), A)
    C = G[:, indices]
    R = G[indices, :]

    skeleton = np.dot(np.transpose(A[:,indices]), A[:,indices])

    u, s, vT = np.linalg.svd(skeleton)

    s_inv = np.zeros(s.shape)
    
    rank = np.linalg.matrix_rank(skeleton)
    
    if full_rank:
    
        s_inv[:rank] = 1 / s[:rank]
        G_app = np.dot(np.dot(C, np.dot(u * s_inv, vT)), R) 
    else:
        s_inv[:k] = 1 / s[:k]
        G_app = np.dot(np.dot(C, np.dot(u * s_inv, vT)), R) 
    
#     skeleton_error = np.eye(s.shape[0]) - np.dot(u * s_inv, vT)
#     error = np.dot(np.dot(C, skeleton), R)
    
#     G_app = np.dot(np.dot(C, np.linalg.inv(skeleton)), R) 
    
    return G_app, rank


class StructureSelector():
    def __init__(self, ce, n_iter = 100, n_step = 5, solver = 'CUR'):
        """
        Args:
            ce: cluster expansion used to calculate feature matrix from
            n_iter: number of iterations used in improving the feature matrix approximation.
            n_step: number of structures to add into the pool each time.
            solver: solver of applied in feature matrix decomposition
    
        Notes:
            feature_matrix contains no Ewald term. Only cluster features included
        
        """
        self.ce = ce
        #self.n_init = n_init
        self.n_iter = n_iter
        self.n_step = n_step

        if self.ce.use_ewald:
            if self.ce.use_inv_r:
                N_sp = sum([len(site.species_and_occu) for site in ce.structure])
                self.N_eweci = 1+N_sp+N_sp*(N_sp-1)//2
            else:
                self.N_eweci = 1
        else:
            self.N_eweci = 0
        self.solver = solver
        
    def _get_femat(self,pool,mat_pool=None):
        if mat_pool is not None and len(mat_pool)==len(pool):
            feature_matrix = []
            for struct,mat in zip(pool,mat_pool):
                cs = self.ce.supercell_from_matrix(mat)
                if self.N_eweci:
                    feature_matrix.append(cs.corr_from_structure(struct)[:-self.N_eweci])
                else:
                    feature_matrix.append(cs.corr_from_structure(struct))
            feature_matrix=np.array(feature_matrix)
        else:
            if self.N_eweci:
                feature_matrix = np.array([self.ce.corr_from_structure(structure)[:-self.N_eweci] for structure in pool])
            else:
                feature_matrix = np.array([self.ce.corr_from_structure(structure) for structure in pool])       
        return feature_matrix        

    def initialization(self, pool, mat_pool = None,n_init = 10):
        # Using only random selection for C and R is enough.
        """
            This method initialize a selection from pool.
        """
        
        feature_matrix = self._get_femat(pool, mat_pool=mat_pool)

        if self.solver == 'CUR':
            selected_ids = self.Nystrom_selection(feature_matrix, n_init= n_init)
        elif self.solver == 'CX':
            selected_ids = self.CX_selection(feature_matrix, n_init= n_init)
        else:
            raise ValueError('Selection algorithm not implemented!')
        
        return selected_ids
            
    def select_new(self, old_pool, new_pool, old_mat_pool = None,\
                   new_mat_pool = None, n_probe = 1):
        """
        Having an existing pool, select the best n_probe structures from a new pool.
        Does not deduplicate. Do that before selection!
        """

        old_feature_matrix = self._get_femat(old_pool,\
                                             mat_pool=old_mat_pool)
        #print('Old pool finished')
        new_feature_matrix = self._get_femat(new_pool,\
                                             mat_pool=new_mat_pool)

        d = old_feature_matrix.shape[1]
        domain = np.eye(d)
        # Using an identity matrix as domain
        
        init_A = old_feature_matrix.copy()
        pool_A = new_feature_matrix.copy()
        
        old_kernel = np.dot(np.transpose(init_A), init_A)
        old_inv = np.linalg.pinv(old_kernel)
        # Used Penrose-Moore inverse
        
        num_pool = pool_A.shape[0] # number of structures in the pool
        reduction = np.zeros(num_pool)

        for i in range(num_pool):
            trial_A = np.concatenate((init_A, pool_A[i].reshape(1, d)), axis =0)

            kernal = np.dot(np.transpose(trial_A), trial_A)
            inv = np.linalg.pinv(kernal)

            reduction[i] = np.sum(np.multiply( (inv-old_inv), domain))

        indices = np.argsort(reduction)

        return indices[:min(n_probe,len(indices))]

    def CX_selection(self, feature_matrix, n_init=10):
                      
        origin_A = feature_matrix.copy()
        A = feature_matrix.copy()

        G = np.dot(origin_A, np.transpose(origin_A))
        total_indices_inall = [i for i in range(origin_A.shape[0])]
        L = leverage_score(A=A[:,1:])

        trial_indices = []
        total_indices = [i for i in range(A.shape[0])]

        for n in range(self.n_step,min(n_init,A.shape[0]),self.n_step):

            error = np.linalg.norm(origin_A)*10000
            return_indices_step = None

            for i in range(self.n_iter):
                trial_indices_step = random.sample(total_indices, self.n_step)
                trial_indices_current = trial_indices+trial_indices_step
                
                C = G[:, trial_indices_current]

                X = CX_decompose(G, C)

                re = np.linalg.norm(G - np.dot(C, X))

                if re < error:
                    return_indices_step = trial_indices_step
                    error = re
            #print(error)
            trial_indices = trial_indices+return_indices_step
            total_indices = [idx for idx in total_indices if idx not in return_indices_step]
            
        return trial_indices

    def Nystrom_selection(self, feature_matrix, n_init = 10):
               
        origin_A = feature_matrix.copy()
        A = feature_matrix.copy()

        G = np.dot(origin_A, np.transpose(origin_A))
        total_indices_inall = [i for i in range(origin_A.shape[0])]
        L = leverage_score(A=A[:,1:])

        trial_indices = []
        total_indices = [i for i in range(A.shape[0])]

        for n in range(self.n_step,min(n_init,A.shape[0]),self.n_step):

            error = np.linalg.norm(origin_A)*10000
            return_indices_step = None

            for i in range(self.n_iter):
                trial_indices_step = random.sample(total_indices, self.n_step)
                trial_indices_current = trial_indices+trial_indices_step
                
                C = G[:, trial_indices_current]
                R = G[trial_indices_current,:]

                U = CUR_decompose(G, C, R)

                re = np.linalg.norm(G - np.dot(np.dot(C, U),R))

                if re < error:
                    return_indices_step = trial_indices_step
                    error = re
            #print(error)
            trial_indices = trial_indices+return_indices_step
            total_indices = [idx for idx in total_indices if idx not in return_indices_step]
            
        return trial_indices
