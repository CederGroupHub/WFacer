__author___ = 'Fengyu Xie'

"""
Mathematic utilities, including linear algebra, combinatorics and integerization
"""

import numpy as np
from itertools import combinations,product
from functools import reduce
from copy import deepcopy
import random
import math

from sympy.ntheory import factorint

#Linear algebra
def CUR_decompose(G, C, R):
    """ calcualate U s.t. G = CUR """
    
    C_inv = np.linalg.pinv(C)
    R_inv = np.linalg.pinv(R)
    U = np.dot(np.dot(C_inv, G), R_inv)
    
    return U

def select_rows(femat,n_select=10,old_femat=[],method='CUR',keep=[]):

    """
    Selecting a certain number of rows that recovers maximum amount of kernel
    information from a matrix, or given an old feature matrix, select a certain
    number of rows that maximizes information gain from a new feature matrix.

    Inputs: 
        femat(2D arraylike):
            New feature matrix to select from
        n_select(int):
            Number of new rows to select
        old_femat(2D arraylike):
            Old feature matrix to compare with
        method(str):
            Row selection method.
            'CUR'(default):
                select by recovery of CUR score
            'random':
                select at full random
        keep(List of ints):
            indices of rows that will always be selected. By default, no row
            has that priviledge.
    Outputs:
        List of ints. Indices of selected rows in femat.

    """
    A = np.array(femat)
    n_pool,d = A.shape
    domain = np.eye(d)
    # Using an identity matrix as domain matrix
    
    if len(keep) > n_pool:
        raise ValueError("Rows to keep more than rows you have!")      
    n_add = max(min(n_select-len(keep),n_pool-len(keep)),0)

    trial_indices = deepcopy(keep)
    total_indices = [i for i in range(n_pool) if i not in keep]

    if len(old_femat) == 0: #init mode

        G = A@A.T

        for i in range(n_add):
            err = 1e8
            return_index = None

            if method == 'CUR':             
                for i in range(100):  
                #Try one row each time, optimize for 100 iterations
                    trial_index = random.choice(total_indices)
                    trial_indices_current = trial_indices+[trial_index]
                    
                    C = G[:, trial_indices_current]
                    R = G[trial_indices_current,:]
        
                    U = CUR_decompose(G, C, R)
        
                    err_trial = np.linalg.norm(G - np.dot(np.dot(C, U),R))
        
                    if err_trial < err:
                        return_index = trial_index
                        err = err_trial

            elif method == 'random':
                return_index = random.choice(total_indices)
 
            else:
                raise NotImplementedError

            trial_indices.append(return_index)
            total_indices.remove(return_index)

    else: #add mode
        
        old_A = np.array(old_femat)        
        old_cov = old_A.T @ old_A
        old_inv = np.linalg.pinv(old_cov)
        # Used Penrose-Moore inverse

        reduction = np.zeros(len(total_indices))

        if method == 'CUR':
            for i_id, i in enumerate(total_indices):
                trial_A = np.concatenate((old_A, A[i].reshape(1, d)), axis =0)

                trial_cov = trial_A.T @ trial_A
                trial_inv = np.linalg.pinv(trial_cov)

                reduction[i_id] = np.sum(np.multiply( (trial_inv-old_inv), domain))

            add_indices = [total_indices[iid] for iid in np.argsort(reduction)[:n_add]]

        elif method == 'random':
            add_indices = sorted(random.sample(total_indices,n_add))

        else:
            raise NotImplementedError

        trial_indices = trial_indices + add_indices
        total_indices = [i for i in total_indices if i not in add_indices]

    trial_indices = sorted(trial_indices)
    total_indices = sorted(total_indices)
               
    return trial_indices


#Number theory
def get_diag_matrices(n,d=3):
    """
    Get d dimensional positive integer diagonal matrices with
    det(M)=n
    """
    factors = factorint(n)
    normv = [1 for i in range(d)]

    prime_partitions = []
    for factor,num in factors.items():
        limiters = [(0,num) for i in range(d)]
        partitions = get_integer_grid(normv,right_side=num,limiters=limiters)
        prime_partitions.append(partitions)

    factor_partitions = []
    for p_combo in product(*prime_partitions):
        factor_partition = [1 for i in range(d)]
        for f_id,factor in enumerate(factors):
            for p_id, power in enumerate(p_combo[f_id]):
                factor_partition[p_id]*=(factor**p_combo[f_id][p_id])
        factor_partitions.append(factor_partition)

    mats = [np.diag(f_part).tolist() for f_part in sorted(factor_partitions)]
    return mats


def GCD(a,b):
    """ The Euclidean Algorithm, giving positive GCD's """
    if round(a)!=a or round(b)!=b:
        raise ValueError("GCD input must be integers!")
    a = abs(a)
    b = abs(b)
    while a:
        a, b = b%a, a
    return b    


def GCD_list(l):
    """ Find GCD of a list of numbers """
    if len(l)<1:
        return None
    elif len(l)==1:
        return l[0]
    else:
        return reduce(lambda a,b:GCD(a,b),l)


def LCM(a,b):
    if a==0 and b==0:
        return 0
    elif a==0 and b!=0:
        return b
    elif a!=0 and b==0:
        return a
    else:
        return a*b // GCD(a,b)


def LCM_list(l):
    if len(l)<1:
        return None
    elif len(l)==1:
        return l[0]
    else:
        return reduce(LCM,l)


# Combinatoric and intergerization
def reverse_ordering(l,ordering):
    """
    Given a mapping order of list, reverse that order
    and return the original list
    """
    original_l = [0 for i in range(len(l))]
    for cur_id,ori_id in enumerate(ordering):
        original_l[ori_id]=l[cur_id]
    return original_l


def combinatorial_number(n,m):
    """
    Calculate the combinatorial number when choosing m instances from a set of n instances.
    m,n: 
        integers.
    """
    if m>n:
        return 0
    return math.factorial(n)//(math.factorial(m)*math.factorial(n-m))


# Partition selection tools
def choose_section_from_partition(probs):
    """
    This function choose one section from a partition based on each section's
    normalized probability.
    Input:
        probs: 
            array-like, probabilities of each sections. If not normalized, will
            be normalized.
    Output:
        id: 
            The id of randomly chosen section.   
    """
    N_secs = len(probs)
    if N_secs<1:
        raise ValueError("Segment can't be selected!")

    norm_probs = np.array(probs)/np.sum(probs)
    upper_bnds = np.array([sum(norm_probs[:i+1]) for i in range(N_secs)])
    rand_seed = np.random.rand()

    for sec_id,sec_upper in enumerate(upper_bnds):
        if sec_id==0:
            sec_lower = 0
        else:
            sec_lower = upper_bnds[sec_id-1]
        if rand_seed>=sec_lower and rand_seed<sec_upper:
            return sec_id
    
    raise ValueError("Segment can't be selected.")


def enumerate_partitions(n_part,enum_fold,constrs=None,quota=1.0):
    """
    Recursivly enumerates possible partitions of a line section from 0.0 to 
    quota or from lower-bound to upper-bound if constrs is not None.
    Inputs:
        n_part(Int): 
            Number of partitions to be enumerated
        enum_fold(Int):
            Step of enumeration = quota/enum_fold.
        constrs(List of float tuples):
            lower and upper bound coustraints of each partition cut point.
            If None, just choose (0.0,quota) for each tuple.
        quota(float):
            Length of the line section to cut on.
    """
    if constrs is None:
        constrs = [(0.0,quota) for i in range(n_part)]

    lb,ub = constrs[0]
    ub = min(quota,ub)
    lb_int = int(np.ceil(lb*enum_fold))
    ub_int = int(np.floor(ub*enum_fold))

    if n_part < 1:
        raise ValueError("Can't partition less than 1 sections!")
    if n_part == 1:
        if quota == ub:
            return [[float(ub_int)/enum_fold]]
        else:
            return []

    this_level = [float(i)/enum_fold for i in range(lb_int,ub_int+1)]
    accumulated_enums = []
    for enum_x in this_level:
        next_levels = enumerate_partitions(n_part-1,enum_fold,\
                            constrs[1:],quota=quota-enum_x)
        if len(next_levels)!=0 and len(next_levels[0])==n_part-1:
            accumulated_enums.extend([[enum_x]+xs for xs in next_levels])

    return accumulated_enums


#Grid
def get_center_grid(center,step_per_dim,n_step_per_dim):
    """
    Generates a d-dimensional grid around center
    Args:
        center(1D arraylike):
            d-dimensional vector
        step_per_dim(float or List[float]):
            step sizes of grid per dimension
        n_step_per_dim(int or List[int]):
            number of steps of grid per dimension
    Returns: 
        List[List[float]], all grid points
    """
    d = len(center)
    center = np.array(center)

    if isinstance(step_per_dim,float):
        step_per_dim = [step_per_dim for i in range(d)]

    if isinstance(n_step_per_dim,(int,np.int64)):
        n_step_per_dim = [n_step_per_dim for i in range(d)]

    if len(step_per_dim)!=d or len(n_step_per_dim)!=d:
        raise ValueError("Length of given list parameters does not match dim of grid.")

    dims = [np.linspace(-s*(n-1)/2,s*(n-1)/2,n).tolist() \
            for s,n in zip(step_per_dim,n_step_per_dim)]

    return (np.array(list(product(*dims)))+center).tolist()
