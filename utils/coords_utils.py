import numpy as np

def Rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],\
                             [0,0,1]])
def Rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],\
                             [0,np.sin(theta),np.cos(theta)]])

def Rot_Matrix(alpha,beta,gamma):
    """
    Gives the transformation matrix of a Euler rotation.
    transformed_coords (shape=N*3) = untransformed_coords (shape=N*3) @ Rot_matrix
    (Right multiplied)
    """
    return Rz(-1.0*alpha)@Rx(-1.0*beta)@Rz(-1.0*gamma)

def Is_Nonlinear(coords):
    """
    This function checks whether the given atomic cluster is a linear one, and 
    also returns the index of the first atom that lies out of the line of 
    coords[0] and coords[1]
    """
    if len(coords)<2:
        return False,None
    elif len(coords)==2:
        retuen False,1
    else:
        shifted = np.array(coords)-np.average(coords,axis=0)
        for i in range(2,len(coords)):
            if np.linalg.norm(np.cross(coords[1]-coords[0],coords[i]-coords[0]))!=0:
                return True,i
        return False,1

def Standardize_Coords(atom_coords):
    """
    Given a set of catesian atomic coordinates, this function trys to 'standardize'
    the coordinates by aligning the center of coordinates, the first atom, and the
    second atom into xz plane, and the vector from the center to the first atom into
    the z-axis direction.
    If the atomic coordinate set only has one atom, then we just shift it to [0,0,0]
    """
    untransformed_coords = np.array(atom_coords)-np.average(atom_coords,axis=0)
    if len(atom_coords)<2:
        return untransformed_coords
    else:
        v_ref0 = untransformed_coords[0]
        
        is_linear,ref1_id=is_linear(untransformed_coords)
        v_ref1 = untransformed_coords[ref1_id]

        ex = np.array([1,0,0])
        ey = np.array([0,1,0])
        ez = np.array([0,0,1])
    
        ex_p = np.cross(v_ref0,ez)/np.linalg.norm(np.cross(v_ref0,ez))\
               if np.linalg.norm(np.cross(v_ref0,ez))!=0 else ex

        alpha_t = np.arccos(np.dot(np.cross(v_ref0,ez),ex)/np.linalg.norm(np.cross(v_ref0,ez)))\
                  if np.linalg.norm(np.cross(v_ref0,ez))!=0 else 0
        alpha_sgn = np.sign(np.dot(np.cross(ex_p,ex),ez))
        #rotation_result x roration_start_point * rotation_axis

        beta_t  = np.arccos(np.dot(v_ref0,ez)/np.linalg.norm(v_ref0))
        beta_sgn  = np.sign(np.dot(np.cross(v_ref0,ez),ex_p))

        n = np.cross(v_ref0,v_ref1)
        if np.linalg.norm(n) == 0:
            gamma_t = 0
            gamma_sgn = 0
        else:
            en = n/np.linalg.norm(n)
            ex_pp = ex_p-np.dot(ex_p,en)*en
            ex_pp = ex_pp/np.linalg.norm(ex_pp)
            gamma_t = np.arccos(np.dot(ex_p,ex_pp))
            gamma_sgn = np.sign(np.dot(np.cross(ex_pp,ex_p),v_ref0))
    
        Rz_a = Rz(-1.0*alpha_t*alpha_sgn)
        Rx_b = Rx(-1.0*beta_t*beta_sgn)
        Rz_g = Rz(-1.0*gamma_t*gamma_sgn)
        transmat = Rz_a@Rx_b@Rz_g

        return untransformed_coords@transmat
