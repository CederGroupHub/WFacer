import numpy as np


def standardize_coords(atom_coords):
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
        v_ref1 = untransformed_coords[1]
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
    
        def Rz(theta):
            return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],\
                             [0,0,1]])
        def Rx(theta):
            return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],\
                             [0,np.sin(theta),np.cos(theta)]])
    
        Rz_a = Rz(-1.0*alpha_t*alpha_sgn)
        Rx_b = Rx(-1.0*beta_t*beta_sgn)
        Rz_g = Rz(-1.0*gamma_t*gamma_sgn)
        transmat = Rz_a@Rx_b@Rz_g

        return untransformed_coords@transmat
