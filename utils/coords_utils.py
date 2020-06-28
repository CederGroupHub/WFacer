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
        return False
    elif len(coords)==2:
        return False
    else:
        shifted = np.array(coords)-np.average(coords,axis=0)
        for i in range(2,len(coords)):
            if np.linalg.norm(np.cross(coords[1]-coords[0],coords[i]-coords[0]))!=0:
                return True
        return False

def Standardize_Coords(atom_coords,z_id=0,x_id=1):
    """
    Given a set of catesian atomic coordinates, this function trys to 'standardize'
    the coordinates by aligning the vector from the center to the id=z_id atom 
    to z-axis direction, align the vector from the center to the id=x_id atom onto
    xz plane, with projection on x_axis >=0.

    If the atomic coordinate set only has one atom, then we just shift it to [0,0,0]

    Note: the marking atom indices z_id and x_id should be chosen carefully, so that
    none of the two pointing vectors is 0.
    """
    untransformed_coords = np.array(atom_coords)-np.average(atom_coords,axis=0)
    if len(atom_coords)<2:
        return untransformed_coords
    else:
        v_ref0 = untransformed_coords[z_id]
        v_ref1 = untransformed_coords[x_id]

        if np.linalg.norm(v_ref0)==0:
            raise ValueError("z-axis reference atom not chosen correctly.")
        ez = v_ref0/np.linalg.norm(v_ref0)

        if Is_Nonlinear(untransformed_coords):
            ey = np.cross(ez,v_ref1)
            if np.linalg.norm(ey)==0:
                raise ValueError("x-axis reference atom not chosen correctly.")
            ey = ey/np.linalg.norm(ey)
        else:
            if ez[2]==0:
                ey = np.array([0.0,0.0,1.0])
            else:
                eyx = 0.0
                eyy = np.sqrt(ez[2]**2/(ez[1]**2+ez[2]**2))
                eyz = -ez[1]/ez[2]*eyy
                ey = np.array([eyx,eyy,eyz])                

        ex = np.cross(ey,ez)
        transmat = np.linalg.inv(np.vstack((ex,ey,ez)))
    
        return untransformed_coords@transmat

def gram_schmidt(A):
    """
    Do Gram-schmidt orthonormalization to row vectors of a, and returns the result.
    If matrix is over-ranked, will remove redundacies automatically.
    Inputs:
        A: array-like
    Returns:
        A_ortho: array-like, orthonormal matrix.
    """
    n,d = A.shape

    if np.allclose(A[0],np.zeros(d)):
        raise ValueError("First row is a zero vector, can not run algorithm.")

    new_rows = [A[0]/np.linalg.norm(A[0])]
    for row in A[1:]:
        new_row = row
        for other_new_row in new_rows:
            new_row = new_row - np.dot(other_new_row,new_row)*other_new_row
        if not np.allclose(new_row,np.zeros(d)):
            new_rows.append( new_row/np.linalg.norm(new_row) )
    return np.vstack(new_rows)


