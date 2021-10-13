import numpy as np
from scipy.spatial import ConvexHull


def hulls_match(old_hull, new_hull, e_tol, comp_tol=0.05):
    """Compares two hulls based on energy and composition differences.

    Args:
        old_hull, new_hull(pd.DataFrame):
            old and new hulls to compare.
            New hull's comp_id should always contain that of old hull.
            Must contain at least 3 columns: ['comp_id','e_prim','ucoord']
            All coordinates must be normalized.
        e_tol(float):
            Tolerance of ground state energy differnece in eV/prim.
            If new ground state occurs, but its energy difference to old gss
            is smaller than tolerance, will still think converged.
        comp_tol(float):
            Tolerance of ground state composition changes, measured in norm
            of unconstrained coordinates change.
            Default is 0.05.
            If new ground state occurs, but its composition difference to
            old ones is smaller than tolerance, will still think converged.
    Returns:
        Boolean.
    """
    # Compare intersection table
    inner_hull = old_hull.merge(new_hull, how='inner', on='comp_id')
    e_match = np.all(np.abs(inner_hull['e_prim_x'] -
                            inner_hull['e_prim_y']) <= e_tol)

    if not e_match:
        return False

    #Compare new compositions
    outer_hull = old_hull.merge(new_hull,how='outer',on='comp_id')
    comple_hull = outer_hull[outer_hull['e_prim_x'].isnull()]

    if len(comple_hull) == 0:
        return True  #No new GS compositions detected!

    inner_hull = inner_hull.reset_index(drop=True)
    comple_hull = comple_hull.reset_index(drop=True)

    for new_id in range(len(comple_hull)):
        new_ucoord = np.array(comple_hull.iloc[new_id]['ucoord_y'])
        new_e = comple_hull.iloc[new_id]['e_prim_y']
        # Find the closest compositioen
        dists = [np.linalg.norm(ucoord - new_ucoord) for ucoord in
                 inner_hull['ucoord_x'].map(np.array)]
        min_d = np.min(dists)
        if min_d > comp_tol:
            return False
        min_d_idx = np.argmin(dists)
        old_e = inner_hull.iloc[min_d_idx]['e_prim_x']

        if np.abs(new_e - old_e) > e_tol:
        # The new CE is very pronounced.
            return False

    return True


def fix_convex_hull(hull_list):
    """Fix a hull into convex hull.

    Only works for 1D comp space!
    Args:
        hull_list(List[(float, float)]):
            Non convex minimum energies. (compositions, energies)
    Returns:
        List[(float, float)]: convex hull.
    """
    hull = hull_list
    if len(hull[0])<3:
        return hull
    else:
        old_hull = (np.array(hull).T)[:,:2]

        #Counter-clockwise
        cvx = ConvexHull(old_hull)

        new_hull = old_hull[cvx.vertices]
        edges_pos_dir = []
        for i in range(len(new_hull)-1):
            edge = new_hull[i+1]-new_hull[i]
            edges_pos_dir.append(edge[0]>0)
        edges_pos_dir.append((new_hull[0]-new_hull[-1])[0]>0)
        if not(edges_pos_dir[-1]):
            pos_end = None
            pos_begin = None
            for e_id,e_pos in enumerate(edges_pos_dir):
                if e_pos and pos_begin is None and pos_end is None:
                    pos_begin = e_id
                if not e_pos and pos_begin is not None and pos_end is None:
                    pos_end = e_id
                    break
            clean_hull = new_hull[pos_begin:pos_end+1]
        else:
            neg_end = None
            neg_begin = None
            for e_id,e_pos in enumerate(edges_pos_dir):
                if not(e_pos) and neg_begin is None and neg_end is None:
                    neg_begin = e_id
                if e_pos and neg_begin is not None and neg_end is None:
                    neg_end = e_id
                    break
            correct_hull_idx = list(range(neg_end,len(new_hull)))+list(range(0,neg_begin+1))
            clean_hull = new_hull[correct_hull_idx]

        clean_hull = clean_hull.T.tolist()
        return clean_hull
