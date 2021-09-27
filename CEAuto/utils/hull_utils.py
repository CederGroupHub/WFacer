import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import warnings


def hulls_match(old_hull,new_hull, e_tol, comp_tol=0.05):
    """
    This util function compares two hulls based on energy and composition differences.
    Args:
        old_hull, new_hull(pd.DataFrame):
            old and new hulls to compare. New hull's comp_id should always contain
            that of old hull's.
            Must contain at least 3 columns: ['comp_id','e_prim','ucoord']
            All coordinates must be normalized.

            The order of input matters!
            Refer to CEAuto.gs_checker.
        e_tol(float):
            tolerance of ground state energy differnece measured by eV/prim.
            If new ground state occurs, but its energy difference to old gss
            is smaller than tolerance, will still think converged.
        comp_tol(float):
            tolerance of ground state composition changes, measured in norm of unconstrained coordinates change.
            Default is 0.05.
            If new ground state occurs, but its composition difference to old gss
            is smaller than tolerance, will still think converged.
    Returns:
        Boolean.
    """
    #Compare intersection table
    inner_hull = old_hull.merge(new_hull,how='inner',on='comp_id')
    e_match = np.all(np.abs(inner_hull['e_prim_x']-inner_hull['e_prim_y'])<=e_tol)

    if not e_match:
        return False

    #Compare new compositions
    outer_hull = old_hull.merge(new_hull,how='outer',on='comp_id')
    comple_hull = outer_hull[outer_hull['e_prim_x'].isnull()]

    if len(comple_hull)==0:
        return True  #No new GS compositions detected!

    inner_hull = inner_hull.reset_index()
    comple_hull = comple_hull.reset_index()

    for new_id in range(len(comple_hull)):
        new_ucoord = np.array(comple_hull.iloc[new_id]['ucoord_y'])
        new_e = comple_hull.iloc[new_id]['e_prim_y']
        #Find the closest compositioen
        dists = [np.linalg.norm(ucoord-new_ucoord) for ucoord in \
                 inner_hull['ucoord_x'].map(np.array)]
        min_d = np.min(dists)
        if min_d > comp_tol:
            return False
        min_d_idx = np.argmin(dists)
        old_e = inner_hull.iloc[min_d_idx]['e_prim_x']

        if np.abs(new_e-old_e) > e_tol: #The new CE is very pronounced.
            return False

    return True


def fix_convex_hull(hull_list):
    """
    Fix a hull into convex hull. Only works for 1D comp space!
    Args:
        hull_list(List[(float, float)]):
            (compositions, energies)
    Returns:
        Fixed hull list.
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


def plot_hull(hull,axis_id=None, fix_hull=True,\
              title='CE hull plot',x_label = None,\
              y_label='Energy per prim/eV',\
              convert_to_formation = True):
    """
    This util function tries to plot a hull, by formation energy.
    When in high dimensional compositional space, must specify 
    an axis to project to.

    Usually you need to overlay scatter plot on it.
    Args:
        hull(pd.DataFrame):
            Dataframe of the minimum energy hulls.
            Must contain at least 3 columns: 
            ['e_prim','ccoord','comp']
            coordinates must be normalized!
        axis_id(int):
            Index of axis. If none given, will always project on 
            the first axis.
        fix_hull(Boolean):
            Fix the hull when it is not convex. Default to True.
        title(str):
            Title or plot
        x_label(str):
            x axis label
        y_label(str):
            y axis label
        convert_to_formation(Boolean):
            If true, will plot formation energy in eV/prim,
            instead of CE energies.
    Return:
        plt.figure, plt.axes,
        e1, e2: 
           estimated energies of the two extremum
        x_min, x_max: 
           constrained coordinate values at extremums on
           the plotting axis.
    """
    hull_sort = hull.reset_index()
    if len(hull_sort.iloc[0]['ccoord'])<1:
        raise ValueError("No coordinates given!")
    if len(hull_sort.iloc[0]['ccoord'])>1 and axis_id is None:
        warnings.warn("Plotting in multi-dimensional space, but no projection axis specified.")

    hull_sort['ccoord'] = hull_sort['ccoord'].map(lambda x: x[axis_id or 0])
    hull_sort = hull_sort.sort_values(by=['ccoord']).reset_index()
     
   
    hull_list = hull_sort.loc[:,['ccoord','e_prim']].to_numpy().T.tolist()
    if fix_hull:
        hull_list = fix_convex_hull(hull_list)

    if convert_to_formation:
        e1 = hull_list[1][0]
        e2 = hull_list[1][-1]
        x_min = hull_list[0][0]
        x_max = hull_list[0][-1]
        hull_list = np.array(hull_list)
        hull_list[0] = (hull_list[0]-x_min)/(x_max-x_min)
        hull_list[1] = hull_list[1] - (hull_list[0]*e2 + (1-hull_list[0])*e1)
    else:
        e1 = None
        e2 = None
        x_min = None
        x_max = None

    comp1 = hull_sort.iloc[0]['comp']
    comp2 = hull_sort.iloc[-1]['comp']
    fig,ax = plt.subplots()
    ax.plot(hull_list[0],hull_list[1],color='g',label='Min Hull\nx=0:{},x=1:{}'\
                                                      .format(comp1,comp2))
    ax.set_title(title)
    ax.set_xlabel(x_label or 'constrained_composition_axis_{}'.format(axis_id or 0))
    ax.set_ylabel(y_label)
    ax.legend(fontsize=10)

    return fig, ax, e1, e2, x_min, x_max   
