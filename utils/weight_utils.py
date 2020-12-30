"""
Utility functions to calculate weights for structures in CE.
"""

__author__ = 'Fengyu Xie'

import numpy as np

from smol.cofe.wrangling.wrangler import weights_energy_above_composition,\
                                         weights_energy_above_hull

from .format_utils import structure_from_occu

def weights_from_fact(fact, flavor='unweighted', **kwargs):
    """
    Compute weights of each row from a featurized fact table.
    Inputs:
        fact(pd.DataFrame):
            Featurized fact table. For format, see CEAuto.struct_enum
        flavor(str):
            Weighting method. Supporting: 'unweighted', 'e_above_comp',
            'e_above_hull'.
        kwargs:
            Weighting method specific arguments.
            For 'unweighted', nothing is required.
            For 'e_above_comp' and 'e_above_hull':
                prim(Compulsory,pymatgen.Structure):
                    Primitive cell for cluster expansion.
                sc_table(Compulsory,pd.DataFrame):
                    Supercell matrices dimension table,
                    containing all supercell matrices and their indices.
                temperature(Optional,float):
                    Temprature for calculating weights. Default to 2000K.
            Other weighting methods are not implemented yet.
    """
    if flavor == 'unweighted':
        return None
    
    if flavor in ['e_above_comp','e_above_hull']:
        #check compulsory args
        if 'prim' not in kwargs:
            raise ValueError("Missing arg: prim for {}.".format(flavor))
        if 'sc_table' not in kwargs:
            raise ValueError("Missing arg: sc_table for {}.".format(flavor))

        prim = kwargs['prim']
        sc_table = kwargs['sc_table']
        if 'temperature' in kwargs:
            T = kwargs['temperature']
        else:
            T = 2000

        fact_join = fact.merge(sc_table,how='left',on='sc_id')
        structs = fact_join.apply(lambda x:\
                                  structure_from_occu(prim,x['matrix'],x['map_occu']),\
                                  axis = 1)
        #Must use un-normalized energies!
        sc_sizes = fact_join.matrix.map(lambda x: int(abs(round(np.linalg.det(x)))) )
        energies = fact_join.e_prim * sc_sizes

        if flavor == 'e_above_comp':
            return weights_energy_above_composition(structs,energies,temperature=T)
        else:
            return weights_energy_above_hull(structs,energies,prim,temperature=T)

    #### YOUR IMPLEMENTATIONS HERE.

    raise NotImplementedError('Weighting method {} not implemented.'.format(flavor))
