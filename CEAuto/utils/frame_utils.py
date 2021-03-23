"""Pandas Dataframe utilities"""

__author__ = 'Fengyu Xie'

import pandas as pd
import json
import os


def load_dataframes(sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
    """
    Loading dimension tables and the fact table. 
    comp_df needs a little bit de-serialization.
    File names can be changed, but not recommended!
    Notice: pandas loads lists as strings. You have to serialize them!
    """
    list_conv = lambda x: json.loads(x) if x is not None else None
    if os.path.isfile(sc_file):
        sc_df = pd.read_csv(sc_file,converters={'matrix':list_conv})
    else:
        sc_df = pd.DataFrame(columns=['sc_id','matrix'])

    if os.path.isfile(comp_file):
        #De-serialize compositions and list values
        comp_df = pd.read_csv(comp_file,
                                    converters={'ucoord':list_conv,
                                                'ccoord':list_conv,
                                                'cstat':list_conv,
                                                'eq_occu':list_conv,
                                                'comp':deser_comp
                                               })
    else:
        comp_df = pd.DataFrame(columns=['comp_id','sc_id',\
                                        'ucoord','ccoord',\
                                        'comp','cstat',\
                                        'eq_occu'])

    if os.path.isfile(fact_file):
        fact_df = pd.read_csv(fact_file,
                                    converters={'ori_occu':list_conv,
                                                'ori_corr':list_conv,
                                                'map_occu':list_conv,
                                                'map_corr':list_conv,
                                                'other_props':list_conv
                                               })
    else:
         fact_df = pd.DataFrame(columns=['entry_id','sc_id','comp_id',\
                                         'iter_id','module',\
                                         'ori_occu','ori_corr',\
                                         'calc_status',\
                                         'map_occu','map_corr',\
                                         'e_prim','other_props'])

    return sc_df, comp_df, fact_df

