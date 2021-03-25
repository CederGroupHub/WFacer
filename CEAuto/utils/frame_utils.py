"""Pandas Dataframe utilities"""

__author__ = 'Fengyu Xie'

import pandas as pd
import json
import os

from .serial_utils import deser_comp, serialize_comp


def save_dataframes(sc_df, comp_df, fact_df,
                    sc_file='sc_mats.csv', comp_file='comps.csv',
                    fact_file='data.csv'):
        """
        Saving dimension tables and the fact table. Must set index=False,
        otherwise will always add one more row for each save and load.

        comp_df needs a little bit serialization.
        File names can be changed, but not recommended!
        """
        sc_df.to_csv(sc_file,index=False)
        comp_ser = comp_df.copy()
        comp_ser.comp = comp_ser.comp.map(serialize_comp)
        comp_ser.comp = comp_ser.comp.map(json.dumps)  # Dump to string.
        comp_ser.to_csv(comp_file,index=False)
        if fact_df is not None:
            fact_df_ser = fact_df.copy()
            fact_df_ser.other_props = fact_df_ser.other_props.map(json.dumps)
            fact_df_ser.to_csv(fact_file,index=False)


def load_dataframes(sc_file='sc_mats.csv', comp_file='comps.csv',
                    fact_file='data.csv'):
    """
    Loading dimension tables and the fact table. 
    comp_df needs a little bit de-serialization.
    File names can be changed, but not recommended!
    Notice: pandas loads lists as strings. You have to serialize them!
    """
    list_conv = lambda x: json.loads(x) if not pd.isna(x) else None
    comp_conv = lambda x: (deser_comp(json.loads(x)) if not pd.isna(x)
                           else None)

    if os.path.isfile(sc_file):
        sc_df = pd.read_csv(sc_file)
        sc_df.matrix = sc_df.matrix.map(list_conv)
    else:
        sc_df = pd.DataFrame(columns=['sc_id','matrix'])

    if os.path.isfile(comp_file):
        #De-serialize compositions and list values
        comp_df = pd.read_csv(comp_file)
        comp_df.ucoord = comp_df.ucoord.map(list_conv)
        comp_df.ccoord = comp_df.ccoord.map(list_conv)
        comp_df.nondisc = comp_df.nondisc.map(list_conv)
        comp_df.compstat = comp_df.compstat.map(list_conv)
        comp_df.comp = comp_df.comp.map(comp_conv)
        comp_df.eq_occu = comp_df.eq_occu.map(list_conv)

    else:
        comp_df = pd.DataFrame(columns=
                               ['comp_id','sc_id',
                                'ucoord','ccoord',
                                'comp','cstat',
                                'nondisc','eq_occu'])

    if os.path.isfile(fact_file):
        fact_df = pd.read_csv(fact_file)
        fact_df.ori_occu = fact_df.ori_occu.map(list_conv)
        fact_df.ori_corr = fact_df.ori_corr.map(list_conv)
        fact_df.map_occu = fact_df.map_occu.map(list_conv)
        fact_df.map_corr = fact_df.map_corr.map(list_conv)
        fact_df.other_props = fact_df.other_props.map(list_conv)

    else:
        fact_df = pd.DataFrame(columns=['entry_id', 'sc_id', 'comp_id',
                                        'iter_id', 'module',
                                        'ori_occu', 'ori_corr',
                                        'calc_status',
                                        'map_occu', 'map_corr',
                                        'e_prim', 'other_props'])

    return sc_df, comp_df, fact_df
