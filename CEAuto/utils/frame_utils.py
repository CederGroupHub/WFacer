"""Pandas Dataframe utilities"""

__author__ = 'Fengyu Xie'

import pandas as pd
import json
import os

from .serial_utils import deser_comp, serialize_comp


def save_dataframes(sc_df, comp_df, fact_df,
                    sc_file='sc_mats.csv', comp_file='comps.csv',
                    fact_file='data.csv'):
    """Save dimension tables and the fact table.

    List and dict objects will be serialized into strings.
    We choose csv files to ensure cross-environment transferablity.

    Args:
        sc_df(pd.DataFrame):
            Un-serialized supercell dataframe.
        comp_df(pd.DataFrame):
            Un-serialized composition dataframe.
        fact_df(pd.DataFrame):
            Un-serialized fact table.
        sc_file(str):
            Supercell save file. Default to 'sc_mats.csv'.
        comp_file(str):
            Composition save file. Default to 'comps.csv'.
        fact_file(str):
            Fact table save file. Default to 'data.csv'.

    File names can be changed, but not recommended!
    """
    # Note: pd can't store int and None together. It's okay not to
    # be be fussy here as all integer values in our dataframes are
    # indices of supercells, compositions or entree. They can't be
    # null value.

    # In pandas, dicts needs special serialization before saving,
    # otherwise they can't be deserialized by json. For safety,
    # lists are also pre-serialized in case pandas serialization
    # breaks reload.
    if sc_df is not None:
        sc_ser = sc_df.copy()
        sc_ser.matrix = sc_ser.matrix.map(json.dumps)
        sc_ser.to_csv(sc_file, index=False)
    if comp_df is not None:
        comp_ser = comp_df.copy()
        comp_ser.comp = comp_ser.comp.map(serialize_comp)
        comp_ser.comp = comp_ser.comp.map(json.dumps)
        comp_ser.ucoord = comp_ser.ucoord.map(json.dumps)
        comp_ser.ccoord = comp_ser.ccoord.map(json.dumps)
        comp_ser.cstat = comp_ser.cstat.map(json.dumps)
        comp_ser.nondisc = comp_ser.nondisc.map(json.dumps)
        comp_ser.to_csv(comp_file, index=False)
    if fact_df is not None:
        fact_df_ser = fact_df.copy()
        fact_df_ser.ori_occu = fact_df_ser.ori_occu.map(json.dumps)
        fact_df_ser.ori_corr = fact_df_ser.ori_corr.map(json.dumps)
        fact_df_ser.map_occu = fact_df_ser.map_occu.map(json.dumps)
        fact_df_ser.map_corr = fact_df_ser.map_corr.map(json.dumps)
        fact_df_ser.other_props = fact_df_ser.other_props.map(json.dumps)
        fact_df_ser.to_csv(fact_file,index=False)


def load_dataframes(sc_file='sc_mats.csv', comp_file='comps.csv',
                    fact_file='data.csv'):
    """Load dimension tables and the fact table.

    List and dict objects will be de-serialized from string format.

    Args:
        sc_file(str):
            Supercell save file. Default to 'sc_mats.csv'.
        comp_file(str):
            Composition save file. Default to 'comps.csv'.
        fact_file(str):
            Fact table save file. Default to 'data.csv'.

        File names can be changed, but not recommended!

    Returns:
        sc_df, comp_df, fact_df: pd.DataFrame.
    """
    list_conv = lambda x: json.loads(x) if not pd.isna(x) else None
    comp_conv = lambda x: (deser_comp(json.loads(x)) if not pd.isna(x)
                           else None)

    # Note: pd can't store int and None together. It's okay not to
    # be be fussy here as all integer values in our dataframes are
    # indices of supercells, compositions or entree. They can't be
    # null value.
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
