"""
This file specifies paths to the data and calculation 
setting files. You can modify these in paths.yaml, but
you are not recommended to do so
"""
__author__ = 'Fengyu Xie'

import os
import yaml

d = {}
if os.path.isfile('paths.yaml'):
    with open('paths.yaml') as fin:
        d = yaml.load(fin, Loader=yaml.FullLoader)

PRIM_FILE = d.get('prim_file', 'prim.cif')
OPTIONS_FILE = d.get('options_file', 'options.yaml')
SC_FILE = d.get('sc_file', 'sc_mats.csv')
COMP_FILE = d.get('comp_file', 'comps.csv')
FACT_FILE = d.get('fact_file', 'data.csv')
CE_HISTORY_FILE = d.get('ce_history_file', 'ce_history.json')
WRAPPER_FILE = d.get('wrapper_file', 'inputs_wrapper.json')
DECOR_FILE = d.get('decor_file', 'decors.json')
TIME_KEEPER_FILE = d.get('time_keeper_file', 'time_keeper.json')
