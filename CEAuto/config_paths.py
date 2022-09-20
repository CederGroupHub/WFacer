"""Specifies paths to ciritical setting files.

You can modify these in paths_setting.json under the running folder."""

__author__ = 'Fengyu Xie'

import os
import yaml

# Now everything is provided as yaml.
d = {}
if os.path.isfile('paths.yaml'):
    with open('paths.yaml', "r") as fin:
        d = yaml.safe_load(fin)

PRIM_FILE = d.get('prim_file', 'prim.cif')  # Must be present.
OPTIONS_FILE = d.get('options_file', 'options.json')
HISTORY_FILE = d.get('history_file', 'history.json')
INPUTS_FILE = d.get('wrapper_file', 'inputs.json')
DECOR_FILE = d.get('decor_file', 'decorators.json')
