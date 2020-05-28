import sys
import os
this_file_path = os.path.abspath(__file__)
this_file_dir = os.dirname(this_file_path)
parent_dir = os.dirname(this_file_dir)
sys.path.append(parent_dir)

from utils.comp_utils import *


