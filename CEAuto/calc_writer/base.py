"""
Base calculation writer class.
"""
__author__ = "Fengyu Xie"

from ..data_manager import DataManager
from ..inputs_wrapper import InputsWrapper

from ..config_paths import *

from abc import ABC, abstractmethod
import numpy as np

class BaseWriter(ABC):
    """
    A calculation write class, to write ab-initio calculations to various 
    data warehouses. Current implementation
    includes local archive and mongo database+fireworks.
   
    Current implementations only support vasp.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    Note: Use get_calc_writer method in InputsWrapper to get any Writer object,
          or auto_load.
          Direct init not recommended!
    """
    def __init__(self,writer_strain=[1.05,1.03,1.01],ab_setting={},\
                      data_manager=DataManager.auto_load(),\
                      **kwargs):
        """
        Args:
            writer_strain(1*3 or 3*3 arraylike):
                Strain matrix to apply to structure before writing as 
                inputs. This helps breaking symmetry, and relax to a
                more reasonable equilibrium structure.
            ab_setting(Dict):
                Pass ab-initio software options. For vasp,
                look at pymatgen.vasp.io.sets doc.
            data_manager(DataManager):
                An interface to previous calculation and enumerated 
                data.
        """
        self.strain = writer_strain
        self.ab_setting = ab_setting
        self._dm = data_manager
        
    def write_tasks(self,strs_undeformed,entry_ids,*args,**kwargs):
        """
        Write input files or push data to fireworks launchpad.
        Will check status and see if writing is required.
        Inputs:
            strs_undeformed(List of Structure):
                Structures in original lattice.(Not deformed.)
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                Must be provided.       
        No return value.
        """
        if self._dm._schecker.after("write"):
            print("**Writing already finished in current iteration {}."\
                  .format(self._dm._schecker.cur_iter_id))
            return

        for eid,str_undef in zip(entry_ids,structures_undeformed):
            self._write_single(str_undef,eid,*args,**kwargs)

    @abstractmethod
    def _write_single(self,structure,eid,*args,**kwargs):
        return

    def auto_write_entree(self):
        """
        Automatically detect uncomputed entree, writes the entree, and 
        updates the status in the fact table.
 
        No return value. The updated datamanager will be flushed!
        """
        eids = self._dm.get_eid_w_status('NC')

        fact_w_strs = self._dm.fact_df_with_structures
        filt_ = fact_w_strs.entry_id.isin(eids)

        strs_undeformed = fact_w_strs.loc[filt_,'ori_str']

        self.write_tasks(strs_undeformed,eids)
        
        self._dm.set_status(eids,'CC') #Set status to 'computing'
        self._dm.auto_save() #flush data

    @classmethod
    def auto_load(cls,options_file=OPTIONS_FILE,\
                      sc_file=SC_FILE,\
                      comp_file=COMP_FILE,\
                      fact_file=FACT_FILE,\
                      ce_history_file=CE_HISTORY_FILE):
        """
        This method is the recommended way to initialize this object.
        It automatically reads all setting files with FIXED NAMES.
        YOU ARE NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!
        Args:
            options_file(str):
                path to options file. Options must be stored as yaml
                format. Default: 'options.yaml'
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'
            comp_file(str):
                path to compositions file, in csv format.
                Default: 'comps.csv'             
            fact_file(str):
                path to enumerated structures dataframe file, in csv format.
                Default: 'data.csv'             
            ce_history_file(str):
                path to cluster expansion history file.
                Default: 'ce_history.json'
        Returns:
             BaseWriter object.
        """
        options = InputsWrapper.auto_load(options_file=options_file,\
                                          ce_history_file=ce_history_file)

        dm = DataManager.auto_load(options_file=options_file,\
                                   sc_file=sc_file,\
                                   comp_file=comp_file,\
                                   fact_file=fact_file,\
                                   ce_history_file=ce_history_file)

        return cls(data_manager=dm,**options.calc_writer_options)
