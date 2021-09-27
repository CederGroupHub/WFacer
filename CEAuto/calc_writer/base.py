"""
Base calculation writer class.
"""
__author__ = "Fengyu Xie"

from ..config_paths import *
from ..utils.class_utils import derived_class_factory

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
    def __init__(self, writer_strain=[1.05,1.03,1.01], ab_setting={},
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
        """
        self.strain = writer_strain
        self.ab_setting = ab_setting

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
        for eid,str_undef in zip(entry_ids,structures_undeformed):
            self._write_single(str_undef,eid,*args,**kwargs)

    @abstractmethod
    def _write_single(self,structure,eid,*args,**kwargs):
        return

    def write_df_entree(self, data_manager):
        """
        Automatically detect uncomputed entree, writes the entree, and 
        updates the status in the fact table.
 
        No return value. The updated datamanager will be flushed!
        Args:
            data_manager(DataManager):
                An interface to previous calculation and enumerated 
                data.
        Return: 
            Data manager after change.
        """
        if data_manager.schecker.after("write"):
            print("**Writing already finished in current iteration {}."
                  .format(data_manager.schecker.cur_iter_id))
            return

        eids = data_manager.get_eid_w_status('NC')

        fact_w_strs = data_manager.fact_df_with_structures
        filt_ = fact_w_strs.entry_id.isin(eids)

        strs_undeformed = fact_w_strs.loc[filt_,'ori_str']

        self.write_tasks(strs_undeformed,eids)

        data_manager.set_status(eids,'CC')
        # Set status to 'computing'


def writer_factory(writer_name, *args, **kwargs):
    """Create a calculation writer with given name.

    Args:
       writer_name(str):
         Name of a calc writer class.
       *args, **kwargs:
         Arguments used to initialize a manager class.

    Returns:
       BaseWriter.
    """
    name = writer_name.capitalize() + 'Writer'
    return derived_class_factory(name, BaseWriter, *args, **kwargs)
