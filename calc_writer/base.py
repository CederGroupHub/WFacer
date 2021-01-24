"""
Base calculation writer class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

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

    Args:
        writer_strain(1*3 or 3*3 arraylike):
            Strain matrix to apply to structure before writing as 
            inputs. This helps breaking symmetry, and relax to a
            more reasonable equilibrium structure.
        ab_setting(Dict):
            Pass ab-initio software options. For vasp,
            look at pymatgen.vasp.io.sets doc.
    """
    def __init__(self,writer_strain=[1.05,1.03,1.01],ab_setting={},**kwargs):
        self.strain = writer_strain
        self.ab_setting = ab_setting
        
    def write_tasks(self,strs_undeformed,entry_ids,*args,**kwargs):
        """
        Write input files or push data to fireworks launchpad.
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
