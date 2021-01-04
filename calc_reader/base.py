"""
Base calculation reader class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

from abc import ABC, abstractmethod
import numpy as np

class BaseReader(ABC):
    """
    A calculation reader class, to read calculation results from 
    various data warehouses. Current implementation includes local 
    archive+SGE queue and mongo database+fireworks.
   
    Current implementations only support vasp.

    This class only serves as accessor to the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.
    """
    def __init__(self):
        pass

    @abstractmethod
    def check_convergence_status(self,entry_ids):
        """
        Checks convergence status of entree with specific indices.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                Must be provided.
        Returns:
            a list of booleans, each shows whether the calculation
            of the correponding entry have succeeded.
        """
        return

    @abstractmethod
    def load_structures(self,entry_ids):
        """
        Loads relaxed structures.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None, will check all available ids.
        Returns:
            a list of pymatgen.Structure, all composed of 
            pymatgen.Element (undecorated).
        """
        return

    def load_properties(self,entry_ids,prop_names='energy',
                        include_pnames=True):
        """
        Load unnormalized, calculated properties from ab_initio data.
        Inputs:
             entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                Must be provided.
             prop_names(List of str or str):
                property names to extract. Currently supports energies,
                and magenetization. You can add more if required.
                If one single string is given, will return a list only.
             include_pnames(Boolean):
                Include property names in the return value.
                If true, will return a dict with property names as 
                keys. If false, will only return properties in a list
                by the order of entry in prop_names.
       Outputs:
           Dict or list containing properties of specified structures,
           depending on the value of include_params, and the format
           of prop_names.
        """
        if isinstance(prop_names,str):
            p = np.array(self._load_property_by_name(entry_ids,name=prop_names))/normalize_by
            return p.tolist()

        properties = []
        for pname in prop_names:
            p = np.array(self._load_property_by_name(entry_ids,name=pname)/normalize_by
            properties.append(p.tolist())

        if include_pnames:
            return {pname:p for pname,p in zip(prop_names,properties)}
        else:
            return properties

    @abstractmethod
    def _load_property_by_name(self,entry_ids,name='energy'):
        """
        Load a single type of property of structures from the warehouse.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                Must be provided.
            name(str):
                Name of property to be extracted. By default, gives
                energy.
                Must be a member in list: supported_properties.
                (class constant)
        Outputs:
            A list containing extracted proerties of corresponding 
            entree.
        """
        return 
