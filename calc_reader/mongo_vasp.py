"""
MongoDB calculations reader class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

import numpy as np
import os

from pymatgen import Structure
from pymatgen.io.vasp.outputs import Outcar

import atomate
from atomate.vasp.database import VaspCalcDb

from .base import BaseReader

class MongoVaspReader(BaseReader):
    """
    A calculation reader class, to read calculation results from 
    various data warehouses. Current implementation includes local 
    archive+SGE queue and mongo database+fireworks.
   
    Current implementations only support vasp.

    This class only serves as accessor to the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    """

    DEFAULT_MONGO_PATH = os.path.join(atomate.__path__[0],'config/json')

    def __init__(self,md_file=None,**kwargs):
       """
       Args:
           md_file(str):
               Path to mongodb setting file. The calculations will be read from
               this database.
       """
        md_file = md_file or DEFAULT_MONGO_PATH
        self._mongod = VaspCalcDb.from_db_file(md_file)
        self.root_name = os.path.split(os.get_cwd())[-1]

    def check_convergence_status(self,entry_ids):
        """
        Checks convergence status of entree with specific indices.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                Must be provided.
        Returns:
            A list of booleans, each shows whether the calculation
            of the correponding entry have succeeded.
            If inquired calculation does not exist, will raise an
            error. If not calculated (no vasprun.xml under it), will
            regard as not converged.
        """
        status = []
        for eid in entry_ids:
            rd, sd = self.get_single_calc(eid)

            r_conv = rd.get('calcs_reversed',[{}])[0].get('has_vasp_completed',False)
            s_conv = sd.get('calcs_reversed',[{}])[0].get('has_vasp_completed',False)
           
            status.append(r_conv and s_conv)

        return status

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
            If inquired calculation does not exist, will raise an
            error. If not calculated (not vasprun.xml), will try to
            look for Poscar instead. If Poscar does not exist, will
            raise error.
        """
        structures = []
        for eid in entry_ids:
            rd, sd = self.get_single_calc(eid)

            str_d = sd.get('calcs_reversed',[{}])[0].\
                                 get('output',{}).\
                                 get('structure',None) or \
                    rd.get('calcs_reversed',[{}])[0].\
                                get('output',{}).\
                                get('structure',None) or \
                    rd.get('calcs_reversed',[{}])[0].\
                                get('input',{}).\
                                get('structure',None)

            if str_d is None:
                raise ValueError("Specified entry {} does not have structure!".format(eid))
            else:
                structures.append(Structure.from_dict(str_d))

        return structures

    def get_single_calc(self,eid):
        """
        Get calculation data of a single entry.
        Args:
            eid(int):
                index of entry. If the entry does not exist, will raise Error.
        Returns:
            Relaxation calculation data in dictionary, and static calculation data in dictionary.
        """
        entry_name = 'ce_{}_{}'.format(self.root_name,eid)
        relax_name = entry_name+'_optimization'
        static_name = entry_name+'_static'

        relax_calcd = self._mongod.collection.find_one({'task_label':relax_name}) or {}
        static_calcd = self._mongod.collection.find_one({'task_label':static_name}) or {}           


        if len(relax_calcd)==0 and len(static_calcd) == 0:
            raise ValueError("Specified entry {} does not exist!".format(eid))

        return relax_calcd,static_calcd

    def _load_property_by_name(self,entry_ids,name='energy'):
        """
        Load a single type of property of structures from the warehouse.
        Currently only checks and reads Vasprun and Outcar, and loads
        energy and magnetization. Implement more if you want.
        Args:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                Must be provided.
            name(str):
                Name of property to be extracted. By default, gives
                energy.
                Must be a member in list: supported_properties.
                (class constant)
        Returns:
            A list containing extracted proerties of corresponding 
            entree.
        It is your responsibility to ensure that entry_ids are all valid
        and readable calculations!!
        """
        property_per_str = []
        for eid in entry_ids:
            rd,sd = get_single_calc(eid)
            #Add more if you need.

            if name == 'energy':
                e = sd.get('calcs_reversed',[{}])[0].\
                       get('output',{}).\
                       get('energy', None) or \
                    rd.get('calcs_reversed',[{}])[0].\
                       get('output',{}).\
                       get('energy', None)

                if e is None:
                    raise ValueError("Specified entry {} does not have energy!"\
                                     .format(eid))
                property_per_str.append(e)

            if name == 'magnetization':                   
                ocard = sd.get('calcs_reversed',[{}])[0].\
                           get('output',{}).\
                           get('outcar',None) or \
                        rd.get('calcs_reversed',[{}])[0].\
			   get('output',{}).\
			   get('outcar', None)                           
                if ocard is None:
                    raise ValueError("Specified entry {} does not have OUTCAR!"\
                                     .format(eid))               

                ocar = Outcar.from_dict(ocard)
                mag = [st_mag['tot'] for st_mag in ocar.magnetization]
                property_per_str.append(mag)

        return property_per_str
