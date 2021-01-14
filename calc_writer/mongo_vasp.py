"""
Mongo+vasp calculation writer class. These classes DO NOT MODIFY 
fact table!

Configure you atomate and fireworks before using this!
"""
__author__ = "Fengyu Xie"

import os

from pymatgen.io.vasp.sets import MPRelaxSet, MPMetalRelaxSet,\
                                  MPStaticSet
from atomate.vasp.fireworks import OptimizeFW,StaticFW
from fireworks import LaunchPad,Workflow

from .base import BaseWriter

def wf_ce_sample(structure, entry_id, root_name = None, is_metal=False,\
                 relax_set_params = None, static_set_params = None,\
                 **kwargs):
    """
    Create a fireworks.Workflow for an input structure, including an optimization and a 
    static calculation.
    Inputs:
        struture(Structure): 
            The structure to be computed. Usually pre-deformed to break relaxation symmetry.
        entry_id(int):
            index of the entry corresponding to the structure.
        root_name(str):
            Root name of all workflows and fireworks. By default, will set to name
            of the current directory.
        is_metal(Boolean):
            If true, will use vasp parameters specific to metallic computations.
        relax_set_params(dict):
            A dictionary specifying other parameters to overwrite in optimization vasp input
            set.
        static_set_params(dict):
            A dictionary specifying other parameters to overwrite in static vasp input
            set.
        kwargs hosts other paramters you wish to pass into the returned workflow object.
    Output:
        A Workflow object, containing optimization-static calculation of a single structure
        in cluster expansion pool.
    """
    #Current folder name will be used to mark calculation entree!
    root_name = root_name or os.path.split(os.getcwd())[-1]
    entry_name = 'ce_{}_{}'.format(root_name,entry_id)
    opt_setting = relax_set_params or {}
    sta_setting = static_set_params or {}
 
    if is_metal:
        opt_set = MPMetalRelaxSet(structure,**opt_setting)
    else:
        opt_set = MPRelaxSet(structure,**sta_setting)
    sta_set = MPStaticSet(structure)

    #fireworks encoded with root and entry names.
    opt_fw = OptimizeFW(structure,vasp_input_set = opt_set,\
                        name = entry_name+'_optimization') 
    sta_fw = StaticFW(structure,vasp_input_set = sta_set,\
                      overwrite_default_vasp_params = sta_fw,\
                      parents = [opt_fw],\
                      name = entry_name+'_static')

    #workflow encoded with root and entry names
    return Workflow([opt_fw,sta_fw], name=entry_name,**kwargs)

class MongoVaspWriter(BaseWriter):
    """
    A calculation write class, to write ab-initio calculations to various 
    data warehouses. Current implementation
    includes local archive and mongo+fireworks.

    Does not interact with your computing resource.
   
    This class writes vasp input files into local folders. Relaxation and 
    static step will be two separate submissions!!

    Current implementations only support vasp.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    It's your responsibility not to dupe-write a directory and not to waste
    your own time.

    Attributes: 
        path(str):
            path to the calculation archieve.
    """
    def __init__(self):
        #Load is based on the atomate launchpad configuration under your environment!
        self.root_name = os.path.split(os.getcwd())[-1]
        self._lpad = LaunchPad.auto_load()
        
    def write_tasks(self,strs_undeformed,entry_ids,*args, strain=[1.05,1.03,1.01],\
                    is_metal = False,relax_set_params=None,static_set_params=None,\
                    **kwargs):
        """
        Write workflows and add to fireworks launchpad.
        Inputs(Order of arguments matters):
            strs_undeformed(List of Structure):
                Structures in original lattice.(Not deformed.)
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                Must be provided.       
            strain(1*3 or 3*3 arraylike):
                Strain matrix to apply to structure before writing as 
                inputs. This helps breaking symmetry, and relax to a
                more reasonable equilibrium structure.
            is_metal(Boolean):
                Select True if your system is a metallic alloy, then we will optimize 
                relaxation parameters with MPMetalRelaxSet. By default, use 
                false.
            relax_set_params(dict):
                A dictionary specifying other parameters to overwrite in optimization vasp input
                set.
            static_set_params(dict):
                A dictionary specifying other parameters to overwrite in static vasp input
                set.
            kwargs hosts other paramters you wish to pass into the workflows.

            Can pass ab-intio settings as **kwargs. Refer to pymatgen.io.vasp.sets
        for more details.
        No return value.
        """
        super().write_tasks(strs_undeformed,entry_ids,*args,strain=strain,\
                            is_metal = is_metal,\
                            relax_set_params=relax_set_params,\
                            static_set_params=static_set_params,\
                            **kwargs)

    def _write_single(self,structure,eid,*args,strain=[1.05,1.03,1.01],\
                      is_metal = False, relax_set_params = None, static_set_params = None,\
                      **kwargs):
        """
        Write a single computation task to archieve.
        """
        #Apply a slight deformation.
        strain = np.array(strain)
        if strain.shape == (3,):
            strain = np.diag(strain)
   
        if strain.shape != (3,3):
            raise ValueError("Incorrect strain format.")
           
        str_input = Deformation(strain).apply_to_structure(structure)

        wf = wf_ce_sample(str_input, eid, root_name=self.root_name,\
                          is_metal=is_metal,\
                          relax_set_params = relax_set_params,\
                          static_set_params = static_set_params,\
                          **kwargs)               

        self._lpad.add_wf(wf)
        print("****Calculation workflow loaded to launchpad for entry: {}.".format(eid))
