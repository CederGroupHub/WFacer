"""
Mongo+vasp calculation writer class. These classes DO NOT MODIFY 
fact table!

Configure you atomate and fireworks before using this!
"""
__author__ = "Fengyu Xie"

import os
import itertools

from pymatgen.io.vasp.sets import (MPRelaxSet, MPMetalRelaxSet,
                                   MPStaticSet)
from atomate.vasp.fireworks import OptimizeFW,StaticFW
from fireworks import LaunchPad,Workflow

from .base import BaseWriter

def wf_ce_sample(structure, entry_id, root_name=None, is_metal=False,
                 relax_set_params=None, static_set_params=None,
                 **kwargs):
    """
    Create a fireworks.Workflow for an input structure, including an
    optimization and a static calculation.
    Inputs:
        struture(Structure): 
            The structure to be computed. Usually pre-deformed to
            break relaxation symmetry.
        entry_id(int):
            index of the entry corresponding to the structure.
        root_name(str):
            Root name of all workflows and fireworks. By default, will
            set to name of the current directory.
        is_metal(Boolean):
            If true, will use vasp parameters specific to metallic
            computations.
        relax_set_params(dict):
            A dictionary specifying other parameters to overwrite in
            optimization vasp input set.
        static_set_params(dict):
            A dictionary specifying other parameters to overwrite in
            static vasp input set.
        kwargs contains other paramters you wish to pass into the returned
        workflow object.
    Output:
        A Workflow object, containing optimization-static calculation of
        a single structure in cluster expansion pool.
    """
    # Current folder name will be used to mark calculation entree!
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

    Note: Use get_calc_write method in InputsWrapper to get any Writer object,
          or auto_load.
          Direct init not recommended!
    """
    def __init__(self, lp_file=None,
                 writer_strain=[1.05,1.03,1.01],
                 is_metal = False,
                 ab_setting ={},
                 **kwargs):
        """
        Args:
            lp_file(str):
                path to launchpad setting file. Default to None, then
                launchpad will auto load based on configuration.
            writer_strain(1*3 or 3*3 arraylike):
                Strain matrix to apply to structure before writing as 
                inputs. This helps breaking symmetry, and relax to a
                more reasonable equilibrium structure.
            is_metal(Boolean):
                If true, will use vasp set specifically designed for 
                metals calculation (MPMetalRelaxSet)
            ab_setting(Dict):
                Pass ab-initio software options. For vasp,
                look at pymatgen.vasp.io.sets doc.
                May have two keys, 'relax' and 'static'.
                See pymaten.vasp.io.sets for detail.
        """      
 
        super().__init__(writer_strain=writer_strain,ab_setting=ab_setting,
                         **kwargs)

        self.root_name = os.path.split(os.getcwd())[-1]
        self.is_metal = is_metal

        if lp_file is not None:
            self._lpad = LaunchPad.from_file(lp_file)
        else:
            self._lpad = LaunchPad.auto_load()

    def _write_single(self,structure,eid,*args,**kwargs):
        """
        Write a single computation task to archieve.
        """
        #Apply a slight deformation.
        strain = np.array(self.strain)
        if strain.shape == (3,):
            strain = np.diag(strain)
   
        if strain.shape != (3,3):
            raise ValueError("Incorrect strain format.")
           
        relax_set_params = self.ab_setting.get('relax',{})
        static_set_params = self.ab_setting.get('static',{})

        str_input = Deformation(strain).apply_to_structure(structure)

        wf = wf_ce_sample(str_input, eid, root_name=self.root_name,\
                          is_metal=self.is_metal,\
                          relax_set_params = relax_set_params,\
                          static_set_params = static_set_params,\
                          **kwargs)               

        #Check duplicacy. If any duplicacy occurs, will OVERWRITE for a re-run.
        fw_ids = self._lpad.get_fw_ids()

        all_dupe_fw_ids = []
        for fw_id in fw_ids:
            wf_name = self._lpad.get_wf_summary_dict(fw_id)['name']
            if (wf_name == wf.name and
                fw_id not in itertools.chain(*all_dupe_fw_ids)):
                wf_old = self._lpad.get_wf_by_fw_id(fw_id)
                dupe_fw_ids = [fw.fw_id for fw in wf_old.fws]
                all_dupe_fw_ids.append(dupe_fw_ids)
      
        for dupe_fw_ids in all_dupe_fw_ids:
            self._lpad.delete_wf(dupe_fw_ids[0])

        self._lpad.add_wf(wf)
        print("****Calculation workflow loaded to launchpad for entry: {}."
              .format(eid))
