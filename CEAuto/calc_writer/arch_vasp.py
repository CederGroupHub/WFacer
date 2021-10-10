"""
Archieve+vasp calculation writer class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

import os

from pymatgen.io.vasp.sets import (MPRelaxSet, MPMetalRelaxSet,
                                   MPStaticSet)

from .base import BaseWriter

class ArchvaspWriter(BaseWriter):
    """A calculation write class. 

    To write ab-initio calculations to various data warehouses.
    Current implementation includes local archive and mongo+fireworks.

    Does not interact with your computing resource.

    This class writes vasp input files into local folders. Relaxation and 
    static step will be two separate submissions!!

    Current implementations only support vasp.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    It's your responsibility not to dupe-write a directory and not to waste
    your own time.

    Note: 
      1, Use get_calc_write method in InputsWrapper to get any Writer object,
         or auto_load.
         Direct init not recommended!

      2, Relax only. Use the last structure in relax as static. (Different
         from MongovaspWriter).
    """

    def __init__(self, path='vasp_run', 
                 writer_strain=[1.05, 1.03, 1.01],
                 is_metal=False,
                 ab_setting={},
                 **kwargs):
        """Initialize.

        Args: 
            writer_path(str):
                path to the calculation archieve.
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
                May have two keys, 'relax' and 'static'. If you provide both,
                only 'relax' will be used.
                See pymaten.vasp.io.sets for detail.
        """
        super().__init__(writer_strain=writer_strain, ab_setting=ab_setting,
                         **kwargs)
        self.path = path
        self.is_metal = is_metal
               
    def _write_single(self, structure, eid, *args, **kwargs):
        """Write a single computation task to archieve.

        Relax only. If you provide both 'relax' and 'static' options,
        only 'relax' options will be used.
        """
        epath = os.path.join(self.path, str(eid))
        if not os.path.isdir(epath):
            os.makedirs(epath)

        # Apply a slight deformation.
        strain = np.array(self.strain)
        if strain.shape == (3, ):
            strain = np.diag(strain)
   
        if strain.shape != (3, 3):
            raise ValueError("Incorrect strain format.")

        str_input = Deformation(strain).apply_to_structure(structure)

        relax_setting = self.ab_setting.get('relax', {})
        static_setting = self.ab_setting.get('static', {})
        if len(relax_setting) > 0:
            setting = relax_setting
        else:
            setting = static_setting

        if self.is_metal:
            io_set = MPMetalRelaxSet(str_input, **setting)
        else:
            io_set = MPRelaxSet(str_input, **setting)

        #Write inputs.
        io_set.incar.write_file(os.path.join(entry_path, 'INCAR'))
        io_set.poscar.write_file(os.path.join(entry_path, 'POSCAR'))
        io_set.potcar.write_file(os.path.join(entry_path, 'POTCAR'))
        io_set.kpoints.write_file(os.path.join(entry_path, 'KPOINTS'))
        print("****Calculations written for entry: {}.".format(eid))
