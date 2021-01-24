"""
Archieve+vasp calculation writer class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

import os

from pymatgen.io.vasp.sets import MPRelaxSet, MPMetalRelaxSet,\
                                  MPStaticSet

from .base import BaseWriter

class ArchVaspWriter(BaseWriter):
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
    def __init__(self, path = 'vasp_run', 
                       writer_strain=[1.05,1.03,1.01],\
                       is_metal = False,\
                       ab_setting={},\
                       **kwargs):
        super().init_(writer_strain=writer_strain,ab_setting=ab_setting,**kwargs)
        self.path = path
        self.is_metal = is_metal
               
    def write_tasks(self,strs_undeformed,entry_ids,*args,mode='relax',**kwargs):
        """
        Write input files.
        Inputs(Order of arguments matters):
            strs_undeformed(List of Structure):
                Structures in original lattice.(Not deformed.)
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                Must be provided.       
            mode(str):
                Type of inputs to write. Supporting:
                    'relax': 
                        Relaxation. Each initial structure should be relaxed
                        first. (default)
                    'static': 
                        Static calculation. Used after relaxation.
                        If previous relaxation does not exist, or did not converge,
                        will skip this entry.
                        (You can choose either to perform a more accurate static calculation 
                         after relaxation or not, by default, will always do a static.)
                    'force_static': 
                        Force a static calculation regardless of the previous 
                        relaxation. (Might be used when some high-energy space
                        constraints are required, but usually not recommended.)

        No return value.
        """
        if mode == 'force_static':
            print("**Warning: Forcing static calculation without checking relaxation, do this at your own risk!")
        super().write_tasks(strs_undeformed,entry_ids,*args,mode = mode,**kwargs)

    def _write_single(self,structure,eid,*args,mode = 'relax', **kwargs):
        """
        Write a single computation task to archieve.
        """
        epath = os.path.join(self.path,str(eid))
        if not os.path.isdir(epath):
            os.makedirs(epath)

        #Apply a slight deformation.
        strain = np.array(self.strain)
        if strain.shape == (3,):
            strain = np.diag(strain)
   
        if strain.shape != (3,3):
            raise ValueError("Incorrect strain format.")
           
        str_input = Deformation(strain).apply_to_structure(structure)

        if mode == 'relax':
            relax_setting = self.ab_setting.get('relax',{})

            if self.is_metal:
                io_set = MPMetalRelaxSet(str_input,**relax_setting)
            else:
                io_set = MPRelaxSet(str_input,**relax_setting)

        elif mode == 'static':
            static_setting = self.ab_setting.get('static',{})

            vrun_path = os.path.join(entry_path,'vasprun.xml')
            ocar_path = os.path.join(entry_path,'OUTCAR')
            vrun_path_new = os.path.join(entry_path,'vasprun.xml.relax')
            ocar_path_new = os.path.join(entry_path,'OUTCAR.relax')

            if not os.path.isfile(vrun_path):
                print("****Entry {} not relaxed before static run.".format(eid))
                return #Skip this entry
            relax_vasprun = Vasprun(vrun_path)
            if not relax_vasprun.converged:
                print("****Entry {} not relaxed before static run.".format(eid))
                return #Skip this entry
            if os.path.isfile(vrun_path_new):
                #Alredy calcuated static point.
                print("****Entry {} static already written.".format(eid))
                return #Skip this entry

            str_input = relax_vasprun.structures[-1]
            io_set = MPStaticSet(str_input,**static_setting)

            #move and keep old calculation results and necessary inputs.            
            os.rename(vrun_path,vrun_path_new)
            os.rename(ocar_path,ocar_path_new)

        elif task_type == 'force_static':
            static_setting = self.ab_setting.get('static',{})

            vrun_path = os.path.join(entry_path,'vasprun.xml')
            ocar_path = os.path.join(entry_path,'OUTCAR')
            vrun_path_new = os.path.join(entry_path,'vasprun.xml.relax')
            ocar_path_new = os.path.join(entry_path,'OUTCAR.relax')

            #Use undeformed lattice for static calc
            str_input = structure
            if os.path.isfile(vrun_path):
                str_input = Vasprun(vrun_path).structures[-1]
                os.rename(vrun_path,vrun_path_new)
            if os.path.isfile(ocar_path):
                os.rename(ocar_path,ocar_path_new)

            io_set = MPStaticSet(str_input,**static_setting)

        else:
            raise ValueError("Calculation type {} not supported in {}.".\
                              format(mode,self.__class__.__name__))

        #Write inputs.
        io_set.incar.write_file(os.path.join(entry_path,'INCAR'))
        io_set.poscar.write_file(os.path.join(entry_path,'POSCAR'))
        io_set.potcar.write_file(os.path.join(entry_path,'POTCAR'))
        io_set.kpoints.write_file(os.path.join(entry_path,'KPOINTS'))
        print("****{} calculations written for entry: {}.".format(mode,eid))
