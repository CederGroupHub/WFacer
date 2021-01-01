"""
Archieve-queue-VASP management class. Read and writes from local folders.
And interacts with the local computation queue directly.
"""
__author__ = "Fengyu Xie"

import os
import stat
import re
import numpy as np

from pymatgen.io.vasp.sets import *
from pymatgen.io.vasp.outputs import Vasprun,Outcar
from pymatgen import Structure,Lattice
from pymatgen.analysis.elesticity.strain import Deformation

import qstat #For SGE only, use other modules or methods for other queue types

from .base import BaseManager
from .utils.format_utils import structure_from_occu

class NotRelaxedError(ValueError):
    """
    Raise this when doing single point calculation before relaxation in an
    archieve.
    """
    pass

class ArchQueueVaspManager(BaseManager):
    """
    A calculation manager class, to write, call ab-initio calculations, and 
    read calculation results from various data warehouses. Current implementation
    includes local archive+SGE queue and mongo database+fireworks.
   
    Current implementations only support vasp.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    For materials project users, this manager is not recommended!
    """
    supported_queue = ('SGE')

    submission_templates = {
                            "SGE":\
                                "#!/bin/bash\n#$ -cwd\n#$ -j y\n#$ -N {*jobname*}\n#$ -m es\n#$ -V\n"+\
                                "#$ -pe impi {*ncores*}\n#$ -o ll_out\n#$ -e ll_er\n#$ -S /bin/bash\n"+\
                                "\n{*vaspcommand*}"
                           }

    submission_commands = {
                           "SGE":"qsub"
                          }
    ####Your queue information here.

    def __init__(self,path='vasp_run',queue_name='SGE',vasp_command = 'vasp', ncores = 16):
        """
        Attributes:
            path(str in path format):
                path to calculations archieve
            queue_name(str):
                name of queueing system. Currently supports SGE(sun
                grid engine).
            vasp_command(str):
                Command used to call vasp in your system. 
                For example, in SGE+mpiexec environment with vasp 5.4.4, we can use:
                'mpiexec.hydra -n $NSLOTS pvasp.5.4.4.intel >> vasp.out' 
                It is highly recommended that you figure out what your command 
                should be.
            ncores(int):
                Number of cores used in each computation. Default is 16,
                number of CPU's per node in a common SGE machine.     
        """
        self.path = path
        if queue_name not in supported_queue:
            raise ValueError("Provided queue {} is not supported by {}".\
                             format(queue_name,self.__class__.__name__))
        self.submit_command = submission_commands[queue_name]
        self.submit_temp = sumbission_templates[queue_name]
        self.vasp_command = vasp_command
        self.ncores = ncores
        
    def create_tasks(self,prim,sc_table,fact_table,entry_ids,*args,\
                     task_type = 'relax', optimize_metal = False,\
                     \
                     **other_set_settings):
        """
        Write input files and submit calculations.
        Inputs:
            prim(pymatgen.Structure):
                primitive cell used to initialize cluster expansion.
            sc_table(pd.DataFrame):
                supercell dimension table
            fact_table(pd.DataFrame):
                fact table containing current calculation informations.
            entry_ids(List of ints):
                List of entree indices to be calculated in fact table.
            task_type(str):
                'relax': relax the provided structure. Will apply a small
                         deformation to break symmetry by a bit.
                'single': single point calculation. If this is selected,
                         You must have already done a relaxation step in
                         the archieve before, so we will expect to find
                         a vasprun.xml in self.path/{entry_id}, and it 
                         shall be converged! If not, will raise error!
                         If a single computation has been finished, and
                         you do another single computation on it, will skip.
 
                'force_single': force single point calculation, regardless
                         of convergence status. If there is a vasprun.xml,
                         will continue with the last ionic step. If not,
                         will compute with undeformed structure!
        
                         Warning: will overwrite previous data! Not recommended!

            optimize_metal(Boolean):
                If you are calculating inter-metallic system, using True is
                recommended for this option. Otherwise always choose False.
     
            Can pass ab-initio settings into **other_set_settings. Read docs
            for pymatgen.io.vasp.sets.MPRelaxSet for more detail.
        No return value.
        """
        fact_uncalc = fact_table.iloc[entry_ids,:]
        fact_uncalc.merge(sc_table,how='left',on='sc_id')
        fact_uncalc['structure'] = fact_uncalc.apply(lambda x:\
                                   structure_from_occu(prim,x['matrix'],x['ori_occu']))

        #Write and sumbit
        for e_id,s in zip(fact_uncalc.entry_id,fact_uncalc.structure):
            self._write_single(e_id,s,task_type = task_type,**other_set_settings)
            self._submit_single(e_id, ncores=ncores)
        
    def _write_single(entry_id,structure,task_type='relax',**other_set_settings):
        """
        Write a single computation task to archieve.
        """
        entry_path = os.path.join(self.path,str(entry_id))
        if not os.path.isdir(entry_path):
            os.makedirs(entry_path)

        #Apply a slight deformation.
        strain = np.diag([1.01,1.03,1.05])
        str_input = Deformation(strain).apply_to_structure(structure)

        if task_type == 'relax':
            if optimize_metal:
                io_set = MPMetalRelaxSet(str_input,**other_set_settings)
            else:
                io_set = MPRelaxSet(str_input,**other_set_settings)

        elif task_type == 'single':
            vrun_path = os.path.join(entry_path,'vasprun.xml')
            ocar_path = os.path.join(entry_path,'OUTCAR')
            vrun_path_new = os.path.join(entry_path,'vasprun.xml.relax')
            ocar_path_new = os.path.join(entry_path,'OUTCAR.relax')

            if not os.path.isfile(vrun_path):
                raise NotRelaxedError("Structure not relaxed before calculation.")
            relax_vasprun = Vasprun(vrun_path)
            if not relax_vasprun.converged:
                raise NotRelaxedError("Structure not relaxed before calculation.")
            if os.path.isfile(vrun_path_new):
                #Alredy calcuated static point.
                return

            str_input = relax_vasprun.structures[-1]
            io_set = MPStaticSet(str_input,**other_set_settings)
            #move and keep old calculation results and necessary inputs.
            
            os.rename(vrun_path,vrun_path_new)
            os.rename(ocar_path,ocar_path_new)

        elif task_type == 'force_single':
            print("Warning: Forcing static calculation at your own risk!")
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

            io_set = MPStaticSet(str_input,**other_set_settings)

        else:
            raise ValueError("Calculation type {} not supported in {}.".\
                              format(task_type,self.__class__.__name__))

        #Write inputs.
        io_set.incar.write_file(os.path.join(entry_path,'INCAR'))
        io_set.poscar.write_file(os.path.join(entry_path,'POSCAR'))
        io_set.potcar.write_file(os.path.join(entry_path,'POTCAR'))
        io_set.kpoints.write_file(os.path.join(entry_path,'KPOINTS'))
        print("****{} calculations written for entry: {}.".format(task_type,entry_id))

    def _submit_single(self,entry_id,ncores = 16):
        """
        Submit a single computation in archieve. It is your responsibility to check that:
        1, The corresponding folder has all required vasp inputs inside
        2, The corresponding folder is not double-computed for the same type of computation.
        """
        root_path = os.getcwd()
        entry_path = os.path.join(self.path,str(entry_id))
        #Check inputs
        if not os.path.isdir(entry_path):
            raise ValueError("Entry {} does not exist under archieve {}!"\
                             .format(entry_id,self.path))
        if not os.path.isfile(os.path.join(entry_path,'INCAR')) or \
           not os.path.isfile(os.path.join(entry_path,'POSCAR')) or \
           not os.path.isfile(os.path.join(entry_path,'POTCAR')) or \
           not os.path.isfile(os.path.join(entry_path,'KPOINTS')):
            raise ValueError("Entry {} vasp inputs not written!".format(entry_id))

        script = self.submit_temp
        #Jobs will be named after root path (Current directory where you run 
        #CEAuto main program).
        jobname = os.path.split(root_path)[-1]+'ce'
        script = re.sub('\{\*jobname\*\}',jobname,script) 
        script = re.sub('\{\*vaspcommand\*\}',self.vasp_command,script)
        script = re.sub('\{\*ncores\*\}',self.ncores,script)
     
        #change to executable and submit
        os.chdir(entry_path)
        with open('sub.sh','w') as script_file:
            script_file.write(script)
        st = os.stat('sub.sh')
        os.chmod('sub.sh', st.st_mode | stat.S_IEXEC)
        
        os.system(self.submit_command+' sub.sh')
        os.chdir(root_path) #It is essential to move back!
        print('****Submitted vasp for entry: {}.'.format(entry_id))

    def check_tasks_status(self,entry_ids=None):
        """
        Check ab-initio task status for given entree indices.
        'NC' for not submitted, 'RX' for relaxation or waiting for
        relaxation in queue, 'RF' for relaxation finished,'SP' for 
        doing single point or waiting for single point in queue. 
        'CL' for all finished. None for non-existent in archieve.
        (same as in the doc of CEAuto.featurizer.)

        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None, will check all available entree.     
        Returns:
            A list of strings specifying status of each task.
        """
#### TODO
        return

    @abstractmethod
    def check_convergence_status(self,entry_ids=None):
        """
        Checks convergence status of entree with specific indices.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None, will check all available entree.
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

    def load_properties(self,entry_ids=None,normalize_by = 1,prop_names='energy',
                        include_pnames=True):
        """
        Load calculated properties from ab_initio data.
        Inputs:
             entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                If none given, will return all availale entrees.       
             normalize_by(float or 1D-arraylike):
                before returning values, will devide them by this
                value or array. Used to normalize extensive variables.
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
    def _load_property_by_name(self,entry_ids=None,name='energy'):
        """
        Load a single type of property of structures from the warehouse.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                If none given, will return all availale entrees. 
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
