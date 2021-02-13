"""
Archieve-queue management class. Interacts with the local computation 
queue directly.
"""
__author__ = "Fengyu Xie"

import os
import stat
import re
import numpy as np

from abc import ABC, abstractmethod

from .base import BaseManager

####If you ever define your own queue, you can write your own class like this.
class ArchQueueManager(BaseManager,ABC):
    """
    Base queue calculation manager class, to call and monitor ab-initio calculations.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    Does not care what type of calculation it is running, so figure it out with
    calc_writer.

    For materials project users, this manager is not recommended!

    NOTE: This class uses root folder name (where you run CEAuto under) and entry
          ids to name queue jobs, so be sure that:
          1, No two CE folders have the same name!
          2, Better not to include underscore in your CE folder names.

    Note: Use get_calc_manager method in InputsWrapper to get any Manager object,
          or auto_load.
          Direct init not recommended!
    """
    submission_template = ""
    submission_command = ""
    kill_command = ""

    def __init__(self,data_manager,\
                      path='vasp_run', ab_command='vasp', ncores = 16,\
                      time_limit=345600,check_interval=300,\
                      **kwargs):
        """
        Args:
            data_manager(DataManager):
                An interface to the calculated and enumerated data.
            path(str in path format):
                path to calculations archieve
            queue_name(str):
                name of queueing system. Currently supports SGE(sun
                grid engine).
            ab_command(str):
                Command used to call the ab_initio program in your system. 
                For example, in SGE+mpiexec environment with vasp 5.4.4, we can use:
                'mpiexec.hydra -n $NSLOTS pvasp.5.4.4.intel >> vasp.out' 
                It is highly recommended that you figure out what your command 
                should be.
            ncores(int):
                Number of cores used in each computation. Default is 16,
                number of CPU's per node in a common SGE machine.     
            time_limit(float):
                Time limit for all calculations to finish. Unit is second.
                Default is 4 days.
            check_interval(float):
                Interval to check status of all computations in queue. Unit is second.
                Default is every 5 mins.
        """
        super().__init__(time_limit=time_limit,check_interval=check_interval,\
                         data_manager=data_manager)
        self.path = path
        self.ab_command = ab_command
        self.ncores = ncores
        self._root = os.getcwd()
        self._root_name = os.path.split(os.getcwd())[-1] 
        #Assume you're running CEAuto under a fixed directory.
      
    def _submit_all(self,eids):
        """
        Submit all tasks in a queue.
        """
        qstats = self.entree_in_queue(eids)
        for eid,e_in_queue in zip(eids,qstats):
            if not e_in_queue:
                self._submit_single(eid)

    def _submit_single(self,eid):
        """
        Submit a single computation in archieve. It is your responsibility to check that:
        1, The corresponding folder has all required vasp inputs inside
        2, The corresponding folder is not double-computed for the same type of computation.
        """
        epath = os.path.join(self.path,str(eid))
        #Check inputs
        if not os.path.isdir(epath):
            raise ValueError("Entry {} does not exist under archieve {}!"\
                             .format(eid,self.path))
        if not os.path.isfile(os.path.join(epath,'INCAR')) or \
           not os.path.isfile(os.path.join(epath,'POSCAR')) or \
           not os.path.isfile(os.path.join(epath,'POTCAR')) or \
           not os.path.isfile(os.path.join(epath,'KPOINTS')):
            raise ValueError("Entry {} vasp inputs not written!".format(eid))

        script = submission_template
        #Jobs will be named after root path (Current directory where you run 
        #CEAuto main program).
        jobname = self._root_name+'_ce_{}'.format(eid)
        script = re.sub('\{\*jobname\*\}',jobname,script) 
        script = re.sub('\{\*abcommand\*\}',self.ab_command,script)
        script = re.sub('\{\*ncores\*\}',self.ncores,script)
     
        #change to executable and submit
        os.chdir(epath)
        with open('sub.sh','w') as script_file:
            script_file.write(script)
        st = os.stat('sub.sh')
        os.chmod('sub.sh', st.st_mode | stat.S_IEXEC)
        
        os.system(submission_command+' sub.sh')
        os.chdir(self._root)  #It is essential to move back!
        print('****Submitted ab_initio for entry: {}.'.format(entry_id))

class ArchSGEManager(ArchQueueManager):
    """
    SGE queue calculation manager class, to call and monitor ab-initio calculations.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    Does not care what type of calculation it is running, so figure it out with
    calc_writer.

    For materials project users, this manager is not recommended!

    NOTE: This class uses root folder name (where you run CEAuto under) and entry
          ids to name queue jobs, so be sure that:
          1, No two CE folders have the same name!
          2, Better not to include underscore in your CE folder names.

    Note: Use get_calc_manager method in InputsWrapper to get any Manager object,
          or auto_load.
          Direct init not recommended!
    """
    submission_template = "#!/bin/bash\n#$ -cwd\n#$ -j y\n"+\
                               "#$ -N {*jobname*}\n#$ -m es\n#$ -V\n"+\
                               "#$ -pe impi {*ncores*}\n#$ -o ll_out\n"+\
                               "#$ -e ll_er\n#$ -S /bin/bash\n"+\
                               "\n{*abcommand*}"
    submission_command = "qsub"
    kill_command = "qdel"

    def __init__(self,data_manager,\
                      path='vasp_run', ab_command='vasp', ncores = 16,\
                      time_limit=345600,check_interval=300,\
                      **kwargs):
        """
        Args:
            data_manager(DataManager):
                An interface to the calculated and enumerated data.
            path(str in path format):
                path to calculations archieve
            queue_name(str):
                name of queueing system. Currently supports SGE(sun
                grid engine).
            ab_command(str):
                Command used to call the ab_initio program in your system. 
                For example, in SGE+mpiexec environment with vasp 5.4.4, we can use:
                'mpiexec.hydra -n $NSLOTS pvasp.5.4.4.intel >> vasp.out' 
                It is highly recommended that you figure out what your command 
                should be.
            ncores(int):
                Number of cores used in each computation. Default is 16,
                number of CPU's per node in a common SGE machine.     
            time_limit(float):
                Time limit for all calculations to finish. Unit is second.
                Default is 4 days.
            check_interval(float):
                Interval to check status of all computations in queue. Unit is second.
                Default is every 5 mins.
        """

        super().__init__(path=path, ab_command=ab_command, ncores=ncores,\
                      time_limit=time_limit, check_interval=check_interval,\
                      data_manager=data_manager,**kwargs)

    def entree_in_queue(self,entry_ids):
        """
        Check ab-initio task status for given entree indices.
        (same as in the doc of  CEAuto.featurizer.)        
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                Must be provided.
        Returns:
            A list of Booleans specifying status of each task.
            True for in queue (either running or waiting, your 
            responsibility to distinguish between them two)
            ,false for not in queue.

        NOTE: This function does not care the type of work you are doing,
              either 'relaxation' or 'static'. It is your responsibility
              check in calc_writer before submission and logging.
        """
        import qstat #package for SGE only

        waiting, running = qstat.qstat()
        all_jobs = waiting+running

        all_eids_in_q = []
        for job in all_jobs:
            if isinstance(job,dict) and job['JB_name'].find(self._root_name)==0:
                #A job name must begins exactly with the root name!
                #eid is recorded as numbers at the end of a job name.
                all_eids_in_q.append(int(job['JB_name'].split('_')[-1]))

        return [(eid in all_eids_in_q) for eid in entry_ids]       

    def kill_tasks(self,entry_ids=None):
        """
         Kill specified tasks if they are still in queue.
         Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None given, will kill anything in the queue
                with the current job root name.
        No return value.
        """  
        import qstat

        job_ids_to_kill = []
        entry_ids_to_kill = []
        
        waiting, running = qstat.qstat()
        all_jobs = waiting+running
        for job in all_jobs:
            if isinstance(job,dict) and job['JB_name'].find(self._root_name)==0:
                #A job name must begins exactly with the root name!
                #eid is recorded as numbers at the end of a job name.
                eid = int(job['JB_name'].split('_')[-1])
                if (entry_ids is not None and eid in entry_ids) or \
                    entry_ids is None:
                    job_ids_to_kill.append(job['JB_job_number'])
                    entry_ids_to_kill.append(eid)

        for jid,eid in zip(job_ids_to_kill,entry_ids_to_kill):
            os.system(kill_command+' {}'.format(jid))
            print("****Job killed: {}, Entry id: {}.".format(jid,eid))
