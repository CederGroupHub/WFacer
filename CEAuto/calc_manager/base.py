"""
Base calculation manager class.
"""
__author__ = "Fengyu Xie"

from abc import ABC, abstractmethod
import numpy as np
import time
from datetime import datetime

from ..config_paths import *

class BaseManager(ABC):
    """
    A calculation manager class, to write, call ab-initio calculations.
    Current implementation includes local archive+SGE queue and mongo 
    database+fireworks. Interacts with 
    calculation resources.   

    May support any abinito software.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    Note: Use get_calc_manager method in InputsWrapper to get any Manager object,
          or auto_load.
          Direct init not recommended!
    """
    def __init__(self,data_manager,\
                      time_limit=345600,check_interval=300,\
                      **kwargs):
        """
        Args:
            data_manager(DataManager):
                An interface to the calculated and enumerated data.
            time_limit(float):
                Time limit for all calculations to finish. Unit is second.
                Default is 4 days.
            check_interval(float):
                Interval to check status of all computations in queue. Unit is second.
                Default is every 5 mins.
        """
        self.time_limit=time_limit
        self.check_interval=check_interval
        self._dm = data_manager        

    @abstractmethod
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
        return
 
    def run_tasks(self,entry_ids):
        """
        Run all calculation tasks specified by entree ids.
        And monitor status. 
        If CEAuto job is interrupted, for arch+queue, will
        resubmit the specified jobs; for mongodb+atomate,
        will check for unreserved tasks, and resubmit to
        queue.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                Must be provided.
        Return:
            Boolean, indicating whether all entree end in specified time
            limit.

        NOTE: 
           1, This function does not care the type of work you are doing,
              either 'relaxation' or 'static'. It is your responsibility
              check in calc_writer before submission and logging.      
           2, This function waits for the calculation resources, therfore
              will always hang for a long time.  
        """
        if self._dm._schecker.after("calc"):
            print("**Calculation of entree already done in the current iteration {}"\
                  .format(self._dm._schecker.cur_iter_id))
            return True

        self._submit_all(entry_ids)

        #set timer
        t_quota = self.time_limit
        print("**Calculations started at: {}".format(datetime.now()))
        print("**Number of calculations {}".format(len(entry_ids)))

        n_checks = 0
        while t_quota>0:
            time.sleep(self.check_interval)
            t_quota -= self.check_interval
            n_checks += 1
            status = self.entree_in_quque(entry_ids)
            if not np.any(self.entree_in_quque(entry_ids)): 
                break          
            print(">>Time: {}, Remaining(seconds): {}\n  {}/{} calculations finished!".\
                  format(datetime.now(),t_quota,int(np.sum(status)),len(status)))
        
        if t_quota>0:            
            print("**Calculations finished at: {}".format(datetime.now()))
        else:
            self.kill_tasks()
            print("**Warning: only {}/{} calculations finished in time limit {}!".\
                  format(int(np.sum(status)),len(status),self.time_limit))
            print("**You may want to use a longer time limit.")

        return (t_quota>0)

    def auto_run(self):
        """
        Automatically submits entree with calc_status 'CC'(computing), and 
        mark them as 'CL'(computation finished) upon completion.

        The modification in the data repository will be flushed upon update.
        No return value.
        """
        eids = self._dm.get_eid_w_status('CC')
        _ = self.run_tasks(eids)
        #Will mark as finished, even if computations exceed walltime limit.
        self._dm.set_status(eids,'CL')
        self._dm.auto_save()  #flush data

    @abstractmethod
    def kill_tasks(self,entry_ids=None):
        """
         Kill specified tasks if they are still in queue.
         Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None given, will kill anything in the queue
                with the current job root name.
        """  
        return

    @abstractmethod
    def _submit_all(self,eids=None):
        """
        Submit entree to queue.
        """
        return  
