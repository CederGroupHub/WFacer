"""
Base calculation manager class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

from abc import ABC, abstractmethod
import numpy as np
import time
from datetime import datetime

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
   
    Attributes:
        time_limit(float):
            Time limit for all calculations to finish. Unit is second.
            Default is 3 days.
        check_interval(float):
            Interval to check status of all computations in queue. Unit is second.
            Default is every 5 mins.
    """
    def __init__(self,time_limit=259200,check_interval=300):
        self.time_limit = time_limit
        self.interval = check_interval
        
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
        Submit tasks for calculation, and checks finish or time limit.
        Be sure to finished all calculations of all entree, then submit 
        another type of jobs.
        It is your responsibility not to dupe-submit.

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
        q_status = self.entree_in_queue(entry_ids)
        for eid,e_in_queue zip(entry_ids,q_status):
            if not e_in_queue:
                self._submit_single(eid)

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
                  format(datetime.now(),t_quota,int(np.sum(status)),len(status))
        
        if t_quota>0:            
            print("**Calculations finished at: {}".format(datetime.now()))
        else:
            self.kill_tasks()
            print("**Warning: only {}/{} calculations finished in time limit {}!".\
                  format(int(np.sum(status)),len(status),self.time_limit))
            print("**You may want to use a longer time limit.")

        return (t_quota>0)

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
    def _submit_single(self,eid):
        """
        Submit a single calculation.
        """
        return
