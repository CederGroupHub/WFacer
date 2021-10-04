"""
Base calculation manager class.
"""
__author__ = "Fengyu Xie"

import logging
log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import numpy as np
import time
from datetime import datetime

from ..config_paths import *
from ..utils.class_utils import derived_class_factory


class BaseManager(ABC):
    """
    A calculation manager class, to write, call ab-initio calculations.
    Current implementation includes local archive+SGE queue and mongo 
    database+fireworks. Interacts with 
    calculation resources.   

    May support any abinito software.

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will
    not be saved as dictionaries into disk.

    Note: Use get_calc_manager method in InputsWrapper to get any Manager
          object, or auto_load.
          Direct init not recommended!
    """
    def __init__(self, time_limit=345600, check_interval=300,
                 *args, **kwargs):
        """
        Args:
            time_limit(float):
                Time limit for all calculations to finish. Unit is second.
                Default is 4 days.
            check_interval(float):
                Interval to check status of all computations in queue. Unit is second.
                Default is every 5 mins.
        """
        self.time_limit=time_limit
        self.check_interval=check_interval

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
 
    def _run_tasks(self,entry_ids):
        """Run all calculation tasks specified by entree ids.

        Also monitors status. 
        If CEAuto job is interrupted, for arch+queue, will
        resubmit the specified jobs; for mongodb+atomate,
        will check for unreserved tasks, and resubmit to
        queue.

        We do not recommend you call this directly.
        Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                Must be provided.
        Return:
            Float, giving remaining time.

        NOTE: 
           1, This function does not care the type of work you are doing,
              either 'relaxation' or 'static'. It is your responsibility
              check in calc_writer before submission and logging.      
           2, This function waits for the calculation resources, therfore
              will always hang for a long time.  
        """
        self._submit_all(entry_ids)

        #set timer
        t_quota = self.time_limit
        log.critical("**Calculations started at: {}.".format(datetime.now()))
        log.critical("**Number of calculations {}.".format(len(entry_ids)))

        n_checks = 0
        while t_quota>0:
            time.sleep(self.check_interval)
            t_quota -= self.check_interval
            n_checks += 1
            status = self.entree_in_queue(entry_ids)
            if not np.any(status): 
                break          
            log.debug(">>Time: {}, Remaining(seconds): {}, " +
                      "{}/{} calculations finished.".
                      format(datetime.now(), t_quota,
                             int(np.sum(status)), len(status)))
        
        if t_quota>0:            
            log.critical("**Calculations finished at: {}."
                         .format(datetime.now()))
        else:
            self.kill_tasks()
            log.warning("**Only {}/{} calculations finished "
                        .format(int(np.sum(status)), len(status)) +
                        "in time limit {}!".format(self.time_limit) +
                        " You may want to use a LONGER time limit.")

        return t_quota

    def run_df_entrees(self, data_manager):
        """Automatically submits entree.

        Submit entree with calc_status 'CC'(computing), and 
        mark them as 'CL'(computation finished) upon completion.

        The modification in the data repository will be flushed upon update.

        Note: Use your TimeKeeper to check before running this, unless you
              wish to re-run all DFT calculations in this iteration.

        Args:
            data_manager(DataManager):
                A data manager object containing info of all the calculations
                so far.
                Since python uses 'pass object reference' in functions, this
                dataframe will be updated on-the-fly.

        Returns:
            float: remaining time.
        """

        eids = data_manager.get_eid_w_status('CC')
        remain_quota = self._run_tasks(eids)

        # Will mark everything as finished.
        # CalcReader will detect failures later.
        data_manager.set_status(eids, 'CL')

        return remain_quota

    @abstractmethod
    def kill_tasks(self, entry_ids=None):
        """Kill specified tasks if they are still in queue.

        Use caution.
        Args:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None given, will kill anything in the queue
                with the current job root name.
        """  
        return

    @abstractmethod
    def _submit_all(self, eids=None):
        """Submit entree to queue."""
        return


def manager_factory(manager_name, *args, **kwargs):
    """Create a calculation manager with given name.

    Args:
       manager_name(str):
         Name of a calc manager class.
       *args, **kwargs:
         Arguments used to initialize a manager class.

    Returns:
       BaseManager.
    """
    name = manager_name.capitalize() + 'Manager'
    return derived_class_factory(name, BaseManager, *args, **kwargs)
