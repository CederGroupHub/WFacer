"""
Launchpad calculation manager class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

import os
import numpy as np
from itertools import chain

from fireworks import LaunchPad, FWorker
from fireworks.queue.queue_launcher import rapidfire
from fireworks.fw_config import QUEUEADAPTER_LOC,LAUNCHPAD_LOC,FWORKER_LOC
from fireworks.utilities.fw_serializers import load_object_from_file

from .base import BaseManager

class MongoFWManager(BaseManager):
    """
    A calculation manager class, to write, call ab-initio calculations.
    Current implementation includes mongo database+fireworks. Interacts with 
    calculation resources.   

    This class only interacts with the data warehouse, and will not change 
    the fact table. Everything in this class shall be temporary, and will not 
    be saved as dictionaries into disk.

    NOTE: For the current implementation, each workflow will only be launched ONCE!
          I have not implemented any function for checking and re-submission, so
          take care if you may have sudden computer power-off!

    Args:
        kill_command(str):
            Killing command of your specific queue. If you queue system belongs to:
            PBS, SGE, Cobalt, SLURM, LoadLeveler, LoadSharingFacility, or MOAB, and
            you have set up your atomate configurations correctly, you shouldn't need
            to specify this value.
    """
    default_kill_commands = 
    {
        "PBS": "qdel",
        "SGE": "qdel",
        "Cobalt": "qdel",
        "SLURM": "scancel",
        "LoadLeveler": "llcancel",
        "LoadSharingFacility": "bkill",
        "MOAB": "canceljob"
    }

    def __init__(self,kill_command=None):
        self.root_name = os.path.split(os.getcwd())[-1]
        self._lpad = LaunchPad.from_file(LAUNCHPAD_LOC)
        self._fworker = FWorker.from_file(FWORKER_LOC)
        self._qadapter = load_object_from_file(QUEUEADAPTER_LOC)
        #If you define any other qadapter than CommonQadapter, make sure
        #to implement a q_type attribute!
        kill_command = kill_command or default_kill_commands.get(self._qadapter.q_type,None)
        if kill_command is None:
            raise ValueError("Queue type {} not supported, but no kill command given in init!"\
                             .format(self._qadapter.q_type))

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
        entree_status = []
        for eid in entry_ids:
            entry_name = 'ce_{}_{}'.format(self.root_name,eid)
            #refer to CEAuto.calc_writer.mongo_vasp
            wf_d = self._lpad.workflows.find_one({'name':entry_name}) or {}
            if len(wf_d) == 0:
                entree_status.append(False)
                continue
            if wf_d.get('state',None) not in ['READY' or 'WAITING'\
                                              or 'RESERVED' or 'RUNNING']:
                entree_status.append(False)
                continue
            entree_status.append(True)

    def _submit_all(self,eids=None):
        """
        Submit all ready workflows to queue, in reserve mode. Do not call this explicitly!
        """
        print("**Submitting entree"+(': {}'.format(eids) if eids is not None else '.'))
        rapidfire(self._lpad,self._fworker,self._qadapter,reserve=True)

               
    def kill_tasks(self,entry_ids=None,kill_command='qdel'):
        """
         Kill specified workflows of entrees, if they are still in the queue.
         The workflows will be defused!

         Inputs:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None given, will kill anything in the queue
                with the current job root name.
            kill_command(str):
                killing command for your specific queue. Default is qdel,
                but for slurm, can be different one.
        """ 
        import re

        regex_ = re.compile("^ce_{}".format(self.root_name))
        all_wids = [wf['nodes'] for wf in self._lpad.fireworks.find({'name':regex_})]
        all_qids = [[self._lpad.get_reservation_id_from_fw_id(fw_id) for fw_id in e_fwids]\
                     for e_fwids in all_fwids]
        all_eids = [int(wf['name'].split('_')[-1])\
                    for wf in self._lpad.workflows.find({'name':regex_})]        

        entry_ids = entry_ids or all_eids
        qstats = self.entree_in_queue(entry_ids) 
        entry_ids = [eid for eid,e_in_q in zip(entry_ids,qstats) if e_in_q]      

        print("Killing tasks associated with")
        all_eids_2k = []
        all_qids_2k = []
        all_wids_2k = []
        for wids,qids,eid in zip(all_wids,all_qids,all_eids):
            if eid in entry_ids:
                all_eids_2delete.append(eid)
                all_qids_2delete.append(qids)
                all_wids_2delete.append(wids)  

        for wids,qids,eid in zip(all_wids_2k,all_qids_2k,all_eids_2k):
            for qid in qids:
                os.system(kill_command+' {}'.format(qid))
            #defuse the workflow correspoding to eid
            self._lpad.defuse_wf(wids[0])
            print("****Tasks corresponding to entry: {} has been killed and defused!".format(eid))

        return