"""
Archieve calculations reader class. These classes DO NOT MODIFY 
fact table!
"""
__author__ = "Fengyu Xie"

import logging
log = logging.getLogger(__name__)

import numpy as np
import os

from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.io.vasp.inputs import Poscar

from .base import BaseReader


class ArchvaspReader(BaseReader):
    """
    A calculation reader class, to read calculation results from 
    various data warehouses. Current implementation includes local 
    archive+SGE queue and mongo database+fireworks.
   
    Current implementations only support vasp.

    This class only serves as accessor to the data warehouse, and will
    not change the fact table. Everything in this class shall be
    temporary, and will not be saved as dictionaries into disk.
    """
    def __init__(self, path='vasp_run', **kwargs):
        """Initialize.

        Args:
            path(str):
                path to the archieve. By difault, will be under ./vasp_run
        """
        self.path = path
        # Class cache for easy access to Vasprun and Outcar.
        self._vruns = {}
        self._ocars = {}

    def check_convergence_status(self, entry_ids):
        """Checks convergence status of entree with specific indices.

        Args:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
                Must be provided.
        Returns:
            A list of booleans, each shows whether the calculation
            of the correponding entry have succeeded.
            If inquired calculation does not exist, will raise an
            error. If not calculated (no vasprun.xml under it), will
            regard as not converged.
        """
        status = []
        for eid in entry_ids:
            e_path = os.path.join(self.path, str(eid))
            if not os.path.isdir(e_path):
                raise ValueError("Specified entry {} does not exist!"
                                 .format(eid))
            vrunpath = os.path.join(e_path, 'vasprun.xml')
            # For relax-static workflow, this vasprun is for static;
            # For relax-only workflow, this vasprun is for relax.
            if not os.path.isfile(vrunpath):
                status.append(False)
            else:
                # Read and write cache
                if entry_id in self._vruns:
                    vrun = self._vruns[entry_id]
                else:
                    vrun = Vasprun(vrunpath)
                    self._vruns[entry_id] = vrun
                status.append(vrun.converged)

        return status

    def load_structures(self, entry_ids):
        """Loads relaxed structures.

        Args:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0
                If None, will check all available ids.
        Returns:
            List[pymatgen.Structure]:
            All composed of pymatgen.Element (undecorated).

        If inquired calculation does not exist, will raise an
        error. If not calculated (not vasprun.xml), will try to
        look for Poscar instead. If Poscar does not exist, will
        raise error.
        """
        structures = []
        for eid in entry_ids:
            e_path = os.path.join(self.path, str(eid))
            if not os.path.isdir(e_path):
                raise ValueError("Specified entry {} does not exist!"
                                 .format(eid))
            vrunpath = os.path.join(e_path, 'vasprun.xml')
            if not os.path.isfile(vrunpath):
                log.warning("****Entry {} may not be optimal.".format(eid) +
                            " Using un-optimized structure.")
                pospath = os.path.join(e_path, 'POSCAR')
                if not os.path.isfile(pospath):
                    raise ValueError("Specified entry {} does not exist!"
                                     .format(eid))
                structures.append(Poscar(pospath).structure)
            else:
                #Read and write cache
                if entry_id in self._vruns:
                    vrun = self._vruns[entry_id]
                else:
                    vrun = Vasprun(vrunpath)
                    self._vruns[entry_id] = vrun
                if not vrun.converged:
                    log.warning("****Entry {} may not be optimal."
                                .format(eid) +
                                " Using un-optimized structure."
                                .format(eid))
                structures.append(vrun.structures[-1])

        return structures

    def _load_property_by_name(self, entry_ids, name='energy'):
        """Load a single type of property.

        Currently only checks Vasprun and Outcar, and loads
        energy and magnetization. Implement more if you want.

        Args:
            entry_ids(List of ints):
                list of entry indices to be checked. Indices in a
                fact table starts from 0.
            name(str):
                Name of property to be extracted. By default, gives
                energy.
                Must be a member in list: supported_properties.
                (class constant)
        Returns:
            A list containing extracted proerties of corresponding 
            entree.
        """
        property_per_str = []
        for eid in entry_ids:
            e_path = os.path.join(self.path, str(eid))
            vrunpath = os.path.join(e_path, 'vasprun.xml')
            ocarpath = os.path.join(e_path, 'OUTCAR')
            # Add more if you need.

            if name == 'energy':
                if eid in self._vruns:
                    vrun = self._vruns[eid]
                    property_per_str.append(vrun.final_energy)
                elif os.path.isfile(vrunpath):
                    vrun = Vasprun(vrunpath)
                    self._vruns[eid] = vrun
                    property_per_str.append(vrun.final_energy)
                else:
                    raise ValueError("Specified entry {} does not have "
                                     .format(eid) +
                                     "vasprun.xml!")

            if name == 'magnetization':
                if eid in self._ocars:
                    ocar = self._ocars[eid]
                    mags = [st_mag["tot"] for st_mag in ocar.magnetization]
                    property_per_str.append(mags)
                elif os.path.isfile(ocarpath):
                    ocar = Outcar(ocarpath)
                    self._ocars[eid] = ocar
                    mags = [st_mag["tot"] for st_mag in ocar.magnetization]
                    property_per_str.append(mags)
                else:
                    raise ValueError("Specified entry {} ".format(eid) +
                                     "does not have OUTCAR!")

        return property_per_str
