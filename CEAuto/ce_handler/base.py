"""
Base CE handler. A CE handler processes computed cluster expansion
with either monte-carlo or pseudo boolean methods, to generate 
useful configurational space information, such as critical samples
and ground states.
"""
__author__ == 'Fengyu Xie'

from abc import ABC,abstractmethod

class BaseHandler(ABC):
    """
    Base CE Handler class. Provides solution to ground states under 
    a single chempot or composition, and if is MC, can provide sampling,
    plus auto equilibration.

    Object does not need serialization.
    """
    def __init__(self,ce,sc_mat,**kwargs):
        """
        Args:
            ce(ClusterExpansion):
                A cluster expansion object, stroring cluster subspace
                and all ecis.
            sc_mat(3*3 Arraylike):
                Supercell matrix to solve on.
        """
        self.ce = ce
        self.sc_mat = sc_mat

    @abstractmethod
    def solve(self):
        """
        Provide ground state solution at current mu or comp.
        Returns:
            gs_occu, gs_e:
                Ground state occupation, ground state energy.
        """
        return
