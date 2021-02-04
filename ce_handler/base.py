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

class MCHandler(BaseHandler,ABC):
    """
    Base monte-carlo handler class. Provides ground states, de-freeze
    sampling.
    Note: In the future, will support auto-equilibration.
    """
    def __init__(self,ce,sc_mat,**kwargs):
        """
        Args:
            ce(ClusterExpansion):
                A cluster expansion object to solve on.
            sc_mat(3*3 ArrayLike):
                Supercell matrix to solve on.
        """
        super().__init__(ce,sc_mat,**kwargs)
 
    @abstractmethod
    def get_unfreeze_sample(self,unfreeze_series=[500,1500,5000]):
        """
        Starting from the ground state, get samples under a series 
        if increasing temperatures.
        Will take 100 entree under each temperature.
       
        Args:
            unfreeze_series(List[float]):
                A series of increasing temperatures to sample on.
                By default, will sample under 500, 1500 and 5000 K.

        Return:
            sample_occus(List[List[int]]):
                A list of sampled encoded occupation arrays. The first
                one will always be the one with lowest energy!
        """
        return
