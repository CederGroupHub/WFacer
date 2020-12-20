__author__='Fengyu Xie'

"""
This file defines a generic propertie assigner class to assign properties 
to undecorated species, and returns their decorated forms.

Possible decorations includes charge (most commonly used), spin polarization.
If the user wishes to define other properties assignment methods, just derive
a new class Assignment class, and write assignment methods accordingly.
"""
from abc import ABC, abstractmethod

class Assignment(ABC):
    """
    Abstract assignment class.
    Attributes:
        labels_av(Dict{Element: List[int|float]...}):
            A dictionary, specifying the elements, and the labels
            that we should assign to this specific element.
            By default, Vacancy will not appear in structure pool when
            directly read from vasp output, so no need to consider.
            For example:
            OrderedDict({Element.from_string('Li'):[1]})
            when assigning charge +1 to Li in a structure.
            When there are multiple assigned property values possible, 
            the values in the list[int|float] should be sorted by the
            order of their cluster centers in the properties axis.
            For example, If I have Mn2+, Mn3+ and Mn4+ (all on high spin), 
            and we assign charges to Mn atoms by magnetization, then we expect 
            a order of [4,3,2], because this is the order of the magnetic moment
            in these three types of oxidation states.
    """
    def __init__(self):
        

    @abstractmethod
    def train(self,str_pool,properties):
        """
        Train a properties assignment model. Model or model parameters
        should be stored in a property of the object.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            properties(2D ArrayLike):
                Numerical properties used to classify sites.
                Shape should be N_strs*N_sites
        """
        return

    @abstractmethod
    def assign(self,str_pool,properties):
        """
        Give assignment to structures. If an assigned structure is not valid,
        for example, in charge assignment, if an assigned structure is not
        charge neutral, then this structure will be returned as None.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            properties(2D ArrayLike):
                Numerical properties used to classify sites.
                Shape should be N_strs*N_sites       
        Return:
            List[Structure|Nonetype], a list of assigned structures, consisting of
        Species|Element, or None. Vacancies will be handled by structure matcher
        in smol.ClusterSubspace, so there is no need to explicitly add them.
        """
        return

    @abstractmethod
    def as_dict(self):
        """
        Serialization method. Please save the trained property partition or clustering here.
        """
        return
