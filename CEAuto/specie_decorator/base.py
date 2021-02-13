__author__='Fengyu Xie'

"""
This file defines a generic propertie assigner class to assign properties 
to undecorated species, and returns their decorated forms.

Possible decorations includes charge (most commonly used), spin polarization.
If the user wishes to define other properties assignment methods, just derive
a new class Assignment class, and write assignment methods accordingly.
"""
from abc import ABC, abstractmethod
from monty.json import MSONable

class BaseDecorator(ABC,MSONable):
    """
    Abstract decorator class.
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
        pass

    @property
    @abstractmethod
    def trained(self):
        """
        Gives whether this decorator is trained or not. If trained, will not be trained
        again.
        """
        return

    @abstractmethod
    def train(self,str_pool,properties,reset=False):
        """
        Train a properties assignment model. Model or model parameters
        should be stored in a property of the object.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            properties(3D ArrayLike):
                Numerical properties used to classify sites.
                Shape should be N_different_proerties*N_strs*N_sites
            reset(Boolean):
                If you want to re-train the decorator model, set this value
                to true. Otherwise we will skip training if self.trained is 
                true.
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
        Returns:
            A dictionary, specifying name of assigned properties and their
            values by structure and by site. If assignment failed for a
            structure, will give None for it.
            For example: 
            {'charge':[[1,4,2,...],None,[...],...]}
            Currently, in pymatgen.Specie's
            other_properties, only 'spin' is allowed. If you want to add more, do
            your own study!
            The de-serialization of property names is given in CEAuto.featurizer.
            featurize.
        """
        return

    @abstractmethod
    def as_dict(self):
        """
        Serialization method. Please save the trained property partition or clustering here.
        """
        return

    @classmethod
    @abstractmethod
    def from_dict(cls,d):
        return
