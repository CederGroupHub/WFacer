"""For charge assignment.

Charges will be assigned by magnitudes of magnetization
vectors.
"""

__author__='Julia Yang, Fengyu Xie'

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element, Specie, DummySpecies

from smol.cofe.space.domain import get_species

from .base import BaseDecorator

import numpy as np
from sklearn.mixture import GaussianMixture
from skopt import gp_minimize
from copy import deepcopy
from functools import partial

import warnings
import logging

def get_section_id(x, cuts):
    """Get index of a section from section cut values.

    Args:
        x(float):
            A value on axis.
        cuts(List[float]):
            Points on axis that separates sections.
            Must be pre-sorted.
            Sections are: -inf~cuts[0], cuts[0]~cuts[1],...,
            cuts[-1]~+inf.
    Returns:
        index of section where x is in.
    """
    return sorted(cuts.copy() + [x]).index(x)


class MagchargeDecorator(BaseDecorator):
    """
    Assign charges from magnitudes of magentic moments. Partition dividers
    will be initialized by a mixture of gaussians model, then optimized with
    maximum neutral structures number by gp_maximize.
    Takes in a pool of structures, gives assigned strutures.

    Args:
        labels_table(Dict{STRING of element: List[int|float]...}):
            A dictionary, specifying the elements, and the labels
            that we should assign to this specific element.
            By default, Vacancy will not appear in structure pool when
            directly read from vasp output, so no need to consider.
            For example:
            {'Li':[1]}
            when assigning charge +1 to Li in a structure.
            When there are multiple assigned property values possible,
            the values in the list[int|float] should be sorted by the
            order of their cluster centers in the properties axis.
            For example, If I have Mn2+, Mn3+ and Mn4+ (all on high spin),
            and we assign charges to Mn atoms by magnetization, then we
            expect a order of [4,3,2], because this is the order of the
            magnetic moment in these three types of oxidation states.
         Note:
            1, All elements in structure pool must be present in this table!
            2, If oxidation state is 0, will assign to Element, not an ox=0
               specie!

         maximize_balance:
            Whether to optimiz cut values with gp_maximize(n_balanced_strs).
            Default to True.
    """
    required_props = ['magnetization']

    def __init__(self, labels_table, maximize_balance=True):
        self.labels_table = labels_table
        self.maximize_balance = maximize_balance
        self._cuts_by_elements = None

    @property
    def trained(self):
        """Indicates trained or not."""
        return (self._cuts_by_elements is not None)

    @property
    def n_params(self):
        return sum(len(labels)-1 for e, labels in
                   self.labels_table.items())

    def _evaluate_cut_params(self, str_pool, properties, cut_params):
        """Compute number of unbalanced structures."""
        if len(cut_params) != self.n_params:
            raise ValueError("Number of cut parameters does not match" +
                             " requirement!")
        cuts_by_elements = {}
        p_id = 0
        for e, labels in sorted(self.labels_table.items()):
            cuts_by_elements[e] = cut_params[p_id : p_id + len(labels) - 1]
            p_id += len(labels) - 1

        oxi_assign = self._assign(str_pool, properties, cuts_by_elements)
        n_fails = 0
        for assign in oxi_assign:
            if assign is None:
                n_fails += 1

        return n_fails  # To minimize.

    def _optimize_cuts(self, cuts_by_elements_init,
                       str_pool, properties, search_range=0.1):
        """Optimize cuts with gaussian optimization."""
        cut_params_init = []
        for e, labels in sorted(self.labels_table.items()):
            cut_params_init.extend(cuts_by_elements_init[e])

        domains = [(c - search_range, c + search_range)
                   for c in cut_params_init]
        objective = partial(self._evaluate_cut_params, str_pool, properties)
        res = gp_minimize(objective, domains, n_calls=50,
                          acq_optimizer='sampling',
                          noise=0.001)
        cut_params_opt = res.x

        cuts_by_elements = {}
        p_id = 0
        for e, labels in sorted(self.labels_table.items()):
            cuts_by_elements[e] = cut_params_opt[p_id :
                                                 p_id + len(labels) - 1]
            p_id += len(labels) - 1

        return cuts_by_elements

    def train(self, str_pool, properties, search_range=0.1, reset=False):
        """
        Train a properties assignment model. Model or model parameters
        should be stored in a property of the object.
        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            properties(Dict{String: 2D ArrayLike}):
                Numerical properties used to classify sites, and property
                names.
                Each property array has shape N_strs*N_sites.
                In this case, only use magnetization.
            search_range(float):
                Optimizing range for cutting bounds. Default to 0.1
            reset(Boolean):
                If you want to re-train the decorator model, set this value
                to true. Otherwise we will skip training if self.trained is 
                true.
        No return value.
        """
        if self.trained and (not reset):
            print("Decorator model trained! Skip training.")
            return

        sites_by_elements = self._get_sites_info_by_element(str_pool, properties)

        _cuts_by_elements_gm = {}
        for e in sites_by_elements:
            # Only use magmoms.
            e_mags = np.array(sites_by_elements[e])[:, 0].reshape((-1, 1))
            gm = GaussianMixture(n_components=
                                 len(self.labels_table[e])).fit(e_mags)

            test_e_mags = np.linspace(min(e_mags) - 0.2, max(e_mags) + 0.2,
                                       1000).reshapce((-1, 1))
            test_labels = gm.predict(test_e_mags)

            cut_ids = []
            cut_id = 0
            labels_appeared = []
            for i in range(len(test_labels)):
                if (test_labels[i] != test_labels[sep_id] and not
                    (test_labels[i] in labels_appeared)):
                    cut_ids.append(i)
                    cut_id = i
                    labels_appeared.append(test_labels[i])

            _cuts_by_elements_gm[e] = [(test_e_mags[cut_id - 1] +
                                         test_e_mags[cut_id]) / 2
                                         for cut_id in cut_ids]

            if len(cut_ids) < len(self.labels_table[e]) - 1:
                warnings.warn("Element {} need ".format(e) +
                              "{} species instances, "
                              .format(len(labels_table)) +
                              "but only {} detected in training structures.\n"
                              .format(len(cut_ids) + 1) +
                              "Training magnetizations: {}"
                              .format(sorted(e_mags)))
                n_deficit = len(self.labels_table[e]) - 1 - len(cut_ids)
                _cuts_by_elements_gm[e] += [max(e_mags) + i * 0.2
                                            for i in range(n_deficit)]

        if not self.maximize_balance:
            self._cuts_by_elements = _cuts_by_elements_gm
        else:
            self._cuts_by_elements = self._optimize_cuts(_cuts_by_elements_gm,
                                                         search_range=
                                                         search_range)

        logging.log("Trained magnetization separators:\n {}."
                    .format(self._cuts_by_elements))

    @staticmethod
    def _assign(str_pool, properties, cuts_by_elements):
        """Assign charges to all sites in a structure pool.

        Will check charge neutrality. If not, the specific structure will
        be marked with None.
        """
        sites_by_elements = self._get_sites_info_by_element(str_pool,
                                                            properties)

        #Assign for each site
        sites_by_elements_assigned = {e:[] for e in self.labels_table.keys()}
        assignments = [[None for st in s] for s in str_pool]

        for e in sites_by_elements:
            mags_e = np.array(sites_by_elements[e])[:, 0]
            for m_id,m in enumerate(mags_e):
                a = self.labels_table[e][get_section_id(m,
                                         cuts_by_elements[e])]
                s_id, st_id = tuple(sites_by_elements[e][m_id][-2: ])
                assignments[s_id][st_id] = a

        oxi_assigned = []
        for s_id,s in enumerate(str_pool):
            if np.sum(assignments[s_id])==0:
                oxi_assigned.append(assignments[s_id])
            else:
                oxi_assigned.append(None)

        return oxi_assigned

    def assign(self, str_pool, properties):
        """Assign charges to all sites in a structure pool.

        Args:
            str_pool(List[Structure]):
                Unassigned structures, must contain only pymatgen.Element
            properties(Dict{String: 2D ArrayLike}):
                Numerical properties used to classify sites, and property
                names.
                Each property array has shape N_strs*N_sites.
                In this classifier, only uses magnetization.
            check_neutral(Bool):
                Check whether the assigned structures are charge neutral.
                Non-neutral structures will be returned as None.
        Returns:
            A dictionary, specifying name of assigned properties and their
            values by structure and by site. If assignment failed for a
            structure, will give None for it.
            For example: 
            {'charge':[[1,4,2,...],None,[...],...]}
            Currently, in pymatgen.Specie's
            other_properties, only 'spin' is allowed. If you want to add more, do
            your own study!
        """
        if self._cuts_by_elements is None:
            raise ValueError("Model not trained, can not assign!")

        oxi_assigned = self._assign(str_pool, properties,
                                    self._cuts_by_elements)

        n_all = len(oxi_assigned)
        n_success = n_all-n_fails

        logging.log("****{}/{} Structures Assigned. ".format(n_success,n_all))
                
        return {'charge':oxi_assigned}    

    def as_dict(self):
        """
        Serialize into dictionary.
        """
        return {'labels_table': self.labels_table,
                'cuts_by_elements': self.cuts_by_elements,
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__
               }
    
    @classmethod
    def from_dict(cls,d):
        """
        Recover from dict.
        """
        socket = cls(d['labels_table'])
        socket._cuts_by_elements = d.get('cuts_by_elements')
        return socket
