#!/usr/bin/env python
"""
Warning:
Current charge assignement tool is based on total magnetic momentum of an
atom, so it is only applicable for the fourth row transition metal ions
under high spin configurations, because only then the ions charges are
monotonious with their spin states.

(Ti(II-IV,decrease), V(II-V,decrease)，Cr(II-VI,decrease), Mn(II-IV,decrease),
Fe(II-III,increase), Co(II-IV,increase), Ni(II-IV,increase))

For non-transtion metal elements, we assume that their charges are fixed.
Thought O2- oxidation is important, we currently can not consider O-/O2-
because their magnetic momentums are similar. We force them all to be 2-
here, will come back later.

We don't recommend using charged cluster expansion in other systems till
we've found a better way to assign oxidation states.

(Notes by FYX: Maybe it would be bader charges? I will test later)
"""

from __future__ import division
__author__ = 'Julia Yang & Fengyu  Xie'

from charge_data import Domains_dict, Fix_Charges_dict

from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
#from pymatgen.core.periodic_table import Element

from skopt import gp_minimize
import re

"""
    Format for a Domains_dict:
    {
    'mag':
      {
        'Mn':
           {
              4:(l1-u1),
              3:(l2-u2),
              2:(l3-u3) (a dict, l2>=u1, l3>=u2 must be satisfied to get a correct ordering)
           },
        ...
      },
    'chg':
    ...
    }

    A Fix_Charges_dict:
    {
      'F':-1,
      'S':-2,
      ...
    }
"""

def make_specie_string(element,oxi):
    if oxi>0:  return element+str(oxi)+'+'
    if oxi==0: return element
    if oxi<0:  return element+str(abs(oxi))+'-'

def make_element_string(specie):
    return re.sub(r'\d*(\+|\-)$','',specie)

def assign_single(s_ori,prop,cutoffs,v_species):
    remade_sites = []
    for site, site_prop in zip(s_ori,prop):
        site_element = make_element_string(str(site.specie)) 
        if site_element in Fix_Charges_dict:
            # No bayers process required.
            oxi = Fix_Charges_dict[site_element]
            site_sp = make_specie_string(site_element,oxi)
            #print('site_sp:',site_sp)  
        else:
            site_v_sps = []
            site_cutoffs = []
            for cutoff,v_sp in zip(cutoffs,v_species):
                if make_element_string(v_sp)== site_element:
                    site_cutoffs.append(cutoff)
                    site_v_sps.append(v_sp)

            if site_prop > cutoffs[-1]:
                site_sp = None #site_property overflowing! We assume this is not very frequent.
                return None
            elif site_prop < cutoffs[0]:
                site_sp = v_species[0]
            else:
                for idx in range(1,len(cutoffs)):
                    if site_prop>cutoffs[idx-1] and site_prop<cutoffs[idx]:
                        site_sp = v_species[idx]
                        break

        remade_site = PeriodicSite(site_sp,site.frac_coords,site.lattice)
        #print(remade_site)
        remade_sites.append(remade_site)

    return Structure.from_sites(remade_sites)

class ChargeAssign(object):
    """
    Here we assign charges to a pool of structures at the same time.
    We will optimize cutoff values on domains using scikit.optimize.
    gp_minimize to choose optimum cutoff values that minimize the
    number of charge unbalanced structures.
    When algo=magmom, we read the domain values from Domains_dict['mag'],
    and assign charges by magnetic moments. (default.)
    When alago=chg, we read the domain values from Domains_dict['chg'],
    and assign charges by bader charges.
    The species in Fix_Charges_dict will always be assigned to the charges
    in the dict, while the species in Domains_dict will be given a charge
    based on the optimized cutoff value. After assignment, charge unbalanced
    structures would be thrown away.
    """
    def __init__(self,structure_pool, site_properties, algo='mag'):
        """
        Structure_pool: a pool of structures to assign charges to.
        site_properties: structure properties by site. Can be magmoms, or
        bader charges of site.
        """
        print('#### Assignment call ####')
        self._pool = structure_pool
        self.site_properties = site_properties
        self.algo = algo
        self._cutoffs = None

        self.elements = []
        for struct in structure_pool:
            for specie in struct.composition.keys():
                element = make_element_string(str(specie))
                if element not in self.elements:
                    self.elements.append(element)
        self.elements = sorted(self.elements)
        #Sort this to get a one-to-one mapping from elements to their domains.
        
        domain_data = Domains_dict[self.algo]

        self.domains = []
        #Domains to find a cutoff values within.
        self.v_species = []
        #species with 'V'ariable charges, each corresponds to an element in self.domain.
        # eg. self.domain = [(0.2,3.5),(3.6,5.0)] self.v_species = ['Fe2+','Fe3+'], etc.
        for element in self.elements:
            if element in domain_data:
                domains_for_element = sorted(domain_data[element].items(),\
                                             key=lambda i:i[1])
                for oxi,domain in domains_for_element:
                    self.domains.append(domain)
                    self.v_species.append(make_specie_string(element,oxi))
            elif element not in Fix_Charges_dict:
                raise ValueError('Warning: Element {} is not a fixed charge element, but its \
                                  optimization domains has not been provided for algorithm {}!\
                                  Structures can not be assigned.'\
                                  .format(element,self.algo))

        self._assigned_structures = None
        print('Initialized assignment with elements: {}\nDomains: {}\nMulti-valance species: {}'.format(self.elements,self.domains,self.v_species))
  
    @property
    def cutoffs(self):
        if self._cutoffs is None:
            if len(self.domains)==0:
                print('Assigning from fixed charges. Domains not required!')
                self._cutoffs = []
            else:
                self._cutoffs = gp_minimize(self.evaluate,self.domains)
        return self._cutoffs

    def evaluate(self,cutoffs):
        #returns number of non zero charges
        n_nonbal = 0
        for s_ori,prop in zip(self._pool,self.site_properties):
            try:
                s_assigned = assign_single(s_ori,prop,cutoffs,self.v_species)
            except:
                s_assigned = None 
            if s_assigned is None or s_assigned.charge != 0:
                n_nonbal+=1
        return n_nonbal

    @property
    def assigned_structures(self):
        if self._assigned_structures is None:
            self._assigned_structures = []
            for struct,prop in zip(self._pool,self.site_properties):
                try:
                    assigned_struct = assign_single(struct,prop,self.cutoffs,self.v_species)
                    if assigned_struct.charge !=0:
                        print('A structure not charge balanced. Dropping.')
                        assigned_struct = None
                except:
                    print('A structure failed assignment. Skipping.')
                    assigned_struct = None
                self._assigned_structures.append(assigned_struct)
        return self._assigned_structures

    def extend_assignments(self,another_pool,another_props):
        #Must be used only for two pools generated in a same system.
        assigned_pool = [] 
        for struct,prop in zip(another_pool,another_props):
            try:
                assigned = assign_single(struct,prop,self.cutoffs,self.v_species) 
                if assigned.charge!=0:
                    print('A structure extension failed charge balance.')
                    assigned = None      
            except:
                print('A structure extension failed assignment.')
                assigned = None
            assigned_pool.append(assigned)
        return assigned_pool
