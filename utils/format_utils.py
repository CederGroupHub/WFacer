"""
Formatting utilities to convert between formats of data.
"""

__author__ = 'Fengyu Xie'

from monty.json import MontyDecoder
import json

from pymatgen import Composition,Specie,DummySpecie

from smol.cofe.configspace.domain import Vacancy
from smol.moca import CEProcessor

# Monty decoding any dict
def decode_from_dict(d):
    return MontyDecoder().decode(json.dumps(d))

# flatten and de-flatten a 2d array
def flatten_2d(2d_lst,remove_nan=True):
    """
    Sequentially flatten any 2D list into 1D, and gives a deflatten rule.
    Inputs:
        2d_lst(List of list):
            A 2D list to flatten
        remove_nan(Boolean):
            If true, will disregard all None values in 2d_lst when compressing.
            Default is True
    Outputs:
        flat(List):
            Flatten list
        deflat_rule(List of int):
            Deflatten rule. Each integer speciefies the length of each sub-list
            in the original 2D list. When deflattening, will serially read and
            unpack to each sublist based on these numbers.
    """
    deflat_rule = []
    flat = []
    for sl in 2d_lst:
        if remove_nan:
            sl_return = [item for item in sl if item is not None]
        else:
            sl_return = sl
        flat.extend(sl_return)
        deflat_rule.append(len(sl_return))

    return flat,deflat_rule

def deflat_2d(flat_lst,deflat_rule,remove_nan=True):
    """
    Sequentially decompress a 1D list into a 2D list based on rule.
    Inputs:
        flat(List):
            Flatten list
        deflat_rule(List of int):
            Deflatten rule. Each integer speciefies the length of each sub-list
            in the original 2D list. When deflattening, will serially read and
            unpack to each sublist based on these numbers.
        remove_nan(Boolean):
            If true, will first deflat on deflat rule, then remove all 
            None values in resulting 2d_lst.
            Default is True.

    Outputs:
        2d_lst(List of list):
            Deflatten 2D list
    """
    if sum(deflat_rule)!= len(flat_lst):
        raise ValueError("Deflatten length does not match original length!")

    2d_lst = []
    it = 0
    for sl_len in deflat_rule:
        sl = []
        for i in range(it,it+sl_len):
            if flat_lst[i] is None and remove_nan:
                continue
            sl.append(flat_lst[i])
        2d_lst.append(sl)
        it = it + sl_len
    return 2d_lst

#serialize and de-serialize compositions
def serialize_comp(comp):
    """
    Serialize a pymatgen.Composition or a list of pymatgen.Composition.
    Composition.as_dict only keeps species string. This will keep all
    property informations of a specie.
    Inputs:
        comp(Composition or List of Composition):
            the composition to serialize
    Outputs:
        A serialzed composition, in form: [(specie_dict,num_specie)] or 
        a list of such.
    """
    if type(comp) == list:
        return [serialize_comp(sl_comp) for sl_comp in comp]

    return [(sp.as_dict(),n) for sp,n in comp.items()]

def deser_comp(comp_ser):
    """
    Deserialize to a pymatgen.Composition or a list of pymatgen.Composition.
    Composition.as_dict only keeps species string. This will keep all
    property informations of a specie.
    Inputs:
        comp_ser(List of tuples or List of List of tuples):
        A serialzed composition, in form: [(specie_dict,num_specie)] or 
        a list of such.
        
    Outputs:
        comp(Composition or List of Composition):
            the composition to serialize
    """
    if type(comp_ser) == list and type(comp_ser[0]) == list:
        return [deser_comp(sl_comp_ser) for sl_comp_ser in comp_ser]

    return {decode_from_dict(sp_d):n for sp_d,n in comp_ser}


# Utilities for parsing occupation, used in charge-neutral semigrand flip table
def occu_to_species_stat(sublattices,occupancy,normalize=False):
    """
    Get a statistics table of each specie on sublattices from an encoded 
    occupancy array.
    Inputs:
        sublattices(A list of smol.moca.Sublattice):
            Sublattice objects of the current system, storing attibutes of
            site indices and site spaces of each sublattice.
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        normalize(Boolean):
            Whether or not to normalize species_stat into fractional 
            compositions. By default, we will not normalize.
    Returns:
        species_stat(2D list of ints/floats)
            Is a statistics of number of species on each sublattice.
            1st dimension: sublattices
            2nd dimension: number of each specie on that specific sublattice.
            Dimensions same as moca.sampler.mcushers.CorrelatedUsher.bits          
    """
    bits = [sl.species for sl in sublattices]
    occu = np.array(occupancy)
    species_stat = [[0 for i in range(len(sl_bits))] for sl_bits in bits]
    for s_id,sp_code in enumerate(occu):
        sl_id = None
        for i,sl in enumerate(sublattices):
            if s_id in sl.sites:
                sl_id = i
                break
        if sl_id is None:
            raise ValueError("Occupancy site {} can not be matched to a sublattice!".format(s_id))   
        species_stat[sl_id][sp_code]+=1
     
    if normalize:
        species_stat_norm = \
            [[float(species_stat[sl_id][sp_id])/sum(species_stat[sl_id])
              for sp_id in range(len(bits[sl_id]))]
              for sl_id in range(len(bits))]
        species_stat = species_stat_norm

    return species_stat

def occu_to_species_list(sublattices,occupancy):
    """
    Get table of the indices of sites that are occupied by each specie on sublattices,
    from an encoded occupancy array.
    Inputs:
        sublattices(A list of smol.moca.Sublattice):
            Sublattice objects of the current system, storing attibutes of
            site indices and site spaces of each sublattice.
        occupancy(np.ndarray like):
            An array representing encoded occupancy
    Returns:
        species_list(3d list of ints):
            Is a statistics of indices of sites occupied by each specie.
            1st dimension: sublattices
            2nd dimension: species on a sublattice
            3rd dimension: site ids occupied by that specie
    """
    bits = [sl.species for sl in sublattices]
    occu = np.array(occupancy)
    species_list = [[[] for i in range(len(sl_bits))] for sl_bits in bits]

    for site_id,sp_id in enumerate(occu):
        sl_id = None
        for i,sl in enumerate(sublattices):
            if s_id in sl.sites:
                sl_id = i
                break
        if sl_id is None:
            raise ValueError("Occupancy site {} can not be matched to a sublattice!".format(s_id))   

        species_list[sl_id][sp_id].append(site_id)

    return species_list

#Wrap up structure_from_occu method from processor module
def structure_from_occu(ce,sc_matrix,occu):
    decode_proc = CEProcessor(ce.cluster_subspace,sc_matrix,ce.coefs)
    return decode_proc.structure_from_occupancy(occu)


