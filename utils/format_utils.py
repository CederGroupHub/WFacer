"""
Formatting utilities to convert between formats of data.
"""

__author__ = 'Fengyu Xie'

from monty.json import MontyDecoder
import json

from pymatgen import Composition,Specie,DummySpecie

from smol.cofe import ClusterSubspace
from smol.cofe.space.domain import Vacancy
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

# Utilities to get supercell sublattice sites from primitive sublattice sites.
def get_sc_sllist_from_prim(sl_list_prim,sc_size=1):
    """
    Get supercell sublattice sites list from primitive cell sublattice sites list.
    Args:
        sl_list_prim(List[List[int]]):
             sublattice sites indices in primitive cell
        sc_size(int):
             supercell size. Default to 1.
    Returns:
        List[List[int]]:
             sublattice sites indices in supercell.
    """
    sl_list_sc = []
    for sl_prim in sl_list_prim:
        sl_sc = []
        for sid_prim in sl_prim:
            all_sid_sc = list(range(sid_prim*sc_size,(sid_orim+1)*sc_size))
            sl_sc.extend(all_sid_sc)
        sl_list_sc.append(sl_sc)

    return sl_list_sc

# Utilities for parsing occupation, used in charge-neutral semigrand flip table
def occu_to_species_stat(bits,sublat_list,occupancy,normalize=False):
    """
    Get a statistics table of each specie on sublattices from an encoded 
    occupancy array.
    Args:
        bits(list of lists of Specie):
            A list of species on each sublattice, must correspond to occupancy
            encoding table
        sublat_list(List of lists of ints):
            A list storing sublattice sites in a PRIM cell
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
    occu = np.array(occupancy)
    if len(occu)%sum([len(sl) for sl in sublat_list])!=0:
        raise ValueError("Occupancy size not multiple of primitive cell size.")
    sc_size = len(occu)//sum([len(sl) for sl in sublat_list])

    species_stat = [[0 for i in range(len(sl_bits))] for sl_bits in bits]
    sublat_list = get_sc_sllist_from_prim(sublat_list,sc_size=sc_size)

    for s_id,sp_code in enumerate(occu):
        sl_id = None
        for i,sl in enumerate(sublat_list):
            if s_id in sl:
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

def occu_to_species_list(bits,sublat_list,occupancy,sc_size=1):
    """
    Get table of the indices of sites that are occupied by each specie on sublattices,
    from an encoded occupancy array.
    Inputs:
        bits(list of lists of Specie):
            A list of species on each sublattice, must correspond to occupancy
            encoding table
        sublat_list(List of lists of ints):
            A list storing sublattice sites in a PRIM cell
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        sc_size(int):
            Size of supercell. Default to 1.
    Returns:
        species_list(3d list of ints):
            Is a statistics of indices of sites occupied by each specie.
            1st dimension: sublattices
            2nd dimension: species on a sublattice
            3rd dimension: site ids occupied by that specie
    """
    occu = np.array(occupancy)
    species_list = [[[] for i in range(len(sl_bits))] for sl_bits in bits]
    sublat_list = get_sc_sllist_from_prim(sublat_list,sc_size=sc_size)

    for site_id,sp_id in enumerate(occu):
        sl_id = None
        for i,sl in enumerate(sublat_list):
            if s_id in sl:
                sl_id = i
                break
        if sl_id is None:
            raise ValueError("Occupancy site {} can not be matched to a sublattice!".format(s_id))   

        species_list[sl_id][sp_id].append(site_id)

    return species_list

#Wrap up structure_from_occu method from processor module
def structure_from_occu(occu,prim,sc_matrix):
    """
    Decodes structure from encoded occupation array.
    Args:
        occu(1D arraylike):
            encoded occupation string
        prim(pymatgen.Structure):
            primitive cell containing all occupying species information.
            It is your responisibility to ensure that it is exactly the
            one you used to initialize cluster expansion.
        sc_matrix(3*3 arraylike):
            Supercell matrix. It is your responsibility to check size
            matches the length of occu
    Returns:
        Decoded pymatgen.Structure object.
    """
    dummy_cspace = ClusterSubspace.from_cutoffs(prim, cutoffs={2:0.01})
    dummy_coefs = np.zeros(dummy_cspace.num_corr_functions)
    dummy_proc = CEProcessor(dummy_cspace,sc_matrix,dummy_coefs)
    return dummy_proc.structure_from_occupancy(occu)
