from coords_util import *

from monty.json import MSONable
from pymatgen.core.periodic_table import Specie
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule

import numpy as np

def get_oxi(ion):
    """
    This tool function helps to read the charge from a given specie(in string format).
    Inputs:
        ion: a string specifying a specie.
    """
    #print(ion)
    if ion[-1]=='+':
        return int(ion[-2]) if ion[-2].isdigit() else 1
    elif ion[-1]=='-':
        #print(ion[-2])
        return int(-1)*int(ion[-2]) if ion[-2].isdigit() else -1
    else:
        return 0

def element_to_ion(element_str,oxi=0):
    """
    Given oxidation state, this tool converts an element string to its specie string.
    """
    if oxi==0:
        return element_str
    elif oxi<0:
        if oxi==-1:
            return element_str+'-'
        else:
            return elelemt_str+str(int(abs(oxi)))+'-'
    else:
        if oxi==1:
            return element_str+'+'
        else:
            return element_str+str(int(oxi))+'+'

class CESpecie(MSONable):
    supported_properties = ['spin']

    def __init__(self,atom_symbols:list,atom_coords=np.array([0.0,0.0,0.0]),\
                 #ref_atoms=[0,0], \
                 heading=np.array([0.0,0.0,0.0]),\
                 oxidation_state:int=0,\
                 other_properties:dict=None):
        """
        This class defines a 'specie' in CE, which can be a group of closely
        connected atoms, such as PO4 3-, O2 -, etc, or just a single ion.

        Inputs:
            atom_coords: 
                CARTESIAN COORDINATES of atoms in this cluster;(Use cartesian
                only!)
                In the initialization, all the atom_coords will be standardized.
                See coords_utils.py for the rule of standardization.
            atom_symbols: 
                Element symbol of the atoms, written in a list of strings;
            heading:
                Array like. Indicating the Euler angles of the atomic cluster,
                in (alpha,beta,gamma)/pi.
                For example, heading = (0.5,0.25,0.20) means that the cluster
                is heading towards Euler angles alpha=pi/2, beta=pi/0.25, 
                gamma=pi/5. When adding it into a lattice, we shall first add
                a 'standard' direction cluster to the lattice, put its center
                on the lattice point, then apply a euler rotation given by
                'heading' around the lattice to get the final result.
                point.
            oxidation_state: 
                Oxidation_state of this cluster
            other_properties: 
                Other properties of this cluster, such as spin, etc.
                For 'spin', you may denote it with 1 or -1, for a 
                spin polarized CE. 
                (We may support vectorized spin in the future, but currently
                 we don't,so the value of 'spin' can only be +1 or -1; 
                 Also, currently we don't support properties other than 'spin')

            Note that only two CESpecies object with the same atom_symbols,
            equivalent atom_coords , the same oxidation_state, symmetrically equivalent 
            headings, and equivalent other_properties dictionary are considered equivalent.
            Otherwise they will be expanded as two different species.
        """
        if len(atom_symbols)!= len(atom_coords):
            raise ValueError("The length of atomic symbols are not equvalent to that of 
                              atomic coordinates!")

        self.symbols = atom_symbols
        self.oxidation_state = oxidation_state
        self.coords = Standardize_Coords(np.array(atom_coords))

        self.heading = np.array(heading)

        if len(atom_symbols)==1:
            self.heading = np.array([0.0,0.0,0.0])
            #An atom has no heading.
            if atom_symbols[0]!='Vac':
                try:
                    self.pmg_specie = Specie(symbol,float(oxidation_state))
                except:
                    raise ValueError("The input symbol is not a valid element!")  
            elif oxidation_state==0:
                    self.pmg_specie = None
            else:
                raise ValueError("Vacancy specie can not have non-zero charge!")           

        else:
            if not Is_Nonlinear(self.coords):
                self.heading[2]=0.0
            #Linear atomic clusters have no rolls.
            self.pmg_specie = None

        for k in other_properties.keys():
            if k not in supported_properties:
                raise ValueError('Property {} not yet supported!'.format(k))
            if k == 'spin' and (other_propertie[k] not in [-1,1]):
                raise ValueError('Magnetization type not supported!')

        self.other_properties = other_properties
        self.molecule = Molecule(self.symbols,self.coords,charge=self.oxidation_state)

    def coords_in_supercell(self,lattice_point,sc_lat_matrix):
        """
        This will give the fractional coordnates of the atoms
        on a specified lattice point.
        Inputs:
            lattice point: array_like, in fractional coordinates
            sc_lat_matrix: lattice matrix of the SUPERCELL
        Outputs:
            a set of fractional coordinates, representing atom loacations in the
            supercell
        """
        alpha,beta,gamma = self.heading*np.pi 
        
        coords_cart = self.coords@Rot_Matrix(alpha,beta,gamma)
        coords_frac = coords_cart@np.linalg.inv(sc_lat_matrix)+lattice_point

        return coords_frac

    def __eq__(self,other):
        symbols_eq = (self.symbols == other.symbols)
        ox_eq = (self.oxidation_state == other.oxidation_state) 
        geo_eq = np.allclose(self.coords,other.coords)
        props_eq = (self.other_properties==other.other_properties)
        
        if len(self.coords)<2:
            heading_eq = True
        else:
            point_group = PointGroupAnalyzer(self.molecule)
            symops = point_group.get_symmetry_operations()

            heading_eq = False
            for symop in symops:
                if np.allclose(symop.operate(other.heading),self.heading):
                    heading_eq = True
                    break
        
        return symbols_eq and ox_eq and geo_eq and props_eq and heading_eq

    def __str__(self):
        return self.molecule.__str__()+'\nHeading direction:{}'.format(self.heading)+\
               '\nOther properties:\n {}'.format(self.other_properties)

    def __repr__(self):
        return self.__str__()
