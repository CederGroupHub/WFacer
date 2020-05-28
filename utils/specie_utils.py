from coords_utils import *

from monty.json import MSONable
from pymatgen.core.periodic_table import Specie
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule

import numpy as np

def get_element_oxi(ion):
    """
    This tool function helps to read the charge from a given specie(in string format).
    Inputs:
        ion: a string specifying a specie.
    """
    #print(ion)
    if ion[-1]=='+':
        oxi =  int(ion[-2]) if ion[-2].isdigit() else 1
        element = ion[:-2] if ion[-2].isdigit() else ion[:-1]
    elif ion[-1]=='-':
        #print(ion[-2])
        oxi = int(-1)*int(ion[-2]) if ion[-2].isdigit() else -1
        element = ion[:-2] if ion[-2].isdigit() else ion[:-1]      
    else:
        oxi = 0
        element = ion
    return element,oxi

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
            return element_str+str(int(abs(oxi)))+'-'
    else:
        if oxi==1:
            return element_str+'+'
        else:
            return element_str+str(int(oxi))+'+'

class CESpecie(MSONable):
    supported_properties = ['spin']

    def __init__(self,atom_symbols:list,atom_coords=np.array([[0.0,0.0,0.0]]),\
                 z_ref=0, x_ref=1, \
                 heading=np.array([0.0,0.0,0.0]),\
                 oxidation_state:int=0,\
                 other_properties:dict={}):
        """
        This class defines a 'specie' in CE, which can be a group of closely
        connected atoms (molecular fragments), such as PO4 3-, O2 -, etc, 
        or just a single atom/ion.

        Inputs:
            atom_coords: 
                CARTESIAN COORDINATES of atoms in this cluster;(Use cartesian
                only!)
                In the initialization, all the atom_coords will be standardized.
                See coords_utils.py for the rule of standardization.
            z_ref:
                The atom index used to calibrate z-axis in standardization
            x_ref:
                The atom index used to calibrate x-axis.
                Both z-ref and x-ref should be carefully chosen.
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

            Note:
            1, only two CESpecies object with the same atom_symbols,
            equivalent atom_coords , the same oxidation_state, symmetrically equivalent 
            headings, and equivalent other_properties dictionary are considered equivalent.
            Otherwise they will be expanded as two different species.

            2, Since we did not implement auto-detection of atomic permutational invariance,
            when you define a molecular fragment, please choose the ordering
            of atoms carefully! If you want two fagments to be recognized as the same, you'd
            better use the same ordering of atoms when defining them!
        """
        if len(atom_symbols)!= len(atom_coords):
            raise ValueError("The length of atomic symbols are not equvalent to that of"+\
                              " atomic coordinates!")

        self.symbols = atom_symbols
        self.oxidation_state = oxidation_state
        self.z_ref = z_ref
        self.x_ref = x_ref

        self.coords = Standardize_Coords(np.array(atom_coords),z_ref,x_ref)

        self.heading = np.array(heading)

        if len(self.symbols)==1:
            self.heading = np.array([0.0,0.0,0.0])
            #An atom has no heading.
            if self.symbols[0]!='Vac':
                try:
                    self.pmg_specie = Specie(self.symbols[0],float(oxidation_state))
                except:
                    raise ValueError("The input symbol is not a valid element!")  
            elif oxidation_state==0:
                    self.pmg_specie = None
            else:
                raise ValueError("Vacancy specie can not have non-zero charge!")        

        else:
            is_nonlinear = Is_Nonlinear(self.coords)
            if not is_nonlinear:
                self.heading[2]=0.0
            #Linear atomic clusters have no rolls.
            self.pmg_specie = None

        for k in other_properties.keys():
            if k not in CESpecie.supported_properties:
                raise ValueError('Property {} not yet supported!'.format(k))
            #Currently only support integer spin, namely z-axis polarized 
            #calculations only.

        self.other_properties = other_properties
        if len(self.symbols)>1 or self.symbols[0]!='Vac':
            self.molecule = Molecule(self.symbols,self.coords,charge=self.oxidation_state)
        else:
            self.molecule = None

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
        """
        Warning:
            Since we did not implement smart detection of rotational and 
            translational invariance between fragments,
        """
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

    @property
    def specie_string(self):
        """
        Returns a string rep of the composition of this specie.
        """
        if len(self.symbols)==1:
            sp_string = element_to_ion(self.symbols[0],oxi=self.oxidation_state)
            oxi_string = ''

        else:
            element_cnt = {}
            for el in self.symbols:
                if el not in element_cnt:
                    element_cnt[el]=1
                else:
                    element_cnt[el]+=1
            sp_string = ''
            for k,v in element_cnt.items():
                sp_string+=k+str(v)

            if self.oxidation_state == 0:
                oxi_string = ''
            elif self.oxidation_state < 0:
                oxi_string = ' {}-'.format(int(abs(self.oxidation_state)))
            else:
                oxi_string = ' {}+'.format(int(abs(self.oxidation_state)))
        #This does not give support encoding of other properties.

        return sp_string + oxi_string

    def __str__(self):
        if self.molecule is not None:
            return self.molecule.__str__()+'\nHeading direction:{}'.format(self.heading)+\
               '\nOther properties:\n {}'.format(self.other_properties)
        else:
            return 'Vacancy'+'\nHeading direction:{}'.format(self.heading)+\
               '\nOther properties:\n {}'.format(self.other_properties)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_string(cls,sp_string,other_properties:dict={}):
        """
        Initialize a specie from its string representation.
        Molecular fragments are marked with a prefix f-.
        This
        """
        if len(sp_string)>=2 and sp_string[:2]=='f-':
            raise NotImplementedError
            #Will be implemented as a standard storage dictionary
        else:
            #Mono-atomic species
            symbol,oxi = get_element_oxi(sp_string) 
            return cls([symbol],oxidation_state=oxi,other_properties=other_properties)
