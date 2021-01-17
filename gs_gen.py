"""
Ground state generator class. THIS FILE CHANGES FACT, and MAY CHANGE
COMP TABLE. All generated structures will be considered into the 
NEXT iteration, NOT the CURRENT one.
"""
from monty.json import MSONable
import multiprocessing as mp
import numpy as np
import pandas as pd

from smol.cofe.space.domain import get_allowed_species

from .comp_space import CompSpace
from .gs_solver import *
from .utils.format_utils import deser_comp,structure_from_occu
from .utils.hull_utils import estimate_mu_from_hull
from .utils.math_utils import get_center_grid

NODATAERROR = RuntimeError("No dataframes. You may call data loading methods to load the calculation data first.")

class GSGenerator(MSONable):
    """
    Ground state generator class. Generates grounds states from 
    enumerate supercell matrices, compositions (or guessed chemical 
    potentials), and writes them to the fact and composition table.

    You may not want to call __init_ directly.

    Also, do not call this twice in an iteration.
    Args:
        ce(smol.ClusterExpansion):
            Latest cluster expansion object to solve GS from.
        sc_table(pd.DataFrame):
            supercell matrices dataframe.
            Will not be modified in this module.
        comp_table(pd.DataFrame):
            compositions dataframe.
            Might be modified if using with 'Grand' canonical solver.
        fact_table(pd.DataFrame):
            fact table dataframe, including all previous enumerated
            structures.
            Might be modified in this class, if not using 'MCCanonical'
            solver.
        solver_flavor(str):
            Specifies class name of the solver to be used.
            Check available solvers in CEAuto.gs_solver module.
            Will use 'Canonical' as default.
            When selecting Canoncial as solver, will not add anything,
            because in structure eneumerator, we are already keeping
            canoncial ground states.
        **grand_solver_args takes in semi-grand solver parameters, such
        as:
            mu_grid_step(float, or list[float]):
                grid density on each dimesion of mu in the constrained 
                compositional coordinate.(see comp_space.py)
                Default is 0.4.
            mu_grid_num(int, or List[int]):
                Number of grid points to scan on each dimension of mu.
                Default is 5.
                We recommend to set a lower value, when you are dealing
                with extremely high-dimensional systems, or give a list
                to emphasize some dimensions.
            n_proc(int):
                Number of mu points to compute simultaneously. Default
                is 4.
 
        Central mu will be estimated from previous CE hull.
    """
    #Modify this when you implement new solvers
    supported_solvers = ('Canonical','MCGrand','PBGrand')

    def __init__(self,ce,\
                 sc_table=None,comp_table=None,fact_table=None,\
                 solver_flavor='Canonical',**grand_solver_args):

        self.ce = ce
        self.prim = ce.cluster_subspace.structure

        bits = get_allowed_species(self.prim)
        self.sublat_list = []
        self.bits = []
        for s_id,s_bits in enumerate(bits):
            if s_bits in self.bits:
                s_bits_id = self.bits.index(s_bits)
                self.sublat_list[s_bits_id].append(s_id)
            else:
                self.sublat_list.append([s_id])
                self.bits.append(s_bits)
        self.sl_sizes = [len(sl) for sl in self.sublat_list]
        self.comp_space = CompSpace(self.bits,sl_sizes = self.sl_sizes)

        self._sc_df = sc_table
        self._comp_df = comp_table
        self._fact_df = fact_table

        if solver_flavor not in supported_solvers:
            raise ValueError("Solver {} not supported!".format(solver_flavor))

        self.flavor = solver_flavor
        if 'Grand' in self.flavor:
            self._grand_solver_args = grand_solver_args
        else:
            self._grand_solver_args = {}

        self._mu_center = None
        self._gs_df = None

    @property
    def n_iter(self):
        """
        This function gets the current CE iteration number from the fact table.
        Returns:
            An integer index.
        """
        if self._fact_df is None:
            raise NODATAERROR
        _n_it = self._fact_df[self._fact_df.module=='enum'].iter_id.max()
        if pd.isnull(_n_it):
            return -1
        else:
            return _n_it

    @property
    def comp_dim(self):
        """
        This function gets the dimension of the constrained comp space
        from the comp table.
        Returns:
            An integer index.
        """
        if self._comp_df is None:
            raise NODATAERROR
        return len(self._comp_df.reset_index().iloc[0]['ccoord'])

    def solve_gss(self):
        """
        Solve for new ground state entree.
        """
        if self._gs_df is not None:
            return self._gs_df

        if self.flavor == 'Canonical':
            self._gs_df = pd.DataFrame(columns=['sc_id','sc','comp','ccoord','ucoord',])

        else:
            #Estimate mu
            if self._mu_center is None:
                history = [{"coefs":self.ce.coefs}]
                _checker = GSChecker(self.ce.cluster_subspace,history,\
                                     sc_table=self._sc_df,comp_table=self._comp_df,\
                                     fact_table=self._fact_df)
                _cehull = self._checker.curr_ce_hull
                self._mu_center = estimate_mu_from_hull(self._cehull)

            _mu_step = self._grand_solver_args.get('mu_grid_step',0.4)
            _mu_num = self._grand_solver_args.get('mu_grid_num',5)
            _mu_grid = get_center_grid(self._mu_center,_mu_step,_mu_num)
            _n_proc = self._grand_solver_args.get('n_proc',4)
  
            def grand_solver_call(flavor,ce,sc_matrix,mu):
                _solver = globals()[flavor](ce,sc_matrix,mu)
                gs_occu,gs_e = _solver.solve()
                gs_str = structure_from_occu(self.ce.structure,sc_matrix,gs_occu)
                gs_corr = self.ce.cluster_subspace.corr_from_structure(gs_str,\
                                                   scmatrix=sc_matrix)
                return gs_occu,gs_corr,gs_str,gs_e

                       
             
        return self._new_comps, self._new_facts

    def add_gss_to_df(self):

    #Serializations and de-serializations
    def as_dict(self):
        """
        Serialize this class. Saving and loading of the star schema are moved to other functions!
        """
        #Serialization
        d={}
        d['ce']=self.ce.as_dict()
        d['flavor']=self.flavor
        d['grand_solver_args']=self._grand_solver_args
        d['mu_center']=self._mu_center
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__

        return d

    @classmethod
    def from_dict(cls,d):
        """
        De-serialze from a dictionary.
        """
        ce = ClusterExpansion.from_dict(d.get('ce'))
        socket = cls(ce = ce,\
                    solver_flavor=d.get('flavor','MCCanonical'),\
                    **d.get('grand_solver_args',{}))
        socket._mu_center = d.get('mu_center')
        return socket

    def _save_data(self,sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Saving dimension tables and the fact table. Must set index=False, otherwise will always add
        One more row for each save and load.
        comp_df needs a little bit serialization.
        File names can be changed, but not recommended!
        """
        if self._sc_table is not None:
            self._sc_table.to_csv(sc_file,index=False)
        if self._comp_table is not None:
            comp_ser = self._comp_table.copy()
            comp_ser.comp = comp_ser.comp.map(lambda c: serialize_comp(c))
            comp_ser.to_csv(comp_file,index=False)
        if self._fact_df is not None:
            self._fact_df.to_csv(fact_file,index=False)

    def _load_data(self,sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Loading dimension tables and the fact table. 
        comp_df needs a little bit de-serialization.
        File names can be changed, but not recommended!
        Notice: pandas loads lists as strings. You have to serialize them!
        """
        list_conv = lambda x: json.loads(x) if x is not None else None
        if os.path.isfile(sc_file):
            self._sc_table = pd.read_csv(sc_file,converters={'matrix':list_conv})
        if os.path.isfile(comp_file):
            #De-serialize compositions and list values
            self._comp_table = pd.read_csv(comp_file,
                                        converters={'ucoord':list_conv,
                                                    'ccoord':list_conv,
                                                    'cstat':list_conv,
                                                    'eq_occu':list_conv,
                                                    'comp':deser_comp
                                                   })
        if os.path.isfile(fact_file):
            self._fact_table = pd.read_csv(fact_file,
                                        converters={'ori_occu':list_conv,
                                                    'ori_corr':list_conv,
                                                    'map_occu':list_conv,
                                                    'map_corr':list_conv,
                                                    'other_props':list_conv
                                                   })


    def auto_save(self,gen_file='ce_gsgen.json',\
                  sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Automatically save object data into specified files.
        Args:
            gen_file(str):
                GSGenerator object file path.
            sc_file(str):
                supercell matrix file path
            comp_file(str):
                composition file path
            fact_file(str):
                fact table file path
        All optional, but I don't recommend you to change the paths.
        """
        with open(gen_file,'w') as fout:
            json.dump(self.as_dict(),fout)

        self._save_data(sc_file=sc_file,comp_file=comp_file,fact_file=fact_file)

    @classmethod
    def auto_load(cls,gen_file='ce_gsgen.json',\
                  sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Automatically load object data from specified files, and returns an object.
        Args:
            gen_file(str):
                GSGenerator object file path.
            sc_file(str):
                supercell matrix file path
            comp_file(str):
                composition file path
            fact_file(str):
                fact table file path
        All optional, but I don't recommend you to change the paths.
        """
        with open(gen_file) as fin:
            socket = cls.from_dict(json.load(fin))

        socket._load_data(sc_file=sc_file,comp_file=comp_file,fact_file=fact_file)
        return socket
