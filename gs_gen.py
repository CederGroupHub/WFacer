"""
Ground state generator class. THIS FILE CHANGES FACT, and MAY CHANGE
COMP TABLE. All generated structures will be considered into the 
NEXT iteration, NOT the CURRENT one.
"""
from monty.json import MSONable
import multiprocessing as mp
import numpy as np
import pandas as pd
import itertools

from smol.cofe.space.domain import get_allowed_species

from .ce_handler import *
from .data_manager import DataManager
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
        prim(Structure):
            Primitive cell of the system. Should be modified 
            to charge neutral and has every specie in it.
        bits(List[List[Specie]]):
            Species on each sublattice, including Vacancy.
        sublat_list(List[List[int]]):
            List of prim cell site indices in a sublattice.
        compspace(CompSpace):
            Compositional space of the current system.
        handler_flavor(str):
            Specifies class name of the handler to be used.
            Check available handlers in CEAuto.gs_handler module.
            Will use 'Canonical' as default.
            When selecting Canoncial as handler, will not add anything,
            because in structure eneumerator, we are already keeping
            canoncial ground states.
        handler_args(Dict):
            Takes in handler parameters, such as:
            For semi-grand
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
 
        data_manager(DataManager):
            DataManager object of the current CE project.

        Central mu will be estimated from previous CE hull.
    """
    #Modify this when you implement new handlers
    supported_handlers = ('CanonicalHandler','MCGrandHandler','PBGrandHandler')

    def __init__(self,ce,\
                      prim,\
                      bits,\
                      sublat_list,\
                      compspace,\
                      handler_flavor='CanonicalHandler',\
                      handler_args={},\
                      data_manger=DataManager.auto_load()):

        self.ce = ce
        self.prim = prim
        self.bits = bits
        self.sublat_list = sublat_list
        self.comp_space = compspace

        if handler_flavor not in supported_handlers:
            raise ValueError("Solver {} not supported!".format(handler_flavor))

        self.flavor = handler_flavor
        self._handler_args = handler_args

        self._mu_center = None
        self._gss = None

        self._dm = data_manager

    @property
    def comp_dim(self):
        """
        This function gets the dimension of the constrained comp space
        from the comp table.
        Returns:
            An integer index.
        """
        return self.comp_space.dim

    @property
    def sc_df(self):
        """
        Supercell dataframe.
        """
        return self._dm.sc_df
 
    @property
    def comp_df(self):
        """
        Compositional dataframe.
        """
        return self._dm.comp_df

    @property
    def fact_df(self):
        """
        Fact dataframe containing all entree.
        """
        return self._dm.fact_df

    def solve_gss(self):
        """
        Solve for new ground state entree.
        """
        if self._gss is not None: #Already solved, no need to do again.
            return

        if 'Canonical' in self.flavor:
            self._gss = []
            return

        #Estimate mu
        if self._mu_center is None:
            history = [{"coefs":self.ce.coefs}]
            _checker = GSChecker(self.ce.cluster_subspace,ce_history=history,\
                                 data_manager=self._dm)
            _cehull = _checker.curr_ce_hull
            self._mu_center = estimate_mu_from_hull(self._cehull)

        _mu_step = self._handler_args.get('mu_grid_step',0.4)
        _mu_num = self._handler_args.get('mu_grid_num',5)
        _mu_grid = get_center_grid(self._mu_center,_mu_step,_mu_num)
        _n_proc = self._handler_args.get('n_proc',4)

        def grand_handler_call(flavor,ce,sc_matrix,mu,**handler_args):
            _handler = globals()[flavor](ce,sc_matrix,mu,**handler_args)
            gs_occu,gs_e = _handler.solve()
            gs_occu = list(gs_occu)
            return gs_occu,gs_e
       
        sc_mus = list(itertools.product(self.sc_df.matrix,_mu_grid))
        pool = mp.Pool(_nproc)
        gs_occu_es = pool.map(sc_mus,\
                            lambda p: grand_handler_call(self.flavor,self.ce,\
                                                         p[0],p[1],\
                                                         **self._handler_args))

        self._gss = []
        #Insert new ground states.
        for (gs_occu,gs_e),(sc,mu) in zip(gs_occu_es,sc_mus):
            self._gss.append((gs_occu,sc))
            self._dm.insert_one_occu(gs_occu,sc_mat=sc,module_name='gs')

    #Serializations and de-serializations

    def auto_save(self,sc_file='sc_mats.csv',\
                       comp_file='comps.csv',\
                       fact_file='data.csv'):
        """
        Automatically save object data into specified files.
        Args:
            sc_file(str):
                supercell matrix file path
            comp_file(str):
                composition file path
            fact_file(str):
                fact table file path
        All optional, but I don't recommend you to change the paths.
        """
        self._dm.auto_save(sc_file=sc_file,comp_file=comp_file,fact_file=fact_file)

    @classmethod
    def auto_load(cls,\
                  options_file='options.yaml',\
                  sc_file='sc_mats.csv',\
                  comp_file='comps.csv',\
                  fact_file='data.csv',\
                  ce_history_file='ce_history.json'):
        """
        This method is the recommended way to initialize this object.
        It automatically reads all setting files with FIXED NAMES.
        YOU ARE NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!
        Args:
            options_file(str):
                path to options file. Options must be stored as yaml
                format. Default: 'options.yaml'
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'             
            sc_file(str):
                path to supercell matrix dataframe file, in csv format.
                Default: 'sc_mats.csv'             
            ce_history_file(str):
                path to cluster expansion history file.
                Default: 'ce_history.json'
        Returns:
            GSChecker object.
        """
        options = InputsWrapper.auto_load(options_file=options_file,\
                                          ce_history_file=ce_history_file)

        dm = DataManager.auto_load(options_file=options_file,\
                                   sc_file=sc_file,\
                                   comp_file=comp_file,\
                                   fact_file=data_file,\
                                   ce_history_file=ce_history_file)

        return cls(options.last_ce,\
                   options.prim,\
                   options.bits,\
                   options.sublat_list,\
                   options.comp_space,\
                   data_manager=dm,\
                   **options.gs_generator_options)
