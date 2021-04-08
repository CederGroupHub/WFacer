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

from .utils.hull_utils import estimate_mu_from_hull
from .utils.math_utils import get_center_grid

from .config_paths import *

NODATAERROR = RuntimeError("No dataframes. You may call data loading methods to load the calculation data first.")

class GSGenerator(MSONable):
    """
    Ground state generator class. Generates grounds states from 
    enumerate supercell matrices, compositions (or guessed chemical 
    potentials), and writes them to the fact and composition table.

    You may not want to call __init_ directly.

    Also, do not call this twice in an iteration.

    Note: Since when studying ground state occupations, we actually care
          about INTERNAL DISTRIBUTION on sublattices, we MUST use DISCRIMINATIVE
          handler!
    """
    #Modify this when you implement new handlers
    supported_handlers = ('CanonicalMCHandler','CanonicalPBHandler',
                          'SemigrandDiscMCHandler','SemigrandDiscPBHandler')

    def __init__(self, ce, prim, bits, sublat_list, compspace, data_manager,
                 handler_flavor='CanonicalMCHandler',
                 handler_args={}):
        """
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
            data_manager(DataManager):
                DataManager object of the current CE project.
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
    
            Central mu will be estimated from previous CE hull.
        """
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

        self._sc_load_path = SC_FILE
        self._comp_load_path = COMP_FILE
        self._fact_load_path = FACT_FILE

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

    def solve_gss(self,insert_to_df = True):
        """
        Solve for new ground state entree.
        Args:
            insert_to_df(Boolean):
                If true(default), will insert generated ground state structures to
                the dataframe.
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
            self._mu_center = estimate_mu_from_hull(_cehull)

        _mu_step = self._handler_args.get('mu_grid_step',0.4)
        _mu_num = self._handler_args.get('mu_grid_num',5)
        _mu_grid = get_center_grid(self._mu_center,_mu_step,_mu_num)
        _n_proc = self._handler_args.get('n_proc',4)

        def grand_handler_call(flavor,ce,sc_matrix,mu,**handler_args):
            _handler = globals()[flavor](ce,sc_matrix,mu,**handler_args)
            gs_occu,gs_e = _handler.solve()
            gs_occu = list(gs_occu)
            return gs_occu,gs_e
       
        #This number can be huge!! How to lower this?
        sc_mus = list(itertools.product(self.sc_df.matrix,_mu_grid))
        pool = mp.Pool(_nproc)
        gs_occu_es = pool.map(lambda p: grand_handler_call(self.flavor,self.ce,\
                                                         p[0],p[1],\
                                                         **self._handler_args),\
                              sc_mus)

        self._gss = []
        #Insert new ground states.
        for (gs_occu,gs_e),(sc,mu) in zip(gs_occu_es,sc_mus):
            self._gss.append((gs_occu,sc))
            if insert_to_df:
                self._dm.insert_one_occu(gs_occu,sc_mat=sc,module_name='gs')

    #Serializations and de-serializations

    def auto_save(self,sc_file=SC_FILE,\
                       comp_file=COMP_FILE,\
                       fact_file=FACT_FILE,\
                       to_load_paths=True):
        """
        Automatically save object data into specified files.
        Args:
            sc_file(str):
                supercell matrix file path
            comp_file(str):
                composition file path
            fact_file(str):
                fact table file path
            to_load_paths(Boolean):
                If true, will save to the paths from which this object is loaded.
                Default is true.
        All optional, but I don't recommend you to change the paths.
        """
        if to_load_paths:
            sc_file = self._sc_load_path
            comp_file = self._comp_load_path
            fact_file = self._fact_load_path

        self._dm.auto_save(sc_file=sc_file,comp_file=comp_file,fact_file=fact_file)

    @classmethod
    def auto_load(cls, data_manager,
                  options_file=OPTIONS_FILE,
                  ce_history_file=CE_HISTORY_FILE):
        """
        This method is the recommended way to initialize this object.
        It automatically reads all setting files with FIXED NAMES.
        YOU ARE NOT RECOMMENDED TO CHANGE THE FILE NAMES, OTHERWISE 
        YOU MAY BREAK THE INITIALIZATION PROCESS!
        Args:
            data_manager(DataManager):
                Data manager to read and write.
            options_file(str):
                path to options file. Options must be stored as yaml
                format. Default: 'options.yaml'           
            ce_history_file(str):
                path to cluster expansion history file.
                Default: 'ce_history.json'
        Returns:
            GSChecker object.
        """
        options = InputsWrapper.auto_load(options_file=options_file,\
                                          ce_history_file=ce_history_file)

        socket = cls(options.last_ce,\
                     options.prim,\
                     options.bits,\
                     options.sublat_list,\
                     options.comp_space,\
                     data_manager=data_manager,\
                     **options.gs_generator_options)

        socket._sc_load_path = sc_file
        socket._comp_load_path = comp_file
        socket._fact_load_path = fact_file

        return socket
