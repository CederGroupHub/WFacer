"""
Ground state scanner that checks grounds states for convergence.
THIS CLASS DOES NOT TOUCH DATA TABLES!
"""
__author__ == "Fengyu Xie"


import json
import numpy as np
import pandas as pd

from smol.cofe import ClusterSubspace

from utils.format_utils import *
from utils.hull_utils import hulls_match, plot_hull

NODATAERROR = RuntimeError("No dataframes. You may call data loading methods to load the calculation data first.")
 
class GSChecker:
    """
    A ground state checker class. This class only checks grand
    canoncial ground state convergence, and does not modify the
    fact table.
    Only energies will be checked.
    Args:
        cluster_subspace(smol.ClusterSubspace):
            Previous ClusterSubspace object used in featurizer.
        ce_history(List of CEFitter dicts):
            Previous cluster expansion fitting history.
        sc_table(pd.DataFrame):
            supercell matrices dataframe.
            Can not be modified
        comp_table(pd.DataFrame):
            compositions dataframe.
            Can be modified by this class, if new grounds states
            are detected.       
        fact_table(pd.DataFrame):
            facts dataframe.
            Can be modified by this class, if new grounds states
            are detected.  
    Notice:
        We consider structures added in this module as belongs to 
        the last structure enumeration iteration.
    You may not want to call this function directly.
    """
    def __init__(self,cluster_subspace=None,ce_history = [],\
                      sc_table=None, comp_table=None, fact_table=None):
        self.cspc = cluster_subspace
        self.ce_history = ce_history
        self._sc_table = sc_table
        self._comp_table = comp_table
        self._fact_table = fact_table
        self._dft_hulls_ahead = {}  #Will not be saved
        self._ce_hulls_ahead = {}


    @property
    def n_iter(self):
        """
        This function gets the current CE iteration number from the fact table.
        Returns:
            An integer index.
        """
        if self._fact_table is None:
            raise NODATAERROR
        _n_it = self._fact_table[self._fact_table.module=='enum'].iter_id.max()
        if pd.isnull(_n_it):
            return -1
        else:
            return _n_it


    def get_hull_n_iters_ahead(self,n_it_ahead=0,mode='dft'):
        """
        This function computes a minimum hull for CE energies (not DFT energies) from
        successful entree in n iterations ahead, to compare with the current minimum 
        CE hull.
        Args:
            n_it_ahead(int):
                Number of iterations ahead. Default is 0, meaning current hull.
            mode(str):
                Type of hull to compute. Can be either 'dft' or 'ce'.
                Default is 'dft'.
        Returns:
            A pd.DataFrame, containing composition indices and minimum energies of 
            a composition.
        """
        if mode=='ce':
            if n_it_ahead in self._ce_hulls_ahead:
                return self._ce_hulls_ahead[n_it_ahead]
        if mode=='dft':
            if n_it_ahead in self._dft_hulls_ahead:
                return self._dft_hulls_ahead[n_it_ahead]      
        if mode not in ['ce','dft']:
            raise NotImplementedError("Hull mode {} not implemented yet."\
                                      .format(mode))           

        if self._fact_table is None:
            raise NODATAERROR

        filt_ = (self._fact_table.iter_id <= self.n_iter-n_it_ahead) & \
                (self._fact_table.calc_status=='SC') &
                (~self._fact_table.e_prim.isna())

        if filt_.sum()==0: #Might be the first iteration, or fact_table is empty
            return None
        if len(self.ce_history)<n_it_ahead+1:
            return None

        fact_prev = self._fact_table[filt_]

        if mode=='ce':
            coef_prev = self.ce_history[-(n_it_ahead+1)]['coefs']['e_prim']
            fact_prev['e_ce'] = np.array(fact_prev.map_corr.tolist())@np.array(coef_prev)

            #If multiple GSs have the same energy, only one will be taken.
            _prev_hull = fact_prev.groupby('comp_id').agg(lambda df: df.loc[df['e_ce'].idxmin()])\
                         .reset_index()
            _prev_hull = _prev_hull.loc[:,['comp_id','e_ce']]
            _prev_hull = _prev_hull.merge(self._comp_table,how='left',on='comp_id')a
            _prev_hull = _prev_hull.rename(columns={'e_ce':'e_prim'})
            self._ce_hulls_ahead[n_it_ahead] = _prev_hull.copy()
            return self._ce_hulls_ahead[n_it_ahead]

        if mode=='dft':
            _prev_hull = fact_prev.groupby('comp_id').agg(lambda df: df.loc[df['e_prim'].idxmin()])\
                         .reset_index()
            _prev_hull = _prev_hull.loc[:,['comp_id','e_prim']]
            _prev_hull = _prev_hull.merge(self._comp_table,how='left',on='comp_id')
            self._dft_hulls_ahead[n_it_ahead] = _prev_hull.copy()
            return self._dft_hulls_ahead[n_it_ahead]


    @property
    def prev_ce_hull(self):
        """
        This function computes a minimum hull for CE energies (not DFT energies) from
        successful entree in previous iterations, to compare with the current minimum 
        CE hull.
        Returns:
            A pd.DataFrame, containing composition indices and minimum energies of 
            a composition.
        """
        return self.get_hull_n_iters_ahead(n_it_ahead = 1,mode='ce')


    @property
    def curr_ce_hull(self):
        """
        This function computes a minimum hull for CE energies (not DFT energies) from
        successful entree in current iteration.
        Returns:
            A pd.DataFrame, containing composition indices and minimum energies of 
            a composition.
        """   
        return self.get_hull_n_iters_ahead(n_it_ahead = 0,mode='ce')

 
    @property
    def prev_dft_hull(self):
        """
        This function computes a minimum hull for DFT energies from
        successful entree in previous iterations, to compare with the current minimum 
        CE hull.
        Returns:
            A pd.DataFrame, containing composition indices and minimum energies of 
            a composition.
        """
        return self.get_hull_n_iters_ahead(n_it_ahead = 1,mode='dft')


    @property
    def curr_dft_hull(self):
        """
        This function computes a minimum hull for DFT energies from
        successful entree in current iteration.
        Returns:
            A pd.DataFrame, containing composition indices and minimum energies of 
            a composition.
        """   
        return self.get_dft_hull_n_iters_ahead(n_it_ahead = 0,mode='dft')


    def check_convergence(self, e_tol=3, comp_tol = 0.05):
        """
        This function checks whether the ground states for a cluster expansion have 
        converged from reading history and the fact table.
        MUST have both DFT and CE hulls match to claim a convergence.

        Args:
            e_tol(float):
                tolerance of ground state energy differnece measured by CV value.
                absolute tolerance = e_tolerance * cv of last expansion in history.
                Default is 3
                If new ground state occurs, but its energy difference to old gss
                is smaller than tolerance, will still think converged.
            comp_tol(float):
                tolerance of ground state composition changes, measured in percent.
                Default is 5%.
                If new ground state occurs, but its composition difference to old gss
                is smaller than tolerance, will still think converged.
        Returns:
            Boolean.
        """
        if self.prev_ce_hull is None or self.curr_ce_hull is None or \
           self.prev_dft_hull is None or self.curr_dft_hull is None:
            return False

        cv = self.ce_history[-1].get('cv',{'e_prim':0.001})['e_prim']

        return hulls_match(self.prev_ce_hull,self.curr_ce_hull,\
                           e_tol=e_tol*cv,comp_tol=comp_tol) and \
               hulls_match(self.prev_dft_hull,self.curr_dft_hull,\
                           e_tol=e_tol*cv,comp_tol=comp_tol)


    def plot_hull_scatter(self,mode='dft',\
                          axis_id=None,title='hull and scatter plot',\
                          x_label=None,y_label='Energy per prim/eV',\
                          convert_to_formation=True):
        """
        Plot hull and scatter of a physical quantity.
        When in high dimensional compositional 
        space, must specify an axis to project to.

        Args:
            mode(str): 
                The type of hull you wish to plot. Can be 'dft' or 
                'ce'.
                Will plot 'dft' by default.
            axis_id(int):
                Index of axis. If none given, will always project on 
                the first axis.
            fix_hull(Boolean):
                Fix the hull when it is not convex. Default to True.
            title(str):
                Title or plot
            x_label(str):
                x axis label
            y_label(str):
                y axis label
            convert_to_formation(Boolean):
                If true, will plot formation energy in eV/prim,
                instead of CE energies.
        Return:
            plt.figure, plt.axes
        """

        filt_ = (self._fact_table.iter_id <= self.n_iter) & \
                (self._fact_table.calc_status=='SC') &
                (~self._fact_table.e_prim.isna())           
        fact_cur = self._fact_table[filt_]
        fact_cur = fact_cur.merge(self._comp_table,how='left',on='comp_id')
        fact_cur['ccoord'] = fact_cur['ccoord'].map(lambda x: x[axis_id or 0])

        title = title+'_'+mode

        if mode == 'dft':
            hull = self.curr_dft_hull
            scatter = fact_cur.loc[:,['ccoord','e_prim']].to_numpy().T
        elif mode == 'ce':
            coef_ = self.ce_history[-1]['coefs']['e_prim']
            fact_cur['e_ce'] = np.array(fact_cur.map_corr.tolist())@np.array(coef_)
            hull = self.curr_ce_hull
            scatter = fact_cur.loc[:,['ccoord','e_ce']].to_numpy().T
        else:
            raise NotImplementedError("Hull mode {} not implemented yet."\
                                      .format(mode))           

        fig, ax, e1, e2, x_min, x_max = plot_hull(hull,axis_id=axis_id,\
                                            title=title,x_label=x_label,\
                                            y_label=y_label,\
                                            convert_to_formation=convert_to_formation)

        if convert_to_formation:
            scatter[0] = (scatter[0]-x_min)/(x_max-x_min)
            scatter[1] = scatter[1] - (scatter[0]*e1 + scatter[0]*e2)

        ax.scatter(scatter[0],scatter[1],color='b')
        
        return fig, ax


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


    @classmethod
    def auto_load(cls,fitter_file='ce_fitter.json',\
                  sc_file='sc_mats.csv',comp_file='comps.csv',fact_file='data.csv'):
        """
        Automatically initializes this object from data and history files.
        This is more frequently called during initailization.
        Args:
            fitter_file(str):
                path to the fitter file.
            sc_file(str):
                supercell matrix data file path.
            comp_file(str):
                compositions data file path.
            fact_file(str):
                calculation entree data file.
            You can specify these paths, but we don't recommend so.
        """
        if not os.path.isfile(fitter_file):
            raise ValueError("No previous fitter record exists!")
        with open(fitter_file) as fin:
            d = json.load(fin)
        cspc = d.get('cluster_subspace')
        cspc = ClusterSubspace.from_dict(cspc) if cspc is not None else None
        history = d.get('history',[])
        socket = cls(cluster_subspace = cspc, ce_history=history)
        socket._load_data(sc_file,comp_file,fact_file)

        return socket
