"""Defines the data schema for CEAuto jobs."""
import numpy as np
from pydantic import BaseModel, Field

from smol.cofe import ClusterSubspace

from .wrangling import CeDataWrangler
from .convergence import ce_converged


class CeOutputsDocument(BaseModel):
    """Summary of cluster expansion workflow as outputs."""

    project_name: str = Field("ceauto_work",
                              description="The name of cluster expansion"
                                          " project.")
    cluster_subspace: ClusterSubspace = Field(None,
                                              description="The cluster subspace"
                                                          " for expansion.")
    prim_specs: dict = Field({},
                             description="Computed specifications of the primitive"
                                         " cell.")
    data_wrangler: CeDataWrangler = Field(CeDataWrangler(cluster_subspace),
                                          description="The structure data"
                                                      " wrangler, including"
                                                      " all successfully"
                                                      " computed and mapped"
                                                      " structures.")
    ce_options: dict = Field({}, description="Cluster expansion workflow"
                                             " options.")
    coefs_history: list = Field([],
                                description="All historical coefficients.")
    cv_history: list = Field([],
                             description="All historical cross validation"
                                         " errors in meV/site.")
    cv_std_history: list = Field([],
                                 description="All historical cross validation"
                                             " standard deviations in"
                                             " meV/site.")
    rmse_history: list = Field([],
                               description="All historical cross training"
                                           " errors in meV/site.")
    params_history: list = Field([],
                                 description="All historical fitting hyper-"
                                             "parameters, if needed by model.")

    # Enumerated data.
    supercell_matrices: list = Field([],
                                     description="Enumerated supercell"
                                                 " matrices.")
    compositions: list = Field([],
                               description="Enumerated composition in species"
                                           " counts per sub-lattice.")
    enumerated_structures: list = Field([],
                                        description="All enumerated structures"
                                                    " till the last"
                                                    " iteration.")
    enumerated_matrices: list = Field([],
                                      description="Supercell matrices for each"
                                                  " enumerated structure.")
    enumerated_features: list = Field(np.array([]).reshape(0,
                                                           cluster_subspace
                                                           .num_corr_functions
                                                           + len(cluster_subspace
                                                                 .external_terms)),
                                      description="Feature vectors for each"
                                                  " enumerated structure.")
    undecorated_entries: list = Field([],
                                      description="Computed structure entry"
                                                  " for each enumerated"
                                                  " structure. If failed, will"
                                                  " be None-type.")
    computed_properties: list = Field([],
                                      desciption="Other properties extracted"
                                                 " for each enumerated"
                                                 " structure. If failed, will"
                                                 " be None-type.")

    @property
    def last_iter_id(self):
        """Index of the last iteration.

        Returns:
            int.
        """
        if (len(self.coefs_history) != len(self.cv_history) or
                len(self.coefs_history) != len(self.cv_std_history) or
                len(self.coefs_history) != len(self.params_history) or
                len(self.coefs_history) != len(self.rmse_history)):
            raise ValueError("Length of history records in coefs, cv,"
                             "cv_std and params must match!")
        if self.data_wrangler.max_iter_id is None:
            if len(self.coefs_history) > 0:
                raise ValueError("Length of history and the record in"
                                 "data wrangler must match!")
        elif (self.data_wrangler.max_iter_id
              != len(self.coefs_history) - 1):
            raise ValueError("Length of history and the record in"
                             "data wrangler must match!")
        return len(self.coefs_history) - 1

    @property
    def converged(self):
        """Check convergence based on given output doc.

        Returns:
            bool.
        """
        return ce_converged(self.coefs_history,
                            self.cv_history,
                            self.cv_std_history,
                            self.data_wrangler,
                            self.ce_options)
