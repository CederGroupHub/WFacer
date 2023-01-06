"""Defines the data schema for CEAuto jobs."""

from pydantic import BaseModel, Field

from smol.cofe import ClusterSubspace

from .wrangling import CeDataWrangler


class CeOutputsDocument(BaseModel):
    """Summary of cluster expansion workflow as outputs."""

    cluster_subspace: ClusterSubspace = Field(None,
                                              description="The cluster subspace"
                                                          " for expansion.")
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
                                         " errors in meV/atom.")
    cv_std_history: list = Field([],
                                 description="All historical cross validation"
                                             " standard deviations in"
                                             " meV/atom.")
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
    enumerated_features: list = Field([],
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

