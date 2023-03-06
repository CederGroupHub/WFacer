"""Defines the data schema for WFacer jobs."""
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from smol.cofe import ClusterSubspace

from .convergence import ce_converged
from .wrangling import CeDataWrangler


class CeOutputsDocument(BaseModel):
    """Summary of cluster expansion workflow as outputs."""

    project_name: str = Field(
        "ace-work", description="The name of cluster expansion" " project."
    )
    cluster_subspace: ClusterSubspace = Field(
        None, description="The cluster subspace" " for expansion."
    )
    prim_specs: Dict[str, Any] = Field(
        None, description="Computed specifications of the primitive" " cell."
    )
    data_wrangler: CeDataWrangler = Field(
        None,
        description="The structure data"
        " wrangler, including"
        " all successfully"
        " computed and mapped"
        " structures.",
    )
    ce_options: Dict[str, Any] = Field(
        None, description="Cluster expansion workflow options."
    )
    coefs_history: List[List[float]] = Field(
        None, description="All historical coefficients."
    )
    cv_history: List[float] = Field(
        None, description="All historical cross validation" " errors in meV/site."
    )
    cv_std_history: List[float] = Field(
        None,
        description="All historical cross validation"
        " standard deviations in"
        " meV/site.",
    )
    rmse_history: List[float] = Field(
        None, description="All historical cross training errors in meV/site."
    )
    params_history: List[Union[Dict[str, Any], None]] = Field(
        None,
        description="All historical fitting hyper-parameters, if needed by model.",
    )

    # Enumerated data.
    supercell_matrices: List[List[List[int]]] = Field(
        None, description="Enumerated supercell matrices."
    )
    compositions: List[List[int]] = Field(
        None, description="Enumerated composition in species counts per sub-lattice."
    )
    enumerated_structures: List[Structure] = Field(
        None, description="All enumerated structures till the last" " iteration."
    )
    enumerated_matrices: List[List[List[int]]] = Field(
        None, description="Supercell matrices for each enumerated structure."
    )
    # Needs to be reshaped when initialized.
    enumerated_features: List[List[float]] = Field(
        None, description="Feature vectors for each enumerated structure."
    )
    undecorated_entries: List[Union[ComputedStructureEntry, None]] = Field(
        None,
        description="Computed structure entry"
        " for each enumerated"
        " structure. If failed, will"
        " be None-type.",
    )
    computed_properties: List[Union[Dict[str, Any], None]] = Field(
        None,
        description="Other properties extracted"
        " for each enumerated"
        " structure. If failed, will"
        " be None-type.",
    )

    # This is to make feature matrix validated correctly.
    class Config:
        """Setting configuration for schema."""

        arbitrary_types_allowed = True

    @property
    def last_iter_id(self):
        """Index of the last iteration.

        Returns:
            int.
        """
        is_none = [
            self.coefs_history is None,
            self.cv_history is None,
            self.cv_std_history is None,
            self.params_history is None,
            self.rmse_history is None,
        ]
        all_none = all(is_none)
        one_none = any(is_none)

        if all_none:
            return -1
        if one_none:
            raise ValueError(
                "Missing some of the following required history records:"
                "coefficients, CV, standard error of CV, optimal hyper-parameters,"
                "and RMSE!"
            )

        if (
            len(self.coefs_history) != len(self.cv_history)
            or len(self.coefs_history) != len(self.cv_std_history)
            or len(self.coefs_history) != len(self.params_history)
            or len(self.coefs_history) != len(self.rmse_history)
        ):
            raise ValueError(
                "Length of history records in coefs, cv,"
                "cv_std and params must match!"
            )
        if self.data_wrangler.max_iter_id is None:
            if len(self.coefs_history) > 0:
                raise ValueError(
                    "Length of history and the record in" "data wrangler must match!"
                )
        elif self.data_wrangler.max_iter_id != len(self.coefs_history) - 1:
            raise ValueError(
                "Length of history and the record in" "data wrangler must match!"
            )
        return len(self.coefs_history) - 1

    @property
    def converged(self):
        """Check convergence based on given output doc.

        Returns:
            bool.
        """
        return ce_converged(
            self.coefs_history,
            self.cv_history,
            self.cv_std_history,
            self.data_wrangler,
            self.ce_options,
        )
