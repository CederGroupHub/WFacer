"""Automatic jobflow maker."""
from dataclasses import dataclass, field
from jobflow import Maker, Response, Flow, job
import logging

from .schema import CeOutputsDocument
from .jobs import (enumerate_structures,
                   calculate_structures,
                   parse_calculations,
                   fit_calculations,
                   update_document,
                   initialize_document)


@job
def ce_step_trigger(last_ce_document):
    """Trigger a step in CE iteration.

    Args:
        last_ce_document(CeOutputsDocument):
            The cluster expansion outputs document from the
            latest step.
    Returns:
        Response:
            Either a CeOutputsDocument if converged, or a
            response to replace with another step.
    """
    iter_id = last_ce_document.last_iter_id + 1
    max_iter = last_ce_document.ce_options["max_iter"]
    project_name = last_ce_document.project_name
    if iter_id >= max_iter and not last_ce_document.converged:
        logging.warning(f"Maximum number of iterations: {max_iter}"
                        f" reached, but cluster expansion model is"
                        f" still not converged!")
    if iter_id >= max_iter or last_ce_document.converged:
        return last_ce_document
    else:
        # enumerate_new structures.
        enumeration = enumerate_structures(last_ce_document)
        enumeration.name = project_name + f"_iter_{iter_id}" + "_enumeration"

        # Create calculations for all structures, and extract outputs.
        calculation = calculate_structures(enumeration.output,
                                           last_ce_document)
        calculation.name = project_name + f"_iter_{iter_id}" + "_calculation"

        # Analyze outputs and wrap up all necessary datas into wrangler.
        parsing = parse_calculations(calculation.output,
                                     enumeration.output,
                                     last_ce_document)
        parsing.name = project_name + f"_iter_{iter_id}" + "_parsing"

        # fit from wrangler.
        fitting = fit_calculations(parsing.output,
                                   last_ce_document)
        fitting.name = project_name + f"_iter_{iter_id}" + "_fitting"

        # Wrapping up.
        updating = update_document(enumeration.output,
                                   parsing.output,
                                   fitting.output,
                                   last_ce_document)
        updating.name = project_name + f"_iter_{iter_id}" + "_updating"

        # Point to next iteration.
        trigger = ce_step_trigger(updating.output)
        trigger.name = project_name + f"_iter_{iter_id}" + "_trigger"

        flow = Flow([enumeration,
                     calculation,
                     parsing,
                     updating,
                     trigger],
                    output=trigger.output,
                    name=project_name + f"_iter_{iter_id}")

        # Always return last_ce_document in this output.
        return Response(output=last_ce_document, replace=flow)  # TODO: replace? addition?


@dataclass
class CeAutoMaker(Maker):
    """The cluster expansion automatic maker.

    Attributes:
        name(str):
            Name of the cluster expansion project. Will be used
            to set the project_name in ce document.
        options(dict):
            A dictionary including all options to set up the automatic
            workflow.
            For available options, see docs in preprocessing.py.
    """
    name: str = "ceauto_work"
    options: dict = field(default_factory=dict)

    def make(self, prim):
        """Make the workflow.

        Args:
            prim(Structure):
                A primitive cell structure (no need to be reduced) with
                partial occupancy on some sub-lattice. This defines the
                lattice model of your cluster expansion.
        Returns:
            Flow:
                The iterative cluster expansion workflow.
        """
        initialize = initialize_document(prim,
                                         project_name=self.name,
                                         options=self.options)
        initialize.name = self.name + "_initialize"

        # Enter iteration.
        ce_trigger = ce_step_trigger(initialize.output)
        ce_trigger.name = self.name + f"_iter_0_trigger"
        return Flow([initialize, ce_trigger],
                    ce_trigger.output,
                    name=self.name)
