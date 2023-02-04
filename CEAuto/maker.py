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
        trigger.name = project_name + f"_iter_{iter_id + 1}" + "_trigger"

        flow = Flow([enumeration,
                     calculation,
                     parsing,
                     fitting,
                     updating,
                     trigger],
                    output=trigger.output,
                    name=project_name + f"_iter_{iter_id}")

        # Always return last_ce_document in this output.
        return Response(output=last_ce_document, replace=flow)


@dataclass
class CeAutoMaker(Maker):
    """The cluster expansion automatic workflow maker.

    Attributes:
        name(str):
            Name of the cluster expansion project. Since the underscore
            will be used to separate fields of job names, it should not
            appear in the project name!
        options(dict):
            A dictionary including all options to set up the automatic
            workflow.
            For available options, see docs in preprocessing.py.
    """
    name: str = "ceauto-work"
    options: dict = field(default_factory=dict)

    def make(self, prim,
             last_document=None,
             add_num_iterations=None):
        """Make the workflow.

        Args:
            prim(Structure):
                A primitive cell structure (no need to be reduced) with
                partial occupancy on some sub-lattice. This defines the
                lattice model of your cluster expansion.
            last_document(CeOutputsDocument): optional
                A cluster expansion outputs maker to continue from. Used
                for warm restart. Will run at maximum another "max_iter"
                number of iterations.
                Default is None, which means to start from scratch.
            add_num_iterations(int): optional
                When the last document has reached the maximum allowed
                number of iterations, add this many more iterations.
                Default is None. When given None or 0, will simply double
                max_iter in options.
        Returns:
            Flow:
                The iterative cluster expansion workflow.
        """
        if last_document is None:
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

        # Warm restart.
        else:
            prev_max_iter = last_document.ce_options["max_iter"]
            prev_iter = last_document.last_iter_id
            # Do max_iter iterations again.
            if prev_max_iter is not None and prev_iter >= prev_max_iter - 1:
                last_document.ce_options["max_iter"] += (add_num_iterations
                                                         or prev_max_iter)

            ce_trigger = ce_step_trigger(last_document)
            ce_trigger.name = self.name + f"_iter_{prev_iter + 1}_trigger"
            return Flow([ce_trigger],
                        ce_trigger.output,
                        name=self.name)
