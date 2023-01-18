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


# You should always only define job I/O connections in a maker.
# The actual operation and processing is always performed in @job
# decorated functions. Otherwise, you may get empty ResponseReference,
# since a job's method may not be executed yet at the moment you pass
# it's output to processing.
@job
def _finalize_ce_step(last_ce_document):
    """Iteratively run next CE step if needed.

    Note: do not import this function.
    Args:
        last_ce_document(CeOutputsDocument):
            The cluster expansion outputs document from the
            latest step.
    Returns:
        Response.
    """
    iter_id = last_ce_document.last_iter_id + 1
    max_iter = last_ce_document.ce_options["max_iter"]
    project_name = last_ce_document.project_name
    # prepared but not executed.
    if iter_id >= max_iter and not last_ce_document.converged:
        logging.warning(f"Maximum number of iterations: {max_iter}"
                        f" reached, but cluster expansion model is"
                        f" still not converged!")
    if iter_id >= max_iter or last_ce_document.converged:
        return last_ce_document
    else:
        next_ce_step = _CeStepMaker(name=
                                    project_name + f"_iter_{iter_id}") \
            .make(last_ce_document)
        # TODO: addition? or replace?
        return Response(output=last_ce_document,
                        replace=next_ce_step)


class _CeStepMaker(Maker):
    """Make a step in CE iteration.

    Note: Do not import this class.
    Attribute:
        name(str):
            Name of the cluster expansion step.
    """
    name: str = "ceauto_work"

    # Will always be set to project name carried in ce document once
    # make() is called.

    def make(self, last_ce_document):
        """Make a step in CE iteration.

        Args:
            last_ce_document(CeOutputsDocument):
                The cluster expansion outputs document from the
                latest step.
        Returns:
            Flow:
                A step in the cluster expansion iteration.
        """
        iter_id = last_ce_document.last_iter_id + 1
        project_name = last_ce_document.project_name

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

        finalizing = _finalize_ce_step(updating.output)
        finalizing.name = project_name + f"_iter_{iter_id}" + "_finalizing"

        return Flow([enumeration,
                     calculation,
                     parsing,
                     updating,
                     finalizing],
                    output=finalizing.output,
                    name=project_name)


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
        # Todo: Can a job output can be used as an argument of .make()?
        #  It seems to be okay for makers that returns Job, but will that
        #  work for maker than returns Flow as well?
        ce_iterate = _CeStepMaker(name=self.name + "_iter_0") \
            .make(initialize.output)
        return Flow([initialize, ce_iterate],
                    ce_iterate.output,
                    name=self.name)
