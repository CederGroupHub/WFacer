"""Generate structures for a next iteration."""
from warnings import warn

from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow
from monty.serialization import dumpfn
from pydantic import parse_file_as

from WFacer.jobs import enumerate_structures, get_structure_calculation_flows
from WFacer.schema import CeOutputsDocument


def __main__():
    document = parse_file_as(CeOutputsDocument, "document.json")

    iter_id = document.last_iter_id + 1
    max_iter = document.ce_options["max_iter"]
    if iter_id >= max_iter and not document.converged:
        warn(
            f"Maximum number of iterations: {max_iter}"
            f" reached, but cluster expansion model is"
            f" still not converged!"
        )
        return
    if document.converged:
        warn("Model already converged! No need for further operation!")
        return

    if document.enumerated_structures is None:
        pass
    else:
        len(document.enumerated_structures)

    print("Enumerating structures!")
    enum_output = enumerate_structures(last_ce_document=document)
    flows = get_structure_calculation_flows(enum_output, document)
    workflows = [flow_to_workflow(f) for f in flows]

    # Add workflows to launchpad to launch.
    # Remember to set my_qadapter.yaml to rlaunch singleshot, then
    # use qlaunch rapidfire to launch.
    print("Adding workflows!")
    lpad = LaunchPad.auto_load()
    lpad.bulk_add_wfs(workflows)

    print("Saving enumerated structures.")
    enum_fname = f"enum_iter_{iter_id}.json"
    dumpfn(enum_output, enum_fname)


if __name__ == "__main__":
    __main__()
