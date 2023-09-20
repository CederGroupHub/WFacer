from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow
from pymatgen.core import Structure

from WFacer.maker import AutoClusterExpansionMaker

# construct a rock salt Ag-Li structure
agli_prim = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=[
        {"Ag": 0.5, "Li": 0.5},
        {"Ag": 0.5, "Li": 0.5},
    ],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)
# Use default for every option.
ce_flow = AutoClusterExpansionMaker(name="agli_fcc_ce", options={}).make(agli_prim)

# convert the flow to a fireworks WorkFlow object
# If argument "store" is not specified, all documents will be saved to the JOB_STORE
# Defined by the local configuration files where you run THIS script from.
wf = flow_to_workflow(ce_flow)

# submit the workflow to the FireWorks launchpad
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
