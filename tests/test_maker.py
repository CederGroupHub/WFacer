"""Test the trigger job and the flow maker."""
import numpy as np
import pytest
from jobflow import Flow, Job, Response

from WFacer.jobs import initialize_document
from WFacer.maker import AutoClusterExpansionMaker, ce_step_trigger
from WFacer.preprocessing import get_prim_specs

from .utils import execute_job_function


@pytest.fixture
def initial_document(prim):
    specs = get_prim_specs(prim)
    options = {
        "cutoffs": {2: 7, 3: 4},
        "comp_enumeration_step": 4,
        "num_structs_per_iter_init": 50,
        "num_structs_per_iter_add": 30,
        "other_properties": ["some_test"],
        "estimator_type": "lasso",
        "optimizer_type": "grid-search",
        "param_grid": {"alpha": 2 ** np.linspace(-25, 3, 15)},
        "max_iter": 5,
    }
    if specs["charge_decorated"]:
        options["decorator_types"] = ["pmg-guess-charge"]

    return initialize_document(prim, options=options)


# Test the internal structure of a trigger job.
def test_trigger(initial_document):
    trigger = ce_step_trigger(initial_document)
    # Not really running the jobs, just returning a response reference.
    response = execute_job_function(trigger)

    # Starting from empty so should trigger a replace.
    assert isinstance(response, Response)
    assert isinstance(response.replace, Flow)

    # Check the structure of a flow.
    flow = response.replace
    assert flow.name == "ace-work_iter_0"
    for job in flow.jobs:
        assert isinstance(job, Job)
    # enum, calc, parse, fit, update,
    assert len(flow.jobs) == 6
    assert flow.jobs[0].name == "ace-work_iter_0_enumeration"
    assert flow.jobs[1].name == "ace-work_iter_0_calculation"
    assert flow.jobs[2].name == "ace-work_iter_0_parsing"
    assert flow.jobs[3].name == "ace-work_iter_0_fitting"
    assert flow.jobs[4].name == "ace-work_iter_0_updating"
    assert flow.jobs[5].name == "ace-work_iter_1_trigger"


def test_ace_maker(prim, initial_document):
    maker = AutoClusterExpansionMaker(name="goodluck")
    flow = maker.make(prim)

    # Initialize, and the first trigger.
    # Check names as well.
    assert flow.name == "goodluck"
    assert len(flow.jobs) == 2
    for job in flow.jobs:
        assert isinstance(job, Job)
    assert flow.jobs[0].name == "goodluck_initialize"
    assert flow.jobs[1].name == "goodluck_iter_0_trigger"

    flow2 = maker.make(prim, last_document=initial_document)
    assert len(flow2.jobs) == 1
    assert flow2.jobs[0].name == "goodluck_iter_0_trigger"
    # Since we restart form a workflow that has not reached max_iter yet,
    # will not update max_iter. So it should still be 5.
    assert initial_document.ce_options["max_iter"] == 5
