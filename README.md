WorkFlow for Automated Cluster Expansion Regression (WFACER)
===================================================

*Modulated automation of cluster expansion based on atomate2 and Jobflow*

-----------------------------------------------------------------------------

**WFacer** is a light-weight package based on [**smol**](https://github.com/CederGroupHub/smol.git)
to automate the building of energy models in crystalline material systems, based on the
*cluster expansion* method from alloy theory. Beyond metallic alloys, **WFacer** is also designed
to handle ionic systems through enabling charge **Decorator**s and external **EwaldTerm**. With the
support of [**Atomate2**](https://github.com/materialsproject/atomate2.git),
[**Jobflow**](https://github.com/materialsproject/jobflow.git)
and [**Fireworks**](https://github.com/materialsproject/fireworks.git), **WFacer** is able to fully automate the
cluster expansion building process on super-computing clusters, and can easily interface
with materials-project style MongoDB data storage.

Functionality
-------------
**WFacer** currently supports the following functionalities:

- Preprocess setup to a cluster expansion workflow as dictionary.
- Enumerating and choosing the least aliasing super-cell matrices with given number of sites;
  enumerating charge balanced compositions in super-cells; Enumerating and selecting low-energy,
  non-duplicated structures into the training set at the beginning of each iteration.
- Computing enumerated structures using **Atomate2** **VASP** interfaces.
- Extracting and saving relaxed structures information and energy in **Atomate2** schemas.
- Decorating structures. Currently, supports charge decoration from fixed labels, from Pymatgen guesses,
  or from [a gaussian process](https://doi.org/10.1038/s41524-022-00818-3) to partition site magnetic moments.
- Fitting effective cluster interactions (ECIs) from structures and energies with sparse linear
  regularization methods and model selection methods provided by
  [**sparse-lm**](https://github.com/CederGroupHub/sparse-lm.git),
  except for overlapped group Lasso regularization.
- Checking convergence of cluster expansion model using the minimum energy convergence per composition,
  the cross validation error, and the difference of ECIs (if needed).
- Creating an **atomate2** style workflow to be executed locally or with **Fireworks**.

Installation
------------
1. Install the latest [**smol**](https://github.com/CederGroupHub/smol.git)
   and [**sparse-lm**](https://github.com/CederGroupHub/sparse-lm.git) from repository.
   (Deprecate after **smol**>=0.3.2 and **sparse-lm**>=0.3.2 update).
2. Install WFacer:
    * From pypi: `pip install WFacer`
    * From source: `Clone` the repository. The latest tag in the `main` branch is the stable version of the
code. The `main` branch has the newest tested features, but may have more
lingering bugs. From the top level directory: `pip install .`

Post-installation configuration
------------
Specific configurations are required before you can properly use **WFacer**.

* **Firework** job management is optional but not required.
  To use job management with **Fireworks** and **Atomate2**,
  configuring **Fireworks** and **Atomate2** with your MongoDB storage is necessary.
  Users are advised to follow the guidance in
  [**Atomate2**](https://materialsproject.github.io/atomate2/user/install.html) and
  [**Atomate**](https://atomate.org/installation.html#configure-database-connections-and-computing-center-parameters)
  installation guides, and run a simple [test workflow](https://materialsproject.github.io/atomate2/user/fireworks.html)
  to see if it is able to run on your queue.
* A mixed integer programming (MIP) solver would be necessary when a MIQP based
  regularization method is used. A list of available MIP solvers can be found in
  [**cvxpy** documentations](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver).
  Commercial solvers such as **Gurobi** and **CPLEX** are typically pre-compiled
  but require specific licenses to run on a super-computing system. For open-source solvers,
  the users are recommended to install **SCIP** in a dedicated conda environment following
  the installation instructions in [**PySCIPOpt**](https://github.com/scipopt/PySCIPOpt.git).

Quick example
-------------------------------
A simple workflow to run automatic cluster expansion in a Ag-Li alloy on FCC lattice is as follows
(see other available options in preprocessing documentations.):
```python

from fireworks import LaunchPad
from WFacer.maker import AutoClusterExpansionMaker
from jobflow.managers.fireworks import flow_to_workflow
from pymatgen.core import Structure

# construct a rock salt Ag-Li structure
agli_prim = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=[{"Ag": 0.5, "Li": 0.5},
             {"Ag": 0.5, "Li": 0.5},],
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
```

After running this script, a workflow with the name *"agli_fcc_ce"* should have been added to **Fireworks**'
launchpad.

Submit the workflow to queue using the following command after you have correctly configured **Fireworks**
queue adapter,
```bash
qlaunch rapidfire -m {n_jobs} --sleep {time}
```
where `n_jobs` is the number of jobs you want to keep in queue, and `time` is the amount of sleep
time between two queue submission attempts. `qlaunch` will keep submitting jobs to the queue until
no unfinished job could be found on launchpad.

After the workflow is finished, use the following codes to retrieve the computed results from MongoDB,
(Assume you run the workflow generation script and the following dataloading script
on the same machine, otherwise you will have to figure out which `JOB_STORE` to use!):
```python

from jobflow import SETTINGS
from pydantic import parse_obj_as
from WFacer.schema import CeOutputsDocument

store = SETTINGS.JOB_STORE
store.connect()

# Just a random example. You have to check what is your maximum iteration on your own.
max_iter = 10
# Find the output of a trigger job, which should be the CeOutputDocument of the final
# iteration.
job_return = store.query_one({"name": f"agli_fcc_ce_iter_{max_iter}_trigger"})
raw_doc = job_return["output"]
# De-serialize everything.
doc = parse_obj_as(CeOutputsDocument, raw_doc)

# See WFacer.schema for more.
print("Cluster subspace:", doc.cluster_subspace)
print("Wrangler:", doc.data_wrangler)
print("coefficients:", doc.coefs_history[-1])
```
