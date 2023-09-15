WorkFlow for Automated Cluster Expansion Regression (WFacer)
===================================================

*Modulated automation of cluster expansion model construction based on atomate2 and Jobflow*

-----------------------------------------------------------------------------

**WFacer** ("Wall"Facer) is a light-weight package based on [smol](https://github.com/CederGroupHub/smol.git)
to automate the fitting of lattice models in disordered crystalline solids using
*cluster expansion* method. Beyond metallic alloys, **WFacer** is also designed
to handle ionic systems through enabling charge **Decorator** and the external **EwaldTerm**. Powered by [Atomate2](https://github.com/materialsproject/atomate2.git),
[Jobflow](https://github.com/materialsproject/jobflow.git)
and [Fireworks](https://github.com/materialsproject/fireworks.git), **WFacer** is able to fully automate the
cluster expansion building process on super-computing clusters, and can easily interface
with MongoDB data storage in the **Materials Project** style .

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
  or from [a gaussian optimization process](https://doi.org/10.1038/s41524-022-00818-3) based on partitioning site magnetic moments.
- Fitting effective cluster interactions (ECIs) from structures and energies with sparse linear
  regularization methods and model selection methods provided by
  [sparse-lm](https://github.com/CederGroupHub/sparse-lm.git),
  except for overlapped group Lasso regularization.
- Checking convergence of cluster expansion model using the minimum energy convergence per composition,
  the cross validation error, and the difference of ECIs (if needed).
- Creating an **atomate2** style workflow to be executed locally or with **Fireworks**.

Installation
------------
* From pypi: `pip install WFacer`
* From source: `Clone` the repository. The latest tag in the `main` branch is the stable version of the
code. The `main` branch has the newest tested features, but may have more
lingering bugs. From the top level directory, do `pip install -r requirements.txt`, then `pip install .` If
you wish to use **Fireworks** as the workflows manager, do `pip install -r requirements-optional.txt` as well.

Post-installation configuration
------------
Specific configurations are required before you can properly use **WFacer**.

* **Fireworks** job management is highly recommended but not required.
  To use job management with **Fireworks** and **Atomate2**,
  configuring **Fireworks** and **Atomate2** with your MongoDB storage is necessary.
  Users are advised to follow the guidance in
  [**Atomate2**](https://materialsproject.github.io/atomate2/user/install.html) and
  [**Atomate**](https://atomate.org/installation.html#configure-database-connections-and-computing-center-parameters)
  installation guides, and run a simple [test workflow](https://materialsproject.github.io/atomate2/user/fireworks.html)
  to see if it is able to run on your queue.

  Instead of writing in **my_qadapter.yaml** as
  ```commandline
     rlaunch -c <<INSTALL_DIR>>/config rapidfire
  ```
  we suggest using:
  ```commandline
     rlaunch -c <<INSTALL_DIR>>/config singleshot
  ```
  instead, because by using *singleshot* with rlaunch, a task in the submission queue will
  be terminated once a structure is finished instead of trying to fetch another structure
  from the launchpad. This can be used in combination with:
  ```commandline
     qlaunch rapidfire -m <number of tasks to keep in queue>
  ```
  to guarantee that each structure is able to use up the maximum wall-time in
  its computation.

* A mixed integer programming (MIP) solver would be necessary when a MIQP based
  regularization method is used. A list of available MIP solvers can be found in
  [cvxpy documentations](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver).
  Commercial solvers such as **Gurobi** and **CPLEX** are typically pre-compiled
  but require specific licenses to run on a super-computing system. For open-source solvers,
  the users are recommended to install **SCIP** in a dedicated conda environment following
  the installation instructions in [PySCIPOpt](https://github.com/scipopt/PySCIPOpt.git).

A quick example for fully automating cluster expansion
-------------------------------
A simple workflow to run automatic cluster expansion in a Ag-Li alloy on FCC lattice is as follows
(see other available options in the documentations of [*preprocessing.py*](WFacer/preprocessing.py).):
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

Submit the workflow to queue using the following command once you have correctly configured **Fireworks**
queue adapter,
```bash
nohup qlaunch rapidfire -m {n_jobs} --sleep {time} > qlaunch.log
```
where `n_jobs` is the number of jobs you want to keep in queue, and `time` is the amount of sleep
time between two queue submission attempts. `qlaunch` will keep submitting jobs to the queue until
no unfinished job could be found on launchpad.

>Note: You may still need to qlaunch manually after every cluster expansion iteration
 because for Fireworks could occasionally set the enumeration job to the READY state
 but fails to continue executing the job.

After the workflow is finished, use the following codes to retrieve the computed results from MongoDB:
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
>Note: Check that the **Jobflow** installations on the computer cluster and the query
 terminal are configured to use the same **JOB_STORE**.

License
-------------------------------
    Workflows For Automated Cluster Expansion Regression (WFACER) Copyright (c) 2023,
    The Regents of the University of California, through Lawrence Berkeley National
    Laboratory (subject to receipt of any required approvals from the U.S.
    Dept. of Energy) and the University of California, Berkeley. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    (1) Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    (3) Neither the name of the University of California, Lawrence Berkeley
    National Laboratory, U.S. Dept. of Energy, University of California,
    Berkeley nor the names of its contributors may be used to endorse or
    promote products derived from this software without specific prior written
    permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    You are under no obligation whatsoever to provide any bug fixes, patches,
    or upgrades to the features, functionality or performance of the source
    code ("Enhancements") to anyone; however, if you choose to make your
    Enhancements available either publicly, or directly to Lawrence Berkeley
    National Laboratory, without imposing a separate written license agreement
    for such Enhancements, then you hereby grant the following license: a
    non-exclusive, royalty-free perpetual license to install, use, modify,
    prepare derivative works, incorporate into other computer software,
    distribute, and sublicense such enhancements or derivative works thereof,
    in binary and source code form.
