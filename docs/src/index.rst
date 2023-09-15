.. WFacer documentation master file, created by
   sphinx-quickstart on Thu Sep 14 13:17:04 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root toctree directive.

:notoc:

.. toctree::
   :maxdepth: 1
   :hidden:

   examples
   contributing


=================
WorkFlow for Automated Cluster Expansion Regression (WFacer)
=================

*Modulated automation of cluster expansion model construction based on atomate2 and Jobflow*

**************

WFacer ("Wall"Facer) is a light-weight package based on `smol <https://github.com/CederGroupHub/smol.git>`_
to automate the fitting of lattice models in disordered crystalline solids using
*cluster expansion* method. Beyond metallic alloys, **WFacer** is also designed
to handle ionic systems through enabling charge **Decorator** and the **EwaldTerm**.
Powered by `Atomate2 <https://github.com/materialsproject/atomate2.git>`_,
`Jobflow <https://github.com/materialsproject/jobflow.git>`_
and `Fireworks <https://github.com/materialsproject/fireworks.git>`_, **WFacer** is able to fully automate the
cluster expansion building process on super-computing clusters, and can easily interface
with MongoDB data storage in the **Materials Project** style.

Functionality
-------------
**WFacer** currently offers the following functionalities:

- Preprocess setup to a cluster expansion workflow as dictionary.
- Enumerating and choosing the least aliasing super-cell matrices with given number of sites;
  enumerating charge balanced compositions in super-cells; Enumerating and selecting low-energy,
  non-duplicated structures into the training set at the beginning of each iteration.
- Computing enumerated structures using **Atomate2** **VASP** interfaces.
- Extracting and saving relaxed structures information and energy in **Atomate2** schemas.
- Decorating structures. Currently, supports charge decoration from fixed labels, from Pymatgen guesses,
  or from `a gaussian optimization process <https://doi.org/10.1038/s41524-022-00818-3>`_ based on partitioning
  site magnetic moments.
- Fitting effective cluster interactions (ECIs) from structures and energies with sparse linear
  regularization methods and model selection methods provided by
  `sparse-lm <https://github.com/CederGroupHub/sparse-lm.git>`_,
  except for overlapped group Lasso regularization.
- Checking convergence of cluster expansion model using the minimum energy convergence per composition,
  the cross validation error, and the difference of ECIs (if needed).
- Creating an **atomate2** style workflow to be executed locally or with **Fireworks**.

Installation
------------
* From pypi: :code:`pip install WFacer`
* From source: :code:`git clone` the repository. The latest tag in the *main* branch is the stable version of the
  code. The **main** branch has the latest tested features, but may have more
  lingering bugs. From the top level directory, do :code:`pip install -r requirements.txt`, then :code:`pip install .` If
  you wish to use **Fireworks** as the calculation manager, do :code:`pip install -r requirements-optional.txt` as well.

Post-installation configuration
------------
Specific configurations are required before you can properly use **WFacer**.

* **Fireworks** job management is highly recommended but not required.
  To use job management with **Fireworks** and **Atomate2**,
  configuring **Fireworks** and **Atomate2** with your MongoDB storage is necessary.
  Users are advised to follow the guidance in
  **Atomate2** and `Atomate <https://atomate.org/installation.html#configure-database-connections-and-computing-center-parameters>`_
  installation guides, and run a simple `test workflow <https://materialsproject.github.io/atomate2/user/fireworks.html>`_
  to see if it is able to run on your queue.

  Instead of writing in **my_qadapter.yaml** as

  .. code-block:: bash

     rlaunch -c <<INSTALL_DIR>>/config rapidfire

  we suggest using:

  .. code-block:: bash

     rlaunch -c <<INSTALL_DIR>>/config singleshot

  instead, because by using **singleshot** within **rlaunch**, a task in the submission queue will
  be terminated once a structure is finished instead of trying to fetch another structure
  from the launchpad. This can be used in combination with:

  .. code-block:: bash

     qlaunch rapidfire -m <number of tasks to keep in queue>

  to guarantee that each structure is able to use up the maximum wall-time in
  its computation.

* A mixed integer programming (MIP) solver would be necessary when a MIQP based
  regularization method is used. A list of available MIP solvers can be found in
  `cvxpy <https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver>`_ documentations.
  Commercial solvers such as **Gurobi** and **CPLEX** are typically pre-compiled
  but require specific licenses to run on a super-computing system. For open-source solvers,
  the users are recommended to install **SCIP** in a dedicated conda environment following
  the installation instructions in `PySCIPOpt <https://github.com/scipopt/PySCIPOpt.git>`_.

Examples
------------
See :ref:`examples` for some typical use cases.

License
-------------------------------

**WFacer** is distributed openly under a modified 3-clause BSD licence.

.. include:: ../../LICENSE
