.. _ full_automation :

=================
Fully automate a cluster expansion
=================

We provide a simple example workflow to run automatic cluster expansion in a Ag-Li alloy on FCC lattice
(see other available options in the documentations of *preprocessing.py*.):

.. literalinclude :: generate_wf.py
   :language: python

After running this script, a workflow with the name **agli_fcc_ce** should have appeared on **Fireworks**'
launchpad.

Make sure you have correctly configured **Fireworks**, **Jobflow** and **atomate2**,
then submit the workflow to computing cluster by running the following command,

.. code-block:: bash

  nohup qlaunch rapidfire -m {n_jobs} --sleep {time} > qlaunch.log

where *n_jobs* is the number of jobs you want to keep in queue, and *time* is the amount of sleep
time in seconds between two queue submission attempts.
*qlaunch* will keep submitting jobs to the queue until no job in the **READY** state could be found
on launchpad.

.. note:: You may still need to qlaunch manually after every cluster expansion iteration
 because for Fireworks could occasionally set the enumeration job to the READY state
 but fails to continue executing the job.

After finishing, use the following code to query the computation results from MongoDB,

.. note:: Check that the **Jobflow** installations on the computer cluster and the query
 terminal are configured to use the same **JOB_STORE**.

.. literalinclude :: analyze_wf.py
   :language: python
