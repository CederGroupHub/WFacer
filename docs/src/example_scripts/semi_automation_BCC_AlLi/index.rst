.. _Semi-automate a basic cluster expansion :

=================================
Semi-automate a cluster expansion
=================================

.. note:: In the context of **WFacer** documentation, **semi-automation** refers to manual execution of scripts for structural generation
 and model fitting, while Jobflow or Fireworks are allowed to manage merely the computation
 of each individual enumerated structure.

The following scripts demonstrate how use classes and utility functions to manually perform semi-automated steps
in a *cluster expansion* iteration.

At the beginning first iteration, parameters for the cluster expansion and first-principles calculations
must be *initialized*. The following script provides an example in doing so:

.. literalinclude :: initialize.py
   :language: python

Using the cluster expansion constructed in the last iteration, you can enumerate new structures to
be added in the current iteration and compute them with **atomate2**:

.. literalinclude :: generate.py
   :language: python

In the final step, you would like to refit a cluster expansion model using the updated training set:

.. literalinclude :: fit_model.py
   :language: python
