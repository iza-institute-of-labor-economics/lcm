.. _dag:

=====================================
Use of Directed Acyclic Graphs (DAGs)
=====================================


Challenges
----------

User provide many functions that might depend on each other. We need to evaluate
functions and their dependencies on the state space or other inputs.

Main design considerations
--------------------------

- We do not need just a dag scheduler as in gettsim but a higher order function that
  creates a function that represents the dag. This function needs to be jax jittable
  and have a signature that can be read via the inspect module. Moreover, it needs to
  work with vmap and our dispatchers.


Current implementation
----------------------

.. automodule:: lcm.dag
    :members: concatenate_functions


Highlights
----------

- The dag function cannot only be used to construct efficient functions based on
  arbitrary building blocks, but generate several versions of such a function,
  for example one that returns intermediate outputs and one that does not.
