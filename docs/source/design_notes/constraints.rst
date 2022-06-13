.. _constraints:

=================================
Constraints on States and choices
=================================

Challenges
----------

When solving a model, there might be the following types of constraints:

- The discrete choice sets vary between states
- Some states are not possible in the model (e.g. having huge wealth in first period)
- Constraints on continuous choices (e.g. budget constraints)

Constraints on the discrete choice sets are typically parameter independent (need to
assert this if we choose design that requires it). All other constraints might depend
on the model parameters. Some constraints might only depend on calibrated parameters
though.

Constraints can only reduce the number of computations but have an ambiguous effect on
memory consumption. However it is always possible to have no extra memory consumption
if we do not try to save computations.


How to handle constraints
-------------------------

- States and discrete choices that are ruled out by constraints that do not depend on
  free parameters can be eliminated once in the beginning. This leads to a higher
  memory consumption for the state_choice space but saves calculations and the memory
  for the value function and/or policies.
- Another way of handling constraints on discrete choices is a masked computation
  on each state. This saves the extra memory requirement for the state_choice space
  but requires more computations. It also requires more memory for everything that is
  defined on a state_choice level, but not more for everything that is only defined on
  the state level.
- For all parameter dependent constraints it is probably best to do masked computations.
  Otherwise there would be a recompilation for each model solution.


GPU vs CPU or JAX vs numba
---------------------------

- On GPU (and thus in JAX in general) we need a static computational graph, i.e array
  shapes cannot vary between runs.
- On GPU the best way of doing reduceat style computations is padding + masking
  (https://github.com/google/jax/issues/2521). On CPU it this would not be the case.
- The successor of ``jax.mask`` should be helpful:
  (https://gist.github.com/mattjj/95438e43c3bf0eef7c3bf0235f314336)
