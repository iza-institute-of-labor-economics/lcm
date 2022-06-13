.. _solvers:

=================================
The (continuous) solver interface
=================================

Challenges
----------

Need for standardization
^^^^^^^^^^^^^^^^^^^^^^^^

Solvers need to be standardized enough such that:

- They can be switched out easily (like an optimizer in estimagic)
- A researcher who develops a new solution method only has to implement one
  the new solver but does not have to make any changes in the likelihood or
  simulate function.
- The backwards induction loop and the forward passes during the simulation or
  likelihood calculation can be implemented once in lcm and do not have to be
  modified for different solvers.

Need for freedom
^^^^^^^^^^^^^^^^

- Different solvers want to make very different trade-offs of memory usage vs.
  redundant calculations. Solvers that optimize memory requirements will store as few
  things as possible on the state or state_choice space level. For example, they might
  only generate the value function on the states grid and everything else
  (e.g. the optimal policies) need to be re-calculated during simulation or likelihood.
- Different solvers require different information to be pre-calculated during the
  backwards induction. While a naive solver only needs the value function on the
  states grid, EGM based methods also require the optimal policies. To save memory
  we do not want redundant information to be required anywhere.

Main Design Considerations
--------------------------

- We do not want to standardize what a solver needs to return but only standardize
  the formats in which things need to be stored in case a solver wants to return them.

Terminology
-----------

- **solver**: The minimal thing that needs to be implemented to solve the continuous
  choice problem, i.e. to calculate all things that depend on continuous choices
  during backwards induction, simulation and likelihood.
- **continuation_values**: The continuation value of a state is the value of being in
  that state, given that all future choices will be optimal. The continuation values
  are defined on the state space, not the state choice space. They are the minimal
  necessary information and solver needs to provide because they are required for
  all kinds of backward induction algorithms.
- **value_function**: Combination of current period utility and discounted continuation
  values. This is defined on the state choice grid.
- **policy**: The optimal continuous choice for a given state and discrete choice.
  Some solvers might need this information during backwards induction. If so, it
  needs to be calculated and stored for the full state_choice grid. Otherwise it
  will be calculated but not stored during the backwards induction.
- **backward_pass**: Applying the solver while going backwards during the state space.
  This is necessary during the backward induction.
- **forward_pass**: Applying the solver while going forward over a collection of states.
  This is necessary during the likelihood calculation or simulation.


Extreme Cases
-------------

Memory Optimized Solver
^^^^^^^^^^^^^^^^^^^^^^^

- Only store the continuation values for the full state space during backwards induction
- Recalculate optimal policies during simulation and likelihood

Compute Optimized Solver
^^^^^^^^^^^^^^^^^^^^^^^^

- Calculate continuation values and policies during backwards induction
- Only do lookups during simulation and likelihood.

Potential Outputs of solvers
----------------------------

- **continuation_values**: n-dimensional array with continuation values. The number and
  order of dimensions is defined by the ``state_space_map`` function.
- **emax_inputs**: ...
- **policies**: n-dimensional array with optimal policies. The number and order of
  dimensions is defined by the ``state_choice_space_map`` function.
- **grids**: the grids that can be used to construct the cartesian grid on which the
  continuation valuers are defined. Only necessary if a solver actually decides which
  grid to use (e.g. for endogenous grid methods or solvers who add a few special
  points around discontinuities)

.. Note: Our requirements for the solver interface might be so general that it would
  not even be necessary to standardize all outputs. We would still want to do it because
  it makes our lives easier during simulation and likelihood estimation.

Implementation 1: Monolithic
----------------------------

- A solver is one function
- The return of the solver is a dictionary that contains everything that needs to be
  passed into the solver again in the next period of the backward induction loop.
- The arguments are:
    - the keys of the return dictionary (if mode == "backward")
    - the state/choice variables
- mode which is "forward" or "backward" and determines what is part of the
  returned dictionary.


Implementation 2: DAG
---------------------

- A solver consists of one function per output it can produce, i.e. at least a function
  to calculate ``policies`` and a function to calculate ``continuation_value`` and
  potentially a function that calculates ``grid``.
- The functions can depend on each other in arbitrary ways as long as the dependency
  structure forms a DAG.
- There can be any number of auxiliary functions that can be used if intermediate
  outputs are needed in several steps. In that case they will only be calculated
  once.
- The mode argument becomes irrelevant because we can determine which outputs we need.
- The signature of the concatenated function that results when the target is equal
  to ``continuation_value`` will be used to determine which pieces of information need
  to be passed back in time.


Discussion
----------

Pro Monolithic:
^^^^^^^^^^^^^^^

- Closer to how people think about solvers
- Result is equally general
- A solver is one function

Pro DAG
^^^^^^^

- Probably a bit more code-reuse when we implement our own function
- No if condition in functions that need to be fast because there is no mode argument.
  On the other hand, this if condition it compile time constant and will probably
  be eliminated by jax anyways.
- I have personally never had an experience where I regretted solving something with
  a dag.
- Better error messages


state_solvers vs. state_choice solvers
--------------------------------------

- A state solver would be a solver that is dispatched over the state space and looks
  at all discrete choices that are possible in that state internally. It cannot produce
  any output at the state_choice level unless all choice variables are ``simple``.
  It's main advantage would be memory efficiency. The memory efficiency does not only
  come from not storing things on state_choice level but also from the fact that no
  state_choice_space is constructed and thus no state_choice_indexer is needed.
- A state_choice solver would be a solver that is dispatched over the state choice
  space. It can only produce outputs on the state_choice level that can be aggregated
  later if necessary. Calculating the Emax would be a separate step after those solvers.
- We definitely need state_solvers because memory is the more stringent resource
  constraint. Whether we even need state_choice_solvers boils down to two questions:
  1. How many calculations can be saved by reducing the choice sets as part of the
  state space construction.
  2. How much memory is saved by storing state_choice level data only for feasible
  choices and not for all choices.
- Since a similar and probably bigger waste of computations is probably incurred anyways
  for the way we implement budget constraints, I currently tend towards no static
  reduction of choice sets, no state_choice_space and thus no state_choice solvers
  (at least in the beginning).
