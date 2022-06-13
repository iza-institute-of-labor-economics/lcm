.. _vectorization:

==============================
Vectorized Function Evaluation
==============================

Challenges
----------

Many functions in lcm (e.g. the user provided utility function) will have to be
evaluated on different collections of inputs:

1. Scalars (e.g. evaluating utility on one state choice combination)
2. Arrays of inputs that contain all values on which the function has to be evaluated
   (e.g. evaluating utility on a grid of possible choices)
3. The cartesian product of a set of grids (e.g. during a grid search)
4. The full state space

Theoretically, 3 and 4 could be reduced to 2 by generating the full array of values
as an array. Naive Approaches would not be possible because of enormous memory
requirements. Batched approaches would be complex.


Main Design Considerations
--------------------------

- All functionality is implemented for the scalar case
- dispatchers such as vmap and functions that build on it are used to achieve the other
  cases.

.. Note:: While one might think that vectorization is only about performance, this is
  absolutely not the case. The dispatchers reduce complexity enormously. Firstly,
  because they are basically a form of encapsulation: The exact representation of the
  state space is irrelevant as long as there is a corresponding dispatcher. Secondly,
  They facilitate the reuse of functions. Thirdly, writing functions for the scalar
  case is much easier and faster.

Current implementation
----------------------


.. automodule:: lcm.dispatchers
    :members: product_map, state_space_map
