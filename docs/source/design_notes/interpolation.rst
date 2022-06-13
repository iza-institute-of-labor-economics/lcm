================================
Function interpolation and Grids
================================

Challenges
----------

Function interpolation plays an important role during the solution of any life cycle
model with continuous choices. However, there are not general functions for fast
linear interpolation in arbitrary dimensions.


The consav package implements interpolation functions for up to four dimensions.


Main Design Considerations
--------------------------

- Need functions that work for arbitrary number of dimensions
- Need ability to exploit special grids structures such as linspace, logspace, ...


Main Insight
------------

Interpolation can be decomposed into two steps.

1. Mapping the interpolation problem into a "similar" interpolation on a Cartesian
   grid, where the distance between grid points is 1 in all dimensions. This step
   contains everything that might exploit certain grid structures. This should also
   work if the spacing is not uniform in any dimension because linear interpolation only
   looks at a very small part of the value grid.
2. Doing the actual interpolation on that grid.

A high performance implementation of step 2 is ``jax.scipy.ndimage.map_coordinates``


How to Convert the Interpolation Problem for ``map_coordinates``.
-----------------------------------------------------------------

Example:

- ``grid = [0.5, 0.75, 1, 1.25]``
- ``point = 0.8``

The index of the next smaller grid point is 1.
The points relative position between the two neighbouring gridpoints is
``(0.8 - 0.75) / (1 - 0.75) = 0.05 / 0.25 = 0.2``

Converted point: 1.2 = index + relative distance from lower point to interpolation point.
