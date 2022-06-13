from functools import partial
from typing import List
from typing import NamedTuple

import jax.numpy as jnp
import lcm.grids as grids_module
import numpy as np
from dags.signature import with_signature
from jax.scipy.ndimage import map_coordinates


class VariableInfo(NamedTuple):
    order: List[str]
    sparse_discrete: List[str] = []
    dense_discrete: List[str] = []
    continuous: List[str] = []


def get_precalculated_function_evaluator(grids, varinfo):  # noqa: U100
    """Create a function to look-up and interpolate a function pre-calculated on a grid.

    An example of a pre-calculated function is a value or policy function. These are
    evaluated on the grid of all sparse and dense discrete state variables as well as
    all continuous state variables.

    This function dynamically generates a function that looks up and interpolates values
    of the pre-calculated function. The arguments of the resulting function can be
    split in two categories: 1. Helper arguments such as information about the grid,
    indexer arrays and the pre-calculated values of the function. 2. The original
    arguments of the function that was pre-calculated on the grid. After partialling
    in all helper arguments, the resulting function will completely hide that it
    is not analytical and feel like a normal function. In particular, it can be
    jitted, differentiated and vmapped with jax.

    The resulting function roughly does the following steps:

    - Translate values of discrete variables into positions in a cartesian product
      of all discrete variables.
    - Translate the positions of sparse variables to the position in our internal sparse
      array representation;
    - Index into the ``values_`` array to extract only the part that is relevant for the
      given values of discrete variables.
    - Index into the ``grids_`` array to extract the grid that is relevant for the given
      values of discrete variables.
    - Translate the values of the continuous variables coordinates needed for
      map_coordinates
    - Do the actual interpolation.

    Depending on the grid, only a subset of these steps is relevant. The chosen
    implementation of each step is also adjusted to the type of grid. The order
    in which the functions are called is determined by a DAG.



    Args:


    Returns:
        function:


    """
    pass


def get_indexer_wrapper(indexer_name, axis_order, out_name=None):
    """Create a function tha enables name based lookups in an indexer array.

    Args:
        indexer_name (str): The name of the indexer. This will be the corresponding
            argument name in the wrapper function.
        axis_order (list): List of strings with names for each axis in the indexer.


    Returns:
        callable: A callable with the keyword-only arguments [axis_order] + [name]
            that looks up values in an indexer array called ``name``.

    """

    if out_name is None:
        out_name = f"{indexer_name.replace('_indexer', '')}_pos"

    @with_signature(kwargs=axis_order + [indexer_name])
    def indexer_wrapper(**kwargs):
        positions = tuple(kwargs[var] for var in axis_order)
        arr = kwargs[indexer_name]
        return arr[positions]

    indexer_wrapper.__name__ = out_name

    return indexer_wrapper


def get_discrete_grid_position_finder(grid, in_name, out_name=None):
    """Create a function

    Args:
        grids (jnp.ndarray): 1d jax array with grid values.
        in_name (str): Name of the variable the grid is representing.

    """
    if out_name is None:
        out_name = f"{in_name}_pos"

    if isinstance(grid, (np.ndarray, jnp.ndarray)):
        # tolist converts jax or numpy specific dtypes to python types
        _grid = grid.tolist()
    elif not isinstance(grid, list):
        _grid = list(grid)
    else:
        _grid = grid

    if _grid == list(range(len(grid))):

        @with_signature(kwargs=[in_name])
        def find_discrete_position(**kwargs):
            return kwargs[in_name]

    else:
        val_to_pos = dict(zip(_grid, range(len(grid))))

        @with_signature(kwargs=[in_name])
        def find_discrete_position(**kwargs):
            return val_to_pos[kwargs[in_name]]

    find_discrete_position.__name__ = out_name

    return find_discrete_position

def get_continuous_coordinate_finder(in_name, grid_type, grid_info=None, out_name=None):

    if out_name is None:
        out_name = f"{in_name}_coordinate"

    grid_info = {} if grid_info is None else grid_info

    raw_func = getattr(grids_module, f"get_{grid_type}_coordinate")
    partialled_func = partial(raw_func, **grid_info)

    @with_signature(kwargs=[in_name])
    def find_coordinate(**kwargs):
        return partialled_func(kwargs[in_name])

    find_coordinate.__name__ = out_name

    return find_coordinate


def get_interpolator(value_name, axis_order, map_coordinates_kwargs=None):

    kwargs = {"order": 1, "mode": "nearest"}
    if map_coordinates_kwargs is not None:
        kwargs = {**kwargs, **map_coordinates_kwargs}

    partialled_map_coordinates = partial(map_coordinates, **kwargs)

    @with_signature(kwargs=[value_name] + axis_order)
    def interpolate(**kwargs):
        coordinates = jnp.array([kwargs[var] for var in axis_order])
        out = partialled_map_coordinates(
            input=kwargs[value_name],
            coordinates=coordinates,
        )
        return out

    return interpolate
