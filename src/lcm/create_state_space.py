"""Create a state space for a given model."""
import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions
from dags import get_ancestors
from lcm import grids as grids_module
from lcm.dispatchers import productmap


def create_state_choice_space(model):
    """Create a state choice space for the model.

    A state_choice_space is a compressed representation of all feasible states and the
    feasible choices within that state. We currently use the following compressions:

    We distinguish between dense and sparse variables (dense_vars and sparse_vars).
    Dense state or choice variables are those whose set of feasible values does not
    depend on any other state or choice variables. Sparse state or choice variables are
    all other state variables. For dense state variables it is thus enough to store the
    grid of feasible values (value_grid), whereas for sparse variables all feasible
    combinations (combination_grid) have to be stored.

    """
    dense_vars, sparse_vars = _find_dense_and_sparse_variables(model)
    grids = _create_grids_from_gridspecs(model)

    space = {
        "combination_grid": _create_combination_grid(
            grids, sparse_vars, model.get("filters", [])
        ),
        "value_grid": _create_value_grid(grids, dense_vars),
    }
    return space


def create_filter_mask(
    grids, filters, fixed_inputs=None, subset=None, aux_functions=None, jit_filter=True
):
    """Create mask for combinations of grid values that is True if all filters are True.

    Args:
        grids (dict): Dictionary containing a one-dimensional grid for each
            variable that is used as a basis to construct the higher dimensional
            grid.
        filters (dict): Dict of filter functions. A filter function depends on
            one or more variables and returns True if a state is feasible.
        fixed_inputs (dict): A dict of fixed inputs for the filters or
            aux_functions. An example would be a model period.
        subset (list): The subset of variables to be considered in the mask.
        aux_functions (dict): Auxiliary functions that calculate derived variables
            needed in the filters.
        jit_filter (bool): Whether the aggregated filter function is jitted before
            applying it.

    Returns:
        jax.numpy.ndarray: Multi-Dimensional boolean array that is True
            for a feasible combination of variables. The order of the
            dimensions in the mask is defined by the order of `grids`.

    """
    # preparations
    _subset = list(grids) if subset is None else subset
    _aux_functions = {} if aux_functions is None else aux_functions
    _axis_names = [name for name in grids if name in _subset]
    _grids = {name: jnp.array(grids[name]) for name in _axis_names}
    _filter_names = list(filters)

    # Create scalar dag function to evaluate all filters
    _functions = {**filters, **_aux_functions}
    _scalar_filter = concatenate_functions(
        functions=_functions,
        targets=_filter_names,
        aggregator=jnp.logical_and,
    )

    # Apply dispatcher to get mask
    _filter = productmap(_scalar_filter, variables=_axis_names)

    # Calculate mask
    if jit_filter:
        _filter = jax.jit(_filter)
    mask = _filter(**_grids, **fixed_inputs)

    return mask


def _find_dense_and_sparse_variables(model):
    state_variables = list(model["states"])
    discrete_choices = [
        name for name, spec in model["choices"].items() if "options" in spec
    ]
    all_variables = set(state_variables + discrete_choices)

    filtered_variables = {}
    filters = model.get("state_filters", [])
    for func in filters:
        filtered_variables = filtered_variables.union(
            get_ancestors(filters, func.__name__)
        )

    dense_vars = all_variables.difference(filtered_variables)
    sparse_vars = all_variables.difference(dense_vars)
    return dense_vars, sparse_vars


def _create_grids_from_gridspecs(model):
    gridspecs = {
        **model["choices"],
        **model["states"],
    }
    grids = {}
    for name, spec in gridspecs.items():
        if "options" in spec:
            grids[name] = np.array(spec["options"])
        else:
            spec = spec.copy()
            func = getattr(grids_module, spec.pop("grid_type"))
            grids[name] = func(**spec)

    return grids


def _create_combination_grid(grids, subset, filters):  # noqa: U100
    """Create the ore state choice space.

    Args:
        grids (dict): Dictionary of grids for all variables in the model
        sparse_vars (set): Names of the sparse_variables
        filters (list): List of filter functions. A filter function depends on one or
            more variables and returns True if a state is feasible.

    Returns:
        dict: Dictionary of arrays where each array represents a column of the core
            state_choice_space.

    """
    if subset or filters:
        # to-do: create state space and apply filters
        raise NotImplementedError()
    else:
        out = {}
    return out


def _create_value_grid(grids, subset):
    return {name: grid for name, grid in grids.items() if name in subset}
