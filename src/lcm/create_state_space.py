"""Create a state space for a given model."""
import inspect
import warnings
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions
from dags import get_ancestors
from lcm import grids as grids_module
from lcm.dispatchers import gridmap
from lcm.dispatchers import productmap


class Grid(NamedTuple):
    """Representation of a dense or sparse grid compatible with grimap."""

    dense_vars: List[str]
    sparse_vars: List[str]
    grid: Dict[str, jnp.ndarray]


class StateChoiceInfo(NamedTuple):
    """Info to work with a function evaluated on a state-choice grid.

    Provides information to look up values, aggregate values over choice dimensions
    or interpolate values over continuous dimensions.

    """

    state_vars: List[str]
    choice_vars: List[str]
    state_indexer: jnp.ndarray
    choice_indexer: Union[jnp.ndarray, None]
    choice_segments: Union[jnp.ndarray, None]
    gridspecs: Dict[str, Union[Dict, jnp.ndarray]]


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


def create_forward_mask(
    initial, grids, next_functions, fixed_inputs=None, aux_functions=None, jit_next=True
):
    """Create a mask for combinations of grid values.

    Args:
        intitial (dict): Dict of arrays with valid combinations of variables.
        grids (dict): Dictionary containing a one-dimensional grid for each
            variable that is used as a basis to construct the higher dimensional
            grid.
        next_functions (dict): Dict of functions for the state space transitions.
            All keys need to start with "next".
        fixed_inputs (dict): A dict of fixed inputs for the next_functions or
            aux_functions. An example would be a model period.
        aux_functions (dict): Auxiliary functions that calculate derived variables
            needed in the filters.
        jit_next (bool): Whether the aggregated next_function is jitted before
            applying it.

    """
    warnings.warn("This is extremely experimental and probably buggy!")
    # preparations
    _state_vars = [
        name for name in grids if f"next_{name}" in next_functions
    ]  # sort in order of grids
    _aux_functions = {} if aux_functions is None else aux_functions
    _shape = tuple(len(grids[name]) for name in _state_vars)
    _next_functions = {
        f"next_{name}": next_functions[f"next_{name}"] for name in _state_vars
    }
    _fixed_inputs = {} if fixed_inputs is None else fixed_inputs

    # find valid arguments
    valid_args = set(initial) | set(_aux_functions) | set(_fixed_inputs)

    # find next functions with only valid arguments
    _valid_next_functions = {}
    for name, func in _next_functions.items():
        present_args = set(inspect.signature(func).parameters)
        if present_args.issubset(valid_args):
            _valid_next_functions[name] = func

    # create scalar next function
    _next = concatenate_functions(
        functions={**_valid_next_functions, **_aux_functions},
        targets=list(_valid_next_functions),
        return_type="dict",
    )

    # apply dispatcher
    needed_args = set(inspect.signature(_next).parameters)
    _needed_initial = {k: val for k, val in initial.items() if k in needed_args}
    _gridmapped = gridmap(_next, dense_vars=[], sparse_vars=list(_needed_initial))

    # calculate next values
    if jit_next:
        _gridmapped = jax.jit(_gridmapped)
    _next_values = _gridmapped(**_needed_initial, **_fixed_inputs)

    # create all-false mask
    mask = np.full(_shape, False)

    # fill with full slices to get indexers
    indices = []
    for i, var in enumerate(_state_vars):
        name = f"next_{var}"
        if name in _next_values:
            indices.append(_next_values[name])
        else:
            indices.append(slice(0, _shape[i]))

    # set mask to True with indexers
    mask[tuple(indices)] = True

    return mask


def create_combination_grid(grids, masks, subset=None):
    # preparations
    _subset = list(grids) if subset is None else subset
    _axis_names = [name for name in grids if name in _subset]
    _grids = {name: jnp.array(grids[name]) for name in _axis_names}

    # get combined mask
    _mask_np = np.array(_combine_masks(masks))

    # Calculate meshgrid
    _all_combis = jnp.meshgrid(*_grids.values(), indexing="ij")

    # Flatten meshgrid entries
    combi_grid = {name: arr[_mask_np] for name, arr in zip(_axis_names, _all_combis)}

    return combi_grid


def _combine_masks(masks):
    if isinstance(masks, (np.ndarray, jnp.ndarray)):
        _masks = [masks]
    else:
        _masks = sorted(masks, key=lambda x: len(x.shape), reverse=True)

    mask = _masks[0]
    for m in _masks[1:]:
        _shape = tuple(list(m.shape) + [1] * (mask.ndim - m.ndim))
        mask = jnp.logical_and(mask, m.reshape(_shape))
    mask = np.array(mask)
    return mask


def create_indexers_and_segments(mask, n_states, fill_value=-1):
    """Create indexers and segment info related to sparse states and choices.

    Args:
        mask (np.ndarray): Boolean array with one dimension per state
            or choice variable that is True for feasible state-choice
            combinations. The state variables occupy the first dimensions.
            I.e. the shape is (n_s1, ..., n_sm, n_c1, ..., n_cm).
        n_states (np.ndarray): Number of state variables.
        fill_value (np.ndarray): Value of the index array for infeasible
            states or choices.


    Returns:
        jax.numpy.ndarray: The state indexer with (n_s1, ..., n_sm). The entries are
            ``fill_value`` for infeasible states and count the feasible states
            otherwise.
        jax.numpy.ndarray: the state-choice indexer with shape
            (n_feasible_states, n_c1, ..., n_cn). The entries are ``fill_value`` for
            infeasible state-choice combinations and count the feasible state-choice
            combinations otherwise.

    """
    mask = np.array(mask)

    choice_axes = tuple(range(n_states, mask.ndim))
    is_feasible_state = mask.any(axis=choice_axes)
    n_feasible_states = np.count_nonzero(is_feasible_state)

    state_indexer = np.full(is_feasible_state.shape, fill_value)
    state_indexer[is_feasible_state] = np.arange(n_feasible_states)

    # reduce mask before doing calculations and using higher dtypes
    reduced_mask = mask[is_feasible_state]

    counter = reduced_mask.cumsum().reshape(reduced_mask.shape) - 1
    state_choice_indexer = np.full(reduced_mask.shape, fill_value)
    state_choice_indexer[reduced_mask] = counter[reduced_mask]

    new_choice_axes = tuple(range(1, mask.ndim - n_states + 1))
    n_choices = np.count_nonzero(reduced_mask, new_choice_axes)
    segments = np.repeat(np.arange(n_feasible_states), n_choices)

    return (
        jnp.array(state_indexer),
        jnp.array(state_choice_indexer),
        jnp.array(segments),
    )


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
    warnings.warn("Outdated function. Just left here to keep a test running.")
    if subset or filters:
        # to-do: create state space and apply filters
        raise NotImplementedError()
    else:
        out = {}
    return out


def _create_value_grid(grids, subset):
    return {name: grid for name, grid in grids.items() if name in subset}
