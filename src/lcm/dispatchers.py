import functools
import inspect

from jax import vmap


def gridmap(func, dense_vars, sparse_vars, dense_first=True):
    """Apply vmap such that func is evaluated on a grid of dense and sparse variables.

    This is achieved by applying a product map for all dense_vars and a vmap for the
    sparse_vars.

    In contrast to vmap, gridmap preserves the function signature and allows the
    function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        dense_vars (list): Names of the dense variables, i.e. those that are simply
            stored as array of possible values in the grid because the possible values
            of those variable does not depend on the value of other variables.
        sparse_vars (list): Names of the sparse variables, i.e. those that are stored
            as arrays of possible combinations of variables in the grid because the
            possible values of these variables does depend on the value of other
            variables.
        dense_first (bool): Whether the dense or sparse dimensions should come first
            in the output of the dispatched function.


    Returns:
        callable: A callable with the same arguments as func (but with an additional
            leading dimension) that returns a jax.numpy.ndarray or pytree of arrays.
            If ``func`` returns a scalar, the dispatched function returns a
            jax.numpy.ndarray with k + 1 dimensions, where k is the length of
            ``dense_vars`` and the additional dimension corresponds to the
            ``sparse_vars``. The order of the dimensions is determined by the order of
            ``dense_vars`` as well as the ``dense_first`` argument.
            If the output of ``func`` is a jax pytree, the usual jax behavior applies,
            i.e. the leading dimensions of all arrays in the pytree are as described
            above but there might be additional dimensions.

    """
    if not set(dense_vars).isdisjoint(sparse_vars):
        raise ValueError("dense_vars and sparse_vars overlap.")

    _all_variables = dense_vars + sparse_vars
    if len(_all_variables) != len(set(_all_variables)):
        raise ValueError("Same argument provided more than once.")

    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    positions = [parameters.index(cv) for cv in sparse_vars]
    in_axes = []
    for i in range(len(parameters)):
        if i in positions:
            in_axes.append(0)
        else:
            in_axes.append(None)

    if dense_first:
        vmapped = vmap(func, in_axes=in_axes)
        vmapped = _product_map(vmapped, dense_vars)
    else:
        vmapped = _product_map(func, dense_vars)
        vmapped = vmap(vmapped, in_axes=in_axes)

    vmapped.__signature__ = signature
    vmapped_with_kwargs = allow_kwargs(vmapped)

    return vmapped_with_kwargs


def productmap(func, variables):
    """Apply vmap such that func is evaluated on the cartesian product of product_axes.

    This is achieved by an iterative application of vmap.

    In contrast to vmap, productmap preserves the function signature and allows the
    function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        variables (list): List with names of arguments that over which the cartesian
            product should be formed.

    Returns:
        callable: A callable with the same arguments as func (but with an additional
            leading dimension) that returns a jax.numpy.ndarray or pytree of arrays.
            If ``func`` returns a scalar, the dispatched function returns a
            jax.numpy.ndarray with k dimensions, where k is the length of ``variables``.
            The order of the dimensions is determined by the order of ``variables``
            which can be different to the order of ``funcs`` arguments. If the output of
            ``func`` is a jax pytree, the usual jax behavior applies, i.e. the leading
            dimensions of all arrays in the pytree are as described above but there
            might be additional dimensions.

    """
    if len(variables) != len(set(variables)):
        raise ValueError("Same argument provided more than once.")

    signature = inspect.signature(func)
    vmapped = _product_map(func, variables)
    vmapped.__signature__ = signature
    vmapped_with_kwargs = allow_kwargs(vmapped)

    return vmapped_with_kwargs


def _product_map(func, product_axes):
    """Do actual product map without signature changes."""
    signature = inspect.signature(func)
    parameters = list(signature.parameters)
    if product_axes is None:
        product_axes = parameters

    positions = [parameters.index(ax) for ax in product_axes]

    vmap_specs = []
    for pos in reversed(positions):
        spec = [None] * len(parameters)
        spec[pos] = 0
        vmap_specs.append(spec)

    vmapped = func
    for spec in vmap_specs:
        vmapped = vmap(vmapped, in_axes=spec)

    return vmapped


def allow_kwargs(func):
    @functools.wraps(func)
    def allow_kwargs_wrapper(*args, **kwargs):
        parameters = list(inspect.signature(func).parameters)

        positional = list(args) if args is not None else []

        kwargs = {} if args is None else kwargs
        if len(args) + len(kwargs) != len(parameters):
            raise ValueError("Not enough or too many arguments provided.")

        positional += convert_kwargs_to_args(kwargs, parameters)

        return func(*positional)

    return allow_kwargs_wrapper


def convert_kwargs_to_args(kwargs, parameters):
    sorted_kwargs = dict(sorted(kwargs.items(), key=lambda kw: parameters.index(kw[0])))
    args = list(sorted_kwargs.values())
    return args
