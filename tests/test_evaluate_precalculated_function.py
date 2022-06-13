from functools import partial

import jax.numpy as jnp
from lcm.dispatchers import productmap
from lcm.evaluate_precalculated_function import get_continuous_coordinate_finder
from lcm.evaluate_precalculated_function import get_discrete_grid_position_finder
from lcm.evaluate_precalculated_function import get_indexer_wrapper
from lcm.evaluate_precalculated_function import get_interpolator


def test_get_indexer_wrapper():
    indexer = jnp.arange(6).reshape(3, 2)
    func = get_indexer_wrapper(
        indexer_name="my_indexer", axis_order=["a", "b"], out_name="state_pos"
    )

    pure_lookup_func = partial(func, my_indexer=indexer)
    calculated = pure_lookup_func(a=2, b=0)
    assert calculated == 4
    assert func.__name__ == "state_pos"


def test_get_discrete_grid_position_finder():
    grid = jnp.array([9, 10, 13])

    func = get_discrete_grid_position_finder(
        grid=grid,
        in_name="schooling",
    )

    assert func(schooling=10) == 1
    assert func.__name__ == "schooling_pos"


def test_get_coordinate_finder():

    find_coordinate = get_continuous_coordinate_finder(
        in_name="wealth",
        grid_type="linspace",
        grid_info={"start": 0, "stop": 10, "n_points": 21},
    )

    calculated = find_coordinate(wealth=5.75)
    assert calculated == 11.5


def test_get_interpolator():

    interpolate = get_interpolator(value_name="vf", axis_order=["wealth", "working"])

    def _utility(wealth, working):
        return 2 * wealth - working

    prod_utility = productmap(_utility, variables=["wealth", "working"])

    values = prod_utility(wealth=jnp.arange(4), working=jnp.arange(3))

    calculated = interpolate(vf=values, wealth=2.5, working=2)

    assert calculated == 3
