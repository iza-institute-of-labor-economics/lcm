import jax.numpy as jnp
import numpy as np
from lcm.solve_brute import contsolve_last_period
from numpy.testing import assert_array_almost_equal as aaae


def test_contsolve_last_period():
    state_choice_space = {
        "value_grid": {"a": jnp.array([0, 1.0]), "b": jnp.array([2, 3.0])},
        "combination_grid": {"c": jnp.array([4, 5, 6])},
    }

    def _utility_and_feasibility(a, c, b, d):
        util = d
        feasib = d <= a + b + c
        return util, feasib

    continuous_choice_grids = {"d": jnp.arange(12)}

    expected = np.array([[[6, 7, 8], [7, 8, 9]], [[7, 8, 9], [8, 9, 10]]])

    calculated = np.array(
        contsolve_last_period(
            state_choice_space,
            _utility_and_feasibility,
            continuous_choice_grids,
        )
    )

    aaae(calculated, expected)
