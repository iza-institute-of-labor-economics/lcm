import jax.numpy as jnp
import numpy as np
import pytest
from lcm.create_state_space import create_combination_grid
from lcm.create_state_space import create_filter_mask
from lcm.create_state_space import create_forward_mask
from lcm.create_state_space import create_indexers_and_segments
from lcm.create_state_space import create_state_choice_space
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS
from numpy.testing import assert_array_almost_equal as aaae


def test_create_state_choice_space_only_simple_variables():
    calculated = create_state_choice_space(PHELPS_DEATON_WITH_SHOCKS)
    expected_value_grid = {
        "retirement": jnp.array([0, 1]),
        "wealth": jnp.linspace(0, 1e4, 11),
    }

    assert calculated["combination_grid"] == {}
    assert set(calculated["value_grid"]) == set(expected_value_grid)
    for key, grid in calculated["value_grid"].items():
        aaae(grid, expected_value_grid[key])


@pytest.fixture()
def filter_mask_inputs():
    def age(period):
        return period + 18

    def mandatory_retirement_filter(retirement, age):
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_filter(lagged_retirement, age):
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_filter(retirement, lagged_retirement):
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    filters = {
        "mandatory_retirement": mandatory_retirement_filter,
        "mandatory_lagged_retirement": mandatory_lagged_retirement_filter,
        "absorbing_retirement": absorbing_retirement_filter,
    }

    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    out = {"filters": filters, "grids": grids, "aux_functions": {"age": age}}

    return out


PARAMETRIZATION = [
    (50, jnp.array([[False, False], [False, True]])),
    (10, jnp.array([[True, True], [False, True]])),
]


@pytest.mark.parametrize("period, expected", PARAMETRIZATION)
def test_create_filter_mask(filter_mask_inputs, period, expected):

    calculated = create_filter_mask(
        fixed_inputs={"period": period}, **filter_mask_inputs
    )

    aaae(calculated, expected)


def test_create_combination_grid():
    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    mask = jnp.array([[True, False], [True, True]])

    calculated = create_combination_grid(grids=grids, masks=mask)

    expected = {
        "lagged_retirement": jnp.array([0, 1, 1]),
        "retirement": jnp.array([0, 0, 1]),
    }

    for key in expected:
        aaae(calculated[key], expected[key])


def test_create_combination_grid_2_masks():
    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    masks = [
        jnp.array([[True, False], [True, True]]),
        jnp.array([[True, True], [False, True]]),
    ]

    calculated = create_combination_grid(grids=grids, masks=masks)

    expected = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    for key in expected:
        aaae(calculated[key], expected[key])


def test_create_combination_grid_multiple_masks():
    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    masks = [
        jnp.array([[True, False], [True, True]]),
        jnp.array([[True, False], [True, True]]),
        jnp.array([[True, True], [False, True]]),
    ]

    calculated = create_combination_grid(grids=grids, masks=masks)

    expected = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    for key in expected:
        aaae(calculated[key], expected[key])


def test_create_forward_mask():
    """We use the following simplified test case (that does not make economic sense).

    - People can stay at home (work=0), work part time (work=1) or full time (work=2)
    - Experience is measured in work units
    - Initial experience is only [0, 1] but in total one can accumulate up to 6 points
    - People can only work full time if they have no previous work experience
    - People have to work at least part time if they have no previous experience

    """
    grids = {
        "experience": jnp.arange(6),
        "working": jnp.array([0, 1, 2]),
    }

    initial = {
        "experience": jnp.array([0, 0, 1, 1]),
        "working": jnp.array([1, 2, 0, 1]),
    }

    def next_experience(experience, working):
        return experience + working

    calculated = create_forward_mask(
        initial=initial,
        grids=grids,
        next_functions={"next_experience": next_experience},
        jit_next=True,
    )

    expected = jnp.array([False, True, True, False, False, False])

    aaae(calculated, expected)


def test_create_forward_mask_multiple_next_funcs():
    """We use another simple example.

    - People can stay at home (work=0), work part time (work=1) or full time (work=2)
    - Experience is measured in work units
    - Initial experience is only [0, 1] but in total one can accumulate up to 6 points
    - People can only work full time if they have no previous work experience
    - People have to work at least part time if they have no previous experience
    - People get bad health after they have more than 1 experience

    """
    grids = {
        "experience": jnp.arange(6),
        "working": jnp.array([0, 1, 2]),
        "health": jnp.array([0, 1]),
    }

    initial = {
        "experience": jnp.array([0, 0, 1, 1]),
        "working": jnp.array([1, 2, 0, 1]),
        "health": jnp.array([0, 0, 0, 0]),
    }

    def next_experience(experience, working):
        return experience + working

    def next_health(experience, working):
        return ((experience + working) > 1).astype(int)

    calculated = create_forward_mask(
        initial=initial,
        grids=grids,
        next_functions={"next_experience": next_experience, "next_health": next_health},
        jit_next=True,
    )

    expected = jnp.array(
        [
            [False, False],
            [True, False],
            [False, True],
            [False, False],
            [False, False],
            [False, False],
        ]
    )

    aaae(calculated, expected)


def test_forward_mask_w_aux_function():
    """We use another simple example.

    - People can stay at home (work=0), work part time (work=1) or full time (work=2)
    - Experience is measured in work units
    - Initial experience is only [0, 1] but in total one can accumulate up to 6 points
    - People have to work at least part time if they have no previous experience
    - People get bad health after they work full time.
    - In bad health additional work experience does not add anything to experience.


    """
    grids = {
        "experience": jnp.arange(6),
        "working": jnp.array([0, 1, 2]),
        "health": jnp.array([0, 1]),
    }

    initial = {
        "experience": jnp.array([0, 0, 1, 1, 2]),
        "working": jnp.array([1, 2, 0, 1, 2]),
        "health": jnp.array([0, 0, 1, 1, 1]),
    }

    def healthy_working(health, working):
        return jnp.where(health == 0, working, 0)

    def next_experience(experience, healthy_working):
        return experience + healthy_working

    def next_health(working):
        return (working == 2).astype(int)

    calculated = create_forward_mask(
        initial=initial,
        grids=grids,
        next_functions={"next_experience": next_experience, "next_health": next_health},
        aux_functions={"healthy_working": healthy_working},
        jit_next=False,
    )

    expected = jnp.array(
        [
            [False, False],
            [True, False],
            [False, True],
            [False, False],
            [False, False],
            [False, False],
        ]
    )

    aaae(calculated, expected)


def test_create_indexers_and_segments():
    mask = np.full((3, 3, 2), False)
    mask[1, 0, 0] = True
    mask[1, -1, -1] = True
    mask[2] = True
    mask = jnp.array(mask)

    state_indexer, choice_indexer, segments = create_indexers_and_segments(
        mask=mask, n_states=2
    )

    expected_state_indexer = jnp.array([[-1, -1, -1], [0, -1, 1], [2, 3, 4]])

    expected_choice_indexer = jnp.array([[0, -1], [-1, 1], [2, 3], [4, 5], [6, 7]])

    expected_segments = jnp.array([0, 1, 2, 2, 3, 3, 4, 4])

    aaae(state_indexer, expected_state_indexer)
    aaae(choice_indexer, expected_choice_indexer)
    aaae(segments, expected_segments)
