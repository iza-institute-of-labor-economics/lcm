from itertools import product

import jax.numpy as jnp
import pytest
from lcm.aggregate_conditional_continuation_values import _put_first_axis_last
from lcm.aggregate_conditional_continuation_values import _put_last_axis_first
from lcm.aggregate_conditional_continuation_values import (
    aggregate_conditional_continuation_values,
)
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture()
def values():
    return jnp.arange(20).reshape(2, 2, 5) / 2


@pytest.fixture()
def segment_info():
    info = {
        "segment_ids": jnp.array([0, 0, 1, 1, 1]),
        "num_segments": 2,
    }
    return info


test_cases = list(product([True, False], range(6)))


@pytest.mark.parametrize("collapse, n_leading_axes", test_cases)
def test_aggregation_without_shocks(values, segment_info, collapse, n_leading_axes):
    values, agg_axes = _get_reshaped_values_and_agg_axes(
        values, collapse, n_leading_axes
    )

    calculated = aggregate_conditional_continuation_values(
        values=values,
        shock_type=None,
        choice_axes=agg_axes,
        choice_segments=segment_info,
    )

    expected = jnp.array([8, 9.5])

    expected_shape = tuple([1] * n_leading_axes + [2])
    assert calculated.shape == expected_shape
    aaae(calculated.flatten(), expected)


scaling_factors = [0.3, 0.6, 1, 2.5, 10]
expected = [
    [8.051974, 9.560844],
    [8.225906, 9.800117],
    [8.559682, 10.265875],
    [10.595821, 12.880228],
    [25.184761, 30.494621],
]
test_cases = []
for scale, expected in zip(scaling_factors, expected):
    for collapse in [True, False]:
        for n_axes in range(6):
            test_cases.append((scale, expected, collapse, n_axes))


@pytest.mark.parametrize("scale, expected, collapse, n_leading_axes", test_cases)
def test_aggregation_with_extreme_value_shocks(
    values, segment_info, scale, expected, collapse, n_leading_axes
):
    values, agg_axes = _get_reshaped_values_and_agg_axes(
        values, collapse, n_leading_axes
    )

    calculated = aggregate_conditional_continuation_values(
        values=values,
        shock_type="extreme_value",
        choice_axes=agg_axes,
        choice_segments=segment_info,
        shock_scale=scale,
    )

    expected_shape = tuple([1] * n_leading_axes + [2])
    assert calculated.shape == expected_shape
    aaae(calculated.flatten(), jnp.array(expected), decimal=5)


@pytest.mark.parametrize("shape", [5, (4, 5), (3, 2, 5, 3, 8), (8, 4, 3)])
def test_transposing_cancels_out(shape):
    a = jnp.arange(jnp.prod(jnp.array(shape))).reshape(shape)
    b = _put_last_axis_first(a)
    c = _put_first_axis_last(b)
    aaae(a, c)


def _get_reshaped_values_and_agg_axes(values, collapse, n_leading_axes):
    if collapse:
        values = values.reshape(4, 5)
        agg_axes = n_leading_axes
    else:
        agg_axes = (n_leading_axes, n_leading_axes + 1)

    new_shape = tuple([1] * n_leading_axes + list(values.shape))
    values = values.reshape(new_shape)

    return values, agg_axes
