import numpy as np
from lcm.grids import get_linspace_coordinate
from lcm.grids import linspace
from lcm.grids import logspace
from numpy.testing import assert_array_almost_equal as aaae


def test_linspace():
    calculated = linspace(start=1, stop=2, n_points=6)
    expected = np.array([1, 1.2, 1.4, 1.6, 1.8, 2])
    aaae(calculated, expected)


def test_linspace_mapped_value():
    calculated = get_linspace_coordinate(
        value=1.3,
        start=1,
        stop=2,
        n_points=6,
    )

    expected = 1.5

    assert np.allclose(calculated, expected)


def test_logspace():
    calculated = logspace(start=1, stop=100, n_points=7)
    expected = np.array(
        [1.0, 2.15443469, 4.64158883, 10.0, 21.5443469, 46.41588834, 100.0]
    )
    aaae(calculated, expected)
