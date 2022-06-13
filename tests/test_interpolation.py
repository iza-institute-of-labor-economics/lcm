import numpy as np
import pytest
from lcm.interpolation import linear_interpolation
from numpy.testing import assert_array_almost_equal as aaae
from scipy.interpolate import RegularGridInterpolator


def assert_point_wise(grids, grid_info, points, values):
    for point in points:
        calculated = linear_interpolation(
            values=values,
            point=point,
            grid_info=grid_info,
        )

        scipy_func = RegularGridInterpolator(
            points=grids, values=values, method="linear"
        )
        scipy_res = scipy_func(point)

        aaae(calculated, scipy_res)


def _calc_values_2d(a, b):
    return a / b


def _calc_values_3d(a, b, c):
    return 2 * a ** 3 + 3 * b ** 2 - c


def _calc_values_4d(a, b, c, d):
    return _calc_values_3d(a, b, c) - d


def _calc_values_5d(a, b, c, d, e):
    return _calc_values_4d(a, b, c, d) + e ** 5


@pytest.mark.parametrize(
    "info, points, calc_values",
    [
        (
            [(1, 5, 5), (2, 4, 3)],
            np.array([[2.5, 3.5], [2.1, 3.8], [2.7, 3.3]]),
            _calc_values_2d,
        ),
        (
            [(1, 5, 5), (4, 7, 4), (7, 9, 4)],
            np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1], [2.7, 4.3, 7]]),
            _calc_values_3d,
        ),
        (
            [(1, 5, 5), (4, 7, 4), (7, 9, 4), (10, 11, 2)],
            np.array([[2.1, 6.2, 8.3, 10.4], [3.3, 5.2, 7.1, 10], [2.7, 4.3, 7, 11.0]]),
            _calc_values_4d,
        ),
        (
            [(1, 5, 5), (4, 7, 4), (7, 9, 4), (10, 11, 2), (-3, 4, 10)],
            np.array(
                [
                    [2.1, 6.2, 8.3, 10.4, -3],
                    [3.3, 5.2, 7.1, 10, 0],
                    [2.7, 4.3, 7, 11.0, -1.7],
                ]
            ),
            _calc_values_5d,
        ),
    ],
)
def test_linear_interpolation_monotone_grid(info, points, calc_values):
    grids = [np.linspace(*i) for i in info]
    grid_info = [("linspace",) + (i,) for i in info]

    values = calc_values(*np.meshgrid(*grids, indexing="ij", sparse=False))

    assert_point_wise(grids, grid_info, points, values)


@pytest.mark.parametrize(
    "grid, info, points, calc_values",
    [
        (
            [(np.log10(1), np.log10(10), 7), (np.log10(1), np.log10(10), 3)],
            [(1, 10, 7), (1, 10, 3)],
            np.array([[9.8, 2.3], [2.1, 8.2], [2.7, 1.1]]),
            _calc_values_2d,
        ),
        (
            [
                (np.log10(1), np.log10(5), 5),
                (np.log10(4), np.log10(7), 4),
                (np.log10(7), np.log10(9), 2),
                (np.log10(10), np.log10(11), 2),
                (np.log10(3), np.log10(4), 10),
            ],
            [(1, 5, 5), (4, 7, 4), (7, 9, 2), (10, 11, 2), (3, 4, 10)],
            np.array(
                [
                    [2.1, 6.2, 8.3, 10.4, 3],
                    [3.3, 5.2, 7.1, 10, 3.6],
                    [2.7, 4.3, 7, 10.5, 4],
                ]
            ),
            _calc_values_5d,
        ),
    ],
)
def test_linear_interpolation_logarithmic_scale(grid, info, points, calc_values):
    grids = [np.logspace(*g) for g in grid]
    grid_info = [("logspace",) + (i,) for i in info]

    values = calc_values(*np.meshgrid(*grids, indexing="ij", sparse=False))

    assert_point_wise(grids, grid_info, points, values)
