from lcm.create_params import create_params
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS


def test_create_params_phelps_deaton_with_shocks():
    params = create_params(PHELPS_DEATON_WITH_SHOCKS)

    expected_index = [
        ("discounting", "beta"),
        ("function_parameter", "delta"),
        ("function_parameter", "interest_rate"),
        ("function_parameter", "wage"),
        ("wage_shock", "mean"),
        ("wage_shock", "sd"),
        ("additive_utility_shock", "mode"),
        ("additive_utility_shock", "scale"),
    ]

    assert sorted(params.index) == sorted(expected_index)
