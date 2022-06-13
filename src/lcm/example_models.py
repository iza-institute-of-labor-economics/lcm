"""Define example model specifications."""
import jax.numpy as jnp


def phelps_deaton_utility_with_shock(
    consumption, working, delta, additive_utility_shock
):
    return jnp.log(consumption) + additive_utility_shock - delta * working


def phelps_deaton_utility(consumption, working, delta):
    return jnp.log(consumption) - delta * working


def working(retirement):
    return 1 - retirement


def next_wealth_with_shock(
    wealth, consumption, working, wage, wage_shock, interest_rate
):
    return interest_rate * (wealth - consumption) + wage * wage_shock * working


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return interest_rate * (wealth - consumption) + wage * working


def next_wealth_constraint(next_wealth):
    return next_wealth >= 0


def age(period):
    return period + 18


def mandatory_retirement_filter(retirement, age):
    return retirement == 1 | age < 65


def absorbing_retirement_filter(retirement, lagged_retirement):
    return retirement == 1 | lagged_retirement == 0


PHELPS_DEATON = {
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "next_wealth_constraint": next_wealth_constraint,
        "working": working,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 1e4,
            "n_points": 11,
        },
    },
    "states": {
        "wealth": {"grid_type": "linspace", "start": 0, "stop": 1e4, "n_points": 11}
    },
    "n_periods": 20,
}


PHELPS_DEATON_WITH_SHOCKS = {
    **PHELPS_DEATON,
    "functions": {
        "utility": phelps_deaton_utility_with_shock,
        "next_wealth": next_wealth_with_shock,
        "next_wealth_constraint": next_wealth_constraint,
        "working": working,
    },
    "shocks": {
        "wage_shock": "lognormal",
        # special name to signal that this shock can be set to zero to calculate
        # expected utility
        "additive_utility_shock": "extreme_value",
    },
}


PHELPS_DEATON_WITH_FILTERS = {
    **PHELPS_DEATON,
    "state_filters": [mandatory_retirement_filter, absorbing_retirement_filter],
}
