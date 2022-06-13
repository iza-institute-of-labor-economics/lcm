"""Create a parameters DataFrame fora model specification."""
import inspect

import numpy as np
import pandas as pd
from lcm import distributions


def create_params(model):
    to_concat = [
        _create_standard_params(),
        _create_function_params(model),
    ]

    if "shocks" in model:
        to_concat.append(_create_shock_params(model["shocks"])),

    params = pd.concat(to_concat)
    params["value"] = np.nan
    params = params[["value", "lower_bound", "upper_bound"]]
    return params


def _create_standard_params():
    ind_tups = [("discounting", "beta")]
    index = pd.MultiIndex.from_tuples(ind_tups, names=["category", "name"])
    params = pd.DataFrame(index=index)
    params["lower_bound"] = 0
    params["upper_bound"] = 1

    return params


def _create_function_params(model):
    variables = {
        *model["functions"],
        *model["choices"],
        *model["states"],
        *model["shocks"],
    }

    all_arguments = set()
    for func in model["functions"].values():
        all_arguments = all_arguments.union(inspect.signature(func).parameters)

    parameters = sorted(all_arguments.difference(variables))

    tuples = [("function_parameter", p) for p in parameters]
    index = pd.MultiIndex.from_tuples(tuples, names=["category", "name"])
    params = pd.DataFrame(index=index)
    return params


def _create_shock_params(shocks):
    to_concat = []
    for name, dist in shocks.items():
        func = getattr(distributions, f"get_{dist}_params")
        to_concat.append(func(name))

    return pd.concat(to_concat)
