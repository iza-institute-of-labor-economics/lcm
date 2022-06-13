"""Collection of distributions for shocks."""
import numpy as np
import pandas as pd


def lognormal(mean, sd, size):
    return np.random.lognormal(mean, sd, size)


def get_lognormal_params(name):
    ind_tups = [(name, "mean"), (name, "sd")]
    index = pd.MultiIndex.from_tuples(ind_tups)
    params = pd.DataFrame(index=index)
    params["lower_bound"] = [-np.inf, 0]
    return params


def extreme_value(mode, scale, size):
    return np.random.gumbel(mode, scale, size)


def get_extreme_value_params(name):
    ind_tups = [(name, "mode"), (name, "scale")]
    index = pd.MultiIndex.from_tuples(ind_tups)
    params = pd.DataFrame(index=index)
    params["lower_bound"] = [-np.inf, 0]
    return params
