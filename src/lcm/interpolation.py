import jax.numpy as jnp
import lcm.grids as grids_module
from jax.scipy.ndimage import map_coordinates


def linear_interpolation(values, point, grid_info):
    """"""

    mapped_values = []
    for i, (grid_type, args) in enumerate(grid_info):
        func = getattr(grids_module, f"get_{grid_type}_coordinate")
        mapped_values.append(func(point[i], *args))

    mapped_point = jnp.array(mapped_values)
    res = map_coordinates(
        input=values, coordinates=mapped_point, order=1, mode="nearest"
    )

    return res
