import dask.array
import numpy as np
import pandas as pd
import xarray as xr


def parse_group(group):
    """Return dimension and property."""
    if '.' in group:
        return group.split('.')
    else:
        return group, None


def group_apply(func, x, group, window=1, **kwargs):
    """Group values by time, then compute function.

    Parameters
    ----------
    func : str
      DataArray method applied to each group.
    x : DataArray
      Data.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping criterion. If only coordinate is given (e.g. 'time') no grouping will be done.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      useful for time.dayofyear grouping.
    **kwargs : dict
      Arguments passed to function.

    Returns
    -------

    """
    dim, prop = parse_group(group)

    dims = dim
    if '.' in group:
        if window > 1:
            # Construct rolling window
            x = x.rolling(center=True, **{dim: window}).construct(window_dim="window")
            dims = ("window", dim)

        sub = x.groupby(group)

    else:
        sub = x


    out = getattr(sub, func)(dim=dims, **kwargs)

    # Save input parameters as attributes of output DataArray.
    out.attrs["group"] = group
    out.attrs["window"] = window
    return out


def get_correction(x, y, kind):
    """Return the additive or multiplicative correction factor."""
    with xr.set_options(keep_attrs=True):
        if kind == "+":
            out = y - x
        elif kind == "*":
            out = y / x
        else:
            raise ValueError("kind must be + or *.")

    out.attrs["kind"] = kind
    return out


def apply_correction(x, factor, kind):
    with xr.set_options(keep_attrs=True):
        if kind == "+":
            out = x + factor
        elif kind == "*":
            out = x * factor
        else:
            raise ValueError

    out.attrs["bias_corrected"] = True
    return out

def nodes(n):
    """Return nodes with `n` equally spaced points within [0, 1] plus two end-points.

    E.g. for nq=4 :  0---x------x------x------x---1
    """
    dq = 1 / n / 2
    q = np.linspace(dq, 1 - dq, n)
    return sorted(np.append([0.0001, 0.9999], q))


# TODO: use xr.pad once it's implemented.
def add_cyclic(da, att):
    """Reindex the scaling factors to include the last time grouping
    at the beginning and the first at the end.

    This is done to allow interpolation near the end-points.
    """
    gc = da.coords[att]
    i = np.concatenate(([-1], range(len(gc)), [0]))
    qmf = da.reindex({att: gc[i]})
    qmf.coords[att] = range(len(i))
    return qmf


# TODO: use xr.pad once it's implemented.
# Rename to extrapolate_q ?
def add_q_bounds(qmf, method="constant"):
    """Reindex the scaling factors to set the quantile at 0 and 1 to the first and last quantile respectively.

    This is a naive approach that won't work well for extremes.
    """
    att = "quantile"
    q = qmf.coords[att]
    i = np.concatenate(([0], range(len(q)), [-1]))
    qmf = qmf.reindex({att: q[i]})
    if method == "constant":
        qmf.coords[att] = np.concatenate(([0], q, [1]))
    else:
        raise ValueError
    return qmf


def get_index(da, dim, prop, interp):
    # Compute the `dim` value for indexing along grouping dimension.
    # TODO: Adjust for different calendars if necessary.

    if prop == "season" and interp:
        raise NotImplementedError

    ind = da.indexes[dim]
    i = getattr(ind, prop)

    if interp:
        if dim == "time":
            if prop == "month":
                i = ind.month - 0.5 + ind.day / ind.daysinmonth
            elif prop == "dayofyear":
                i = ind.dayofyear
            else:
                raise NotImplementedError

    xi = xr.DataArray(i, dims=dim, coords={dim: da.coords[dim]}, name=dim + " group index")

    # Expand dimensions of index to match the dimensions of xq
    # We want vectorized indexing with no broadcasting
    return xi.expand_dims(**{k: v for (k, v) in da.coords.items() if k != dim})
