import numpy as np
import xarray as xr
from xarray.core.missing import get_clean_interp_index
import dask.array
import pandas as pd


def nodes(n):
    """Return nodes with `n` equally spaced points within [0, 1] plus two end-points.

    E.g. for nq=4 :  0---x------x------x------x---1
    """
    dq = 1 / n / 2
    q = np.linspace(dq, 1 - dq, n)
    return sorted(np.append([0.0001, 0.9999], q))


def group_quantile(x, group, q, window=1):
    """Group values by time, then compute quantiles. The resulting array will
    have new time and quantile dimensions.

    Parameters
    ----------
    x : DataArray
      Data.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping criterion. If only coordinate is given (e.g. 'time') no grouping will be done.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      useful for time.dayofyear grouping.

    Returns
    -------

    """
    if '.' in group:
        dim, prop = group.split('.')
    else:
        dim = group

    if '.' in group:
        if window > 1:
            # Construct rolling window
            x = x.rolling(center=True, **{dim: window}).construct(window_dim="window")
            dims = ("window", dim)
        else:
            dims = dim

        xq = x.groupby(group).quantile(q, dim=dims)

    else:
        xq = x.quantile(q, dim=dim)

    return xq


def train(x, y, nq, group='time.dayofyear', window=1):
    """Compute quantile bias-adjustment factors.

    Parameters
    ----------
    x : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    y : xr.DataArray
      Training target, usually an observed at-site time-series.
    nq : int
      Number of quantiles.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time'}
      Grouping criterion. If only coordinate is given (e.g. 'time') no grouping will be done.
    window : int
      Length of the rolling window centered around the time of interest used to estimate the quantiles. This is mostly
      useful for time.dayofyear grouping.

    Returns
    -------
    xr.Dataset
      Quantiles for the source and target series.
    """
    q = nodes(nq)
    xq = group_quantile(x, group, q, window)
    yq = group_quantile(y, group, q, window)

    out = xr.Dataset(data_vars={"source": xq, "target": yq},
                     attrs={"group": group, "window": window})

    return out


def predict(x, qm, interp=False, detrend_order=4):
    """Apply quantile mapping delta to an array.

    Parameters
    ----------
    x : xr.DataArray
      Data to predict on.
    qm : xr.DataArray
      Quantile mapping computed by the `train` function.
    interp : bool
      Whether to interpolate between the groupings.

    Returns
    -------
    xr.DataArray
      Input array with delta applied.
    """
    if '.' in qm.group:
        dim, prop = qm.group.split('.')
    else:
        dim, prop = qm.group, None

    if prop == "season" and interp:
        raise NotImplementedError

    coord = x.coords[dim]

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        qm = add_cyclic(qm, prop)
        qm = add_q_bounds(qm)

    # Invert the source quantiles
    isource = invert(qm.source, prop)

    # Compute the `dim` value for indexing along grouping dimension
    # TODO: Adjust for different calendars if necessary.
    ind = x.indexes[dim]
    y = getattr(ind, prop)
    if interp:
        if dim == "time":
            if prop == "month":
                y = ind.month - 0.5 + ind.day / ind.daysinmonth
            elif prop == "dayofyear":
                y = ind.dayofyear
            else:
                raise NotImplementedError

    xt = xr.DataArray(y, dims=dim, coords={dim: coord}, name=dim + " group index")

    # Expand dimensions of index to match the dimensions of xq
    # We want vectorized indexing with no broadcasting
    xt = xt.expand_dims(**{k: v for (k, v) in x.coords.items() if k != dim})

    # Extract the correct quantile for each time step.
    if interp:  # Interpolate both the time group and the quantile.
        factor = qm.interp({prop: xt, "x": x})
    else:  # Find quantile for nearest time group and quantile.
        factor = qm.sel({prop: xt, "x": x}, method="nearest")

    # Apply the correction factors
    out = x.copy()

    if qm.kind == "+":
        out += factor
    elif qm.kind == "*":
        out *= factor

    out.attrs["bias_corrected"] = True

    # Remove time grouping and quantile coordinates
    return out.drop(["quantile", prop])


# TODO: use xr.pad once it's implemented.
def add_cyclic(qmf, att):
    """Reindex the scaling factors to include the last time grouping
    at the beginning and the first at the end.

    This is done to allow interpolation near the end-points.
    """
    gc = qmf.coords[att]
    i = np.concatenate(([-1], range(len(gc)), [0]))
    qmf = qmf.reindex({att: gc[i]})
    qmf.coords[att] = range(len(i))
    return qmf


# TODO: use xr.pad once it's implemented.
def add_q_bounds(qmf):
    """Reindex the scaling factors to set the quantile at 0 and 1 to the first and last quantile respectively.

    This is a naive approach that won't work well for extremes.
    """
    att = "quantile"
    q = qmf.coords[att]
    i = np.concatenate(([0], range(len(q)), [-1]))
    qmf = qmf.reindex({att: q[i]})
    qmf.coords[att] = np.concatenate(([0], q, [1]))
    return qmf


def invert(qm, dim):
    """Invert a quantile array. Quantiles become coordinates and q becomes the data."""

    newx = np.unique(qm.values.flat)
    # TODO: cluster the values to reduce size.

    func = lambda x: xr.Variable(dims="x", data=np.interp(newx, x, qm["quantile"].values))

    out = qm.groupby(dim).map(func, shortcut=True)
    return out.assign_coords(x=newx)


