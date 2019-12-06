import numpy as np
import xarray as xr
import dask.array
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

"""
Basic univariate quantile mapping post-processing algorithms.

"""


def train(x, y, nq, group, kind="+", detrend_order=4):
    """Compute quantile bias-adjustment factors.

    Parameters
    ----------
    x : xr.DataArray
      Training data, usually a model output whose biases are to be corrected.
    y : xr.DataArray
      Training target, usually an observed at-site time-series.
    nq : int
      Number of quantiles.
    group : {'time.season', 'time.month', 'time.dayofyear', 'time.time'}
      Grouping criterion. Use an accessor that is identical to all values to skip grouping (e.g. time.time for daily
      values).
    kind : {'+', '*'}
      The transfer operation, + for additive and * for multiplicative.
    detrend : int, None
      Polynomial order of detrending curve. Set to None to skip detrending.

    Returns
    -------
    xr.DataArray
      Delta factor computed over time grouping and quantile bins.
    """
    # Detrend
    if detrend_order is not None:
        x, _ = xdetrend(x, dim="time", deg=detrend_order)
        y, _ = xdetrend(y, dim="time", deg=detrend_order)


    # Define nodes. Here n equally spaced points within [0, 1]
    # E.g. for nq=4 :  0---x------x------x------x---1
    dq = 1 / nq / 2
    q = np.linspace(dq, 1 - dq, nq)

    # Group values by time, then compute quantiles. The resulting array will have new time and quantile dimensions.
    xq = x.groupby(group).quantile(q)
    yq = y.groupby(group).quantile(q)

    # Compute the correction factor
    if kind == "+":
        out = yq - xq
    elif kind == "*":
        out = yq / xq
    else:
        raise ValueError("kind must be + or *.")

    # Save input parameters as attributes of output DataArray.
    out.attrs["kind"] = kind
    out.attrs["group"] = group

    return out


def predict(x, qmf, interp=False, detrend_order=4):
    """Apply quantile mapping delta to an array.

    Parameters
    ----------
    x : xr.DataArray
      Data to predict on.
    qmf : xr.DataArray
      Quantile mapping factors computed by the `train` function.
    interp : bool
      Whether to interpolate between the groupings.
    detrend : str, None
      Dimension over which to detrend before apply quantile mapping factors. Set to None to skip detrending.

    Returns
    -------
    xr.DataArray
      Input array with delta applied.
    """

    if "time" not in qmf.group:
        raise NotImplementedError

    if "season" in qmf.group and interp:
        raise NotImplementedError

    # Detrend
    if detrend_order is not None:
        x, trend = xdetrend(x, dim="time", deg=detrend_order)


    # Find the group indexes
    ind, att = qmf.group.split(".")

    time = x.coords["time"]
    gc = qmf.coords[att]

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        qmf = add_cyclic(qmf, att)
        qmf = add_q_bounds(qmf)

    # Compute the percentile time series of the input array
    q = x.groupby(qmf.group).apply(xr.DataArray.rank, pct=True, dim=ind)
    iq = xr.DataArray(q, dims="time", coords={"time": time}, name="quantile index")

    # Create DataArrays for indexing
    # TODO: Adjust for different calendars if necessary.

    if interp:
        time = q.indexes["time"]
        if att == "month":
            y = time.month - 0.5 + time.day / time.daysinmonth
        elif att == "dayofyear":
            y = time.dayofyear
        else:
            raise NotImplementedError

    else:
        y = getattr(q.indexes[ind], att)

    it = xr.DataArray(y, dims="time", coords={"time": time}, name="time group index")

    # Extract the correct quantile for each time step.

    if interp:  # Interpolate both the time group and the quantile.
        factor = qmf.interp({att: it, "quantile": iq})
    else:  # Find quantile for nearest time group and quantile.
        factor = qmf.sel({att: it, "quantile": iq}, method="nearest")

    # Apply the correction factors
    out = x.copy()
    if qmf.kind == "+":
        out += factor
    elif qmf.kind == "*":
        out *= factor

    out.attrs["bias_corrected"] = True

    # Add trend back
    if detrend_order is not None:
        out += trend

    # Remove time grouping and quantile coordinates
    return out.drop(["quantile", att])


# TODO: use pad
def add_cyclic(qmf, att):
    """Reindex the scaling factors to include the last time grouping
    at the beginning and the first at the end.

    This is done to allow interpolation near the end-points.
    """
    gc = qmf.coords[att]
    i = np.concatenate(([-1], range(len(gc)), [0]))
    qmf = qmf.reindex({att: gc[i]})
    qmf.coords[att] = range(len(qmf))
    return qmf


# TODO: use pad ?
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


def _calc_slope(x, y):
    """Wrapper that returns the slop from a linear regression fit of x and y."""
    from scipy import stats
    slope = stats.linregress(x, y)[0]  # extract slope only
    return slope


def polyfit(da, deg=1, dim="time"):
    """
    Least squares polynomial fit.

    Fit a polynomial ``p(x) = p[deg] * x ** deg + ... + p[0]`` of degree `deg`
    Returns a vector of coefficients `p` that minimises the squared error.
    Parameters
    ----------
    da : xarray.DataArray
        The array to fit
    deg : int, optional
        Degree of the fitting polynomial, Default is 1.
    dim : str
        The dimension along which the data will be fitted. Default is `time`.

    Returns
    -------
    output : xarray.DataArray
        Polynomial coefficients with a new dimension to sort the polynomial
        coefficients by degree
    """

    y = da.dropna(dim, how="all")
    coord = y.coords[dim]

    if pd.api.types.is_datetime64_dtype(coord.data):
        # Use the 1e-9 to scale nanoseconds to seconds (by default, xarray use
        # datetime in nanoseconds
        x = pd.to_numeric(coord) * 1e-9
    else:
        x = coord

    # Fit the parameters (lazy computation)
    coefs = dask.array.apply_along_axis(
            np.polyfit, da.get_axis_num("time"), x, y, deg=deg, shape=(deg+1, ), dtype=float)

    coords = dict(da.coords.items())
    coords.pop(dim)
    coords['degree'] = range(deg, -1, -1)

    dims = list(da.dims)
    dims.remove(dim)
    dims.insert(0, "degree")

    out = xr.DataArray(data=coefs, coords=coords, dims=dims)
    return out


def polyval(coord, coefs):
    if pd.api.types.is_datetime64_dtype(coord.data):
        # Use the 1e-9 to scale nanoseconds to seconds (by default, xarray use
        # datetime in nanoseconds
        x = pd.to_numeric(coord) * 1e-9
    else:
        x = coord

    y = xr.apply_ufunc(np.polyval, coefs, x, input_core_dims=[['degree'], []], dask='allowed')

    return y


def xdetrend(obj, dim='time', deg=1):
    coefs = polyfit(obj, dim=dim, deg=deg)

    trend = polyval(obj[dim], coefs)
    detrended = obj - trend
    return detrended, trend


def _order_and_stack(obj, dim):
    """
    Private function used to reorder to use the work dimension as the first
    dimension, stack all the dimensions except the first one
    """
    dims_stacked = [di for di in obj.dims if di != dim]
    new_dims = [dim, ] + dims_stacked
    if obj.ndim > 2:
        obj_stacked = (obj.transpose(*new_dims)
                       .stack(temp_dim=dims_stacked)
                       .dropna('temp_dim'))
    elif obj.ndim == 2:
        obj_stacked = obj.transpose(*new_dims)
    else:
        obj_stacked = obj
    return obj_stacked


def _unstack(obj):
    """
    Private function used to reorder to use the work dimension as the first
    dimension, stack all the dimensions except the first one
    """
    if 'temp_dim' in obj.dims:
        obj_unstacked = obj.unstack('temp_dim')
    else:
        obj_unstacked = obj
    return obj_unstacked


