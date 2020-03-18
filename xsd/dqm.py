import numpy as np
import xarray as xr
from xarray.core.missing import get_clean_interp_index
import dask.array
import pandas as pd
from .utils import add_cyclic, add_q_bounds

"""
Basic univariate quantile mapping post-processing algorithms.

Use `train` to estimate the correction factors, and then use `predict`
to apply the factors to a series.

"""


def train(x, y, nq, group='time.dayofyear', kind="+", window=1, detrend_order=1):
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
    kind : {'+', '*'}
      The transfer operation, + for additive and * for multiplicative.
    detrend_order : int, None
      Polynomial order of detrending curve. Set to None to skip detrending.

    Returns
    -------
    xr.DataArray
      Delta factor computed over time grouping and quantile bins.
    """
    if '.' in group:
        dim, prop = group.split('.')
    else:
        dim = group

    # Detrend
    if detrend_order is not None:
        x, _, cx = detrend(x, dim=dim, deg=detrend_order)
        y, _, cy = detrend(y, dim=dim, deg=detrend_order)

    # Define nodes. Here n equally spaced points within [0, 1]
    # E.g. for nq=4 :  0---x------x------x------x---1
    dq = 1 / nq / 2
    q = np.linspace(dq, 1 - dq, nq)
    q = sorted(np.append([0.0001, 0.9999], q))

    # Group values by time, then compute quantiles. The resulting array will have new time and quantile dimensions.
    if '.' in group:

        if window > 1:
            # Construct rolling window
            x = x.rolling(center=True, **{dim: window}).construct(window_dim="window")
            y = y.rolling(center=True, **{dim: window}).construct(window_dim="window")
            dims = ("window", dim)
        else:
            dims = dim

        xq = x.groupby(group).quantile(q, dim=dims)
        yq = y.groupby(group).quantile(q, dim=dims)
    else:
        xq = x.quantile(q, dim=dim)
        yq = y.quantile(q, dim=dim)

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
    out.attrs["window"] = window

    if detrend_order is not None:
        out.attrs["detrending_poly_coeffs_x"] = cx
        out.attrs["detrending_poly_coeffs_y"] = cy

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
    detrend_order : str, None
      Polynomial order of detrending curve. Set to None to skip detrending.

    Returns
    -------
    xr.DataArray
      Input array with delta applied.
    """
    if '.' in qmf.group:
        dim, prop = qmf.group.split('.')
    else:
        raise NotImplementedError
        dim, prop = qmf.group, None

    if prop == "season" and interp:
        raise NotImplementedError

    # Detrend
    if detrend_order is not None:
        x, trend, coeffs = detrend(x, dim=dim, deg=detrend_order)

    coord = x.coords[dim]

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        qmf = add_cyclic(qmf, prop)
        qmf = add_q_bounds(qmf)

    # Compute the percentile of the input array along `dim`
    xq = x.groupby(qmf.group).apply(xr.DataArray.rank, pct=True, dim=dim)
    # iq = xr.DataArray(q, dims=q.dims, coords=q.coords, name="quantile index")

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
        factor = qmf.interp({prop: xt, "quantile": xq})
    else:  # Find quantile for nearest time group and quantile.
        factor = qmf.sel({prop: xt, "quantile": xq}, method="nearest")

    # Apply the correction factors
    out = x.copy()

    if qmf.kind == "+":
        out += factor
    elif qmf.kind == "*":
        out *= factor

    out.attrs["bias_corrected"] = True
    if detrend_order is not None:
        out.attrs["detrending_poly_coeffs"] = coeffs

    # Add trend back
    if detrend_order is not None:
        out += trend

    # Remove time grouping and quantile coordinates
    return out.drop(["quantile", prop])


def _calc_slope(x, y):
    """Wrapper that returns the slop from a linear regression fit of x and y."""
    from scipy import stats
    slope = stats.linregress(x, y)[0]  # extract slope only
    return slope


# TODO: use xr.polyfit once it's implemented.
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
    # Remove NaNs
    y = da.dropna(dim=dim, how="any")

    # Compute the x value.
    x = get_clean_interp_index(da, dim)

    # Fit the parameters (lazy computation)
    coefs = dask.array.apply_along_axis(
        np.polyfit, da.get_axis_num(dim), x, y, deg=deg, shape=(deg + 1, ), dtype=float
    )

    coords = dict(da.coords.items())
    coords.pop(dim)
    coords['degree'] = range(deg, -1, -1)

    dims = list(da.dims)
    dims.remove(dim)
    dims.insert(0, "degree")

    out = xr.DataArray(data=coefs, coords=coords, dims=dims)
    return out


# TODO: use xr.polyval once it's implemented.
def polyval(coefs, coord):
    """
    Evaluate polynomial function.

    Parameters
    ----------
    coord : xr.Coordinate
      Coordinate (e.g. time) used as the independent variable to compute polynomial.
    coefs : xr.DataArray
      Polynomial coefficients as returned by polyfit.
    """
    x = xr.Variable(data=get_clean_interp_index(coord, coord.name), dims=(coord.name,))

    y = xr.apply_ufunc(np.polyval, coefs, x, input_core_dims=[['degree'], []], dask='allowed')

    return y


def detrend(obj, dim="time", deg=1, kind="+"):
    """Detrend a series with a polynomial.

    The detrended object should have the same mean as the original.



    """

    # Fit polynomial coefficients using Ordinary Least Squares.
    coefs = polyfit(obj, dim=dim, deg=deg)

    # Set the 0th order coefficient to 0 to preserve the original mean
    # coefs = xr.where(coefs.degree == 0, 0, coefs)

    # Compute polynomial
    trend = polyval(coefs, obj[dim])

    # Remove trend from original series while preserving means
    # TODO: Get the residuals directly from polyfit
    if kind == "+":
        detrended = obj - trend - trend.mean() + obj.mean()
    elif kind == "*":
        detrended = obj / trend / trend.mean() * obj.mean()

    return detrended, trend, coefs


def get_index(coord):
    """Return x coordinate for polynomial fit."""
    if pd.api.types.is_datetime64_dtype(coord.data):
        x = pd.to_numeric(coord) * 1E-9
    elif 'calendar' in coord.encoding:
        dt = xr.coding.cftime_offsets.get_date_type(coord.encoding['calendar'])
        offset = dt(1970, 1, 1)
        x = xr.Variable(data=xr.core.duck_array_ops.datetime_to_numeric(coord, offset, datetime_unit="s"), dims=("time",))
    else:
        x = coord

    return x
