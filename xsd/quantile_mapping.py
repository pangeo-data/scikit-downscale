from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da
import xarray as xr
import numpy as np
from scipy import stats
from xarray.core.pycompat import dask_array_type

# from storylines.tools.encoding import attrs, encoding, make_gloabl_attrs

SYNTHETIC_MIN = -1e20
SYNTHETIC_MAX = 1e20

variables = ['pcp', 't_mean', 't_range']
detrend = {'pcp': False, 't_mean': True, 't_range': True}
extrapolate = {'pcp': 'max', 't_mean': 'both', 't_range': 'max'}
zeros = {'pcp': True, 't_mean': False, 't_range': False}


def quantile_mapping(input_data, ref_data, data_to_match,
                     alpha=0.4, beta=0.4, detrend=False,
                     extrapolate=None, n_endpoints=10,
                     use_ref_data=True):
    '''quantile mapping between `input_data` and `data_to_match`

    Parameters
    ----------
    input_data : xr.DataArray
        Input data to be quantile mapped to match the distribution of
        `data_to_match`
    ref_data : xr.DataArray
        Reference data to be used to adjust `input_data`
    data_to_match : xr.DataArray
        Target data for quantile mapping
    alpha, beta : float
        Plotting positions parameter. Default is 0.4.
    detrend : bool
        If True, detrend `input_data` prior to performing quantile mapping.
        Default is False.
    extrapolate : str
        Option specifying how to handle endpoints/extreme values. Valid options
        are {'max', 'min', 'both', None}. If `extrapolate` is not `None`, the
        end point(s) of the CDF (0, 1) will be linearly extrapolated using the
        last `n_endpoints` from the tail of the distribution. Default is None.
    n_endpoints : int
        Number of data points to use when the `extrapolate` option is set.

    Returns
    -------
    new : xr.DataArray
        Quantile mapped data with shape from `input_data` and probability
            distribution from `data_to_match`.

    See Also
    --------
    scipy.stats.mstats.plotting_positions
    '''

    if detrend:
        # remove linear trend, saving the slope/intercepts for use later
        input_data, input_data_trend = remove_trend(input_data)
        data_to_match, _ = remove_trend(data_to_match)
        if ref_data is not None:
            ref_data, _ = remove_trend(ref_data)

    # arguments to qmap
    kwargs = dict(alpha=alpha, beta=beta, extrapolate=extrapolate,
                  n_endpoints=n_endpoints)

    new = qmap_grid(input_data, data_to_match, ref_data, **kwargs)

    # put the trend back
    if detrend:
        new += input_data_trend

    return new


def quantile_mapping_by_group(input_data, ref_data, data_to_match,
                              grouper='time.month', **kwargs):
    '''quantile mapping between `input_data` and `data_to_match by group`

    Parameters
    ----------
    input_data : xr.DataArray
        Input data to be quantile mapped to match the distribution of
        `data_to_match`
    ref_data : xr.DataArray
        Reference data to be used to adjust `input_data`
    data_to_match : xr.DataArray
        Target data for quantile mapping
    grouper : str, array, Grouper
        Object to pass to `DataArray.groupby`, default ``'time.month'``
    kwargs : any
        Additional named arguments to `quantile_mapping`

    Returns
    -------
    new : xr.DataArray
        Quantile mapped data with shape from `input_data` and probability
            distribution from `data_to_match`.

    See Also
    --------
    quantile_mapping
    scipy.stats.mstats.plotting_positions

    Note
    ----
    This function will use `dask.array.map_blocks` if the input arguments are
    of type `dask.array.Array`.
    '''

    # Allow grouper to be None
    if grouper is None:
        return quantile_mapping(input_data, ref_data, data_to_match, **kwargs)

    # Create the groupby objects
    obs_groups = data_to_match.groupby(grouper)
    input_groups = input_data.groupby(grouper)

    # Iterate over the groups, calling the quantile method function on each
    results = []
    if ref_data is not None:
        ref_groups = ref_data.groupby(grouper)
        for (_, g_obs), (_, g_ref), (_, g_input) in zip(obs_groups, ref_groups,
                                                        input_groups):
            results.append(quantile_mapping(g_input, g_ref, g_obs, **kwargs))
    else:
        for (_, group_obs), (_, group_input) in zip(obs_groups, input_groups):
            results.append(quantile_mapping(group_input, None, group_obs,
                                            **kwargs))

    # put the groups back together
    new_concat = xr.concat(results, dim='time')
    # Now sort the time dimension again
    new_concat = new_concat.sortby('time')

    return new_concat


def plotting_positions(n, alpha=0.4, beta=0.4):
    '''Returns a monotonic array of plotting positions.

    Parameters
    ----------
    n : int
        Length of plotting positions to return.
    alpha, beta : float
        Plotting positions parameter. Default is 0.4.

    Returns
    -------
    positions : ndarray
        Quantile mapped data with shape from `input_data` and probability
            distribution from `data_to_match`.

    See Also
    --------
    scipy.stats.mstats.plotting_positions

    '''
    return (np.arange(1, n + 1) - alpha) / (n + 1. - alpha - beta)


def make_x_and_y(y, alpha, beta, extrapolate,
                 x_min=SYNTHETIC_MIN, x_max=SYNTHETIC_MAX):
    '''helper function to calculate x0, conditionally adding endpoints'''
    n = len(y)

    temp = plotting_positions(n, alpha, beta)

    x = np.empty(n + 2)
    y_new = np.full(n + 2, np.nan)
    rs = slice(1, -1)
    x[rs] = temp

    # move the values from y to the new y array
    # repeat the first/last values to make everything consistant
    y_new[rs] = y
    y_new[0] = y[0]
    y_new[-1] = y[-1]

    # Add endpoints to x0
    if (extrapolate is None) or (extrapolate == '1to1'):
        x[0] = temp[0]
        x[-1] = temp[-1]
    elif extrapolate == 'both':
        x[0] = x_min
        x[-1] = x_max
    elif extrapolate == 'max':
        x[0] = temp[0]
        x[-1] = x_max
    elif extrapolate == 'min':
        x[0] = x_min
        x[-1] = temp[-1]
    else:
        raise ValueError('unknown value for extrapolate: %s' % extrapolate)

    return x, y_new, rs


def _extrapolate(y, alpha, beta, n_endpoints, how='both', ret_slice=False,
                 x_min=SYNTHETIC_MIN, x_max=SYNTHETIC_MAX):

    x_new, y_new, rs = make_x_and_y(y, alpha, beta,
                                    extrapolate=how, x_min=x_min, x_max=x_max)
    y_new = calc_endpoints(x_new, y_new, how, n_endpoints)

    if ret_slice:
        return x_new, y_new, rs
    else:
        return x_new, y_new


def _custom_extrapolate_x_data(x, y, n_endpoints):
    lower_inds = np.nonzero(-np.inf == x)[0]
    upper_inds = np.nonzero(np.inf == x)[0]
    if len(lower_inds):
        s = slice(lower_inds[-1] + 1, lower_inds[-1] + 1 + n_endpoints)
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        x[lower_inds] = (y[lower_inds] - intercept) / slope
    if len(upper_inds):
        s = slice(upper_inds[0] - n_endpoints, upper_inds[0])
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        x[upper_inds] = (y[upper_inds] - intercept) / slope
    return x


def calc_endpoints(x, y, extrapolate, n_endpoints):
    '''extrapolate the tails of the CDF using linear interpolation on the last
    n_endpoints

    This function modifies `y` in place'''

    if n_endpoints < 2:
        raise ValueError('Invalid number of n_endpoints, must be >= 2')

    if extrapolate in ['min', 'both']:
        s = slice(1, n_endpoints + 1)
        # fit linear model to slice(1, n_endpoints + 1)
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        # calculate the value of y at x[0]
        y[0] = intercept + slope * x[0]
    if extrapolate in ['max', 'both']:
        s = slice(-n_endpoints - 1, -1)
        # fit linear model to slice(-n_endpoints - 1, -1)
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        # calculate the value of y at x[-1]
        y[-1] = intercept + slope * x[-1]

    return y


def qmap(data, like, ref=None, alpha=0.4, beta=0.4, extrapolate=None,
         n_endpoints=10):
    '''quantile mapping for a single point'''

    # fast track if data has nans
    if np.isnan(np.sum(data)):
        return np.full_like(data, np.nan)

    # x is the percentiles
    # y is the sorted data
    sort_inds = np.argsort(data)
    x_data, y_data, rs = _extrapolate(data[sort_inds], alpha, beta,
                                      n_endpoints,
                                      how=extrapolate, ret_slice=True,
                                      x_min=0, x_max=1)

    x_like, y_like = _extrapolate(np.sort(like), alpha, beta,
                                  n_endpoints, how=extrapolate,
                                  x_min=-1e15, x_max=1e15)

    # map the quantiles from ref-->data
    # TODO: move to its own function
    if ref is not None:
        x_ref, y_ref = _extrapolate(np.sort(ref), alpha, beta, n_endpoints,
                                    how=extrapolate, x_min=-1e10, x_max=1e10)

        left = -np.inf if extrapolate in ['min', 'both'] else None
        right = np.inf if extrapolate in ['max', 'both'] else None
        x_data = np.interp(y_data, y_ref, x_ref, left=left, right=right)

    if np.isinf(x_data).any():
        # Extrapolate the tails beyond 1.0 to handle "new extremes"
        x_data = _custom_extrapolate_x_data(x_data, y_data, n_endpoints)

    # empty array, prefilled with nans
    new = np.full_like(data, np.nan)

    # Do the final mapping
    new[sort_inds] = np.interp(x_data, x_like, y_like)[rs]

    # If extrapolate is 1to1, apply the offset between ref and like to the
    # tails of new
    if ref is not None and extrapolate == '1to1':
        ref_max = ref.max()
        ref_min = ref.min()
        inds = (data > ref_max)
        if inds.any():
            if len(ref) == len(like):
                new[inds] = like.max() + (data[inds] - ref_max)
            elif len(ref) > len(like):
                ref_at_like_max = np.interp(x_like[-1], x_ref, y_ref)
                new[inds] = like.max() + (data[inds] - ref_at_like_max)
            elif len(ref) < len(like):
                like_at_ref_max = np.interp(x_ref[-1], x_like, y_like)
                new[inds] = like_at_ref_max + (data[inds] - ref_max)
        inds = (data < ref_min)
        if inds.any():
            if len(ref) == len(like):
                new[inds] = like.min() + (data[inds] - ref_min)
            elif len(ref) > len(like):
                ref_at_like_min = np.interp(x_like[0], x_ref, y_ref)
                new[inds] = like.min() + (data[inds] - ref_at_like_min)
            elif len(ref) < len(like):
                like_at_ref_min = np.interp(x_ref[0], x_like, y_like)
                new[inds] = like_at_ref_min + (data[inds] - ref_min)

    return new


def _calc_slope(x, y):
    '''wrapper that returns the slop from a linear regression fit of x and y'''
    slope = stats.linregress(x, y)[0]  # extract slope only
    return slope


def remove_trend(obj):
    time_nums = xr.DataArray(obj['time'].values.astype(np.float),
                             dims='time',
                             coords={'time': obj['time']},
                             name='time_nums')
    trend = xr.apply_ufunc(_calc_slope, time_nums, obj,
                           vectorize=True,
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[[]],
                           output_dtypes=[np.float],
                           dask='parallelized')

    trend_ts = (time_nums * trend).transpose(*obj.dims)
    detrended = obj - trend_ts
    return detrended, trend_ts


def qmap_grid(data, like, ref, **kwargs):

    if isinstance(data.data, dask_array_type):
        kws = dict(dtype=data.dtype, chunks=data.chunks, **kwargs)
        if ref is not None:
            new = da.map_blocks(
                _inner_qmap_grid, data.data, like.data, ref.data,
                token="qmap_grid", use_ref_data=True, **kws)
        else:
            new = da.map_blocks(
                _inner_qmap_grid, data.data, like.data, None,
                token="qmap_grid", use_ref_data=False, **kws)
    else:
        # don't use dask map blocks
        if ref is not None:
            new = _inner_qmap_grid(data.data, like.data, ref.data,
                                   use_ref_data=True, **kwargs)
        else:
            new = _inner_qmap_grid(data.data, like.data, ref.data,
                                   use_ref_data=False, **kwargs)

    return xr.DataArray(new, dims=data.dims, coords=data.coords,
                        attrs=like.attrs, name=like.name)


def _inner_qmap_grid(data, like, ref, use_ref_data=False, **kwargs):
    new = np.full_like(data, np.nan)
    if use_ref_data:
        for i, j in np.ndindex(data.shape[1:]):
            new[:, i, j] = qmap(data[:, i, j], like[:, i, j], ref=ref[:, i, j],
                                **kwargs)
    else:
        for i, j in np.ndindex(data.shape[1:]):
            new[:, i, j] = qmap(data[:, i, j], like[:, i, j], **kwargs)
    return new
