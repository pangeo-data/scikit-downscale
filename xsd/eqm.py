"""
Empirical quantile mapping
==========================

References
----------

"""

import numpy as np
import xarray as xr
from .utils import group_apply, parse_group, get_correction, apply_correction, add_cyclic, get_index, nodes, \
    add_q_bounds


def train(x, y, nq, group='time.dayofyear', kind="+", window=1, extrapolation="constant"):
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
    xq = group_apply("quantile", x, group, window, q=q)
    yq = group_apply("quantile", y, group, window, q=q)

    qqm = get_correction(xq, yq, kind)
    xqm = reindex(qqm, xq, extrapolation)
    return xqm


def reindex(qm, xq, extrapolation="constant"):
    """Create a mapping between x values and y values based on their respective quantiles.

    Notes
    -----
    The original qm object has `quantile` coordinates and some grouping coordinate (e.g. month). This function
    reindexes the array based on the values of x, instead of the quantiles. Since the x values are different from
    group to group, the index can get fairly large.
    """
    dim, prop = parse_group(xq.group)
    ds = xr.Dataset({'xq': xq, 'qm': qm})
    gr = ds.groupby(prop)

    # X coordinates common to all groupings
    xs = list(map(lambda x: extrapolate_qm(x[1].qm, x[1].xq, extrapolation)[0], gr))
    newx = np.unique(np.concatenate(xs))

    # Interpolation from quantile to values.
    def func(d):
        x, y = extrapolate_qm(d.qm, d.xq, extrapolation)
        return xr.DataArray(dims="x", data=np.interp(newx, x, y), coords={'x': newx})

    out = gr.map(func, shortcut=True)
    return out



def extrapolate_qm(qm, xq, method="constant"):
    """Extrapolate quantile correction factors beyond the computed quantiles.

    Parameters
    ----------
    qm : xr.DataArray
      Correction factors over `quantile` coordinates.
    xq : xr.DataArray
      Values at each `quantile`.
    method : {"constant"}
      Extrapolation method. See notes below.

    Returns
    -------
    array, array
        Extrapolated correction factors and x-values.

    Notes
    -----
    constant
      The correction factor above and below the computed values are equal to the last and first values
      respectively.
    constant_iqr
      Same as `constant`, but values are set to NaN if farther than one interquartile range from the min and max.
    """
    if method == "constant":
        x = np.concatenate(([-np.inf, ], xq, [np.inf, ]))
        q = np.concatenate((qm[:1], qm, qm[-1:]))
    elif method == "constant_iqr":
        iqr = np.diff(xq.interp(quantile=[.25, .75]))[0]
        x = np.concatenate(([-np.inf, xq[0] - iqr], xq, [xq[-1] + iqr, np.inf]))
        q = np.concatenate(([np.nan, qm[0]], qm, [qm[-1], np.nan]))
    else:
        raise ValueError

    return x, q


