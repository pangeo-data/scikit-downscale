"""
Quantile delta mapping
======================

1. Detrend individual quantiles
2. Apply QM to detrended series
3. Apply trends to bias adjusted quantiles

"""

from .utils import group_apply, parse_group, get_correction, apply_correction, add_cyclic, get_index, nodes, \
    add_q_bounds


def train(x, y, nq, group='time.dayofyear', kind="+", window=1):
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
    xq = group_apply("quantile", x, group, q, window, q=q)
    yq = group_apply("quantile". y, group, q, window, q=q)

    return get_correction(xq, yq, kind)


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
    dim, prop = parse_group(qm.group)

    coord = x.coords[dim]

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        qm = add_cyclic(qm, prop)
        qm = add_q_bounds(qm)

    # Invert the source quantiles
    isource = invert(qm.source, prop)

    xt = get_index(x, dim, prop, interp)

    # Extract the correct quantile for each time step.
    if interp:  # Interpolate both the time group and the quantile.
        factor = qm.interp({prop: xt, "x": x})
    else:  # Find quantile for nearest time group and quantile.
        factor = qm.sel({prop: xt, "x": x}, method="nearest")

    # Apply the correction factors
    out = apply_correction(x, factor, qm.kind)

    # Remove time grouping and quantile coordinates
    return out.drop(["quantile", prop])

