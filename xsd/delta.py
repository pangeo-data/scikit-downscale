"""
Delta method
============

TODO


"""

from .utils import group_apply, parse_group, correction, add_cyclic, get_index


def train(x, y, group="time.month", kind="+", window=1):
    """Compute mean adjustment factors."""
    sx = group_apply("mean", x, group, window)
    sy = group_apply("mean", y, group, window)

    return correction(sx, sy, kind)


def predict(x, obj, interp=False):
    """Apply correction to data.
    """
    dim, prop = parse_group(obj.group)

    # Add cyclical values to the scaling factors for interpolation
    if interp:
        obj = add_cyclic(obj, prop)

    index = get_index(x, dim, prop, interp)

    if interp:  # Interpolate the time group correction
        factor = obj.interp({prop: index, "x": x})
    else:  # Find quantile for nearest time group
        factor = obj.sel({prop: index, "x": x}, method="nearest")

    if obj.kind == "+":
        out = x + factor
    elif obj.kind == "*":
        out = x * factor
    else:
        raise ValueError

    out.attrs["bias_corrected"] = True
    return out.drop([prop, ])


