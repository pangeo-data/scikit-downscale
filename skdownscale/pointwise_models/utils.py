import numpy as np
import pandas as pd


def check_max_features(array, n=1):
    if array.ndim == 1:
        pass
    elif array.ndim == 2:
        n_features = array.shape[1]
        if n_features > n:
            raise ValueError(
                'Found array with %d features (shape=%s) while '
                'a maximum of %d is required' % (n_features, array.shape, n)
            )

    else:
        raise ValueError(
            'Found array with %d dimensions. Unclear which should '
            'be the feature dim.' % array.ndim
        )
    return array


def ensure_samples_features(obj):
    """helper function to ensure sammples conform to sklearn format
    requirements
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            return obj
        if obj.ndim == 1:
            return obj.reshape(-1, 1)
    return obj  # hope for the best, probably better to raise an error here


def default_none_kwargs(kwargs, copy=False):
    if kwargs is not None:
        if copy:
            return kwargs.copy()
        else:
            return kwargs
    else:
        return {}
