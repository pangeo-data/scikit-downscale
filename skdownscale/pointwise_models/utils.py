from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def check_max_features(array: ArrayLike, n: int = 1) -> ArrayLike:
    if array.ndim == 1:
        pass
    elif array.ndim == 2:
        n_features = array.shape[1]
        if n_features > n:
            raise ValueError(
                f'Found array with {n_features} features (shape={array.shape}) while '
                f'a maximum of {n} is required'
            )

    else:
        raise ValueError(
            f'Found array with {array.ndim} dimensions. Unclear which should be the feature dim.'
        )
    return array


def ensure_samples_features(
    obj: pd.DataFrame | pd.Series | NDArray[Any],
) -> pd.DataFrame | NDArray[Any]:
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


def default_none_kwargs(kwargs: dict[str, Any] | None, copy: bool = False) -> dict[str, Any]:
    if kwargs is not None:
        if copy:
            return kwargs.copy()
        else:
            return kwargs
    else:
        return {}
