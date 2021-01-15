import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .quantile import plotting_positions
from .utils import check_max_features, default_none_kwargs

try:
    import pwlf
except:
    pwlf = None


def arrm_breakpoints(X, y, window_width, max_breakpoints):
    '''Calculate breakpoints in x and y

    Parameters
    ----------
    X, y : array_like
        1-D arrays of data
    window_width : float
        Fraction of the length of x (or y) to use for a window width
    max_breakpoints : int
        Maximum number of breakpoints

    Returns
    -------
    breakpoints : ndarray
    '''
    min_width = 10

    npoints = len(X)
    assert len(X) == len(y)
    assert X.shape[1] == 1

    X = np.sort(X[:, 0])
    y = np.sort(y)
    quantiles = plotting_positions(len(X))

    # temporary array, set initial values to 2 (mask value - must be greater than 1)
    r2 = np.zeros_like(X) + 2
    breakpoints = []

    # upper half of distribution (why do they call this the upper half when start is 0.4?)
    # start at 0.4
    start = np.argmin(np.absolute(quantiles - 0.4))

    # TODO: figure out if width is ever larger than the min_width of 10 points
    width = max(round(window_width * npoints), min_width)

    # iterate over all windows, calculating r2
    # store r2 values in middle of window
    # TODO: need to figure out how this works when width is not an even number
    for right in range(start, len(X) + 1):
        left = right - width
        s = slice(left, right)
        mid = round((left + right) / 2)
        r2[mid] = np.corrcoef(X[s], y[s])[0, 1] ** 2

    # select breakpoints for the upper half of the distribution
    for bp in range(max_breakpoints // 2):  # this means max_breakpoints must always be even
        mind = np.argmin(r2)  # find minimum r2 index location
        breakpoints.append(mind)

        # break points cannot be adjacent to one another
        # mask r2 values with a buffer of min_width
        # todo: should be the width of the window at mind (which may or may not be min_width)
        r2[mind - min_width : mind + min_width + 1] = 1

    # lower half of distribution
    # start at 0.4 or the first breakpoint
    start = min(breakpoints) if breakpoints else start
    # likely need this to avoid recomputing r2 and picking the same breakpoint twice
    start -= (min_width // 2) + 1

    # iterate over all windows, calculating r2
    # this time, the window trails the percentile
    for left in range(start, -1, -1):
        right = left + width
        s = slice(left, right)
        mid = round((left + right) / 2)
        r2[mid] = np.corrcoef(X[s], y[s])[0, 1] ** 2

    # find the last three breakpoints
    for bp in range(max_breakpoints // 2):  # this means max_breakpoints must always be even
        mind = np.argmin(r2[:start])  # find minimum r2, only look at the first part of the array
        breakpoints.append(mind)  # breakpoint is in the center of the window

        # break points cannot be adjacent to one another
        # mask r2 values with a buffer of min_width
        # todo: should be the width of the window at mind (which may or may not be min_width)
        r2[mind - min_width : mind + min_width + 1] = 1

    # TODO: handle cases where we don't have breakpoints in one half of the distribution

    return X[np.sort(breakpoints)]


class PiecewiseLinearRegression(RegressorMixin, BaseEstimator):
    """ Piecewise Linear Regression

    Parameters
    ----------
    n_segments : int, default=7
        The desired number of line segments.
    fit_option : {"auto", "fast", or "arrm" }, default='auto'
        The method to use for fitting the piecewise linear regression.
    pwlf_kwargs : dict, default=None
        Additional keyword arguments to pass to the PiecewiseLinFit init method.

    Attributes
    ----------
    TODO

    See Also
    --------
    TODO
    """

    _fit_attributes = ['model_', 'fit_breaks_']

    def __init__(self, n_segments=7, fit_option='auto', pwlf_kwargs=None):

        if pwlf is None:
            raise ImportError('pwlf is not installed')

        self.n_segments = n_segments
        self.fit_option = fit_option
        self.pwlf_kwargs = pwlf_kwargs

    def fit(self, X, y, **kwargs):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=True)
        X = check_max_features(X)

        pwlf_kws = default_none_kwargs(self.pwlf_kwargs)

        self.model_ = pwlf.PiecewiseLinFit(X[:, 0], y, **pwlf_kws)

        if self.fit_option == 'auto':
            self.fit_breaks_ = self.model_.fit(self.n_segments, **kwargs)
        elif self.fit_option == 'arrm':
            self.fit_breaks_ = arrm_breakpoints(X, y, 0.05, self.n_segments)
            _ = self.model_.fit_with_breaks(self.fit_breaks_, **kwargs)
        elif self.fit_option == 'fast':
            self.fit_breaks_ = self.model_.fitfast(self.n_segments, **kwargs)
        else:
            raise ValueError(f"unsupported fit_option '{self.fit_option}'")

        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = check_max_features(X)

        return self.model_.predict(X[:, 0])
