import numpy as np

from scipy.spatial import cKDTree

from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


class AnalogBase(LinearModel, RegressorMixin):
    _fit_attributes = ['kdtree_', 'y_']

    def fit(self, X, y):

        self.kdtree_ = cKDTree(X, **self.kdtree_kwargs)
        self.y_ = y
        
        return self


class AnalogRegression(AnalogBase):
    ''' AnalogRegression

    Parameters
    ----------
    n_analogs: int
        Number of analogs to use when building linear regression
    kdtree_kwargs : dict
        Keyword arguments to pass to the scipy.spatial.cKDTree constructor
    query_kwargs : dict
        Keyword arguments to pass to the scipy.spatial.cKDTree.query method
    lr_kwargs : dict
        Keyword arguments to pass to the sklear.linear_model.LinearRegression
        constructor

    Attributes
    ----------
    kdtree_ : scipy.spatial.cKDTree
        KDTree object
    '''

    def __init__(self, n_analogs=200, kdtree_kwargs={}, query_kwargs={}, lr_kwargs={}):

        self.n_analogs = n_analogs
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs
        self.lr_kwargs = lr_kwargs

    def predict(self, X):
        check_is_fitted(self, self._fit_attributes)

        predicted = np.empty(len(X))

        # TODO - extract from lr_model's below.
        self.stats = {}

        for i, (_, row) in enumerate(X.iterrows()):
            # predict for this time step
            predicted[i] = self._predict_one_step(row.values)

        return predicted

    def _predict_one_step(self, X):
        # get analogs
        _, inds = self.kdtree_.query(X, k=self.n_analogs, **self.query_kwargs)

        # extract data to train linear regression model
        x = self.kdtree_.data[inds]
        y = self.y_[inds]

        # train linear regression model
        lr_model = LinearRegression(**self.lr_kwargs).fit(x, y)

        # predict for this time step
        predicted = lr_model.predict(X)
        return predicted


class PureAnalog(AnalogBase):
    ''' PureAnalog

    Attributes
    ----------
    kdtree_ : scipy.spatial.cKDTree
        KDTree object
    '''

    def __init__(self, n_analogs=200, kdtree_kwargs={}, query_kwargs={}):

        self.n_analogs = n_analogs
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs

    def predict(self, X, n_analogs=200, kind='best_analog', thresh=None, stats=True):
        '''Predict using the PureAnalog model

        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Samples.

        Returns
        -------
        C : pd.DataFrame, shape (n_samples, 1)
            Returns predicted values.
        '''
        check_is_fitted(self, self._fit_attributes)

        if kind == 'best_analog':
            n_analogs = 1
        else:
            n_analogs = n_analogs

        dist, inds = self.kdtree_.query(X, k=n_analogs, **self.query_kwargs)

        analogs = self.y_[inds]

        if thresh is not None:
            # TODO: rethink how the analog threshold is applied.
            # There are certainly edge cases not dealt with properly here
            # particularly in the weight analogs case
            analog_mask = analogs > thresh
            masked_analogs = analogs[analog_mask]

        if kind == 'best_analog':
            predicted = analogs[:, 0]
        
        elif kind == 'sample_analogs':
            # get 1 random index to sample from the analogs
            rand_inds = np.random.randint(
                low=0, high=self.n_analogs, size=X.shape)
            # select the analog now
            predicted = analogs[rand_inds]
        
        elif kind == 'weight_analogs':
            # take weighted average
            # work around for zero distances (perfect matches)
            tiny = 1e-20
            weights = 1. / np.where(dist == 0, tiny, dist)
            if thresh:
                predicted = np.average(
                    masked_analogs, weights=weights, axis=1)
            else:
                predicted = np.average(analogs, weights=weights, axis=1)

        elif kind == 'mean_analogs':
            if thresh is not None:
                predicted = masked_analogs.mean(axis=1)
            else:
                predicted = analogs.mean(axis=1)
        
        else:
            raise ValueError('got unexpected kind %s' % kind)

        if thresh is not None:
            # for mean/weight cases, this fills nans when all analogs
            # were below thresh
            predicted = np.nan_to_num(predicted, nan=0.)

        if stats:
            self.stats_ = {}
            # calculate the standard deviation of the anlogs
            if thresh is not None:
                self.stats['error'] = masked_analogs.std(axis=1)
            else:
                self.stats['error'] = analogs.where(analog_mask).std(axis=1)

            # calculate the probability of precip
            if thresh is not None:
                self.stats['pop'] = np.where(analog_mask, 1, 0).mean(axis=1)

        return predicted
