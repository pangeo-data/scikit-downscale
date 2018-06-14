import xarray as xr


class StatDown(object):
    '''generic class for statistical downscaling'''
    def set_params(self, **kwargs):
        '''Set the parameters of this estimator.'''
        self.params = kwargs

    def train(self, target):
        '''train the statistical model'''
        # do some things
        pass

    def predict(self, predictors):
        '''predict using this method'''
        # do some things, return new
        pass

    def post_process(self, how=None):
        '''Post process predicted data'''
        # do some things
        pass

    def score(self, X, y, sample_weight=None, dim=None):
        '''Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual sum
        of squares ((y_true - y_true.mean()) ** 2).sum(). Best possible score
        is 1.0 and it can be negative (because the model can be arbitrarily
        worse). A constant model that always predicts the expected value of y,
        disregarding the input features, would get a R^2 score of 0.0.
        '''
        from .metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


@xr.register_dataarray_accessor('xsd')
class Xsd(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._sd = None

    def set_params(self, how=None, **kwargs):
        pass

    def train(self, target):
        # do some things
        pass

    def predict(self):
        # do some things, return new
        pass

    def post_process(self, how=None):
        # do some things
        pass

    def score(self):
        pass
