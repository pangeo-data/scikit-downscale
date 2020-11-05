import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from skdownscale.pointwise_models.grouping import GroupedRegressor, PaddedDOYGrouper


def test_groupers():

    estimator = LinearRegression
    fit_grouper = PaddedDOYGrouper

    def predict_grouper(x):
        return x.dayofyear

    model = GroupedRegressor(
        estimator, fit_grouper, predict_grouper, fit_grouper_kwargs={'window': 5}
    )

    n = 1234
    index = pd.date_range('2019-01-01', periods=n)

    X = pd.DataFrame({'foo': np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 10}, index=index)
    y = X + 2

    model.fit(X, y)
    y_hat = model.predict(X)
    assert len(y_hat) == len(X)
