import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.utils.estimator_checks import parametrize_with_checks

from skdownscale.pointwise_models import (
    AnalogRegression,
    BcsdPrecipitation,
    BcsdTemperature,
    LinearTrendTransformer,
    PaddedDOYGrouper,
    PureAnalog,
    QuantileMapper,
    QuantileMappingReressor,
    ZScoreRegressor,
)


@parametrize_with_checks(
    [
        # Regressors
        AnalogRegression(),
        BcsdPrecipitation(),
        BcsdTemperature(),
        PureAnalog(),
        ZScoreRegressor(),
        QuantileMappingReressor(n_endpoints=2),
        # transformers
        LinearTrendTransformer(),
        QuantileMapper(),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_linear_trend_roundtrip():
    # TODO: there is probably a better analytic test here
    n = 100
    trend = 1
    yint = 15

    trendline = trend * np.arange(n) + yint
    trendline = trendline.reshape(-1, 1)
    noise = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 10
    noise = noise.reshape(-1, 1)
    data = trendline + noise

    ltt = LinearTrendTransformer()

    # remove trend
    d_no_trend = ltt.fit_transform(data)

    # assert detrended data is equal to noise
    np.testing.assert_almost_equal(d_no_trend, noise, decimal=0)
    # assert linear coef is equal to trend
    np.testing.assert_almost_equal(ltt.lr_model_.coef_, trend, decimal=0)
    # assert roundtrip
    np.testing.assert_array_equal(ltt.inverse_transform(d_no_trend), data)


def test_quantile_mapper():
    n = 100
    expected = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 10
    expected = expected.reshape(-1, 1)
    with_bias = expected + 2

    mapper = QuantileMapper()
    mapper.fit(expected)
    actual = mapper.transform(with_bias)
    np.testing.assert_almost_equal(actual, expected)


@pytest.mark.xfail(reason='Need 3 part QM routine to handle bias removal')
def test_quantile_mapper_detrend():
    n = 100
    trend = 1
    yint = 15

    trendline = trend * np.arange(n) + yint
    base = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 10
    expected = base + trendline

    with_bias = expected + 2

    mapper = QuantileMapper(detrend=True)
    mapper.fit(base)
    actual = mapper.transform(with_bias)
    np.testing.assert_almost_equal(actual.squeeze(), expected)


@pytest.mark.parametrize(
    'model_cls',
    [BcsdTemperature, PureAnalog, AnalogRegression, ZScoreRegressor, QuantileMappingReressor],
)
def test_linear_model(model_cls):

    n = 365
    # TODO: add test for time other time ranges (e.g. < 365 days)
    index = pd.date_range('2019-01-01', periods=n)

    X = pd.DataFrame({'foo': np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 10}, index=index)
    y = X + 2

    model = model_cls()
    model.fit(X, y)
    y_hat = model.predict(X)
    assert len(y_hat) == len(X)


@pytest.mark.parametrize('model_cls', [BcsdPrecipitation])
def test_linear_model_prec(model_cls):

    n = 365
    # TODO: add test for time other time ranges (e.g. < 365 days)
    index = pd.date_range('2019-01-01', periods=n)

    X = pd.DataFrame({'foo': np.random.random(n)}, index=index)
    y = X + 2

    model = model_cls()
    model.fit(X, y)
    y_hat = model.predict(X)
    assert len(y_hat) == len(X)


def test_zscore_scale():
    time = pd.date_range(start='2018-01-01', end='2020-01-01')
    data_X = np.linspace(0, 1, len(time))
    data_y = data_X * 2

    X = xr.DataArray(data_X, name='foo', dims=['index'], coords={'index': time}).to_dataframe()
    y = xr.DataArray(data_y, name='foo', dims=['index'], coords={'index': time}).to_dataframe()

    data_scale_expected = [2 for i in np.zeros(364)]
    scale_expected = xr.DataArray(
        data_scale_expected, name='foo', dims=['day'], coords={'day': np.arange(1, 365)}
    ).to_series()

    zscore = ZScoreRegressor()
    zscore.fit(X, y)

    np.testing.assert_allclose(zscore.scale_, scale_expected)


def test_zscore_shift():
    time = pd.date_range(start='2018-01-01', end='2020-01-01')
    data_X = np.zeros(len(time))
    data_y = np.ones(len(time))

    X = xr.DataArray(data_X, name='foo', dims=['index'], coords={'index': time}).to_dataframe()
    y = xr.DataArray(data_y, name='foo', dims=['index'], coords={'index': time}).to_dataframe()

    shift_expected = xr.DataArray(
        np.ones(364), name='foo', dims=['day'], coords={'day': np.arange(1, 365)}
    ).to_series()

    zscore = ZScoreRegressor()
    zscore.fit(X, y)

    np.testing.assert_allclose(zscore.shift_, shift_expected)


def test_zscore_predict():
    time = pd.date_range(start='2018-01-01', end='2020-01-01')
    data_X = np.linspace(0, 1, len(time))

    X = xr.DataArray(data_X, name='foo', dims=['index'], coords={'index': time}).to_dataframe()

    shift = xr.DataArray(
        np.zeros(364), name='foo', dims=['day'], coords={'day': np.arange(1, 365)}
    ).to_series()
    scale = xr.DataArray(
        np.ones(364), name='foo', dims=['day'], coords={'day': np.arange(1, 365)}
    ).to_series()

    zscore = ZScoreRegressor()
    zscore.shift_ = shift
    zscore.scale_ = scale

    i = int(zscore.window_width / 2)
    expected = xr.DataArray(
        data_X, name='foo', dims=['index'], coords={'index': time}
    ).to_dataframe()
    expected[0:i] = 'NaN'
    expected[-i:] = 'NaN'

    out = zscore.predict(X)

    np.testing.assert_allclose(out.astype(float), expected.astype(float))


def test_paddeddoygrouper():
    index = pd.date_range(start='1980-01-01', end='1982-12-31')
    X = pd.DataFrame({'foo': np.random.random(len(index))}, index=index)
    day_groups = PaddedDOYGrouper(X)
    doy_group_list = dict(list(day_groups))

    day_of_year = 123
    days_included = np.arange(day_of_year - 15, day_of_year + 16)
    np.testing.assert_array_equal(
        np.unique(doy_group_list[day_of_year].index.dayofyear), days_included
    )


def test_BcsdTemperature_nasanex():
    index = pd.date_range(start='1980-01-01', end='1982-12-31')
    X = pd.DataFrame({'foo': np.random.random(len(index))}, index=index)
    y = pd.DataFrame({'foo': np.random.random(len(index))}, index=index)
    model_nasanex = BcsdTemperature(time_grouper='daily_nasa-nex', return_anoms=False).fit(X, y)
    assert issubclass(model_nasanex.time_grouper, PaddedDOYGrouper)
