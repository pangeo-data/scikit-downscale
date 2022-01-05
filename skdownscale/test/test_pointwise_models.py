import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.utils.estimator_checks import parametrize_with_checks

from skdownscale.pointwise_models import (
    AnalogRegression,
    BcsdPrecipitation,
    BcsdTemperature,
    EquidistantCdfMatcher,
    LinearTrendTransformer,
    PaddedDOYGrouper,
    PureAnalog,
    PureRegression,
    QuantileMapper,
    QuantileMappingReressor,
    ZScoreRegressor,
)


@pytest.fixture(scope='module')
def sample_X_y(n=365):
    index = pd.date_range('2019-01-01', periods=n)
    X = pd.DataFrame(
        {'foo': np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 10, 'bar': np.random.rand((n))},
        index=index,
    )
    y = X['foo'] + 2
    return X, y


@parametrize_with_checks(
    [
        # Regressors
        AnalogRegression(),
        BcsdPrecipitation(),
        BcsdTemperature(),
        PureAnalog(),
        PureRegression(),
        ZScoreRegressor(),
        QuantileMappingReressor(n_endpoints=2),
        EquidistantCdfMatcher(kind='difference', n_endpoints=2),
        EquidistantCdfMatcher(kind='ratio', n_endpoints=2),
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
    'model',
    [
        BcsdTemperature(),
        PureAnalog(),
        AnalogRegression(),
        PureRegression(),
        ZScoreRegressor(),
        QuantileMappingReressor(),
        QuantileMappingReressor(extrapolate='min'),
        QuantileMappingReressor(extrapolate='max'),
        QuantileMappingReressor(extrapolate='both'),
        QuantileMappingReressor(extrapolate='1to1'),
        EquidistantCdfMatcher(),
        EquidistantCdfMatcher(extrapolate='min'),
        EquidistantCdfMatcher(extrapolate='max'),
        EquidistantCdfMatcher(extrapolate='both'),
        EquidistantCdfMatcher(extrapolate='1to1'),
    ],
)
def test_linear_model(model):

    n = 365
    # TODO: add test for time other time ranges (e.g. < 365 days)
    index = pd.date_range('2019-01-01', periods=n)

    X = pd.DataFrame({'foo': np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 10}, index=index)
    y = X + 2

    model.fit(X, y)
    y_hat = model.predict(X)
    assert len(y_hat) == len(X)


@pytest.mark.parametrize(
    'model_cls', [PureAnalog, AnalogRegression, PureRegression],
)
def test_models_with_multiple_features(sample_X_y, model_cls):
    X, y = sample_X_y
    model = model_cls()
    model.fit(X, y)
    y_hat = model.predict(X)
    assert len(y_hat) == len(X)


@pytest.mark.parametrize(
    'kind', ['best_analog', 'sample_analogs', 'weight_analogs', 'mean_analogs'],
)
def test_gard_analog_models(sample_X_y, kind):
    X, y = sample_X_y

    # test non threshold modeling
    model = PureAnalog(kind=kind, n_analogs=3)
    model.fit(X, y)
    out = model.predict(X)
    y_hat = out['pred']
    error = out['prediction_error']
    prob = out['exceedance_prob']
    assert len(prob) == len(error) == len(y_hat) == len(X)
    assert (prob == 1).all()

    # test threshold modeling
    model = PureAnalog(kind=kind, n_analogs=3, thresh=0)
    model.fit(X, y)
    out = model.predict(X)
    y_hat = out['pred']
    error = out['prediction_error']
    prob = out['exceedance_prob']
    assert len(prob) == len(error) == len(y_hat) == len(X)
    assert (prob <= 1).all()
    assert (prob >= 0).all()


@pytest.mark.parametrize('thresh', [None, 3])
def test_gard_analog_regression_models(sample_X_y, thresh):
    X, y = sample_X_y

    model = AnalogRegression(thresh=thresh)
    model.fit(X, y)
    out = model.predict(X)
    y_hat = out['pred']
    error = out['prediction_error']
    prob = out['exceedance_prob']
    assert len(prob) == len(error) == len(y_hat) == len(X)
    if model.thresh:
        assert (prob <= 1).all()
        assert (prob >= 0).all()
    else:
        assert (prob == 1).all()


@pytest.mark.parametrize('thresh', [None, 3])
def test_gard_pure_regression_models(sample_X_y, thresh):
    X, y = sample_X_y

    model = PureRegression(thresh=thresh)
    model.fit(X, y)
    out = model.predict(X)
    y_hat = out['pred']
    error = out['prediction_error']
    prob = out['exceedance_prob']
    assert len(prob) == len(error) == len(y_hat) == len(X)
    if model.thresh:
        assert (prob <= 1).all()
        assert (prob >= 0).all()
    else:
        assert (prob == 1).all()


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


def test_EquidistantCdfMatcher():
    x = np.arange(1, 22)
    projected_change = 2
    bias = 3

    X_train = pd.DataFrame(x)
    y_train = pd.DataFrame(x + bias)

    for kind in ['difference', 'ratio']:
        if kind == 'difference':
            X_test = pd.DataFrame(x + projected_change)
        elif kind == 'ratio':
            X_test = pd.DataFrame(x * projected_change)

        bias_correction_model = EquidistantCdfMatcher(kind=kind)
        bias_correction_model.fit(X=X_train, y=y_train)
        y_test = bias_correction_model.predict(X_test)

        if kind == 'difference':
            assert (y_test.reshape(-1, 1) == (y_train.values + projected_change)).all()
        elif kind == 'ratio':
            assert (y_test.reshape(-1, 1) == (y_train.values * projected_change)).all()
