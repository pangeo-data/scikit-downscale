import numpy as np
import pandas as pd
import probscale
import scipy
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt


def get_sample_data(kind):

    if kind == 'training':
        data = xr.open_zarr('../data/downscale_test_data.zarr.zip', group=kind)
        # extract 1 point of training data for precipitation and temperature
        df = (
            data.isel(point=0)
            .to_dataframe()[['T2max', 'PREC_TOT']]
            .rename(columns={'T2max': 'tmax', 'PREC_TOT': 'pcp'})
        )
        df['tmax'] -= 273.13
        df['pcp'] *= 24
        return df.resample('1d').first()
    elif kind == 'targets':
        data = xr.open_zarr('../data/downscale_test_data.zarr.zip', group=kind)
        # extract 1 point of training data for precipitation and temperature
        return (
            data.isel(point=0)
            .to_dataframe()[['Tmax', 'Prec']]
            .rename(columns={'Tmax': 'tmax', 'Prec': 'pcp'})
        )
    elif kind == 'wind-hist':
        return (
            xr.open_dataset(
                '../data/uas/uas.hist.CanESM2.CRCM5-UQAM.day.NAM-44i.raw.Colorado.19801990.nc'
            )['uas']
            .sel(lat=40.25, lon=-109.2, method='nearest')
            .squeeze()
            .to_dataframe()[['uas']]
        )
    elif kind == 'wind-obs':
        return (
            xr.open_dataset('../data/uas/uas.gridMET.NAM-44i.Colorado.19801990.nc')['uas']
            .sel(lat=40.25, lon=-109.2, method='nearest')
            .squeeze()
            .to_dataframe()[['uas']]
        )
    elif kind == 'wind-rcp':
        return (
            xr.open_dataset(
                '../data/uas/uas.rcp85.CanESM2.CRCM5-UQAM.day.NAM-44i.raw.Colorado.19902000.nc'
            )['uas']
            .sel(lat=40.25, lon=-109.2, method='nearest')
            .squeeze()
            .to_dataframe()[['uas']]
        )
    else:
        raise ValueError(kind)

    return df


def prob_plots(x, y, y_hat, shape=(2, 2), figsize=(8, 8)):

    fig, axes = plt.subplots(*shape, sharex=True, sharey=True, figsize=figsize)

    scatter_kws = dict(label='', marker=None, linestyle='-')
    common_opts = dict(plottype='qq', problabel='', datalabel='')

    for ax, (label, series) in zip(axes.flat, y_hat.items()):

        scatter_kws['label'] = 'original'
        fig = probscale.probplot(x, ax=ax, scatter_kws=scatter_kws, **common_opts)

        scatter_kws['label'] = 'target'
        fig = probscale.probplot(y, ax=ax, scatter_kws=scatter_kws, **common_opts)

        scatter_kws['label'] = 'corrected'
        fig = probscale.probplot(series, ax=ax, scatter_kws=scatter_kws, **common_opts)
        ax.set_title(label)
        ax.legend()

    [ax.set_xlabel('Standard Normal Quantiles') for ax in axes[-1]]
    [ax.set_ylabel('Temperature [C]') for ax in axes[:, 0]]
    [fig.delaxes(ax) for ax in axes.flat[len(y_hat.keys()) :]]
    fig.tight_layout()

    return fig


def zscore_ds_plot(training, target, future, corrected):
    labels = ['training', 'future', 'target', 'corrected']
    colors = {k: c for (k, c) in zip(labels, sns.color_palette('Set2', n_colors=4))}

    alpha = 0.5

    time_target = pd.date_range('1980-01-01', '1989-12-31', freq='D')
    time_training = time_target[~((time_target.month == 2) & (time_target.day == 29))]
    time_future = pd.date_range('1990-01-01', '1999-12-31', freq='D')
    time_future = time_future[~((time_future.month == 2) & (time_future.day == 29))]

    plt.figure(figsize=(8, 4))
    plt.plot(time_training, training.uas, label='training', alpha=alpha, c=colors['training'])
    plt.plot(time_target, target.uas, label='target', alpha=alpha, c=colors['target'])

    plt.plot(time_future, future.uas, label='future', alpha=alpha, c=colors['future'])
    plt.plot(
        time_future, corrected.uas, label='corrected', alpha=alpha, c=colors['corrected'],
    )

    plt.xlabel('Time')
    plt.ylabel('Eastward Near-Surface Wind (m s-1)')
    plt.legend()

    return


def zscore_correction_plot(zscore):
    training_mean = zscore.fit_stats_dict_['X_mean']
    training_std = zscore.fit_stats_dict_['X_std']
    target_mean = zscore.fit_stats_dict_['y_mean']
    target_std = zscore.fit_stats_dict_['y_std']

    future_mean = zscore.predict_stats_dict_['meani']
    future_mean = future_mean.groupby(future_mean.index.dayofyear).mean()
    future_std = zscore.predict_stats_dict_['stdi']
    future_std = future_std.groupby(future_std.index.dayofyear).mean()
    corrected_mean = zscore.predict_stats_dict_['meanf']
    corrected_mean = corrected_mean.groupby(corrected_mean.index.dayofyear).mean()
    corrected_std = zscore.predict_stats_dict_['stdf']
    corrected_std = corrected_std.groupby(corrected_std.index.dayofyear).mean()

    labels = ['training', 'future', 'target', 'corrected']
    colors = {k: c for (k, c) in zip(labels, sns.color_palette('Set2', n_colors=4))}

    doy = 20

    plt.figure()
    x, y = _gaus(training_mean, training_std, doy)
    plt.plot(x, y, c=colors['training'], label='training')
    x, y = _gaus(target_mean, target_std, doy)
    plt.plot(x, y, c=colors['target'], label='target')
    x, y = _gaus(future_mean, future_std, doy)
    plt.plot(x, y, c=colors['future'], label='future')
    x, y = _gaus(corrected_mean, corrected_std, doy)
    plt.plot(x, y, c=colors['corrected'], label='corrected')
    plt.legend()

    return


def _gaus(mean, std, doy):
    mu = mean[doy]
    sigma = std[doy]

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = scipy.stats.norm.pdf(x, mu, sigma)
    return x, y
