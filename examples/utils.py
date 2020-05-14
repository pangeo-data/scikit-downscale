import matplotlib.pyplot as plt
import pandas as pd
import probscale
import seaborn as sns
import xarray as xr


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
