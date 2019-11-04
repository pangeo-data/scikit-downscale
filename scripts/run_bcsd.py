#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

import pandas as pd
import xarray as xr

from xsd.bcsd import bcsd, disagg

import gc


@click.command()
@click.option('--obs', type=str, help='Obs filename')
@click.option('--ref', type=str, help="Reference filename")
@click.option('--predict', type=str, help="Predict filename")
@click.option('--out_prefix', type=str, help="output_prefix")
@click.option('--kind', type=str, help="domain flag")
def main(obs, ref, predict, out_prefix, kind):
    obs = obs.replace('\\', '')
    ref = ref.replace('\\', '')
    predict = predict.replace('\\', '')
    print(obs, ref, predict)
    # make out directory
    dirname = os.path.dirname(out_prefix)
    os.makedirs(dirname, exist_ok=True)

    if kind == 'ak':
        anoms, out = run_ak(obs, ref, predict)
    elif kind == 'hi':
        anoms, out = run_hi(obs, ref, predict)

    # watch output files
    anoms.load().to_netcdf(os.path.join(out_prefix + 'anoms.nc'))
    out.load().to_netcdf(os.path.join(out_prefix + 'out.nc'))


def run_ak(obs_fname, train_fname, predict_fname):
    # Alaska
    varnames = {'tmax': 'tasmax', 'tmin': 'tasmin', 'pcp': 'pr'}
    chunks = None
    if 'hist' in predict_fname:
        predict_time_bounds = slice('1950', '2006')
    else:
        predict_time_bounds = slice('2006', '2099')

    anoms = xr.Dataset()
    out = xr.Dataset()

    def preproc(ds):
        coords = ['xc', 'yc', 'xv', 'yv']
        ds = ds.rename({'lat': 'nj', 'lon': 'ni'}).drop(['latitude', 'longitude']).set_coords(coords)
        for v in coords:
            ds[v] = ds[v].load()
        return ds

    # get variables from the obs/training/prediction datasets
    print(obs_fname, chunks)
    ds_obs = xr.open_mfdataset(obs_fname, combine='by_coords', preprocess=preproc, coords='minimal', compat='override')
    time_bounds = slice(ds_obs.indexes['time'][0].strftime('%Y-%m-%d'), ds_obs.indexes['time'][-1].strftime('%Y-%m-%d'))
    ds_obs.coords['xc'] = ds_obs['xc'].where(ds_obs['xc'] >= 0,
                                             ds_obs.coords['xc'] + 360)
    attrs_to_delete = ['grid_mapping', 'cell_methods', 'remap', 'FieldType',
                       'MemoryOrder', 'stagger', 'sr_x', 'sr_y']

    for obs_var, gcm_var in varnames.items():

        obs_keep_vars = [obs_var, 'xc', 'yc', 'xv', 'yv']
        ds_obs_daily = ds_obs[obs_keep_vars]
        ds_obs_daily[obs_var] = ds_obs_daily[obs_var].astype('f4')
        times = pd.date_range('1980-01-01', '2017-12-31', freq='D')
        ds_obs_daily = ds_obs_daily.reindex(time=times, method='nearest')
        ds_obs_1var = ds_obs_daily.resample(
            time='MS', keep_attrs=True).mean('time').load()

        for i, v in enumerate(obs_keep_vars):
            ds_obs_1var[v].attrs = ds_obs[v].attrs
            if i:
                ds_obs_1var[v].encoding['_FillValue'] = None
        for v in ds_obs_1var:
            for attr in attrs_to_delete:
                if attr in ds_obs_1var[v].attrs:
                    del ds_obs_1var[v].attrs[attr]

        if 'time' in ds_obs_1var['xv'].dims:
            ds_obs_1var['xv'] = ds_obs_1var['xv'].isel(time=0)
            ds_obs_1var['yv'] = ds_obs_1var['yv'].isel(time=0)

        print('ds_obs_1var')
        ds_obs_1var.info()

        da_train = xr.open_mfdataset(
            train_fname.format(gcm_var=gcm_var), chunks=chunks,
            combine='by_coords', data_vars='minimal')[gcm_var]
        da_train['time'] = da_train.indexes['time'].to_datetimeindex()
        da_train = da_train.sel(time=time_bounds).astype('f4').resample(
            time='MS').mean('time').load()
        da_predict = xr.open_mfdataset(
            predict_fname.format(gcm_var=gcm_var), chunks=chunks,
            combine='by_coords', data_vars='minimal')[gcm_var]
        da_predict['time'] = da_predict.indexes['time'].to_datetimeindex()
        da_predict = da_predict.sel(
                time=predict_time_bounds).astype('f4').resample(
                    time='MS').mean('time').load()

        anoms[obs_var] = bcsd(ds_obs_1var, da_train.to_dataset(name=obs_var),
                              da_predict.to_dataset(name=obs_var),
                              var=obs_var)
        out[obs_var] = disagg(ds_obs_daily[obs_var], anoms[obs_var],
                              var=obs_var)
        for var in ['xv', 'yv']:
            if var not in out.coords:
                out[var] = ds_obs_1var[var]

        gc.collect()
    return anoms, out


def run_hi(obs_fname, train_fname, predict_fname):
    varnames = {'tmax': 'tasmax', 'tmin': 'tasmin', 'pcp': 'pr'}
    chunks = None

    if 'hist' in predict_fname:
        predict_time_bounds = slice('1950', '2006')
    else:
        predict_time_bounds = slice('2006', '2099')

    anoms = xr.Dataset()
    out = xr.Dataset()

    # get variables from the obs/training/prediction datasets
    ds_obs = xr.open_mfdataset(obs_fname, chunks=chunks, combine='by_coords',
                               data_vars='minimal')
    print('obs', ds_obs)
    time_bounds = slice(ds_obs.indexes['time'][0].strftime('%Y-%m-%d'),
                        ds_obs.indexes['time'][-1].strftime('%Y-%m-%d'))

    for obs_var, gcm_var in varnames.items():
        obs_keep_vars = [obs_var, 'lon', 'lat']
        ds_obs_daily = ds_obs[obs_keep_vars]
        ds_obs_daily[obs_var] = ds_obs_daily[obs_var].astype('f4')
        times = pd.date_range('1990-01-01', '2014-12-31', freq='D')
        ds_obs_daily = ds_obs_daily.reindex(time=times, method='nearest')
        ds_obs_1var = ds_obs_daily.resample(
            time='MS', keep_attrs=True).mean('time').load()

        for i, v in enumerate(obs_keep_vars):
            ds_obs_1var[v].attrs = ds_obs[v].attrs
            if i:
                ds_obs_1var[v].encoding['_FillValue'] = None

        print('ds_obs_1var')
        ds_obs_1var.info()

        da_train = xr.open_mfdataset(
            train_fname.format(gcm_var=gcm_var), chunks=chunks,
            combine='by_coords', data_vars='minimal')[gcm_var].pipe(resample, time_bounds)
        da_predict = xr.open_mfdataset(
            predict_fname.format(gcm_var=gcm_var), chunks=chunks,
            combine='by_coords', data_vars='minimal')[gcm_var].pipe(resample, time_bounds)
        print('da_train', da_train)
        print('da_predict', da_predict)
        anoms[obs_var] = bcsd(ds_obs_1var, da_train.to_dataset(name=obs_var),
                              da_predict.to_dataset(name=obs_var),
                              var=obs_var)
        out[obs_var] = disagg(ds_obs_daily[obs_var], anoms[obs_var],
                              var=obs_var)

        gc.collect()
    return anoms, out


def resample(da, time_bounds):
    try:
        da['time'] = da.indexes['time'].to_datetimeindex()
    except:
        pass
    return da.sel(time=time_bounds).astype('f4').resample(time='MS').mean('time').load()


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
