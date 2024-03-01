#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click
import pandas as pd
import xarray as xr
from xsd.bcsd import bcsd, disagg

# Command-line interface setup using Click to parse arguments
@click.command()
@click.option('--obs', type=str, help='Observation dataset filename')
@click.option('--ref', type=str, help='Reference dataset filename')
@click.option('--predict', type=str, help='Prediction dataset filename')
@click.option('--out_prefix', type=str, help='Prefix for output files')
@click.option('--kind', type=str, help='Domain flag indicating the geographical area ("ak" for Alaska, "hi" for Hawaii)')
def main(obs, ref, predict, out_prefix, kind):
    # Sanitize file paths to avoid issues with backslashes
    obs = obs.replace('\\', '')
    ref = ref.replace('\\', '')
    predict = predict.replace('\\', '')

    # Create output directory if it does not exist
    dirname = os.path.dirname(out_prefix)
    os.makedirs(dirname, exist_ok=True)

    # Execute the main processing routine based on the 'kind' flag
    if kind in ['ak', 'hi']:
        anoms, out = run_processing(obs, ref, predict, kind)
        # Save output datasets to NetCDF files
        anoms.load().to_netcdf(os.path.join(out_prefix + 'anoms.nc'))
        out.load().to_netcdf(os.path.join(out_prefix + 'out.nc'))
    else:
        print("Error: Unsupported 'kind' value. Use 'ak' for Alaska or 'hi' for Hawaii.")

# Main processing function that handles both Alaska and Hawaii scenarios
def run_processing(obs_fname, train_fname, predict_fname, kind):
    # Define variable names for different datasets
    varnames = {'tmax': 'tasmax', 'tmin': 'tasmin', 'pcp': 'pr'}
    chunks = {'time': 365}  # Suggested chunk size for daily data

    # Time bounds for prediction, varying by dataset
    predict_time_bounds = slice('1950', '2006') if 'hist' in predict_fname else slice('2006', '2099')

    anoms = xr.Dataset()
    out = xr.Dataset()

    # Open the observation dataset and adjust coordinates if necessary
    ds_obs = xr.open_mfdataset(obs_fname, chunks=chunks, concat_dim='time', data_vars='minimal')
    time_bounds = slice(ds_obs.indexes['time'][0], ds_obs.indexes['time'][-1])
    if kind == 'ak':
        ds_obs.coords['xc'] = ds_obs['xc'].where(ds_obs['xc'] >= 0, ds_obs.coords['xc'] + 360)

    obs_keep_vars = ['xc', 'yc', 'xv', 'yv'] if kind == 'ak' else ['lon', 'lat']

    # Process each variable (tmax, tmin, pcp) in the dataset
    for obs_var, gcm_var in varnames.items():
        process_variable(ds_obs, obs_var, obs_keep_vars, train_fname, predict_fname, time_bounds, predict_time_bounds, anoms, out, kind)

    return anoms, out

# Function to process each variable individually
def process_variable(ds_obs, obs_var, obs_keep_vars, train_fname, predict_fname, time_bounds, predict_time_bounds, anoms, out, kind):
    attrs_to_delete = [
        'grid_mapping', 'cell_methods', 'remap', 'FieldType', 'MemoryOrder', 'stagger', 'sr_x', 'sr_y'
    ]
    
    # Prepare the observation dataset
    ds_obs_daily = ds_obs[obs_keep_vars + [obs_var]].astype('f4')
    times = pd.date_range(start='1980-01-01', end='2017-12-31', freq='D') if kind == 'ak' else pd.date_range(start='1990-01-01', end='2014-12-31', freq='D')
    ds_obs_daily = ds_obs_daily.reindex(time=times, method='nearest').resample(time='MS').mean('time').load()

    # Clean and prepare attributes
    for v in obs_keep_vars + [obs_var]:
        if v in ds_obs:
            ds_obs_daily[v].attrs = ds_obs[v].attrs
            if v != obs_var:
                ds_obs_daily[v].encoding['_FillValue'] = None

    # Remove unnecessary attributes
    for attr in attrs_to_delete:
        if attr in ds_obs_daily[obs_var].attrs:
            del ds_obs_daily[obs_var].attrs[attr]

    # Load training and prediction datasets
    da_train, da_predict = load_datasets(train_fname, predict_fname, gcm_var, time_bounds, predict_time_bounds, kind)

    # Apply Bias Correction and Statistical Downscaling (BCSD) method
    anoms[obs_var] = bcsd(ds_obs_daily.to_dataset(name=obs_var), da_train.to_dataset(name=obs_var), da_predict.to_dataset(name=obs_var), var=obs_var)
    out[obs_var] = disagg(ds_obs_daily[obs_var], anoms[obs_var], var=obs_var)

# Load training and prediction datasets with bias correction
def load_datasets(train_fname, predict_fname, gcm_var, time_bounds, predict_time_bounds, kind):
    chunks = {'time': 365}  # Suggested chunk size for daily data
    da_train = xr.open_mfdataset(train_fname.format(gcm_var=gcm_var), chunks=chunks, concat_dim='time', data_vars='minimal')[gcm_var].sel(time=time_bounds).astype('f4').resample(time='MS').mean('time').load()
    da_predict = xr.open_mfdataset(predict_fname.format(gcm_var=gcm_var), chunks=chunks, concat_dim='time', data_vars='minimal')[gcm_var].sel(time=predict_time_bounds).astype('f4').resample(time='MS').mean('time').load()
    return da_train, da_predict

if __name__ == '__main__':
    main()  # Execute the CLI
