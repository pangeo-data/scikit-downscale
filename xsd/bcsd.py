#!/usr/bin/env python
''' Bias Correction and Statistical Downscaling (BCSD) method'''

# 1. set list of models to use -- will these be the same 97 as in the CONUS
# dataset?  if using new ones, need to make sure that there are both
# retrospective and projection runs for the same model. (edited)
#
# 2.  regrid those model projections (pr, tas, tasmin and tasmax) to common
# (1 degree)? grid -- can also clip to a desired domain in process (edited)
#
# 3. from historical climate model runs, calculate climo PDFs by model, month
# and cell.  We can handle extrapolation by fitting distributions (ie normal
# for temperature; EV1 or EV3 for precip, fit to the upper half of the
# distribution.
#
# 4.  upscale obs historical data to the GCM grid.  Depending on the period of
# record available, this will control the POR of the bias-correction.
#
# 5.  calculate climo PDFs for obs data with extrapolation distrib. fits as in
# step 3.
#
# 6.  BC projection precip at the model grid scale, then calculate
# multiplicative anomalies relative to the historical climo mean.  Finally
# interpolate anomalies to target forcing resolution.
#
# 7.  calculate running mean temperature shift for model projections, save,
# and subtract from model projections.
#
# 8.  BC projection temperatures after mean shift removal, then add back the
# mean shift.  Calculate additive anomalies relative to historical climo mean
# and interpolate to target forcing resolution.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import xarray as xr
import xesmf as xe

from .quantile_mapping import quantile_mapping_by_group


def get_bounds(obj, lat_var='lat', lon_var='lon'):
    ''' Determine the latitude and longitude bounds of a xarray object'''
    return {'lat': (obj[lat_var].values.min(), obj[lat_var].values.max()),
            'lon': (obj[lon_var].values.min(), obj[lon_var].values.max())}


def _make_source_grid(obj):
    ''' Add longitude and latitude bounds to an xarray object

    Note
    ----
    This function is only valid if the object is already on a regular lat/lon
    grid.
    '''

    if obj['lon'].ndim == 2:
        return obj

    lon_step = np.diff(obj['lon'].values[:2])[0]
    lat_step = np.diff(obj['lat'].values[:2])[0]

    lon_bounds = np.append(obj['lon'].values - 0.5*lon_step,
                           obj['lon'].values[-1] + 0.5*lon_step)
    obj.coords['lon_b'] = ('x_b', lon_bounds)
    lat_bounds = np.append(obj['lat'].values - 0.5*lat_step,
                           obj['lat'].values[-1] + 0.5*lat_step)
    obj.coords['lat_b'] = ('y_b', lat_bounds)
    return obj


def _running_mean(obj, **kwargs):
    '''helper function to apply rolling mean to groupby object'''
    return obj.rolling(**kwargs).mean()


def _regrid_to(dest, *objs, method='bilinear'):
    ''' helper function to handle regridding a batch of objects to a common
    grid
    '''
    out = []
    for obj in objs:

        if isinstance(obj, xr.DataArray):
            source = obj.to_dataset(name='array')
        else:
            source = obj

        source = _make_source_grid(source)  # add grid info if needed
        # construct the regridder
        regridder = xe.Regridder(source, dest, method)
        out.append(regridder(obj))  # do the regrid op
    return out


def bcsd(da_obs, da_train, da_predict, var='pr'):
    ''' Apply the Bias Correction and Statistical Downscaling (BCSD) method.

    Parameters
    ----------
    da_obs : xr.DataArray
        Array representing the observed (truth) values.
    da_train : xr.DataArray
        Array representing the training data.
    da_predict : xr.DataArray
        Array representing the prediction data to be corrected using the BCSD
        method.
    var : str
        Variable name triggering particular treatment of some variables. Valid
        options include {'pr', 'tmin', 'tmax', 'trange', 'tavg'}.

    Returns
    -------
    out_coarse : xr.DataArray
        Anomalies on the same grid as ``da_obs``.
    '''

    # regrid to common coarse grid
    bounds = get_bounds(da_obs)
    coarse_grid = xe.util.grid_2d(*bounds['lon'], 1, *bounds['lat'], 1)
    da_obs_coarse, da_train_coarse, da_predict_coarse = _regrid_to(
        coarse_grid, da_obs, da_train, da_predict)

    # Calc mean climatology for obs data
    da_obs_coarse_mean = da_obs_coarse.groupby('time.month').mean(dim='time')

    if var == 'pr':
        # Bias correction
        # apply quantile mapping by month
        da_predict_coarse_qm = quantile_mapping_by_group(
            da_predict_coarse, da_train_coarse, da_obs_coarse,
            grouper='time.month')

        # calculate the amonalies as a ratio of the training data
        # again, this is done month-by-month
        if (da_obs_coarse_mean.min('month') <= 0).any():
            raise ValueError('Invalid value in observed climatology')
        da_predict_coarse_anoms = (da_predict_coarse_qm.groupby('time.month')
                                   / da_obs_coarse_mean)
    else:
        train_mean_regrid = da_train_coarse.groupby(
            'time.month').mean(dim='time')

        # Calculate the 9-year running mean for each month
        da_predict_coarse_rolling_mean = da_predict_coarse.groupby(
            'time.month').apply(_running_mean, time=9, center=True,
                                min_periods=1)

        # calc shift
        da_predict_coarse_shift = da_predict_coarse_rolling_mean.groupby(
            'time.month') - train_mean_regrid

        # remove shift
        da_predict_coarse_no_shift = (da_predict_coarse
                                      - da_predict_coarse_shift)

        # Bias correction
        # apply quantile mapping by month
        da_predict_coarse_qm = quantile_mapping_by_group(
            da_predict_coarse_no_shift, da_train_coarse, da_obs_coarse,
            grouper='time.month')

        # restore the shift
        da_predict_qm_w_shift = da_predict_coarse_qm + da_predict_coarse_shift

        # calc anoms (difference)
        da_predict_coarse_anoms = (da_predict_qm_w_shift.groupby('time.month')
                                   - da_obs_coarse_mean)

    # regrid to obs grid
    out_coarse, = _regrid_to(da_obs, da_predict_coarse_anoms,
                             method='bilinear')

    # return regridded anomalies
    return out_coarse


def main():

    obs_fname = '/glade/u/home/jhamman/workdir/GARD_inputs/newman_ensemble/conus_ens_004.nc'
    train_fname = '/glade/p/ral/RHAP/gutmann/cmip/daily/CNRM-CERFACS/CNRM-CM5/historical/day/atmos/day/r1i1p1/latest/pr/*nc'
    predict_fname = '/glade/p/ral/RHAP/gutmann/cmip/daily/CNRM-CERFACS/CNRM-CM5/rcp45/day/atmos/day/r1i1p1/latest/pr/*nc'

    out = xr.Dataset()
    for var in ['pr']:

        # get variables from the obs/training/prediction datasets
        da_obs_daily = xr.open_mfdataset(obs_fname)[var]
        da_obs = da_obs_daily.resample(time='MS').mean('time').load()
        da_train = xr.open_mfdataset(train_fname)[var].resample(
            time='MS').mean('time').load()
        da_predict = xr.open_mfdataset(predict_fname)[var].resample(
            time='MS').mean('time').load()

        out[var] = bcsd(da_obs, da_train, da_predict, var=var)

    out_file = './test.nc'
    print('writing outfile %s' % out_file)
    out.to_netcdf(out_file)


if __name__ == '__main__':
    main()
