#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import calendar

import numpy as np
import pandas as pd
import xarray as xr

from .quantile_mapping import quantile_mapping_by_group


REGRID = "CDO"
if REGRID == 'CDO':
    import tempfile
    from cdo import Cdo
    cdo = Cdo()
elif REGRID == 'XESMF':
    import xesmf as xe


def get_bounds(obj):
    ''' Determine the latitude and longitude bounds of a xarray object'''
    if 'lat' in obj.coords:
        lat_var = 'lat'
        lon_var = 'lon'
    elif 'xc' in obj.coords:
        lat_var = 'yc'
        lon_var = 'xc'
    else:
        raise ValueError('did not find coordinate variables in %s' % obj)
    return {'lat': (obj[lat_var].values.min(), obj[lat_var].values.max()),
            'lon': (obj[lon_var].values.min(), obj[lon_var].values.max())}


def make_coarse_grid(bounds, res=1.):
    lats = np.arange(*bounds['lat'], step=res)
    lons = np.arange(*bounds['lon'], step=res)
    grid = xr.DataArray(np.ones((len(lats), len(lons)), dtype='i4'),
                        dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons})
    ds = grid.to_dataset(name='grid')
    ds['lon'].attrs = dict(long_name="longitude of grid cell center",
                           units="degrees_east")
    ds['lat'].attrs = dict(long_name="latitude of grid cell center",
                           units="degrees_north")
    ds['lon'].encoding = {'_FillValue': None}
    ds['lat'].encoding = {'_FillValue': None}
    ds.attrs['title'] = 'simple coarse grid for BCSD regridders'
    return ds


def _running_mean(obj, **kwargs):
    '''helper function to apply rolling mean to groupby object'''
    return obj.rolling(**kwargs).mean()


if REGRID == 'CDO':
    def _regrid_to(dest, *objs, name=None):
        ''' helper function to handle regridding a batch of objects to a common
        grid
        '''
        if isinstance(dest, xr.Dataset):
            f = tempfile.NamedTemporaryFile(prefix='bcsd_dest_', suffix='.nc', delete=False)
            f.close()
            dest.to_netcdf(f.name, engine='scipy')
            dest = f.name

        out = []
        for obj in objs:
            if isinstance(obj, xr.DataArray):
                input_ds = obj.to_dataset(name=name)
            else:
                input_ds = obj

            if isinstance(input_ds, xr.Dataset):
                f = tempfile.NamedTemporaryFile(prefix='bcsd_src_', suffix='.nc', delete=False)
                f.close()
                input_ds.to_netcdf(f.name, engine='scipy')
                input_ds = f.name

            out.append(cdo.remapbil(dest, input=input_ds, returnXArray=name).load())
        return out
else:   
    def _regrid_to(dest, *objs, method='bilinear'):
        ''' helper function to handle regridding a batch of objects to a common
        grid
        '''
        out = []
        for obj in objs:

            if isinstance(obj, xr.DataArray):
                source = obj.to_dataset(name='array')
                source['mask'] = obj.isel(time=0).notnull()
            else:
                source = obj
                if 'mask' not in source:
                    var = next(iter(obj.variables))
                    source['mask'] = obj[var].isel(time=0).notnull()

            # construct the regridder
            regridder = xe.Regridder(source, dest, method, reuse_weights=False)
            out.append(regridder(obj))  # do the regrid op
        return out


def bcsd(ds_obs, ds_train, ds_predict, var='pcp'):
    ''' Apply the Bias Correction and Statistical Downscaling (BCSD) method.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Dataset representing the observed (truth) values. Must include a
        variable with name equal to the ``var`` keyword argument.
    ds_train : xr.Dataset
        Dataset representing the training data.
    ds_predict : xr.Dataset
        Dataset representing the prediction data to be corrected using the BCSD
        method.
    var : str
        Variable name triggering particular treatment of some variables. Valid
        options include {'pcp', 'tmin', 'tmax', 'trange', 'tavg'}.

    Returns
    -------
    out_coarse : xr.DataArray
        Anomalies on the same grid as ``ds_obs``.
    '''

    # regrid to common coarse grid
    bounds = get_bounds(ds_obs)
    coarse_grid = make_coarse_grid(bounds, 1.)
    # coarse_grid = 'r360x180'
    (da_obs_coarse, da_train_coarse, da_predict_coarse) = _regrid_to(  # pylint: disable=unbalanced-tuple-unpacking
        coarse_grid, ds_obs, ds_train, ds_predict)

    mask = da_obs_coarse.any(dim='time')
    da_obs_coarse = da_obs_coarse.where(mask)

    # Calc mean climatology for obs data
    da_obs_coarse_mean = da_obs_coarse.groupby('time.month').mean(dim='time')

    if var == 'pcp':
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
    out_coarse, = _regrid_to(ds_obs, da_predict_coarse_anoms)  # pylint: disable=unbalanced-tuple-unpacking

    # return regridded anomalies
    return out_coarse


def get_month_slice(year, month):
    '''helper function to create a slice for 1 month given a year/month'''
    start = '{:04d}-{:02d}-01'.format(year, month)
    last_day = dpm(year, month)
    stop = '{:04d}-{:02d}-{:02d}'.format(year, month, last_day)
    return slice(start, stop)


def dpm(year, month):
    '''days per month'''
    return calendar.monthrange(year, month)[1]


def get_obs_data(obs, month, obs_year, anom_year):
    obs_dpm = dpm(obs_year, month)
    anom_dpm = dpm(anom_year, month)
    # the more common/simple case
    if month != 2 or anom_dpm == obs_dpm:
        tslice = get_month_slice(obs_year, month)
        return obs.sel(time=tslice)
    elif obs_dpm > anom_dpm:
        # drop the last day of the obs
        tslice = slice(
            '{:04d}-{:02d}-01'.format(obs_year, month),
            '{:04d}-{:02d}-{:02d}'.format(obs_year, month, anom_dpm))
        return obs.sel(time=tslice)
    else:
        # repeat the last day so the month is long enough
        repeats = anom_dpm - obs_dpm
        assert repeats == 1
        tslice = get_month_slice(obs_year, month)
        temp = obs.sel(time=tslice)
        last = temp.isel(time=-1)
        last.coords['time'].values = (last.coords['time'].values +
                                      np.timedelta64(1, 'D'))  # pylint: disable=too-many-function-args
        to_concat = [temp, last]
        return xr.concat(to_concat, dim='time')


def disagg(da_obs, da_anoms, var='pcp'):
    # purely random month selection
    years = xr.DataArray(np.random.randint(low=da_obs['time.year'].min(),
                                           high=da_obs['time.year'].max(),
                                           size=len(da_anoms['time'])),
                         dims='time', coords={'time': da_anoms['time']})

    # loop through the months and apply the anomalies
    disag_out = []
    for i, (year, month) in enumerate(zip(years.values,
                                          years['time.month'].values)):
        anom = da_anoms.isel(time=i)
        anom_time = pd.Timestamp(anom['time'].values)
        obs = get_obs_data(da_obs, month, year, anom_time.year)
        if var == 'pcp':
            disag_data = obs * anom
        else:
            disag_data = obs + anom
        if len(disag_data['time']) != dpm(anom_time.year, month):
            raise ValueError(
                'disag_data has length %d whereas this month should have '
                'length %d' % (len(disag_data['time']),
                               dpm(anom_time.year, month)))

        disag_data['time'] = pd.date_range(anom_time, freq='D',
                                           periods=len(disag_data['time']))
        disag_data.coords['%s_anom_year' % var] = xr.Variable(
            'time', [year] * len(disag_data['time']))

        disag_out.append(disag_data)

    print('disag_out', disag_out)
    print('sample', xr.concat(disag_out[:10], dim='time'))
    # concat all months together (they are already sorted)
    return xr.concat(disag_out, dim='time')
