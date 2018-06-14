
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
# 7.  calculate running mean temperature increase for model projections, save,
# and subtract from model projections.
#
# 8.  BC projection temperatures after mean shift removal, then add back the
# mean shift.  Calculate additive anomalies relative to historical climo mean
# and interpolate to target forcing resolution.
#
# And that's it.  From there the other other scripts could handle the daily
# disag to get the final forcings.  I probably should have stuck that all in a
# 1 pager, sorry -- mainly I just though it would be good to be clear on the
# steps if we're thinking of going for gold with xarray.

import numpy as np
import xarray as xr
import xesmf as xe

from .quantile_mapping import quantile_mapping_by_group


BUFFER = 0.5


def get_bounds(ds):
    return {'lat': (ds['lat'].values.min(), ds['lat'].values.max()),
            'lon': (ds['lon'].values.min(), ds['lon'].values.max())}


def make_course_grid(bounds, step=1.0):
    ds = xr.Dataset({'lat': (['lat'], np.arange(bounds['lat'][0] - BUFFER,
                                                bounds['lat'][1] + BUFFER,
                                                step)),
                     'lon': (['lon'], np.arange(bounds['lon'][0] - BUFFER,
                                                bounds['lon'][1] + BUFFER,
                                                step))})
    return ds


def make_source_grid(obj):

    lon_step = np.diff(obj.lon.values[:2])[0]
    lat_step = np.diff(obj.lat.values[:2])[0]

    obj.coords['lon_b'] = ('x_b', np.append(obj.lon.values - 0.5*lon_step,
                           obj.lon.values[-1] + 0.5*lon_step))
    obj.coords['lat_b'] = ('y_b', np.append(obj.lat.values - 0.5*lat_step,
                           obj.lat.values[-1] + 0.5*lat_step))
    return obj


def bcsd(da_obs, da_train, da_predict, var='pr'):

    # add grid information to input arrays
    da_obs = make_source_grid(da_obs)
    da_train = make_source_grid(da_train)
    da_predict = make_source_grid(da_predict)

    # regrid to common course grid
    bounds = get_bounds(da_obs)
    course_grid = course_grid = xe.util.grid_2d(*bounds['lon'], 1, *bounds['lat'], 1)
    regridder = xe.Regridder(da_obs, course_grid, 'bilinear')
    da_obs_regrid = regridder(da_obs)

    # regrid training/predict data to common grid
    regridder = xe.Regridder(da_train, course_grid, 'bilinear')
    da_train_regrid = regridder(da_train)
    da_predict_regrid = regridder(da_predict)

    # Calc means for train/paedict
    da_train_regrid_mean = da_train_regrid.groupby(
        'time.month').mean(dim='time')

    if var == 'pr':
        # Bias correction
        # apply quantile mapping
        da_predict_regrid_qm = quantile_mapping_by_group(
            da_predict_regrid, da_train_regrid, da_obs_regrid,
            grouper='time.month')

        da_predict_regrid_anoms = da_predict_regrid_qm.groupby('time.month') / da_train_regrid_mean
    else:
        # don't do this for training period? check with andy
        da_predict_regrid_run = da_predict_regrid.groupby(
            'time.month').rolling(time='9Y', center=True).mean('time')
        # what do we do with these?
        da_predict_changes = da_predict_regrid - da_predict_regrid_run

        da_predict_regrid_qm = quantile_mapping_by_group(
            da_predict_regrid, da_train_regrid, da_obs_regrid,
            grouper='time.month')

        da_predict_regrid_qm_mean = da_predict_regrid_run + da_predict_regrid_qm
        # calc anoms (difference)

    da_predict_regrid_anoms = da_predict_regrid_qm - da_predict_regrid_qm_mean

    # regrid to obs grid
    regridder = xe.Regridder(da_predict_regrid_anoms, da_obs, 'bilinear')
    out_regrid = regridder(da_predict_regrid_anoms)

    # return regridded anomalies
    return out_regrid


def main():

    obs_fname = '/glade/u/home/jhamman/workdir/GARD_inputs/newman_ensemble/conus_ens_004.nc'
    train_fname = '/glade/p/ral/RHAP/gutmann/cmip/daily/CNRM-CERFACS/CNRM-CM5/historical/day/atmos/day/r1i1p1/latest/pr/*nc'
    predict_fname = '/glade/p/ral/RHAP/gutmann/cmip/daily/CNRM-CERFACS/CNRM-CM5/rcp45/day/atmos/day/r1i1p1/latest/pr/*nc'
    var = 'pr'

    # get variables from the obs/training/prediction datasets
    da_obs_daily = xr.open_mfdataset(obs_fname)[var]
    da_obs = da_obs_daily.resample(time='MS').mean('time').load()
    da_train = xr.open_mfdataset(train_fname)[var].resample(
        time='MS').mean('time').load()
    da_predict = xr.open_mfdataset(predict_fname)[var].resample(
        time='MS').mean('time').load()

    bcsd(da_obs, da_train, da_predict, var=var)


if __name__ == '__main__':
    main()
