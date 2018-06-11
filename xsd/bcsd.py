
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

from storylines.tools import plotting_positions

BUFFER = 0.5


class Qmap(object):

    def __init__(self, **kwargs):
        pass

    def _cdf(self, obj):
        axis = obj.get_axis_num('time')
        dims = list(obj.dims)
        dims[axis] = 'quantile'
        n_quantiles = obj.shape[axis]
        coords = obj.coords
        coords['quantile'] = plotting_positions(n_quantiles)
        del coords['time']

        return xr.DataArray(np.sort(obj.values, axis=axis),
                            dims=dims, coords=coords)

    def fit(self, train, target):

        self.train_cdf = self._cdf(train)
        self.target_cdf = self._cdf(target)

    def predict(self, predict):

        self


def get_bounds(ds):
    return {'lat', (ds['lat'].values.min(), ds['lat'].values.max()),
            'lon', (ds['lon'].values.min(), ds['lon'].values.max())}


def make_course_grid(bounds, step=1.0):
    ds = xr.Dataset({'lat': (['lat'], np.linspace(bounds['lat'][0] - BUFFER,
                                                  bounds['lat'][1] + BUFFER,
                                                  step)),
                     'lon': (['lon'], np.linspace(bounds['lon'][0] - BUFFER,
                                                  bounds['lon'][1] + BUFFER,
                                                  step))})
    return ds


def bcsd(da_obs, da_train, da_predict, var='precip'):

    # regrid to common course grid
    bounds = get_bounds(da_obs)
    course_grid = make_course_grid(bounds)
    regridder = xe.Regridder(da_obs, course_grid, 'conservative')
    da_obs_regrid = regridder(da_obs)

    regridder = xe.Regridder(da_train, course_grid, 'conservative')
    da_train_regrid = regridder(da_train)
    da_predict_regrid = regridder(da_predict)

    # Calc means for train/paedict
    da_train_regrid_mean = da_train_regrid.groupby(
        'time.month').mean(dim='time')

    if var == 'precip':
        # Calc CDFs
        # sort op
        # calc quantiles (x values)
        # do this for all three datasets
        # TODO: figure out extrapolation
        qmap = Qmap()
        qmap.fit(da_train_regrid, da_obs_regrid)

        # Bias correction
        # apply quantile mapping
        da_predict_regrid_qm = qmap.predict(da_predict_regrid)

        da_predict_regrid_anoms = da_predict_regrid_qm.groupby('time.month') / da_train_regrid_mean
    else:
        # don't do this for training period? check with andy
        da_predict_regrid_run = da_predict_regrid.groupby('time.month').rolling(time='9Y', center=True).mean('time')
        da_predict_changes = da_predict_regrid - da_predict_regrid_run
        qmap = Qmap(da_train_regrid, da_obs)
        da_predict_regrid_qm = qmap.predict(da_predict_changes)

        da_predict_regrid_qm_mean = da_predict_regrid_run + da_predict_regrid_qm
        # calc anoms (difference)

    # regrid to obs grid
    regridder = xe.Regridder(da_predict_regrid_anoms, da_obs, 'bilinear')
    out_regrid = regridder(da_predict_regrid_anoms)

    return out_regrid


def main():

    obs_fname = ''
    train_fname = ''
    predict_fname = ''
    var = 'pr'

    # get variables from the obs/training/prediction datasets
    da_obs_daily = xr.open_mfdataset(obs_fname)[var]
    da_obs = da_obs_daily.resample(time='MS').mean('time')
    da_train = xr.open_mfdataset(train_fname)[var].resample(
        time='MS').mean('time')
    da_predict = xr.open_mfdataset(predict_fname)[var].resample(
        time='MS').mean('time')

    bcsd(da_obs, da_train, da_predict, var=var)


if __name__ == '__main__':
    main()
