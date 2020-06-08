import copy

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_FEATURE_DIM = 'variable'


def xenumerate(arr):
    """
    Multidimensional index iterator for xarray objects

    Return an iterator yielding pairs of array indexers (dicts) and values.

    Parameters
    ----------
    arr : xarray.DataArray
        Input array.

    See Also
    --------
    numpy.ndenumerate
    """

    for index, _ in np.ndenumerate(arr):
        xindex = dict(zip(arr.dims, index))
        yield xindex, arr.isel(**xindex)


def _make_mask(da, reduce_dims):
    _reduce = {d: 0 for d in reduce_dims}
    return da.isel(**_reduce, drop=True).notnull()


def _da_to_df(da, feature_dim=DEFAULT_FEATURE_DIM):
    """ manually construct dataframe """
    if feature_dim in da.dims:
        if feature_dim in da.coords:
            columns = da.coords[feature_dim]
        else:
            size_fd = dict(zip(da.dims, da.shape))[feature_dim]
            columns = [f'feature{i}' for i in range(size_fd)]
    else:
        columns = [f'{feature_dim}_0']
    data = da.transpose('time', ...).data
    df = pd.DataFrame(data, columns=columns, index=da.indexes['time'])
    return df


def _fit_wrapper(X, *args, along_dim='time', feature_dim=DEFAULT_FEATURE_DIM, **kwargs):

    if len(args) == 2:
        y, model = args
    else:
        model = args[0]
        y = None

    # create a mask for this block
    reduce_dims = [along_dim, feature_dim]
    mask = _make_mask(X, reduce_dims)

    # create the empty output array
    models = xr.DataArray(np.empty(mask.shape, dtype=np.object), coords=mask.coords, dims=mask.dims)

    scalar_obj = np.empty((1), dtype=np.object)
    for index, val in xenumerate(mask):
        mod = copy.deepcopy(model)
        if not val:
            continue
        xdf = X[index].pipe(_da_to_df, feature_dim)
        if y is not None:
            ydf = y[index].pipe(_da_to_df, feature_dim)
            scalar_obj[:] = [mod.fit(xdf, ydf, **kwargs)]
        else:
            scalar_obj[:] = [mod.fit(xdf, **kwargs)]
        models[index] = scalar_obj.squeeze()
    return models


def _predict_wrapper(X, models, along_dim=None, feature_dim=DEFAULT_FEATURE_DIM, **kwargs):

    ydims = list(X.dims)
    yshape = list(X.shape)
    ycoords = dict(X.coords)
    ycoords.pop(feature_dim)
    if feature_dim in ydims:
        ydims.pop(X.get_axis_num(feature_dim))
        yshape.pop(X.get_axis_num(feature_dim))

    y = xr.DataArray(np.empty(yshape, dtype=X.dtype), coords=ycoords, dims=ydims)

    for index, model in xenumerate(models):
        xdf = X[index].pipe(_da_to_df, feature_dim)
        ydf = model.item().predict(xdf, **kwargs)
        y[index] = ydf.squeeze()

    return y


def _transform_wrapper(X, models, feature_dim=DEFAULT_FEATURE_DIM, **kwargs):

    xtrans = xr.full_like(X, np.nan)

    for index, model in xenumerate(models):
        xdf = X[index].pipe(_da_to_df, feature_dim).drop(models.coords.keys())
        xtrans_df = model.item().transform(xdf, **kwargs)
        xtrans[index] = xtrans_df.squeeze()
    return xtrans


class PointWiseDownscaler:
    """
    Pointwise downscaling model wrapper

    Apply a scikit-learn model (e.g. Pipeline) point-by-point. The pipeline
    must implement the fit and predict methods.

    Parameters
    ----------
    model : sklearn.Pipeline or similar
        Object that implements the scikit-learn fit/predict api.
    dim : str, optional
        Dimension to apply the model along. Default is ``time``.
    """

    def __init__(self, model, dim='time'):
        self._dim = dim
        self._model = model
        self._models = None

        if not hasattr(model, 'fit'):
            raise TypeError(
                'Type %s does not have the fit method required'
                ' by PointWiseDownscaler' % type(model)
            )

    def fit(self, X, *args, **kwargs):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : xarray.DataArray or xarray.Dataset
            Training data. Must fulfill input requirements of first step of
            the pipeline. If an xarray.Dataset is passed, it will be converted
            to an array using `to_array()`.
        y : xarray.DataArray, optional
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        feature_dim : str, optional
            Name of feature dimension.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the this model. If the
            model is a sklearn Pipeline, parameters can be passed to each
            step, where each parameter name is prefixed such that parameter
            ``p`` for step ``s`` has key ``s__p``.
        """
        kws = {'along_dim': self._dim, 'feature_dim': DEFAULT_FEATURE_DIM}
        kws.update(kwargs)

        assert len(args) <= 1
        args = list(args)
        args.append(self._model)

        X = self._to_feature_x(X, feature_dim=kws['feature_dim'])

        if X.chunks:
            reduce_dims = [self._dim, kws['feature_dim']]
            mask = _make_mask(X, reduce_dims)
            template = xr.full_like(mask, None, dtype=np.object)
            self._models = xr.map_blocks(_fit_wrapper, X, args=args, kwargs=kws, template=template)
        else:
            self._models = _fit_wrapper(X, *args, **kws)

    def predict(self, X, **kwargs):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : xarray.DataArray
            Data to predict on. Must fulfill input requirements of first step
            of the model or pipeline.
        feature_dim : str, optional
            Name of feature dimension.
        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

        Returns
        -------
        y_pred : xarray.DataArray
        """

        kws = {'along_dim': self._dim, 'feature_dim': DEFAULT_FEATURE_DIM}
        kws.update(kwargs)

        X = self._to_feature_x(X, feature_dim=kws['feature_dim'])

        if X.chunks:
            return xr.map_blocks(_predict_wrapper, X, args=[self._models], kwargs=kws)
        else:
            return _predict_wrapper(X, self._models, **kws)

    def transform(self, X, **kwargs):
        """Apply transforms to the data, and transform with the final estimator

        Parameters
        ----------
        X : xarray.DataArray
            Data to transform on. Must fulfill input requirements of first step
            of the model or pipeline.
        feature_dim : str, optional
            Name of feature dimension.
        **transform_params : dict of string -> object
            Parameters to the ``transform`` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_trans : xarray.DataArray
        """

        kws = {'feature_dim': DEFAULT_FEATURE_DIM}
        kws.update(kwargs)

        X = self._to_feature_x(X, feature_dim=kws['feature_dim'])

        if X.chunks:
            return xr.map_blocks(_transform_wrapper, X, args=[self._models], kwargs=kws)
        else:
            return _transform_wrapper(X, self._models, **kws)

    def _to_feature_x(self, X, feature_dim=DEFAULT_FEATURE_DIM):
        # xarray.Dataset --> xarray.DataArray
        if isinstance(X, xr.Dataset):
            X = X.to_array(feature_dim)

        if feature_dim not in X.dims:
            X = X.expand_dims(**{feature_dim: [f'{feature_dim}_0']}, axis=1)

        X = X.transpose(self._dim, feature_dim, ...)

        return X

    def __repr__(self):
        summary = ['<skdownscale.{}>'.format(self.__class__.__name__)]
        summary.append('  Fit Status: {}'.format(self._models is not None))
        summary.append('  Model:\n    {}'.format(self._model))
        return '\n'.join(summary)
