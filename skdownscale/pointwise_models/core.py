from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import DTypeLike

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
    """manually construct dataframe"""
    if feature_dim in da.dims:
        if feature_dim in da.coords:
            columns = da.coords[feature_dim]
        else:
            size_fd = dict(zip(da.dims, da.shape))[feature_dim]
            columns = [f'feature{i}' for i in range(size_fd)]
    else:
        columns = [f'{feature_dim}_0']
    data = da.transpose('time', ...).data

    # Get time index, handling cases where it might not be a proper pandas index
    if 'time' not in da.dims:
        raise ValueError('DataArray must have a "time" dimension')

    try:
        time_index = da.indexes['time']
    except (KeyError, AttributeError):
        # If indexes doesn't exist or 'time' is not in indexes, use the coordinate values
        if 'time' in da.coords:
            time_index = da.coords['time'].values
        else:
            # Fallback to range index if no time coordinate exists
            time_index = pd.RangeIndex(da.sizes['time'])

    return pd.DataFrame(data, columns=columns, index=time_index)


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
    models = xr.DataArray(
        np.full(mask.shape, None, dtype=object), coords=mask.coords, dims=mask.dims
    )

    scalar_obj = np.empty((1), dtype=object)
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


def _predict_wrapper(
    X,
    along_dim=None,
    feature_dim=DEFAULT_FEATURE_DIM,
    n_outputs=1,
    output_names=None,
    models=None,
    **kwargs,
):
    # When models is passed via kwargs (for dask), select the subset matching X's coordinates
    if models is not None:
        # Get dimensions that exist in both models and X (excluding feature_dim)
        common_dims = [d for d in models.dims if d in X.dims and d != feature_dim]
        if common_dims:
            # Select models matching this block's coordinates
            sel_dict = {dim: X.coords[dim] for dim in common_dims}
            models = models.sel(sel_dict)

    # determine the dimension/shape/coordinates of the prediction output
    ydims = list(X.dims)
    yshape = list(X.shape)
    ycoords = dict(X.coords)
    # since most models can utilize many features to generate one prediction, remove the
    # dimension of `feature_dim
    ycoords.pop(feature_dim)
    if feature_dim in ydims:
        ydims.pop(X.get_axis_num(feature_dim))
        yshape.pop(X.get_axis_num(feature_dim))

    y = xr.DataArray(np.full(yshape, np.nan, dtype=X.dtype), coords=ycoords, dims=ydims)
    # some models, such as the GARD models generate multiple columns instead of one column of prediction result
    # in the .predict method. This would be set in `n_outputs` and `output_names`
    # if there are multiple output columns, add the `feature_dim` back to accommodate them
    if n_outputs > 1:
        y = y.expand_dims(**{feature_dim: output_names}, axis=1).copy()
        y = y.transpose(along_dim, feature_dim, ...)

    for index, model in xenumerate(models):
        if model.item():
            xdf = X[index].pipe(_da_to_df, feature_dim)
            ydf = model.item().predict(xdf, **kwargs)
            y[index] = ydf.squeeze()

    return y


def _transform_wrapper(
    X, direction='transform', feature_dim=DEFAULT_FEATURE_DIM, models=None, **kwargs
):
    # When models is passed via kwargs (for dask), select the subset matching X's coordinates
    if models is not None:
        # Get dimensions that exist in both models and X (excluding feature_dim and time)
        common_dims = [d for d in models.dims if d in X.dims and d != feature_dim and d != 'time']
        if common_dims:
            # Select models matching this block's coordinates
            sel_dict = {dim: X.coords[dim] for dim in common_dims}
            models = models.sel(sel_dict)

    dims = list(X.dims)
    shape = list(X.shape)
    coords = dict(X.coords)
    xtrans = xr.DataArray(np.full(shape, np.nan, dtype=X.dtype), coords=coords, dims=dims)

    for index, model in xenumerate(models):
        xdf = X[index].pipe(_da_to_df, feature_dim)
        for key in models.coords.keys():
            if key in xdf:
                xdf.drop(key)
        if model.item():
            xtrans_df = getattr(model.item(), direction)(xdf, **kwargs)
            xtrans[index] = xtrans_df
    return xtrans


def _getattr_wrapper(key, dtype, template_output=None, models=None):
    # Note: for getattr_wrapper, models should already be properly selected since
    # it doesn't take X as input, so no coordinate matching needed here
    if template_output is None:
        dims = list(models.dims)
        shape = list(models.shape)
        coords = dict(models.coords)
    else:
        if isinstance(template_output, xr.Dataset):
            example_var = list(template_output.data_vars)[0]
            template_output = template_output[example_var]
        dims = list(template_output.dims)
        shape = list(template_output.shape)
        coords = dict(template_output.coords)

    # construct output dataset
    out = xr.DataArray(np.full(shape, np.nan, dtype), coords=coords, dims=dims)

    # iterate through models to get attribute values
    for index, model in xenumerate(models):
        if model.item():
            out[index] = getattr(model.item(), key)

    return out


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

    def __init__(self, model: Any, dim: str = 'time') -> None:
        self._dim = dim
        self._model = model
        self._models = None

        if not hasattr(model, 'fit'):
            raise TypeError(
                f'Type {type(model)} does not have the fit method required by PointWiseDownscaler'
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
        kws = {'along_dim': self._dim, 'feature_dim': DEFAULT_FEATURE_DIM} | kwargs
        if len(args) > 1:
            raise ValueError(f'Expected at most 1 positional argument, got {len(args)}')
        args = list(args)
        args.append(self._model)

        X = self._to_feature_x(X, feature_dim=kws['feature_dim'])

        if X.chunks:
            reduce_dims = [self._dim, kws['feature_dim']]
            mask = _make_mask(X, reduce_dims)
            # we want the datatype to be an object because it will be populated
            # with fitted models (which have a dtype of object!)
            template = xr.full_like(mask, None, dtype=object)
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

        kws = {'along_dim': self._dim, 'feature_dim': DEFAULT_FEATURE_DIM} | kwargs
        X = self._to_feature_x(X, feature_dim=kws['feature_dim'])

        # check the model type to see if the model returns multiple columns are returned in
        # the .prdict function. notably, the GARD model family returns 3 columns
        try:
            kws['n_outputs'] = self._model.n_outputs
            kws['output_names'] = self._model.output_names
        except AttributeError:
            kws['n_outputs'] = 1

        if X.chunks:
            if kws['n_outputs'] == 1:
                # if there's only one output columns, remove the feature_dim in input to generate output template
                reduce_dims = [kws['feature_dim']]
                mask = _make_mask(X, reduce_dims)
                # we want the datatype to be the same as the input dataset (an object
                # is much bigger unnecessarily and has different behavior)
                template = xr.full_like(mask, None, dtype=X.dtype)
            else:
                # otherwise, maintain the `feature_dim` dimension to accommodate the number of outputs
                ydims = list(X.dims)
                yshape = list(X.shape)
                ycoords = dict(X.coords)
                if kws['feature_dim'] not in ydims:
                    template = xr.DataArray(
                        np.full(yshape, np.nan, dtype=X.dtype), coords=ycoords, dims=ydims
                    )
                    template = template.expand_dims(
                        **{kws['feature_dim']: kws['output_names']}, axis=1
                    ).copy()
                    template = template.transpose(self._dim, kws['feature_dim'], ...)
                else:
                    yshape[X.get_axis_num(kws['feature_dim'])] = kws['n_outputs']
                    ycoords[kws['feature_dim']] = kws['output_names']
                    template = xr.DataArray(
                        np.full(yshape, np.nan, dtype=X.dtype), coords=ycoords, dims=ydims
                    )
                chunksizes = dict(X.chunksizes)
                chunksizes[kws['feature_dim']] = kws['n_outputs']
                template = template.chunk(chunksizes)

            # Pass models via kwargs after computing to avoid xarray rechunking object dtype arrays
            # xarray's map_blocks doesn't support dask collections in kwargs, so compute first
            kws['models'] = (
                self._models.compute() if hasattr(self._models, 'compute') else self._models
            )
            return xr.map_blocks(_predict_wrapper, X, kwargs=kws, template=template)
        else:
            return _predict_wrapper(X, models=self._models, **kws)

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

        kws = {'feature_dim': DEFAULT_FEATURE_DIM} | kwargs
        X = self._to_feature_x(X, feature_dim=kws['feature_dim'])

        if X.chunks:
            # Pass models via kwargs after computing to avoid xarray rechunking object dtype arrays
            # xarray's map_blocks doesn't support dask collections in kwargs, so compute first
            kws['models'] = (
                self._models.compute() if hasattr(self._models, 'compute') else self._models
            )
            return xr.map_blocks(_transform_wrapper, X, kwargs=kws)
        else:
            return _transform_wrapper(X, models=self._models, **kws)

    def inverse_transform(self, X, **kwargs):
        """Apply inverse transforms to the data, and transform with the final estimator

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
        y_inverse_trans : xarray.DataArray
        """

        kws = {'feature_dim': DEFAULT_FEATURE_DIM} | kwargs
        X = self._to_feature_x(X, feature_dim=kws['feature_dim'])

        if X.chunks:
            # Pass models and direction via kwargs after computing to avoid xarray rechunking object dtype arrays
            # xarray's map_blocks doesn't support dask collections in kwargs, so compute first
            kws['models'] = (
                self._models.compute() if hasattr(self._models, 'compute') else self._models
            )
            kws['direction'] = 'inverse_transform'
            return xr.map_blocks(_transform_wrapper, X, kwargs=kws)
        else:
            return _transform_wrapper(X, models=self._models, direction='inverse_transform', **kws)

    def get_attr(
        self, key: str, dtype: DTypeLike, template_output: xr.DataArray | xr.Dataset | None = None
    ) -> xr.DataArray:
        """
        Get attribute values specified in key from each of the pointwise models

        Parameters
        ----------
        key: str
        dtype: expected dtype of the values
        template_output: template data array or dataset of the output dimensions
        """
        # Compute models if chunked to avoid xarray rechunking object dtype arrays
        computed_models = (
            self._models.compute() if hasattr(self._models, 'compute') else self._models
        )

        if template_output is not None:
            return _getattr_wrapper(key, dtype, template_output, models=computed_models)
        else:
            return _getattr_wrapper(key, dtype, models=computed_models)

    def _to_feature_x(self, X, feature_dim=DEFAULT_FEATURE_DIM):
        # xarray.Dataset --> xarray.DataArray
        if isinstance(X, xr.Dataset):
            X = X.to_array(feature_dim)

        if feature_dim not in X.dims:
            X = X.expand_dims(**{feature_dim: [f'{feature_dim}_0']}, axis=1)

        # all features must be in 1 chunk for map_blocks to work later on
        if X.chunks:
            X = X.chunk({feature_dim: -1})
        X = X.transpose(self._dim, feature_dim, ...)

        return X

    def __repr__(self):
        summary = [
            f'<skdownscale.{self.__class__.__name__}>',
            f'  Fit Status: {self._models is not None}',
            f'  Model:\n    {self._model}',
        ]
        return '\n'.join(summary)
