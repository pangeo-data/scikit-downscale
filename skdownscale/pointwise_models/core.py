import copy
from typing import List, Optional, Union

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


def _predict_wrapper(
    X,
    models,
    along_dim=None,
    feature_dim=DEFAULT_FEATURE_DIM,
    n_outputs=1,
    output_names=None,
    **kwargs,
):
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

    y = xr.DataArray(np.empty(yshape, dtype=X.dtype), coords=ycoords, dims=ydims)
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


def _transform_wrapper(X, models, feature_dim=DEFAULT_FEATURE_DIM, **kwargs):

    dims = list(X.dims)
    shape = list(X.shape)
    coords = dict(X.coords)
    xtrans = xr.DataArray(np.empty(shape, dtype=X.dtype), coords=coords, dims=dims)

    for index, model in xenumerate(models):
        xdf = X[index].pipe(_da_to_df, feature_dim)
        for key in models.coords.keys():
            if key in xdf:
                xdf.drop(key)
        xtrans_df = model.item().transform(xdf, **kwargs)
        xtrans[index] = xtrans_df
    return xtrans


def _getattr_wrapper(models, key, dtype, template_output=None):
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
    out = xr.DataArray(np.empty(shape, dtype), coords=coords, dims=dims)

    # iterate through models to get attribute values
    for index, model in xenumerate(models):
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

        kws = {'along_dim': self._dim, 'feature_dim': DEFAULT_FEATURE_DIM}
        kws.update(kwargs)

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
                template = xr.full_like(mask, None, dtype=object)
            else:
                # otherwise, maintain the `feature_dim` dimension to accommodate the number of outputs
                ydims = list(X.dims)
                yshape = list(X.shape)
                ycoords = dict(X.coords)
                if kws['feature_dim'] not in ydims:
                    template = xr.DataArray(
                        np.empty(yshape, dtype=X.dtype), coords=ycoords, dims=ydims
                    )
                    template = template.expand_dims(
                        **{kws['feature_dim']: kws['output_names']}, axis=1
                    ).copy()
                    template = template.transpose(self._dim, kws['feature_dim'], ...)
                else:
                    yshape[X.get_axis_num(kws['feature_dim'])] = kws['n_outputs']
                    ycoords[kws['feature_dim']] = kws['output_names']
                    template = xr.DataArray(
                        np.empty(yshape, dtype=X.dtype), coords=ycoords, dims=ydims
                    )
                template = template.chunk(X.chunksizes)

            return xr.map_blocks(
                _predict_wrapper, X, args=[self._models], kwargs=kws, template=template
            )
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

    def get_attr(
        self, key: str, dtype: str, template_output: Optional[Union[xr.DataArray]] = None
    ) -> xr.Dataset:
        """
        Get attribute values specified in key from each of the pointwise models

        Parameters
        ----------
        key: str
        dtype: expected dtype of the values
        template_output: template data array or dataset of the output dimensions
        """
        if self._models.chunks:
            if template_output is not None:
                template = xr.full_like(template_output, None, dtype=dtype)
                # there is currently a bug in xarray map block such that the template output has to be a dataset instead of a dataarray
                return xr.map_blocks(
                    _getattr_wrapper,
                    self._models,
                    args=[key, dtype, template_output.to_dataset()],
                    template=template,
                )

            else:
                template = xr.full_like(self._models, None, dtype=dtype)
                return xr.map_blocks(
                    _getattr_wrapper, self._models, args=[key, dtype], template=template
                )

        else:
            if template_output is not None:
                return _getattr_wrapper(self._models, key, dtype, template_output)
            else:
                return _getattr_wrapper(self._models, key, dtype)

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
        summary = ['<skdownscale.{}>'.format(self.__class__.__name__)]
        summary.append('  Fit Status: {}'.format(self._models is not None))
        summary.append('  Model:\n    {}'.format(self._model))
        return '\n'.join(summary)
