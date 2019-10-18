import dask.array
import numpy as np
import pandas as pd
import xarray as xr


def _reshape_for_sklearn(vals, columns=None):
    vals = np.atleast_2d(vals).transpose()  # reshape input data
    if columns is not None:
        return pd.DataFrame(data=vals, columns=columns)
    return vals


def _fit_model(X, y, model=None, columns=None, **kwargs):
    X = _reshape_for_sklearn(X, columns=columns)
    y = _reshape_for_sklearn(y)

    # return a len 1 scalar of dtype np.object, there is likely a better way
    # to do this
    # this is required because sklearn pipelines are iterable and can be cast
    # to arrays
    out = np.empty((1), dtype=np.object)
    out[:] = [model.fit(X, y, **kwargs)]
    out = out.squeeze()
    return out


def _predict(model, X, columns=None):
    X = _reshape_for_sklearn(X, columns=columns)
    # pull item() out because model is wrapped in np.scalar
    out = model.item().predict(X).squeeze()
    return out


def _transform(model, X, columns=None):
    X = _reshape_for_sklearn(X, columns=columns)
    # pull item() out because model is wrapped in np.scalar
    out = model.item().transform(X).squeeze()
    return out


def _maybe_use_dask(a, dims, b=None):
    dask_option = "allowed"
    if isinstance(a.data, dask.array.Array):
        dask_option = "parallelized"

    if isinstance(b.data, dask.array.Array):
        dask_option = "parallelized"

    if dask_option == "parallelized":
        a = a.chunk({d: -1 for d in dims if d in a.dims})
        if b is not None:
            b = b.chunk({d: -1 for d in dims if d in b.dims})

    return a, b, dask_option


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

    def __init__(self, model, dim="time"):
        self._dim = dim
        self._model = model
        self._models = None

        if not hasattr(model, "fit"):
            raise TypeError(
                "Type %s does not have the fit and predict methods required"
                " by PointWiseDownscaler" % type(model)
            )

    def fit(self, X, y, feature_dim="variable", **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : xarray.DataArray or xarray.Dataset
            Training data. Must fulfill input requirements of first step of
            the pipeline. If an xarray.Dataset is passed, it will be converted
            to an array using `to_array()`.

        y : xarray.DataArray
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
        self._ydims = y.dims

        kwargs = dict(model=self._model, **fit_params)

        # xarray.Dataset --> xarray.DataArray
        if isinstance(X, xr.Dataset):
            X = X.to_array(feature_dim)
        if isinstance(y, xr.Dataset):
            assert len(y.data_vars) == 1, y.data_vars
            y = y[list(y.data_vars.keys())[0]]

        if feature_dim in X.coords:
            input_core_dims = [[feature_dim, self._dim], [self._dim]]
            kwargs["columns"] = X.coords[feature_dim].data
        else:
            input_core_dims = [[self._dim], [self._dim]]

        X, y, dask_option = _maybe_use_dask(X, (self._dim, feature_dim), b=y)

        self._models = xr.apply_ufunc(
            _fit_model,
            X,
            y,
            vectorize=True,
            dask=dask_option,
            output_dtypes=[np.object],
            input_core_dims=input_core_dims,
            kwargs=kwargs,
        )
        return self

    def predict(self, X, feature_dim="variable", **predict_params):
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

        kwargs = dict(**predict_params)

        # xarray.Dataset --> xarray.DataArray
        if isinstance(X, xr.Dataset):
            X = X.to_array(feature_dim)

        if feature_dim in X.coords:
            input_core_dims = [[], [feature_dim, self._dim]]
            kwargs["columns"] = X.coords[feature_dim].data
        else:
            input_core_dims = [[self._dim], [self._dim]]

        X, _, dask_option = _maybe_use_dask(X, (self._dim, feature_dim), b=self._models)

        return xr.apply_ufunc(
            _predict,
            self._models,
            X,
            vectorize=True,
            dask=dask_option,
            output_dtypes=[X.dtype],
            input_core_dims=input_core_dims,
            output_core_dims=[[self._dim]],
            kwargs=kwargs,
        ).transpose(*self._ydims)

    def predict(self, X, feature_dim="variable", **predict_params):
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

        kwargs = dict(**predict_params)

        # xarray.Dataset --> xarray.DataArray
        if isinstance(X, xr.Dataset):
            X = X.to_array(feature_dim)

        if feature_dim in X.coords:
            input_core_dims = [[], [feature_dim, self._dim]]
            kwargs["columns"] = X.coords[feature_dim].data
        else:
            input_core_dims = [[self._dim], [self._dim]]

        X, _, dask_option = _maybe_use_dask(
            X, (self._dim, feature_dim), b=self._models)

        return xr.apply_ufunc(
            _transform,
            self._models,
            X,
            vectorize=True,
            dask=dask_option,
            output_dtypes=[X.dtype],
            input_core_dims=input_core_dims,
            output_core_dims=[[self._dim]],
            kwargs=kwargs,
        ).transpose(*self._ydims)

    def __repr__(self):
        summary = ["<xsd.{}>".format(self.__class__.__name__)]
        summary.append("  Fit Status: {}".format(self._models is not None))
        summary.append("  Model:\n    {}".format(self._model))
        return "\n".join(summary)
