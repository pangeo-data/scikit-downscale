from sklearn.pipeline import Pipeline
from skdownscale.pointwise_models import PointWiseDownscaler, TrendAwareQuantileMappingRegressor, QuantileMappingReressor
from skdownscale.pointwise_models.bcsd import BcsdBase
import rioxarray
from rasterio.enums import Resampling

class BcsdWrapper:
    def __init__(self, model, feature_list=None, dim='time'):
        """
        Parameters
        ----------
        model: a BCSD model instance to be fitted pointwise 
        feature_list: a list of feature names to be used in predicting 
        dim: dimension to apply the model along. Default is ``time``. 
        """
        self._dim = dim
        # if not isinstance(model, BcsdBase):
        #     raise TypeError('model must be part of the BCSD family of pointwise models ')
        self._features = feature_list

        # NOTE: 
        # alternatively we can also write a "gard_preprocess" class to do the spatial matching required for GARD, which outputs X and y that are matched in resolution 
        # then initialize the entire model as: 
        # gard_pipeline = Pipeline(
        #   [gard_preprocess, 
        #    PointwiseDownscaler(Pipeline([pointwise_bias_correct, gard]))
        #   ])
        # in this word, bcsd_pipeline = Pipeline([bcsd_preprocess, PointwiseDownscaler(bcsd), bcsd_postprocess])
        # the benefit in this is that it allows us to re-use certain elements more easily: new_pipeline = Pipeline([bcsd_preprocess, bias_correct, bcsd, bcsd_postprocess])
        # the downside is that the code in a notebook would be messier (or we put it into a prefect workflow??)
        self._model = model 
        # #Pipeline(
        #     [
        #         ('bcsd', model)
        #     ]
        # )

    def fit(self, X, y, **kwargs):
        """
        Fit the BCSD model as specified over a domain. Some transformations will be spatial and some will be pointwise. 

        Parameters
        ----------
        X : xarray.DataArray or xarray.Dataset
            Independent vairables/features of the training data, typically this is the coarse resolution data (e.g. historical GCM). 
            Must fulfill input requirements of first step of the pipeline. 
            If an xarray.Dataset is passed, it will be converted to an array using `to_array()`. 
            Must contain all features in self._features . 
        y : xarray.DataArray
            Dependent variable/target of the training data, typically this is the fine resolution data (e.g. historical observation). 
            Must fulfill label requirements for all steps of the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the this model. If the
            model is a sklearn Pipeline, parameters can be passed to each
            step, where each parameter name is prefixed such that parameter
            ``p`` for step ``s`` has key ``s__p``.
        """
        
        # could do this instead: https://github.com/pangeo-data/scikit-downscale/blob/ak-hi-round2/xsd/bcsd.py
        # grab random obs month to get daily data (but chose same month for all variables to preserve physical consistency)
        self._validate_data(X=X, y=y)
        # perform preprocess that is done for the entire domain 
        # for gard: simply match the spatial domain & limit to selected features 
        # TODO: extend this to include using spatial features and/or transform the feature space into PCA, etc 
        # store the spatial anomalies resulting from the interpolation
        coarsened_y = y.rio.reproject_match(X, resampling=Resampling.bilinear) 
        # swap out regrid xesmf? because rasterio doesn't return dask arrays
        # could be faster or slower but can save the weights AND has support for dask
        # but need to have complete spatial chunks
        obs_spatial_anomalies = coarsened_y.interp_like(y, 
                                                        kwargs={"fill_value": "extrapolate"}) - y
        self._interpolation_seasonal_anomalies = obs_spatial_anomalies.groupby('time.month').mean()

        # fit point wise models 
        self._pointwise_models = PointWiseDownscaler(model=self._model, dim=self._dim)
        self._pointwise_models.fit(X, coarsened_y, **kwargs)

        return self


    def predict(self, X):
        self._validate_data(X=X)
        # add extrapoliation
        bias_corrected = self._pointwise_models.predict(X)
        # print(self._interpolation_seasonal_anomalies)
        # then do the disaggregation
        bias_corrected_interpolated = bias_corrected.interp_like(self._interpolation_seasonal_anomalies,
                                                kwargs={"fill_value": "extrapolate"})
        #  and add back in the spatial anomalies
        bcsd_results = (bias_corrected_interpolated.groupby('time.month') + self._interpolation_seasonal_anomalies)
        return bcsd_results


    def _validate_data(self, X, y=None):
        for f in self._features:
            assert f in X 

        # TODO: 
        # check that X is coarser than y and roughly in the same domain to avoid extrapolating 
        # check that spatial dimensions are named lat/lon or x/y 

    def _preprocess(self, X):
        return X
        # regridding/chunking

    def _postprocess(self, y_hat):
        # TODO: any postprocessing needed for gard? 
        return y_hat
