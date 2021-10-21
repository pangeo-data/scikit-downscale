from sklearn.pipeline import Pipeline
from skdownscale.pointwise_models import AnalogBase, PointWiseDownscaler, TrendAwareQuantileMappingRegressor, QuantileMappingReressor


class GardWrapper:
    def __init__(self, model, feature_list=None, dim='time', bias_correction_model=None):
        """
        Parameters
        ----------
        model: a GARD model instance to be fitted pointwise 
        feature_list: a list of feature names to be used in predicting 
        dim: dimension to apply the model along. Default is ``time``. 
        """
        self._dim = dim
        if not isinstance(model, AnalogBase):
            raise TypeError('model must be part of the GARD family of pointwise models ')
        self._features = feature_list

        # construct the pipeline to be used point wise 
        if not bias_correction_model:
            bias_correction_model = TrendAwareQuantileMappingRegressor(QuantileMappingReressor())

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
        self._pipeline = Pipeline(
            [
                ('bias_correct', bias_correction_model),
                ('gard', model)
            ]
        )


    def fit(self, X, y, **kwargs):
        """
        Fit the GARD model as specified over a domain. Some transformations will be spatial and some will be pointwise. 

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
        
        self._validate_data(X=X, y=y)

        # perform preprocess that is done for the entire domain 
        # for gard: simply match the spatial domain & limit to selected features 
        # TODO: extend this to include using spatial features and/or transform the feature space into PCA, etc 
        self._lats = y.lat 
        self._lons = y.lon
        resampled_X = X.sel(lat=self._lats, lon=self._lons, method='nearest')[self._features]

        # fit point wise models 
        self._pointwise_models = PointWiseDownscaler(model=self._pipeline, dim=self._dim)
        self._pointwise_models.fit(resampled_X, y, **kwargs)

        return self


    def predict(self, X):
        self._validate_data(X=X)

        resampled_X = X.sel(lat=self._lats, lon=self._lons, method='nearest')[self._features]
        downscaled = self._pointwise_models.predict(resampled_X) 

        return self._postprocess(downscaled)


    def _validate_data(self, X, y=None):
        for f in self._features:
            assert f in X 

        # TODO: 
        # check that X is coarser than y and roughly in the same domain to avoid extrapolating 
        # check that spatial dimensions are named lat/lon or x/y 


    def _postprocess(self, y_hat):
        # TODO: any postprocessing needed for gard? 
        return y_hat
