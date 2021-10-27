from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from skdownscale.pointwise_models import AnalogBase, PureRegression, PointWiseDownscaler, TrendAwareQuantileMappingRegressor, QuantileMappingReressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from skdownscale.pointwise_models.utils import default_none_kwargs





class GardWrapper:
    def __init__(
        self, 
        model, 
        label_name,
        feature_list=None, 
        dim='time', 
        bias_correction_method='quantile_transform', 
        bc_kwargs=None,
        generate_scrf=True,
        # spatial_features=(1, 1)
        ):
        """
        Parameters
        ----------
        model                 : a GARD model instance to be fitted pointwise 
        feature_list          : a list of feature names to be used in predicting 
        dim                   : string. dimension to apply the model along. Default is ``time``. 
        bias_correction_method: string of the name of bias correction model 
        bc_kwargs             : kwargs dict. directly passed to the bias correction model
        generate_scrf         : boolean. indicates whether a spatio-temporal correlated random field (scrf) will be 
                                generated based on the fine resolution data provided in .fit as y. if false, it is 
                                assumed that a pre-generated scrf will be passed into .predict as an argument that 
                                matches the prediction result dimensions.  
        """
        self._dim = dim
        if not isinstance(model, (AnalogBase, PureRegression)):
            raise TypeError('model must be part of the GARD family of pointwise models ')
        self.features = feature_list
        self.label_name = label_name
        self._model = model 

        # TODO: allow different methods for each feature 
        availalbe_methods = ['quantile_transform', 'z_score', 'quantile_map', 'detrended_quantile_map', 'none']
        if bias_correction_method not in availalbe_methods:
            raise NotImplementedError(f'bias correction method must be one of {availalbe_methods}')
        self.bias_correction_method = bias_correction_method
        self.bc_kwargs = bc_kwargs
        self.generate_scrf = generate_scrf

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
        self._lats = y.lat 
        self._lons = y.lon
        X = X.sel(lat=self._lats, lon=self._lons, method='nearest')[self.features]
        X = X.assign_coords({'lat': self._lats, 'lon': self._lons})

        #TODO: spatial features 
        #TODO: extend this to include transforming the feature space into PCA, etc 

        # generate correlation lengths for spatio-temporal correlated random field (scrf) if needed 
        if self.generate_scrf:
            self.len_scales = find_correlation_length_scale(data=y) 

        # point wise transformation on X based on what bias correction method is chosen 
        bc_kws = default_none_kwargs(self.bc_kwargs, copy=True)

        if self.bias_correction_method in ['quantile_map', 'detrended_quantile_map']:
            self.X_train = X 
        else:
            for v in self.features:
                if self.bias_correction_method == 'quantile_transform':
                    # no need to remember the transformer since the quantile transformation itself serves as bias correction 
                    if 'n_quantiles' not in bc_kws:
                        bc_kws['n_quantiles'] = len(X[v])
                    qm = PointWiseDownscaler(model=QuantileTransformer(**bc_kws))
                    qm.fit(X[v])
                    X = qm.transform(X[v])
                elif self.bias_correction_method == 'z_score':
                    # no need to remember the transformer since the zscoring itself serves as bias correction 
                    sc = PointWiseDownscaler(model=StandardScaler(**bc_kws))
                    sc.fit(X[v])
                    X = sc.transform(X[v])

        # point wise downscaling 
        self._pointwise_models = PointWiseDownscaler(model=self._model, dim=self._dim)
        self._pointwise_models.fit(X, y, **kwargs)

        return self


    def predict(self, X, **kwargs):
        self._validate_data(X=X)

        # spatial process 
        X = X.sel(lat=self._lats, lon=self._lons, method='nearest')[self.features]
        X = X.assign_coords({'lat': self._lats, 'lon': self._lons})

        # point wise transformation 
        ref_start = kwargs.get('ref_start', X.time[0])
        ref_stop = kwargs.get('ref_stop', X.time[-1])
        reference_X = X.sel(time=slice(ref_start, ref_stop))
        bc_kws = default_none_kwargs(self.bc_kwargs, copy=True)
        for v in self.features:
            if self.bias_correction_method == 'quantile_transform':
                # no need to remember the CDF since the quantile transformation itself serves as bias correction 
                if 'n_quantiles' not in bc_kws:
                    bc_kws['n_quantiles'] = len(X[v])
                qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kws))
                qt.fit(reference_X[v])
                # TODO: extrapolate at the extremes 
                X = qt.transform(X[v])
            elif self.bias_correction_method == 'z_score':
                scaler = PointWiseDownscaler(model=StandardScaler(**bc_kws))
                scaler.fit(reference_X[v])
                X[v] = scaler.transform(X[v])
            elif self.bias_correction_method == 'quantile_map':
                qm = PointWiseDownscaler(model=QuantileMappingReressor(**bc_kws), dim=self._dim)
                qm.fit(self.X_train[v], reference_X[v])
                X[v] = qm.predict(X[v])
            elif self.bias_correction_method == 'detrended_quantile_map':
                qm = PointWiseDownscaler(TrendAwareQuantileMappingRegressor(QuantileMappingReressor(**bc_kws)))
                qm.fit(self.X_train[v], reference_X[v])
                X[v] = qm.predict(X[v])

        # point wise downscaling
        downscaled = self._pointwise_models.predict(X) 

        return self._postprocess(downscaled)


    def _validate_data(self, X, y=None):
        for f in self.features:
            assert f in X 

        # TODO: 
        # check that X is coarser than y and roughly in the same domain to avoid extrapolating 
        # check that spatial dimensions are named lat/lon or x/y 


    def _postprocess(self, y_hat):
        # TODO: any postprocessing needed for gard? 
        return y_hat


    def find_correlation_length_scale(self, data, seasonality_period=31):
        for d in ['time', 'lat', 'lon']:
            assert d in data

        # remove seasonality before finding correlation length, otherwise the seasonality correlation dominates 
        seasonality = data.rolling({'time': seasonality_period}, center=True, min_periods=1).mean().groupby('time.dayofyear').mean()
        detrended = data.groupby("time.dayofyear") - seasonality

        # find spatial length scale 
        bin_center, gamma = gs.vario_estimate(pos=(detrended.lon.values, detrended.lat.values), field=detrended[self.label_name].values, latlon=True, mesh_type='structured')
        spatial = gs.Gaussian(dim=2, latlon=True, rescale=gs.EARTH_RADIUS)
        spatial.fit_variogram(bin_center, gamma, sill=np.mean(np.var(fields, axis=(1,2))))
        spatial_len_scale = spatial.len_scale

        # find temporal length scale 
        # break the time series into fields of 1 year length, since otherwise the algorithm struggles to find the correct length scale 
        fields = []
        day_in_year = 365
        for yr, group in detrended[self.label_name].groupby('time.year'):
            v = group.isel(time=slice(0, day_in_year)).stack(point=['lat', 'lon']).transpose('point', 'time').values
            fields.extend(list(v))
        t = np.arange(day_in_year) / 1000.
        bin_center, gamma = gs.vario_estimate(pos=t, field=fields, mesh_type='structured')
        temporal = gs.Gaussian(dim=1)
        temporal.fit_variogram(bin_center, gamma, sill=np.mean(np.var(fields, axis=1)))
        temporal_len_scale = temporal.len_scale

        return {'temporal': temporal_len_scale, 'spatial': spatial_len_scale}
