.. _roadmap:

Development Roadmap
===================

Author: Joe Hamman
Date:September 15, 2019

Background and scope
--------------------

Scikit-downscale is a toolkit for statistical downscaling using Xarray. It is
meant to support the development of new and existing downscaling methods in a
common framework. It implements a fit/predict API that accepts Xarray objects,
similiar to Python's Scikit-Learn, for building a range of downscaling models.
For example, implementing a BCSD workflow may look something like this:

.. code-block:: Python

    from skdownscale.pointwise_models import PointWiseDownscaler
    from skdownscale.models.bcsd import BCSDTemperature, bcsd_disaggregator

    # da_temp_train: xarray.DataArray (monthly)
    # da_temp_obs: xarray.DataArray (monthly)
    # da_temp_obs_daily: xarray.DataArray (daily)
    # da_temp_predict: xarray.DataArray (monthly)

    # create a model
    bcsd_model = PointWiseDownscaler(BCSDTemperature(), dim='time')

    # train the model
    bcsd_model.train(da_temp_train, da_temp_obs)

    # predict with the model  (downscaled_temp: xr.DataArray)
    downscaled_temp = bcsd_model.predict(da_temp_predict)

    # disaggregate the downscaled data (final: xr.DataArray)
    final = bcsd_disaggregator(downscaled_temp, da_temp_obs_daily)

We are currently envisioning the project having three componenets (described
in the components section below). While we haven't started work on the deep
learning models component, this is certainly a central motivation to this
package and I am looking forward to starting on this work soon.

Principles
----------

- Open - aim to take the sausage making out downscaling; open-source methods,
  comparable, extensible
- Scalable - plug into existing frameworks (e.g. dask/pangeo) to scale up,
  allow for use a single points to scale down
- Portable - unopionated when it comes to compute platform
- Tested - Rigourously tested, both on the computational and scientific
  implementation

Components
----------

1. `pointwise_models`: a collection of linear models that are intended to be
   applied point-by-point. These may be sklearn Pipelines or custom sklearn-like
   models (e.g. BCSDTemperature).
2. `global_models`: (not implemented) concept space for deep learning-based
   models.
3. `metrics`: (not implemented) concept space for a benchmarking suite

Models
------

Scikit-downscale should provide a collection of a common set of downscaling
models and the building blocks to construct new models. As a starter, I intend
to implement the following models:

Pointwise models
~~~~~~~~~~~~~~~~

1. BCSD_[Temperature, Precipitation]: Wood et al 2002
2. ARRM: Stoner et al 2012
3. Delta Method
4. Hybrid Delta Method
5. GARD: https://github.com/NCAR/GARD
6. ?

Other methods, like LOCA, MACA, BCCA, etc, should also be possible.

Global models
~~~~~~~~~~~~~

This category of methods is really what is motivating the development of this
package. We've seen some early work from TJ Vandal in this area but there is
more work to be done. For now, I'll just jot down what a possible API might
look like:

.. code-block:: Python

    from skdownscale.global_models import GlobalDownscaler
    from skdownscale.global_models.deepsd import DeepSD

    # ...

    # create a model
    model = GlobalDownscaler(DeepSD())

    # train the model
    model.train(da_temp_train, da_temp_obs)

    # predict with the model  (downscaled_temp: xr.DataArray)
    downscaled_temp = model.predict(da_temp_predict)

Dependencies
------------

- Core: Xarray, Pandas, Dask, Scikit-learn, Numpy, Scipy
- Optional: Statsmodels, Keras, PyTorch, Tensorflow, etc.

Related projects
----------------

- FUDGE: https://github.com/NOAA-GFDL/FUDGE
- GARD: https://github.com/NCAR/GARD
- DeepSD: https://github.com/tjvandal/deepsd
