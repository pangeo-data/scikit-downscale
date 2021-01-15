.. currentmodule:: skdownscale.pointwise_models

#############
API reference
#############

This page provides an auto-generated summary of skdownscale's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

Pointwise methods
=================

Xarray Wrappers
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   PointWiseDownscaler

Linear Models
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   BcsdPrecipitation
   BcsdTemperature
   PureAnalog
   AnalogRegression
   PiecewiseLinearRegression
   QuantileMappingReressor
   TrendAwareQuantileMappingRegressor

Transformers
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   LinearTrendTransformer
   QuantileMapper

Grouping
~~~~~~~~
.. autosummary::
   :toctree: generated/

   GroupedRegressor


Groupers
~~~~~~~~

.. autosummary::
   :toctree: generated/

   DAY_GROUPER
   MONTH_GROUPER
   PaddedDOYGrouper
