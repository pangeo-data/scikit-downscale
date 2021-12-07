from .arrm import PiecewiseLinearRegression
from .bcsd import BcsdPrecipitation, BcsdTemperature
from .core import PointWiseDownscaler
from .gard import AnalogRegression, PureAnalog, PureRegression
from .groupers import DAY_GROUPER, MONTH_GROUPER, PaddedDOYGrouper
from .grouping import GroupedRegressor
from .quantile import (
    EquidistantCdfMatcher,
    QuantileMapper,
    QuantileMappingReressor,
    TrendAwareQuantileMappingRegressor,
)
from .trend import LinearTrendTransformer
from .zscore import ZScoreRegressor
