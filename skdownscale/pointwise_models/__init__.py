from .arrm import PiecewiseLinearRegression
from .bcsd import BcsdPrecipitation, BcsdTemperature
from .core import PointWiseDownscaler
from .gard import AnalogRegression, PureAnalog
from .groupers import DAY_GROUPER, MONTH_GROUPER, PaddedDOYGrouper
from .quantile_mapping import QuantileMappingReressor, TrendAwareQuantileMappingRegressor
from .utils import LinearTrendTransformer, QuantileMapper
from .zscore import ZScoreRegressor
