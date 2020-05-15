from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel


class AbstractDownscaler(LinearModel, RegressorMixin):
    pass
