"""定义torch包的入口，统一对外提供各子包"""

from . import dataset, model

__all__ = ["dataset", "model"]