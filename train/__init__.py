"""定义train包的入口，负责对外提供训练函数"""

from .train import digits_train
from .train import evaluate

__all__ = ["digits_train", "evaluate"]
