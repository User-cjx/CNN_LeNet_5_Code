"""定义模型函数包的入口，负责对外提供不同的模型函数"""

from .model import BackBone, Head, LeNet5


__all__ = ["BackBone", "Head", "LeNet5"]