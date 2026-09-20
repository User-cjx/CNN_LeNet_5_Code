"""简单的全连接神经网络模型。"""
import torch
from torch import nn

class DigitNetwork(nn.Module):  # 继承torch的nn.model父类
    """32x32 灰度图 -> 10 类数字的多层感知机。"""

    def __init__(self) -> None:
        super().__init__()  # 初始化父类
        self.fc1 = nn.Linear(1024, 256) # 第一层为1024转化成256个神经元
        self.fc2 = nn.Linear(256, 128)  # 第二层为256转128
        self.fc3 = nn.Linear(128, 64)   # 第三层为128转64
        self.fc4 = nn.Linear(64, 10)    # 输出成为64转10
        self.Relu = nn.ReLU() # 隐藏层用Relu函数
        self.softmax = nn.Softmax(dim=1)    # 输出层用softmax，dim为1则对每个元素进行softmax计算


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # x的形状为（batch, 32，32），保留batch维度，后面压扁，所以新的维度为（batch，1024）
        x = self.Relu(self.fc1(x))
        x = self.Relu(self.fc2(x))
        x = self.Relu(self.fc3(x))   # 先计算线性，再激活
        x = self.fc4(x)
        # 这里不用softmax是因为train函数里计算CrossEntropyLoss()自带softmax计算结果
        return x

    