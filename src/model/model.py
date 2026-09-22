"""LeNet-5 现代版：ReLU + MaxPool，C5/F6 用 Linear 代替原高斯连接。"""
import torch
from torch import nn


class BackBone(nn.Module):
    """卷积部分，维度链：1x32x32 -> 6x28x28 -> 6x14x14 -> 16x10x10 -> 16x5x5。"""

    def __init__(self) -> None:
        super().__init__()
        # conv1：1通道特征图转换为6个5×5卷积核，size由32转28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # pool1：池化，不改变通道数，将尺寸转由28转为14，保留最强响应并减半计算量
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # conv2：由6转16通道，组合低级笔划为数字部件；size由14转10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # pool2：size由10转5，输出 (batch,16,5,5) 供全连接层分类
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()  # 引入非线性，否则多层叠加仍等价于线性变换

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.conv1(x)))    
        x = self.pool2(self.relu(self.conv2(x)))    # 先卷积，再激活，后池化
        return x  # (batch, 16, 5, 5) 输出特征图的shape


class Head(nn.Module):
    """全连接部分：Flatten + Linear(400->120) + Linear(120->84) + Linear(84->10)。"""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()  # 16 * 5 * 5 = 400 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # C5: layer 120 
        self.fc2 = nn.Linear(120, 84)  # F6: layer 84 
        self.fc3 = nn.Linear(84, 10)  # OUTPUT: 10 
        self.relu = nn.ReLU()   # CrossEntropyLoss 自带 softmax，这里不加 Softmax
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x) # 将16个特征图展平成400个特征
        x = self.relu(self.fc1(x))  # 先全连接，将400个特征映射到120个特征，再激活
        x = self.relu(self.fc2(x))  # 将120个特征映射到84个特征
        x = self.fc3(x) # 将84个特征映射到10个特征
        return x    # 返回最终值，无需做任何处理


class LeNet5(nn.Module):
    """BackBone(卷积提特征) + Head(全连接做分类) 组合，输入1×32×32，输出10维得分。"""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = BackBone()
        self.head = Head()  # 继承两个类

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)  # 先提空间特征：(B,1,32,32)→(B,16,5,5)
        x = self.head(x)  # 再映射到10类得分，配合CrossEntropyLoss训练
        return x
