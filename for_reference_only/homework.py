import os
import numpy as np
import cv2
from pathlib import Path


# ==================== 1. 多层神经网络类 ====================
class NeuralNetwork:
    """
    一个标准的全连接神经网络，支持任意数量的隐藏层。
    每一层：线性变换 -> ReLU（隐藏层）或 Sigmoid（输出层）
    """

    def __init__(self, layer_sizes: list, random_state: int = 42):
        """
        参数：
            layer_sizes: 列表，指定每一层的神经元数量。
                         例如 [1024, 256, 128, 64, 10] 表示：
                         输入层 1024，隐藏层_1 256，隐藏层_2 128，隐藏层_3 64，输出层 10，意味着：
                         一张图片有1024个像素，每个像素只有1和0两个值，其中1024是像素点，是输入层的神经元个数，
                         10是输出的数字，是输出层的神经元个数，即0-9一共10个种类的输出
                         一共900多张照片，那么就有900多个样本，同理就有900多个标签
            random_state: 随机种子，保证可复现
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # 权重层数（即线性变换的次数）
        rng = np.random.default_rng(random_state)   # 随机数生成器

        # ---------- 初始化权重和偏置 ----------
        # 用列表存储每一层的 W 和 b
        # W[i] 形状: (layer_sizes[i], layer_sizes[i+1])，即每一层的权重矩阵
        # b[i] 形状: (1, layer_sizes[i+1])，即每一层的偏置
        self.W = []
        self.b = []
        for i in range(self.num_layers):
            # 使用 He 初始化（适合 ReLU）：标准差 sqrt(2 / 输入维度)
            scale = np.sqrt(2.0 / layer_sizes[i])  # 每一层根据自己的输入维度算出不同的 scale
            self.b.append(np.zeros((1, layer_sizes[i+1]))* 0.01)# 偏置，形状为(1, layer_sizes[i+1])，全0
            # 正态分布，均值为0，标准差为scale，形状为(layer_sizes[i], layer_sizes[i+1])的随机数组w，代表这一层的权重
            self.W.append(rng.normal(0, scale, size=(layer_sizes[i], layer_sizes[i+1]))* 0.01)

        # 缓存前向传播的中间结果，用于反向传播
        self.Z = []  # 每一层的线性输出（未激活）
        self.A = []  # 每一层的激活输出（A[0] 是输入 X）

    # ---------- 激活函数 ----------

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        z_shifted = z - np.max(z, axis=1, keepdims=True)  # 减去最大值，防止指数爆炸
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ---------- 前向传播 ----------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        逐层计算，返回最终输出（概率）。
        X 形状: (样本数, 特征数)
        """
        self.Z = []
        self.A = [X]  # A[0] 是输入

        # 遍历所有层（除了最后一层用 Sigmoid，其余用 ReLU）
        for i in range(self.num_layers):    # 遍历每一层（0，1，2，3，4）
            Z = self.A[i] @ self.W[i] + self.b[i]  # 线性输出，Z = X * W + b
            self.Z.append(Z)  # 缓存线性输出

            if i == self.num_layers - 1:  # 最后一层用 softmax
                # 输出层：softmax
                A = self.softmax(Z) # 对线性输出做softmax激活
            else:
                # 隐藏层：sigmoid
                A = self.sigmoid(Z)    # 对线性输出做sigmoid激活

            self.A.append(A)    # 将激活输出缓存，作为下一层的输入

        return self.A[-1]  # 完成所有层的循环后，返回最后一层的激活输出




# ==================== 2. 图片处理 ====================

def load_images_from_folder(folder="data", size=(32, 32)):
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"找不到文件夹{folder}")
    images = []
    for file in folder.iterdir():
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        images.append(img.reshape(-1))
    return np.array(images)

# ==================== 3. 测试运行 ====================
if __name__ == '__main__':
    # ---------- 获得数据 ----------
    X = load_images_from_folder(folder='testing', size=(32, 32))

    # ---------- 创建网络：3 个隐藏层 ----------
    # 输入 1024 -> 隐藏层1 256 -> 隐藏层2 128 -> 隐藏层3 64 -> 输出 10
    layer_sizes = [1024, 256, 128, 64, 10]
    net = NeuralNetwork(layer_sizes, random_state=42)

    # ---------- 输出 ----------
    res = net.forward(X)
    print(f"结果为: {res}")