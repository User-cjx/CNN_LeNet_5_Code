import numpy as np
import os.path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Grayscale, Resize, ToTensor

# ==================== 1. 定义数据集类 ======================

class CustomDataset(Dataset):
    """按文件名 '数字_编号.png' 解析标签的手写数字数据集。"""

    def __init__(self, root: str = r'D:\workspace\Project_learning_number\data', split: str = "training_img") -> None:
        self.root_dir = os.path.expanduser(os.path.join(root, split))
        self.transforms = [
            Grayscale(),
            Resize((32, 32)),
            ToTensor()
        ]
        self.image_paths = []
        self.labels = []
        for file_name in sorted(os.listdir(self.root_dir)):
            if file_name.endswith('.png'):
                label, _ = file_name.split('_')
                image_path = os.path.join(self.root_dir, file_name)
                self.image_paths.append(image_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        label = self.labels[index]
        for transform in self.transforms:
            image = transform(image)
        return image, label




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
            self.b.append(np.zeros((1, layer_sizes[i+1])))  # 偏置，形状为(1, layer_sizes[i+1])，全0
            # 正态分布，均值为0，标准差为scale，形状为(layer_sizes[i], layer_sizes[i+1])的随机数组w，代表这一层的权重
            self.W.append(rng.normal(0, scale, size=(layer_sizes[i], layer_sizes[i+1])))  

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

    # ---------- 反向传播(暂时不需要完成) ----------
    def backward(self, y: np.ndarray) -> dict:
        """
        从输出层往回计算每一层的梯度，用于更新权重和偏置
        y 形状: (样本数, 1)，900多张照片就有900多个样本y
        返回: 包含所有 W 和 b 梯度的列表
        """
        m = y.shape[0]  # 样本数，即y的行数，900多张照片就有900多个样本y
        L = self.num_layers  # 线性层数量，即权重的层数，本题有4层权重

        # 存储每一层的梯度
        dW = [None] * L
        db = [None] * L

        # ----- 输出层梯度 -----
        # 最后一层是 Sigmoid + 交叉熵，导数简化为 (A_last - y)，就是平时的预测值减真实值
        dZ = self.A[-1] - y  # (m, 1)，二维矩阵，表示最后一层的误差A[-1]为最后一层的输出，y为真实值
        """计算最后一层的权重梯度，self.A[-2]为倒数第二层的输出，在本题倒数第二层有64个输出，也是最后一层的输入，shape为（64，1）
        倒置后为（1，64），那么w的shape为（64，10），dZ为最后一层的误差，其shape与输出一致，为（10，1）"""
        dW[-1] = (self.A[-2].T @ dZ) / m  
        db[-1] = np.mean(dZ, axis=0, keepdims=True)  # 计算最后一层的偏置梯度

        # ----- 从倒数第二层往回传 -----
        for i in range(L - 2, -1, -1):    # 遍历每一层（3，2，1，0）,3 那层已经在上面执行过了
            # 当前层的误差：后一层的误差乘以后一层的权重的倒置，再乘当前层激活函数的导数
            dA = dZ @ self.W[i+1].T
            dZ = dA * self.relu_grad(self.Z[i])  # 隐藏层用 ReLU 导数

            dW[i] = (self.A[i].T @ dZ) / m
            db[i] = np.mean(dZ, axis=0, keepdims=True)

        return {"dW": dW, "db": db}

    # ---------- 参数更新 ----------
    def update(self, grads: dict, lr: float):
        """用梯度下降更新所有层的 W 和 b。"""
        for i in range(self.num_layers):
            self.W[i] -= lr * grads["dW"][i]
            self.b[i] -= lr * grads["db"][i]

    # ---------- 预测 ----------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) >= 0.5).astype(int)


# ==================== 2. 损失函数类(暂时不需要完成) ====================
class LossFunction:
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_label = (y_pred >= 0.5).astype(int)
        return float(np.mean(y_label == y_true))


# ==================== 3. 训练函数 ====================
def train(net: NeuralNetwork, X: np.ndarray, y: np.ndarray,
          epochs: int = 2000, lr: float = 0.1, verbose: bool = True):
    losses = []
    for epoch in range(1, epochs + 1):
        # 1. 前向传播
        y_pred = net.forward(X)

        # 2. 计算损失
        loss = LossFunction.binary_cross_entropy(y, y_pred)
        losses.append(loss)

        # 3. 反向传播
        grads = net.backward(y)

        # 4. 更新参数
        net.update(grads, lr)

        # 5. 打印日志
        if verbose and epoch % 200 == 0:
            acc = LossFunction.accuracy(y, y_pred)
            print(f"Epoch {epoch:4d} | Loss = {loss:.6f} | Acc = {acc:.4f}")

    return losses


# ==================== 4. 测试运行 ====================
if __name__ == '__main__':
    # ---------- 造数据 ----------
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(300, 2))  # 300 个样本，2 个特征

    # 真实标签：x1 + x2 > 0 为正类
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

    # ---------- 创建网络：3 个隐藏层 ----------
    # 输入 2 -> 隐藏层1 16 -> 隐藏层2 8 -> 隐藏层3 4 -> 输出 1
    layer_sizes = [2, 16, 8, 4, 1]
    net = NeuralNetwork(layer_sizes, random_state=42)

    # ---------- 训练 ----------
    losses = train(net, X, y, epochs=3000, lr=0.1)

    # ---------- 评估 ----------
    y_pred = net.predict_proba(X)
    acc = LossFunction.accuracy(y, y_pred)
    print(f"\n最终训练集准确率: {acc:.4f}")