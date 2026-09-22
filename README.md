# Handwritten Digit Recognition with LeNet-5 (PyTorch)

PyTorch 复刻 LeNet-5 的手写数字识别：32×32 灰度图输入，10 分类输出。
卷积部分封装为 `BackBone`，全连接部分封装为 `Head`。

## 1. 原理：LeNet-5 前向维度链

输入 `1×32×32`：

```text
1×32×32
→ Conv(1→6, 5×5, s1, p0) + ReLU + MaxPool(2×2, s2)   → 6×14×14
→ Conv(6→16, 5×5, s1, p0) + ReLU + MaxPool(2×2, s2)  → 16×5×5
→ Flatten(400) → Linear(400→120) → ReLU               (C5)
→ Linear(120→84) → ReLU                               (F6)
→ Linear(84→10)                                       (OUTPUT)
```

与原论文的差异：`Tanh → ReLU`，`AvgPool → MaxPool`，
高斯连接层用普通 `Linear` 代替；`CrossEntropyLoss` 自带 softmax，
网络末端不再加 `Softmax`。

训练原理：每轮在训练集上做完整一轮前向 + 交叉熵 + 反向 +
`Adam` 更新；每轮末在测试集上只做前向算 `loss / acc / P / R`，
测试集准确率创新高则存 `best.pt`，每轮覆盖 `last.pt`；
训练结束载回 `best` 权重再做最终评估。

指标口径（One-vs-Rest，对类别 `i`）：

```text
TP = C[i][i]
FP = 该列和 − TP
FN = 该行和 − TP
TN = 总数 − TP − FP − FN
P  = TP / (TP + FP)
R  = TP / (TP + FN)
F1 = 2·P·R / (P + R)
```

## 2. 代码

```text
├── main.py               # 一键训练 + 评估 + 画图
├── src/
│   ├── dataset/data.py   # CustomDataset：按文件名“数字_编号.png”解析标签，Grayscale → Resize((32, 32)) → ToTensor
│   └── model/model.py    # BackBone / Head / LeNet5
├── train/
│   └── train.py          # digits_train / evaluate / metrics_from_confusion
└── utils/
    └── plot.py           # loss/acc/混淆矩阵/报表等全部绘图
```

- `src/model/model.py`：`BackBone` 做两组 `Conv + ReLU + MaxPool` 提特征，
输出 `(batch, 16, 5, 5)`；`Head` 做 `Flatten + 400→120→84→10` 分类；
`LeNet5` 按 `backbone + head` 顺序组合。
- `train/train.py`：`digits_train` 负责训练循环、每轮测试集评估、
`best.pt / last.pt` 保存；`evaluate` 负责最终评估并统计每类正确数与
10×10 混淆矩阵（行=真实，列=预测）；`metrics_from_confusion` 负责
混淆矩阵换算每类 `TP / TN / FP / FN / P / R / F1` 与宏平均。
- `main.py`：调 `digits_train` 拿回 7 元组
（模型 + 训练/测试 loss/acc + 宏平均 P/R 历史），再调两次 `evaluate`
打印训练/测试总体与每类准确率、P/R/F1 全表，最后画
`loss_curve / accuracy_curve / confusion_matrix / pr_table` 等图到 `run/`。
