"""定义train函数，负责训练模型"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.dataset import CustomDataset
from src.model import LeNet5


def pick_device() -> torch.device:
    """有 GPU 用 GPU，否则回退 CPU，保证哪里都能跑"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("没有找到cuda，请检查是否安装。")   # 我的电脑没有安装CPU版本的pytorch
    print(f"使用设备: {device}")
    return device


def metrics_from_confusion(confusion_matrix):
    """计算每类的 TP/FP/FN/TN、精确率 P、召回率 R、F1

    行=真实，列=预测。对类别 i：
      TP = C[i][i]，FP = 列和-TP，FN = 行和-TP，TN = 总数-TP-FP-FN，
      P = TP/(TP+FP)，R = TP/(TP+FN)，F1 = 2PR/(P+R)，无分母时记 0.0。

    返回：
        tp, fp, fn, tn, per_precision, per_recall, per_f1, macro_p, macro_r, macro_f1
    """
    n_class = len(confusion_matrix) # 混淆矩阵数值的长度
    total = sum(sum(row) for row in confusion_matrix)   # 获得样本总数
    tp, fp, fn, tn, per_p, per_r, per_f1 = [], [], [], [], [], [], []
    for i in range(n_class):
        tp_i = confusion_matrix[i][i]
        col_sum = sum(confusion_matrix[r][i] for r in range(n_class))   # 每一列的总和
        row_sum = sum(confusion_matrix[i])  # 每一行的总和
        fp_i = col_sum - tp_i
        fn_i = row_sum - tp_i
        tn_i = total - tp_i - fp_i - fn_i
        p_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        r_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        f1_i = 2 * p_i * r_i / (p_i + r_i) if (p_i + r_i) > 0 else 0.0
        tp.append(tp_i)
        fp.append(fp_i)
        fn.append(fn_i)
        tn.append(tn_i)
        per_p.append(p_i)
        per_r.append(r_i)
        per_f1.append(f1_i)
    macro_p = sum(per_p) / max(n_class, 1)
    macro_r = sum(per_r) / max(n_class, 1)
    macro_f1 = sum(per_f1) / max(n_class, 1)
    # 返回训练结果，为作图准备
    return tp, fp, fn, tn, per_p, per_r, per_f1, macro_p, macro_r, macro_f1


def eval_loader(model, loader, criterion, device):
    """在给定 loader 上只做前向，顺带统计混淆矩阵并算出 P/R。

    返回：
        avg_loss: 平均每个 batch 的损失
        accuracy: 总准确率
        macro_p: 10 类精确率的宏平均
        macro_r: 10 类召回率的宏平均
        per_precision: 每类精确率，长度10
        per_recall: 每类召回率，长度10
        confusion_matrix: 10x10，行=真实，列=预测
    """
    model.eval()    # 将模型转换为测试模式，不会backward计算梯度
    total_loss = 0.0
    correct = 0
    total = 0
    confusion_matrix = [[0] * 10 for _ in range(10)]
    with torch.no_grad():   # 关闭梯度计算，只负责前向传播
        for images, labels in loader:
            images = images.to(device)  # 将数据搬到GPU上
            labels = labels.to(device)  # 将标签搬到GPU上
            output = model(images)  # 将数据集丢进model进行训练
            total_loss += criterion(output, labels).item()  # 计算损失值
            predicted = output.argmax(dim=1)    # 取概率最大的索引作为预测值
            correct += (predicted == labels).sum().item()  # 计算准确率
            total += labels.size(0)
            for t, p in zip(labels.view(-1).tolist(), predicted.view(-1).tolist()):
                confusion_matrix[t][p] += 1 # 获取混淆矩阵的值
    tp, fp, fn, tn, per_p, per_r, _, macro_p, macro_r, _ = metrics_from_confusion(confusion_matrix) # 从混淆矩阵获得每一类的值
    return (total_loss / max(len(loader), 1), correct / max(total, 1),
            macro_p, macro_r, per_p, per_r, confusion_matrix)


def digits_train(batch_size: int, epochs: int, train_split: str = "training_img",
                 test_split: str = "test_img", lr: float = 0.0001,
                 save_dir=None):
    """训练 LeNet-5，每轮末在测试集上评估，存 best.pt + last.pt。

    参数：
        batch_size: 批次大小
        epochs: 训练轮数
        train_split: 训练集文件夹名，默认 training_img
        test_split: 测试集文件夹名，默认 test_img（按测试集准确率选 best）
        lr: Adam 学习率，默认 0.001
        save_dir: 权重保存目录，默认项目根目录下 checkpoints/

    返回：
        model: 已载入 best 权重的模型
        train_loss_hist: 每轮训练 loss
        train_acc_hist: 每轮训练 acc
        test_loss_hist: 每轮测试 loss
        test_acc_hist: 每轮测试 acc
        macro_p_hist: 每轮测试集精确率(宏平均)
        macro_r_hist: 每轮测试集召回率(宏平均)
    """

    device = pick_device()

    '''实例化Dataset功能，包装DataLoader，获取数据'''
    train_dataset = CustomDataset(root='D:/workspace/Project_learning_number/data', split=train_split)
    test_dataset = CustomDataset(root='D:/workspace/Project_learning_number/data', split=test_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    '''创建权重保存目录'''
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    '''创建神经网络实例'''
    model = LeNet5().to(device)   # 将模型搬到GPU
    criterion = nn.CrossEntropyLoss()   # 交叉熵函数，自带softmax计算结果
    optimizer = optim.Adam(model.parameters(), lr=lr)   # Adam 自适应优化器，，可以在此修改β参数

    '''训练函数'''
    train_loss_hist = []    # 记录每次训练的损失值，供绘制损失值折线图
    train_acc_hist = []   # 记录每次训练的准确率，供绘制准确率折线图
    test_loss_hist = [] # 记录每次测试的损失值，供绘制损失值折线图
    test_acc_hist = []   # 记录每次测试的准确率，供绘制准确率折线图
    macro_p_hist = []   # 每轮测试集精确率(宏平均)，供绘制 PR 曲线
    macro_r_hist = []   # 每轮测试集召回率(宏平均)，供绘制 PR 曲线
    best_acc = -1.0   # 记录最佳准确率，初始化为-1.0
    best_epoch = 0   # 记录最佳准确率对应的轮数
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0   # 本轮预测正确的样本数
        epoch_total = 0     # 本轮总样本数
        model.train()   # 确保训练模式

        for images, labels in train_loader:
            images = images.to(device)  # 图片搬去 GPU
            labels = labels.to(device)  # 标签搬去 GPU
            optimizer.zero_grad()   # 每次先进行梯度清零
            output = model(images)  # 将数据集丢进model进行训练
            bath_loss = criterion(output, labels) # 进行交叉熵运算
            bath_loss.backward()    #  反向传播，计算梯度
            optimizer.step()    # 更新参数
            epoch_loss += bath_loss.item()   # 记录每次损失函数的值
            predicted = output.argmax(dim=1)    # 取概率最大的索引作为预测值
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)

        avg_train_loss = epoch_loss / max(len(train_loader), 1)  # 平均到每个batch，方便画梯度下降曲线
        train_acc = epoch_correct / max(epoch_total, 1)    # 本轮训练准确率
        test_loss, test_acc, macro_p, macro_r, _, _, _ = eval_loader(model, test_loader, criterion, device)

        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        macro_p_hist.append(macro_p)
        macro_r_hist.append(macro_r)

        # 每轮都覆盖 last.pt
        torch.save(model.state_dict(), last_path)
        # 测试集准确率最高则更新 best.pt
        tag = ""
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            tag = "  <-- New Best!"
        print(f"第{epoch + 1}/{epochs}轮, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, P: {macro_p:.4f}, R: {macro_r:.4f}{tag}")

    print(f"\n最佳测试准确率: {best_acc:.4f} (第{best_epoch}轮)，已保存到: {best_path}")
    print(f"最后一轮权重已保存到: {last_path}")

    # 把 best 权重载回 model 再返回，后续评估/画混淆矩阵都基于最优模型
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    return model, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, macro_p_hist, macro_r_hist


def evaluate(model, batch_size: int, split: str = "test_img"):
    """定义评估函数，写入dataloader类，负责测试模型的准确度。

    参数：
        model: 训练好的模型
        batch_size: 批次大小
        split: 数据集划分，训练集 "training_img" / 测试集 "test_img"

    返回：
        avg_loss: 平均每个batch的损失
        accuracy: 总准确率
        class_correct: 每个类别预测正确的数量，长度10
        class_total: 每个类别的总样本数，长度10
        confusion_matrix: 10x10 混淆矩阵，行=真实数字(y)，列=预测数字(x)
    """

    device = pick_device()

    '''实例化Dataset功能'''
    test_dataset = CustomDataset(root='D:/workspace/Project_learning_number/data', split=split)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)    # 保证模型和数据在同一设备上
    model.eval()    # pytorch默认模式是model.train()为训练模式，model.eval()为测试模式，不会计算梯度
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0    # 初始化总损失
    correct = 0 # 初始化预测正确的数量
    total = 0   # 初始化总样本数量
    class_correct = [0] * 10    # 每个类别预测正确的数量
    class_total = [0] * 10  # 每个类别的总样本数量，为1×10的矩阵，shape为(10,)
    confusion_matrix = [[0] * 10 for _ in range(10)]    # 混淆矩阵的数据，为10×10的矩阵，shape为(10,10)

    with torch.no_grad():   # 关闭梯度计算，只负责前向传播
        for images, labels in test_loader:  # 按批次拿test数据
            images = images.to(device)
            labels = labels.to(device)  # 将数据搬到GPU上
            output = model(images)  # output的形状是（batch，10）
            bath_loss = criterion(output, labels)   # 对这个batch做损失值计算
            total_loss += bath_loss.item()  # 计算所有batch完成后的损失值
            '''torch.max(output.data, 1)是在output返回一个元组，分别是最大值，最大值的索引，
            而一共只有10个类别，那么出现最大值的那个位置，就是0-9这个返回的值，也就是最大值的索引，相当于预测值了'''

            each_sample, predicted = torch.max(output.data, 1)  # 此函数的意思是返回每个样本的最大值，并返回最大值的索引
            total += labels.size(0) # 统计每个batch的label个数，最后一个batch可能 <= batch_size

            '''统计预测正确的数量，predicted == labels时为true，得到1，false 则为0'''
            correct += (predicted == labels).sum().item()  # item()只是把计算结果换成普通数字方便计算
            for true_label, pred_label in zip(labels.view(-1).tolist(), predicted.view(-1).tolist()):   # 把预测值和真实值压缩成元组，两两配对，用两个变量分别接收
                class_total[true_label] += 1   # 统计每个类别的总样本数量
                if true_label == pred_label:    # 如果预测值=真实值
                    class_correct[true_label] += 1   # 统计每个类别预测正确的数量
                confusion_matrix[true_label][pred_label] += 1   # 统计混淆矩阵的值，行为真实数字，列为预测数字

    avg_loss = total_loss / max(len(test_loader), 1)    # len(test_loader)是一共有多少个batch，这里算出来的是平均每个batch的损失
    '''完成所有计算后，计算准确率'''
    accuracy = correct / max(total, 1)  # 正确数/全样本

    return avg_loss, accuracy, class_correct, class_total, confusion_matrix


def collect_error_samples(model, batch_size: int, confusion_matrix, split: str = "test_img",
                          per_pair: int = 4, top_k_pairs: int = 6):
    """按 Top-K 混淆对收集误分类样本，供 plot_error_samples 画图。

    返回 [(image(HxW numpy), true_label, pred_label), ...]，总数 <= top_k_pairs * per_pair。
    """
    import numpy as np

    n = len(confusion_matrix)
    pairs = [(confusion_matrix[t][p], t, p)
             for t in range(n) for p in range(n) if t != p]
    pairs.sort(reverse=True)
    wanted = {(t, p): per_pair for _, t, p in pairs[:top_k_pairs] if _ > 0}
    if not wanted:
        return []

    device = pick_device()
    dataset = CustomDataset(root='D:/workspace/Project_learning_number/data', split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    samples = []
    remaining = dict(wanted)
    with torch.no_grad():
        for images, labels in loader:
            output = model(images.to(device))
            predicted = output.argmax(dim=1)
            for img, t, p in zip(images.cpu(), labels.tolist(), predicted.cpu().tolist()):
                if t != p and (t, p) in remaining and remaining[(t, p)] > 0:
                    arr = np.asarray(img.squeeze().numpy())
                    samples.append((arr, t, p))
                    remaining[(t, p)] -= 1
                    if all(v <= 0 for v in remaining.values()):
                        # 按混淆次数排序，保证展示顺序稳定
                        order = {(t, p): i for i, (_, t, p) in enumerate(pairs)}
                        samples.sort(key=lambda s: order.get((s[1], s[2]), 999))
                        return samples
    order = {(t, p): i for i, (_, t, p) in enumerate(pairs)}
    samples.sort(key=lambda s: order.get((s[1], s[2]), 999))
    return samples
