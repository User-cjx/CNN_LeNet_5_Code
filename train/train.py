"""定义train函数，负责训练模型"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.dataset import CustomDataset
from src.model import DigitNetwork

def digits_train(batch_size: int, epochs: int):
    """定义train函数，写入dataloader类"""

    '''确认设备'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("GPU不可用，请检查GPU是否正常工作")


    '''实例化Dataset功能，包装DataLoader'''
    train_dataset = CustomDataset(root='D:/workspace/Project_learning_number/data', split='training_img')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # batch_size就是getitem里面的index了


    '''创建神经网络实例'''
    model = DigitNetwork().to(device)   # 将模型搬到GPU
    criterion = nn.CrossEntropyLoss()   # 交叉熵函数，自带softmax计算结果（前提是模型没加softmax）
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    '''训练函数'''
    loss_history = []
    accuracy_history = []   # 记录每次训练的准确率，供绘制准确率折线图
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
            bath_loss.backward()    #  反向传播
            optimizer.step()    # 更新参数
            epoch_loss += bath_loss.item()   # 记录每次损失函数的值
            predicted = output.argmax(dim=1)    # 取概率最大的索引作为预测值
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)

        avg_epoch_loss = epoch_loss / len(train_loader)  # 平均到每个batch，方便画梯度下降曲线
        epoch_accuracy = epoch_correct / epoch_total    # 本轮训练准确率
        loss_history.append(avg_epoch_loss)
        accuracy_history.append(epoch_accuracy)
        print(f"第{epoch + 1}次训练, 损失函数值Loss: {avg_epoch_loss:.4f}, 准确率Acc: {epoch_accuracy:.4f}")

    return model, loss_history, accuracy_history


def evaluate(model, batch_size: int, split: str = "new_data_img"):
    """定义评估函数，写入dataloader类，负责测试模型的准确度。

    参数：
        model: 训练好的模型
        batch_size: 批次大小
        split: 数据集划分，可选 "training_img" 或 "test_img"

    返回：
        avg_loss: 平均每个batch的损失
        accuracy: 总准确率
        class_correct: 每个类别预测正确的数量，长度10
        class_total: 每个类别的总样本数，长度10
        confusion_matrix: 10x10 混淆矩阵，行=真实数字(y)，列=预测数字(x)
    """

    '''确认设备'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("GPU不可用，请检查GPU是否正常工作")

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

    avg_loss = total_loss / len(test_loader)    # len(test_loader)是一共有多少个batch，这里算出来的是平均每个batch的损失
    '''完成所有计算后，计算准确率'''
    accuracy = correct / total  # 正确数/全样本

    return avg_loss, accuracy, class_correct, class_total, confusion_matrix





