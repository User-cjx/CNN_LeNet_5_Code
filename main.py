"""完成main函数，负责函数训练"""
from pathlib import Path
import os
import torch

from train.train import digits_train, evaluate
from utils import plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix


def print_summary(train_loss: float, train_acc: float, train_correct: list,
                  train_total: list, test_loss: float, test_acc: float,
                  test_correct: list, test_total: list) -> None:
    train_n = sum(train_total)
    test_n = sum(test_total)
    print(f"\n--- 评估结果 (训练集 {train_n} 张 | 测试集 {test_n} 张) ---")
    print(f"训练: Acc {train_acc:.4f} ({sum(train_correct)}/{train_n}) | Loss {train_loss:.4f}")
    print(f"测试: Acc {test_acc:.4f} ({sum(test_correct)}/{test_n}) | Loss {test_loss:.4f}")
    print(f"{'类别':<6}{'训练集':<22}{'测试集':<22}")
    for digit in range(10):
        train_a = train_correct[digit] / train_total[digit] if train_total[digit] else 0.0
        test_a = test_correct[digit] / test_total[digit] if test_total[digit] else 0.0
        print(f"{digit:<8}{train_a:.4f} ({train_correct[digit]}/{train_total[digit]:<6})"
              f"{test_a:.4f} ({test_correct[digit]}/{test_total[digit]})")


if __name__ == "__main__":
    batch_size = 32
    epochs = 40


    model, loss_history, accuracy_history = digits_train(batch_size, epochs)

    train_loss, train_acc, train_class_correct, train_class_total, _ = evaluate(
        model, batch_size, split="training_img")
    test_loss, test_acc, test_class_correct, test_class_total, test_confusion = evaluate(
        model, batch_size, split="resize_test_img")

    print_summary(train_loss, train_acc, train_class_correct, train_class_total,
                  test_loss, test_acc, test_class_correct, test_class_total)

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "digit_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

    loss_fig_path = plot_loss_curve(loss_history)
    print(f"损失曲线已保存到: {loss_fig_path}")
    acc_fig_path = plot_accuracy_curve(accuracy_history)
    print(f"准确率曲线已保存到: {acc_fig_path}")
    cm_fig_path = plot_confusion_matrix(test_confusion)
    print(f"测试集混淆矩阵已保存到: {cm_fig_path}")
