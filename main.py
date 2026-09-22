"""main函数，负责函数训练以及格式化输出"""
from pathlib import Path

from train.train import digits_train, evaluate, metrics_from_confusion, collect_error_samples
from utils import (plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix,
                   plot_confusion_matrix_norm, plot_per_class_bar, plot_pr_curve,
                   plot_error_samples, plot_pr_table)


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


def print_pr(tp: list, fp: list, fn: list, tn: list, per_p: list,
             per_r: list, per_f1: list, macro_p: float, macro_r: float, macro_f1: float) -> None:
    print(f"\n--- 测试集 P/R/F1 (宏平均 P {macro_p:.4f} | R {macro_r:.4f} | F1 {macro_f1:.4f}) ---")
    print(f"{'类别':<6}{'TP':<8}{'TN':<8}{'FP':<8}{'FN':<8}{'P':<10}{'R':<10}{'F1':<10}")
    for i in range(10):
        print(f"{i:<8}{tp[i]:<8}{tn[i]:<8}{fp[i]:<8}{fn[i]:<8}"
              f"{per_p[i]:<10.4f}{per_r[i]:<10.4f}{per_f1[i]:<10.4f}")
    print(f"{'Macro':<8}{'-':<8}{'-':<8}{'-':<8}{'-':<8}"
          f"{macro_p:<10.4f}{macro_r:<10.4f}{macro_f1:<10.4f}")


if __name__ == "__main__":
    # 设置batch_size and epochs
    batch_size = 32
    epochs = 30

    # digits_train 内部每轮在测试集上评估，按测试准确率存 checkpoints/best.pt，
    # 每轮覆盖 checkpoints/last.pt，返回的 model 已载入 best 权重
    model, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, macro_p_hist, macro_r_hist = digits_train(
        batch_size, epochs, train_split="training_img", test_split="test_img")

    train_loss, train_acc, train_class_correct, train_class_total, _ = evaluate(
        model, batch_size, split="training_img")
    test_loss, test_acc, test_class_correct, test_class_total, test_confusion = evaluate(
        model, batch_size, split="test_img")

    print_summary(train_loss, train_acc, train_class_correct, train_class_total,
                  test_loss, test_acc, test_class_correct, test_class_total)

    # 测试集混淆矩阵换算 TP/FP/FN/TN + P/R/F1 并打印
    tp, fp, fn, tn, per_p, per_r, per_f1, macro_p, macro_r, macro_f1 = metrics_from_confusion(test_confusion)
    print_pr(tp, fp, fn, tn, per_p, per_r, per_f1, macro_p, macro_r, macro_f1)

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    print(f"best.pt: {checkpoint_dir / 'best.pt'}")
    print(f"last.pt: {checkpoint_dir / 'last.pt'}")

    # loss、acc 随 epoch 变化曲线图（训练+测试同图）
    loss_fig_path = plot_loss_curve(train_loss_hist, test_loss_hist)
    print(f"损失曲线已保存到: {loss_fig_path}")
    acc_fig_path = plot_accuracy_curve(train_acc_hist, test_acc_hist)
    print(f"准确率曲线已保存到: {acc_fig_path}")
    # 混淆矩阵热力图单独成图
    cm_fig_path = plot_confusion_matrix(test_confusion)
    print(f"测试集混淆矩阵已保存到: {cm_fig_path}")
    # 分类报告表
    pr_tab_path = plot_pr_table(tp, fp, fn, tn, per_p, per_r, per_f1)
    print(f"分类报告表已保存到: {pr_tab_path}")
    # 每类 P/R/F1 分组柱状图
    bar_path = plot_per_class_bar(per_p, per_r, per_f1)
    print(f"每类P/R/F1柱状图已保存到: {bar_path}")
    # 归一化混淆矩阵
    norm_cm_path = plot_confusion_matrix_norm(test_confusion)
    print(f"归一化混淆矩阵已保存到: {norm_cm_path}")
    # P/R 随 epoch 变化曲线
    pr_curve_path = plot_pr_curve(macro_p_hist, macro_r_hist)
    print(f"P/R曲线已保存到: {pr_curve_path}")
    # 误分类样本展示
    err_samples = collect_error_samples(model, batch_size, test_confusion, split="test_img")
    err_path = plot_error_samples(err_samples)
    print(f"误分类样本图已保存到: {err_path}")
