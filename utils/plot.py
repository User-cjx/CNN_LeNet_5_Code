"""绘图工具：loss/acc 随 epoch 变化曲线 + 混淆矩阵，统一保存到run文件夹。"""
from pathlib import Path

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def _save_dir() -> Path:
    """保证项目根目录下存在run文件夹并返回其路径。"""
    run_dir = Path(__file__).resolve().parent.parent / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _epochs_axis(train_hist: list, test_hist: list) -> range:
    return range(1, max(len(train_hist), len(test_hist)) + 1)


def plot_loss_curve(train_loss: list, test_loss: list) -> Path:
    """绘制 loss 随 epoch 变化曲线（训练+测试同图），保存到 run/loss_curve.png。"""
    fig_path = _save_dir() / "loss_curve.png"
    plt.figure()
    plt.plot(_epochs_axis(train_loss, test_loss), train_loss, marker=".", label="训练集")
    plt.plot(_epochs_axis(train_loss, test_loss), test_loss, marker=".", label="测试集")
    plt.legend()
    plt.title("Loss 随 epoch 变化曲线")
    plt.xlabel("训练次数(epochs)")
    plt.ylabel("损失函数值(loss)")
    plt.grid(True, linestyle="-", alpha=1)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig_path


def plot_accuracy_curve(train_acc: list, test_acc: list) -> Path:
    """绘制 acc 随 epoch 变化曲线（训练+测试同图），保存到 run/accuracy_curve.png。"""
    fig_path = _save_dir() / "accuracy_curve.png"
    plt.figure()
    plt.plot(_epochs_axis(train_acc, test_acc), train_acc, marker=".", color="orange", label="训练集")
    plt.plot(_epochs_axis(train_acc, test_acc), test_acc, marker=".", color="blue", label="测试集")
    plt.legend()
    plt.title("Acc 随 epoch 的变化曲线")
    plt.xlabel("训练次数(epochs)")
    plt.ylabel("准确率(accuracy)")
    plt.grid(True, linestyle="-", alpha=1)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig_path


def plot_confusion_matrix(confusion_matrix: list) -> Path:
    """混淆矩阵热力图单独成图，保存到 run/confusion_matrix.png。行=真实，列=预测。"""
    fig_path = _save_dir() / "confusion_matrix.png"
    counts = [[float(v) for v in row] for row in confusion_matrix]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(counts, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    vmax = max(max(row) for row in counts)
    threshold = vmax / 2.0 if vmax > 0 else 0
    for i in range(10):
        for j in range(10):
            value = int(counts[i][j])
            color = "white" if counts[i][j] > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig_path


def plot_per_class_bar(per_precision: list, per_recall: list, per_f1: list) -> Path:
    """每类 P/R/F1 分组柱状图，保存到 run/per_class_bar.png。"""
    import numpy as np
    fig_path = _save_dir() / "per_class_bar.png"
    n = len(per_precision)
    x = np.arange(n)  # 横轴为 0~9 十个类别
    w = 0.25  # 组内三根柱(P/R/F1)并排，需错开 ±w 避免重叠
    plt.figure(figsize=(10, 5))
    plt.bar(x - w, per_precision, width=w, label="精确率(P)")
    plt.bar(x, per_recall, width=w, label="召回率(R)")
    plt.bar(x + w, per_f1, width=w, label="F1")
    plt.xticks(x, [str(i) for i in range(n)])
    plt.ylim(0, 1.05)
    plt.xlabel("类别")
    plt.ylabel("分数")
    plt.title("每类 P/R/F1 分组柱状图")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig_path


def plot_pr_curve(macro_p_hist: list, macro_r_hist: list) -> Path:
    """宏平均 P/R 随 epoch 变化曲线，保存到 run/pr_curve.png。"""
    fig_path = _save_dir() / "pr_curve.png"
    plt.figure()
    epochs = _epochs_axis(macro_p_hist, macro_r_hist)
    plt.plot(epochs, macro_p_hist, marker=".", label="精确率(P Macro)")
    plt.plot(epochs, macro_r_hist, marker=".", label="召回率(R Macro)")
    plt.legend()
    plt.title("P/R 随 epoch 变化曲线")
    plt.xlabel("训练次数(epochs)")
    plt.ylabel("分数")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="-", alpha=1)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig_path


def plot_confusion_matrix_norm(confusion_matrix: list) -> Path:
    """按行归一化的混淆矩阵热力图，保存到 run/confusion_matrix_norm.png。行=真实，列=预测。"""
    fig_path = _save_dir() / "confusion_matrix_norm.png"
    n = len(confusion_matrix)
    norm = []
    # 按行归一化：每行除以该真实类别总数，消除各类别样本量差异的影响
    for row in confusion_matrix:
        s = sum(row)
        norm.append([v / s if s > 0 else 0.0 for v in row])

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax.set_title("混淆矩阵(按行归一化)")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    for i in range(n):
        for j in range(n):
            color = "white" if norm[i][j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i][j]:.2f}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig_path


def plot_error_samples(samples: list, ncols: int = 6) -> Path:
    """误分类样本展示图，保存到 run/error_samples.png。

    参数 samples: [(image(HxW array), true_label, pred_label), ...]。
    """
    import numpy as np
    fig_path = _save_dir() / "error_samples.png"
    if not samples:
        plt.figure()
        plt.text(0.5, 0.5, "无误分类样本", ha="center", fontsize=14)
        plt.axis("off")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig_path
    ncols = max(1, min(ncols, len(samples)))
    nrows = (len(samples) + ncols - 1) // ncols  # 向上取整，保证所有样本都有格子
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[len(samples):]:
        ax.axis("off")  # 样本数凑不满整行时，多余子图留空
    for ax, (img, t, p) in zip(axes, samples):
        ax.imshow(np.asarray(img).squeeze(), cmap="gray")
        ax.set_title(f"真实{t}->预测{p}", fontsize=10)
        ax.axis("off")
    fig.suptitle("误分类样本展示", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig_path


def plot_pr_table(tp: list, fp: list, fn: list, tn: list,
                  per_precision: list, per_recall: list, per_f1: list) -> Path:
    """分类报告表：Class/TP/TN/FP/FN/Precision/Recall/F1 + Macro Avg 行，保存到 run/pr_table.png。"""
    fig_path = _save_dir() / "pr_table.png"
    n = len(per_precision)
    macro_p = sum(per_precision) / max(n, 1)
    macro_r = sum(per_recall) / max(n, 1)
    macro_f1 = sum(per_f1) / max(n, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_title("测试集分类报告表", fontsize=14, weight="bold", pad=12)
    col_labels = ["类别（class）", "TP", "TN", "FP", "FN", "精确率（Precision）", "召回率（Recall）", "F1"]
    cell_text = [[str(i), str(tp[i]), str(tn[i]), str(fp[i]), str(fn[i]),
                  f"{per_precision[i]:.4f}", f"{per_recall[i]:.4f}", f"{per_f1[i]:.4f}"]
                 for i in range(n)]
    cell_text.append(["Macro Avg", "-", "-", "-", "-",
                      f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}"])
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif row == len(cell_text):
            cell.set_facecolor("#D9D9D9")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E1F2")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return fig_path
