"""绘图工具：损失下降曲线与训练准确率折线图，统一保存到run文件夹。"""
from pathlib import Path

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def _save_dir() -> Path:
    """保证项目根目录下存在run文件夹并返回其路径。"""
    run_dir = Path(__file__).resolve().parent.parent / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def plot_loss_curve(loss_history: list) -> Path:
    """绘制损失下降曲线，保存到 run/loss_curve.png 并弹窗展示。"""
    fig_path = _save_dir() / "loss_curve.png"
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker=".")
    plt.title("损失下降曲线")
    plt.xlabel("训练次数(epochs)")
    plt.ylabel("损失函数值(loss)")
    plt.grid(True, linestyle="-", alpha=1)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig_path


def plot_accuracy_curve(accuracy_history: list) -> Path:
    """绘制训练准确率折线图，保存到 run/accuracy_curve.png 并弹窗展示。"""
    fig_path = _save_dir() / "accuracy_curve.png"
    plt.figure()
    plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker=".", color="orange")
    plt.title("训练准确率折线图")
    plt.xlabel("训练次数(epochs)")
    plt.ylabel("准确率(accuracy)")
    plt.grid(True, linestyle="-", alpha=1)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig_path


def plot_confusion_matrix(confusion_matrix: list) -> Path:
    """绘制测试集混淆矩阵热力图，保存到 run/confusion_matrix.png 并弹窗展示。

    参数：
        confusion_matrix: 10x10 列表，行=真实数字(y轴)，列=预测数字(x轴)
    """
    fig_path = _save_dir() / "confusion_matrix.png"
    counts = [[float(v) for v in row] for row in confusion_matrix]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(counts, cmap="Blues")

    fig.colorbar(im, ax=ax)
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测数字")
    ax.set_ylabel("真实数字")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    threshold = max(max(row) for row in counts) / 2.0
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
