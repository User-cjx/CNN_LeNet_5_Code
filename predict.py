"""手写数字自测脚本：读取 data/my_digits 中的照片，用 best.pt 预测每个数字。"""
from pathlib import Path

import cv2
import torch

from src.model import LeNet5


def load_model(weights: Path = None) -> torch.nn.Module:
    """载入 LeNet5 结构 + best.pt 权重，切换到评估模式（关闭 dropout/BN 更新）。"""
    if weights is None:
        weights = Path(__file__).resolve().parent / "checkpoints" / "best.pt"
    # 有 GPU 用 GPU，否则回退 CPU，保证任意机器都能跑通预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5()
    # map_location 保证 GPU 训练的权重也能在纯 CPU 机器上加载
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    return model, device


def preprocess(path: Path) -> torch.Tensor:
    """cv2 预处理链（参数已按自测照片调好）：灰度→32×32→均衡化→取反→双边滤波→二值化。

    输入：图片路径（Path 或 str）；输出：1×1×32×32 的 float Tensor（值域 0~1，CPU 上），
    可直接 `.to(device)` 送模型，也可 `.squeeze(0)` 取 1×32×32 存拼图。
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不到图片，请检查路径: {path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
    resize_img = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA)  # resize 处理
    equalize_img = cv2.equalizeHist(resize_img)  # 直方图均衡化
    inverted_image = cv2.bitwise_not(equalize_img)  # 取反
    bilateral_img = cv2.bilateralFilter(inverted_image, 3, 30, 16)  # 双值滤波
    _, threshold_img = cv2.threshold(bilateral_img, 190, 255, cv2.THRESH_TOZERO)  # 二值化
    dilate_img = cv2.dilate(threshold_img,(3, 3), 3)
    t = torch.from_numpy(dilate_img).float() / 255.0  # H×W numpy → Tensor，归一化到 0~1
    return t.unsqueeze(0).unsqueeze(0)  # 补通道维+batch 维 → 1×1×32×32，满足模型输入要求


def main() -> None:
    folder = Path(__file__).resolve().parent / "data" / "my_digits"
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    if not files:
        print(f"{folder} 中没有找到图片")
        return
    model, device = load_model()
    results = []
    # no_grad 只做前向不建计算图，预测更快且不占梯度内存
    with torch.no_grad():
        for p in files:
            x = preprocess(p)  # 1×1×32×32，CPU 上的 Tensor
            out = model(x.to(device))  # 输出 (1,10)，每维对应数字 0~9 的得分
            prob = torch.softmax(out, dim=1)[0]  # 转为概率分布再取最大者为预测
            pred = int(prob.argmax())
            top2_vals, top2_idx = torch.topk(prob, 2)  # 取前二，便于看出模型在犹豫哪两个数字
            print(f"{p.name}: 预测 {pred}，置信度 {prob[pred]:.2%} "
                  f"(Top2: {int(top2_idx[0])} {float(top2_vals[0]):.2%} / "
                  f"{int(top2_idx[1])} {float(top2_vals[1]):.2%})")
            results.append((p.name, x.squeeze(0), pred, float(prob[pred])))
    save_path = Path(__file__).resolve().parent / "run" / "my_predict.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _save_montage(results, save_path)
    print(f"拼图已保存到: {save_path}")


def _save_montage(results: list, save_path: Path) -> None:
    """把所有自测结果拼成一张图：原图 + 预测标签 + 置信度，方便对照检查。"""
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，服务器/终端下也能直接存图不弹窗
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(results)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.2 * nrows))
    axes = [axes] if n == 1 else list(__import__("numpy").atleast_1d(axes).flatten())
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (name, img, pred, conf) in zip(axes, results):
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(f"{name}\n预测{pred} ({conf:.0%})", fontsize=10)
        ax.axis("off")
    fig.suptitle("手写数字自测结果", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


