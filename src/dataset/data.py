"""数据集定义：读取 data/training_img 和 data/test_img 中的手写数字图片。"""
import os.path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Grayscale, Resize, ToTensor


class CustomDataset(Dataset):
    """按文件名‘数字_编号.png’解析标签的手写数字数据集"""

    def __init__(self, root: str = r'D:\workspace\Project_learning_number\data', split: str = 'training_img') -> None:
        self.root_dir = os.path.expanduser(os.path.join(root, split))
        # 与 LeNet-5 输入对齐：转单通道灰度 → 缩到 32×32 → 转 Tensor(值域0~1)
        self.transforms = [Grayscale(), Resize((32, 32)), ToTensor()]   # 进行图片处理

        self.image_paths = []
        self.labels = []
        # 文件名格式固定为“标签_编号.png”
        for file_name in sorted(os.listdir(self.root_dir)):
            if file_name.endswith('.png'):
                label = file_name.split('_')[0]
                image_path = os.path.join(self.root_dir, file_name)
                self.image_paths.append(image_path)
                self.labels.append(int(label))

    def __len__(self):
        # 返回样本总数，决定 DataLoader 能切出多少个 batch
        return len(self.image_paths)

    def __getitem__(self, index):
        # 每次取一张图：PIL 打开 → 依次做灰度/缩放/Tensor，返回(图,标签)
        image = Image.open(self.image_paths[index])
        label = self.labels[index]
        for transform in self.transforms:
            image = transform(image)
        return image, label
