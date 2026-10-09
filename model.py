"""
model.py
手写数字识别 CNN 模型定义
结构: Conv2d -> BN -> ReLU -> MaxPool -> Conv2d -> BN -> ReLU -> MaxPool -> Flatten -> FC -> Dropout -> FC
"""
import torch
import torch.nn as nn


class MNISTNet(nn.Module):
    """一个轻量、参数量小、在 CPU 上也能快速训练的 CNN，用于 MNIST 手写数字识别（0-9）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 第一段卷积: 1 -> 32 通道
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 28 -> 14

        # 第二段卷积: 32 -> 64 通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 14 -> 7

        # 全连接分类头: 7*7*64 -> 128 -> num_classes
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)          # 展平
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


def build_model(num_classes: int = 10) -> MNISTNet:
    """工厂函数，返回模型实例。"""
    return MNISTNet(num_classes=num_classes)
