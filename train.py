"""
train.py
手写数字识别（MNIST）完整训练流程
包含: 数据加载与预处理 -> 模型构建 -> 训练循环 -> 评估 -> 可视化(训练曲线/混淆矩阵) -> 模型保存

运行方式:
    python train.py                      # 默认参数
    python train.py --epochs 15 --batch 128 --lr 0.001 --device cuda
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model


# --------------------------------------------------------------------------- #
# 1. 数据加载与预处理
# --------------------------------------------------------------------------- #
def get_dataloaders(batch_size: int = 64, download: bool = True):
    """下载 MNIST 并返回 (train_loader, test_loader, class_names)。

    MNIST 是灰度 28x28 手写数字，这里用 (0.1307, 0.3081) 作为官方标准归一化均值/方差。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                     # HWC uint8 -> CHW float [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # 归一化到均值0, 方差1
    ])

    train_set = datasets.MNIST(
        root="./data", train=True, download=download, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=download, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# 2. 训练 / 评估
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


# --------------------------------------------------------------------------- #
# 3. 可视化
# --------------------------------------------------------------------------- #
def plot_history(train_loss, train_acc, test_acc):
    """绘制训练损失与准确率曲线，保存为 results/training_curves.png"""
    import matplotlib
    matplotlib.use("Agg")  # 无界面环境
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, "o-", color="#2f80ed", label="train loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_acc, "o-", color="#27ae60", label="train acc")
    ax2.plot(epochs, test_acc, "s--", color="#eb5757", label="test acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    print("[✓] 训练曲线已保存: results/training_curves.png")


def plot_confusion_matrix(all_preds, all_labels, class_names):
    """绘制测试集混淆矩阵，保存为 results/confusion_matrix.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("[✓] 混淆矩阵已保存: results/confusion_matrix.png")


# --------------------------------------------------------------------------- #
# 4. 主入口
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="MNIST 手写数字识别 - CNN 训练")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    train_loader, test_loader = get_dataloaders(batch_size=args.batch)
    class_names = [str(i) for i in range(10)]

    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_hist, train_acc_hist, test_acc_hist = [], [], []
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device, epoch)
        v_loss, v_acc, _, _ = evaluate(model, test_loader, criterion, device)
        train_loss_hist.append(t_loss)
        train_acc_hist.append(t_acc)
        test_acc_hist.append(v_acc)
        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={t_loss:.4f} train_acc={t_acc:.2f}% "
              f"test_acc={v_acc:.2f}%")

    # 最终测试评估 + 可视化
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print(f"\n最终测试准确率: {test_acc:.2f}%")

    plot_history(train_loss_hist, train_acc_hist, test_acc_hist)
    plot_confusion_matrix(all_preds, all_labels, class_names)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mnist_cnn.pt")
    print("[✓] 模型已保存: checkpoints/mnist_cnn.pt")


if __name__ == "__main__":
    main()
