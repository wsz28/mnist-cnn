"""
predict.py
用训练好的模型对单张手写数字图片做推理

运行方式:
    python predict.py --image path/to/img.png
    python predict.py --image path/to/img.png --checkpoint checkpoints/mnist_cnn.pt
"""
import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from model import build_model


def load_image(path: str):
    """读取图片并按 MNIST 训练时的预处理方式处理成 1x1x28x28 张量。"""
    img = Image.open(path).convert("L")            # 转灰度
    img = img.resize((28, 28), Image.BILINEAR)     # 缩放到 28x28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform(img).unsqueeze(0)             # 加 batch 维度


def main():
    parser = argparse.ArgumentParser(description="单张手写数字推理")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", default="checkpoints/mnist_cnn.pt")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"找不到模型 {args.checkpoint}，请先运行 python train.py 训练。"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    x = load_image(args.image).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        conf = prob[0, pred].item()

    print(f"预测数字: {pred}   置信度: {conf:.2%}")


if __name__ == "__main__":
    main()
