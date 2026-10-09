# 手写数字识别（MNIST · CNN）

一个端到端的手写数字识别项目：基于 PyTorch 搭建卷积神经网络（CNN），在 MNIST 数据集上完成训练、评估、可视化与单张图片推理。

> 适合写进简历的亮点：完整工程结构（数据 → 模型 → 训练 → 评估 → 部署推理）、训练曲线与混淆矩阵可视化、CPU 上即可快速复现。

---

## 目录结构

```
mnist-cnn/
├── model.py          # CNN 模型定义（Conv2d + BN + MaxPool + FC + Dropout）
├── train.py          # 完整训练流程（数据加载/训练/评估/可视化/保存）
├── predict.py        # 单张手写数字图片推理
├── requirements.txt  # 依赖清单
└── .gitignore
```

## 环境要求

- Python 3.8+
- 依赖安装：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
python train.py                 # 默认 10 个 epoch
python train.py --epochs 15 --batch 128 --lr 0.001 --device cuda
```

训练完成后会：
- 在终端打印每个 epoch 的训练/测试准确率
- 生成 `results/training_curves.png`（训练损失与准确率曲线）
- 生成 `results/confusion_matrix.png`（测试集混淆矩阵）
- 保存模型权重到 `checkpoints/mnist_cnn.pt`

### 2. 单张图片推理

```bash
python predict.py --image 某张手写数字图片.png
```

输出示例：

```
预测数字: 7   置信度: 98.32%
```

---

## 模型结构

| 层 | 说明 |
|----|------|
| Conv2d(1→32, 3×3) + BN + ReLU | 提取边缘/笔画等低层特征 |
| MaxPool(2×2) | 下采样，28×28 → 14×14 |
| Conv2d(32→64, 3×3) + BN + ReLU | 提取更高层特征 |
| MaxPool(2×2) | 下采样，14×14 → 7×7 |
| FC(7×7×64 → 128) + Dropout | 特征展平后全连接，抑制过拟合 |
| FC(128 → 10) | 输出 10 个类别的 logits |

## 实验配置与结果

- 优化器：Adam，lr=0.001
- 损失函数：CrossEntropyLoss
- 训练/测试集：MNIST 官方划分（60000 / 10000）
- 数据预处理：灰度归一化 `Normalize(0.1307, 0.3081)`

> 在 CPU 上训练约 10 个 epoch 即可达到 **98%+** 测试准确率。

---

## License

MIT
