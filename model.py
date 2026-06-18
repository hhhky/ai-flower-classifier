# -*- coding: utf-8 -*-
"""
统一模型模块 — ResNet-50 迁移学习
====================================
与 train_transfer.py 配套，训练出的模型可直接被 app.py 加载。
模型架构：mindcv ResNet-50（ImageNet 预训练）→ 5 类花卉分类
当前最佳准确率：86.24%
"""

import os, sys, json
from pathlib import Path
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import transforms, vision

import mindcv

# ============================================================
# 配置常量
# ============================================================
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "static" / "results"

IMAGE_SIZE = 224
NUM_CLASSES = 5

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CLASS_NAMES_CN = ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]


# ============================================================
# 模型定义
# ============================================================

class TransferNet(nn.Cell):
    """
    ResNet-50 迁移学习模型。
    架构：ResNet-50 backbone + 5 类分类头
    参数量：~23M（backbone）+ ~10K（分类头）
    """
    def __init__(self):
        super().__init__()
        self.network = mindcv.create_model(
            'resnet50', pretrained=False,
            num_classes=NUM_CLASSES,
            in_channels=3,
        )

    def construct(self, x):
        return self.network(x)


# ============================================================
# 模型加载
# ============================================================

def load_trained_model():
    """加载训练好的 ResNet-50 模型（单例模式供 app.py 调用）"""
    net = TransferNet()
    model_path = MODEL_DIR / "flower_cnn_final.ckpt"

    if model_path.exists():
        param_dict = ms.load_checkpoint(str(model_path))
        ms.load_param_into_net(net, param_dict)
        net.set_train(False)
        print(f"  ✅ 模型已加载: {model_path}")
        print(f"     架构: ResNet-50 (mindcv)")
        print(f"     输入: {IMAGE_SIZE}×{IMAGE_SIZE} RGB")
        print(f"     输出: {NUM_CLASSES} 类花卉")
    else:
        print(f"  ⚠️  未找到模型文件 {model_path}")
        print(f"     请先运行: python train_transfer.py")

    return net


# ============================================================
# 单张图片推理
# ============================================================

def predict_image(net, image_path):
    """
    对单张图片推理，返回 Top-3 预测结果。
    预处理流程与训练时验证集保持一致：Resize(256) → CenterCrop(224) → Normalize
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # (H, W, C) uint8

    # ImageNet 标准化参数（uint8 范围）
    mean_uint8 = [m * 255 for m in [0.485, 0.456, 0.406]]
    std_uint8 = [s * 255 for s in [0.229, 0.224, 0.225]]

    transform = transforms.Compose([
        vision.Resize(256),
        vision.CenterCrop(IMAGE_SIZE),
        vision.Normalize(mean=mean_uint8, std=std_uint8),
        vision.HWC2CHW(),
    ])

    img_tensor = transform(img_np)
    img_tensor = ms.Tensor(img_tensor).expand_dims(0)  # (1, C, H, W)

    # 推理
    net.set_train(False)
    logits = net(img_tensor)
    probabilities = ops.softmax(logits, axis=1).asnumpy()[0]

    # Top-3 预测
    top3_idx = np.argsort(probabilities)[::-1][:3]

    results = []
    for idx in top3_idx:
        results.append({
            "class_name": CLASS_NAMES[idx],
            "class_name_cn": CLASS_NAMES_CN[idx],
            "confidence": round(float(probabilities[idx]) * 100, 2),
        })

    return results


# ============================================================
# 特征图提取（供 Web 可视化）
# ============================================================

def extract_feature_maps(net, image_path):
    """
    提取 ResNet-50 各阶段特征图，用于 Web 端的 AI 可解释性展示。

    ResNet-50 特征提取流程：
      Conv1 (7×7, 64ch)  → 边缘/颜色检测
      Layer1 (256ch)      → 纹理/形状
      Layer2 (512ch)      → 局部特征
      Layer3 (1024ch)     → 花瓣/花蕊
      Layer4 (2048ch)     → 高级语义
    """
    import io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    mean_uint8 = [m * 255 for m in [0.485, 0.456, 0.406]]
    std_uint8 = [s * 255 for s in [0.229, 0.224, 0.225]]

    transform = transforms.Compose([
        vision.Resize(256),
        vision.CenterCrop(IMAGE_SIZE),
        vision.Normalize(mean=mean_uint8, std=std_uint8),
        vision.HWC2CHW(),
    ])

    img_tensor = transform(img_np)
    img_tensor = ms.Tensor(img_tensor).expand_dims(0)

    net.set_train(False)

    # 逐层前向传播，收集中间输出
    x = net.network.conv1(img_tensor)
    x = net.network.bn1(x)
    x = net.network.relu(x)
    conv1_out = x  # 64 channels

    x = net.network.maxpool(x)
    x = net.network.layer1(x)
    layer1_out = x  # 256 channels

    x = net.network.layer2(x)
    layer2_out = x  # 512 channels

    x = net.network.layer3(x)
    layer3_out = x  # 1024 channels

    x = net.network.layer4(x)
    layer4_out = x  # 2048 channels

    layers = [
        ("Conv1 — 边缘/颜色检测器 (64通道)", conv1_out, 64),
        ("Layer1 — 纹理/形状 (256通道)", layer1_out, 256),
        ("Layer2 — 局部特征 (512通道)", layer2_out, 512),
        ("Layer3 — 花瓣/花蕊 (1024通道)", layer3_out, 1024),
        ("Layer4 — 高级语义 (2048通道)", layer4_out, 2048),
    ]

    feature_images = []
    for layer_name, output, total_ch in layers:
        fm = output.asnumpy()[0]
        num_show = min(8, total_ch)
        cols = 4
        rows = 2

        fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
        axes = axes.flatten()

        # 均匀采样通道（覆盖整个通道范围）
        step = max(1, total_ch // num_show)
        for i in range(num_show):
            ch_idx = i * step
            fmap = fm[ch_idx]
            f_min, f_max = fmap.min(), fmap.max()
            if f_max > f_min:
                fmap = (fmap - f_min) / (f_max - f_min)
            axes[i].imshow(fmap, cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"通道 {ch_idx+1}", fontsize=8)
        for i in range(num_show, len(axes)):
            axes[i].axis("off")

        fig.suptitle(layer_name, fontsize=13, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        feature_images.append({
            "name": layer_name,
            "image": f"data:image/png;base64,{img_b64}",
        })

    return feature_images
