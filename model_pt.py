# -*- coding: utf-8 -*-
"""
PyTorch 推理模块 — 替代 MindSpore，用于 HF Spaces 部署
========================================================
优势：轻量 (~100MB vs MindSpore ~1GB)、HF 预装、稳定可靠
"""
import io, base64
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms

# 配置
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "flower_model.pth"
IMAGE_SIZE = 224
NUM_CLASSES = 5

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CLASS_NAMES_CN = ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]

# ImageNet 标准化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 全局模型缓存
_model = None


def get_model():
    """加载 PyTorch ResNet-50 模型（单例）"""
    global _model
    if _model is None:
        _model = resnet50(weights=None, num_classes=NUM_CLASSES)
        state = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
        _model.load_state_dict(state)
        _model.eval()
        print(f"  ✅ PyTorch 模型已加载: {MODEL_PATH}")
    return _model


def get_transform():
    """推理预处理 — 与 MindSpore 训练时一致"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def predict_image(model, image_path):
    """单张图片推理，返回 Top-3"""
    img = Image.open(image_path).convert("RGB")
    t = get_transform()
    x = t(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1).squeeze(0).numpy()

    top3 = np.argsort(probs)[::-1][:3]
    return [
        {"class_name": CLASS_NAMES[i], "class_name_cn": CLASS_NAMES_CN[i],
         "confidence": round(float(probs[i]) * 100, 2)}
        for i in top3
    ]


def extract_feature_maps(model, image_path):
    """提取 ResNet-50 各阶段特征图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    t = get_transform()
    x = t(img).unsqueeze(0)

    # Hook 中间层输出
    features = {}
    def hook(name):
        def fn(module, input, output):
            features[name] = output.detach()
        return fn

    handles = [
        model.conv1.register_forward_hook(hook("conv1")),
        model.layer1.register_forward_hook(hook("layer1")),
        model.layer2.register_forward_hook(hook("layer2")),
        model.layer3.register_forward_hook(hook("layer3")),
        model.layer4.register_forward_hook(hook("layer4")),
    ]

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    layer_info = [
        ("Conv1 — 边缘/颜色 (64ch)", "conv1", 64),
        ("Layer1 — 纹理/形状 (256ch)", "layer1", 256),
        ("Layer2 — 局部特征 (512ch)", "layer2", 512),
        ("Layer3 — 花瓣/花蕊 (1024ch)", "layer3", 1024),
        ("Layer4 — 高级语义 (2048ch)", "layer4", 2048),
    ]

    result = []
    for name, key, total_ch in layer_info:
        fm = features[key].squeeze(0).numpy()
        num_show = min(8, total_ch)
        step = max(1, total_ch // num_show)
        cols, rows = 4, 2

        fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
        axes = axes.flatten()
        for i in range(num_show):
            fmap = fm[i * step]
            f_min, f_max = fmap.min(), fmap.max()
            if f_max > f_min:
                fmap = (fmap - f_min) / (f_max - f_min)
            axes[i].imshow(fmap, cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"通道 {i*step+1}", fontsize=8)
        for i in range(num_show, len(axes)):
            axes[i].axis("off")
        fig.suptitle(name, fontsize=13, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        result.append({
            "name": name,
            "image": f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}",
        })
    return result
