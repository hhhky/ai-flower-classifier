# -*- coding: utf-8 -*-
"""
CNN 特征图可视化模块
=====================
功能：提取并可视化 CNN 各层的特征图，展示 AI "看到了什么"
这是"含金量"的核心体现 —— 让黑盒变白盒

技术原理：
  CNN 的每一层卷积都会产生多个特征图（Feature Map），
  每张特征图对应一种"模式检测器" ——
  低层检测边缘/颜色，高层检测形状/语义。

  通过可视化特征图，可以：
  1. 理解 AI 的"视觉认知过程"
  2. 验证模型是否学到了有意义的特征
  3. 为模型优化提供依据（模型可解释性）
"""

import os
import sys
import io
from pathlib import Path

# Windows GBK 终端 UTF-8 输出支持
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import vision, transforms

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "static" / "results"
IMAGE_SIZE = 224

# 复用 train.py 的模型定义
from train import FlowerCNN, CLASS_NAMES, CLASS_NAMES_CN


class FeatureExtractor(nn.Cell):
    """
    特征提取器 —— 从指定层截取中间输出。
    这是"模型可解释性"的技术实现。
    """

    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}

        # 注册 hook，在前向传播时捕获各层输出
        # MindSpore 2.x 中通过自定义 construct 来实现
        # 这里我们直接暴露各层

    def construct(self, x):
        """前向传播并记录各层输出"""
        features = {}

        # Layer 1
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu1(x)
        features["conv1_relu"] = x
        x = self.model.pool1(x)

        # Layer 2
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu2(x)
        features["conv2_relu"] = x
        x = self.model.pool2(x)

        # Layer 3
        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu3(x)
        features["conv3_relu"] = x
        x = self.model.pool3(x)

        # Layer 4
        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu4(x)
        features["conv4_relu"] = x

        return features


def load_model():
    """加载训练好的模型"""
    net = FlowerCNN()
    model_path = MODEL_DIR / "flower_cnn_final.ckpt"
    if model_path.exists():
        param_dict = ms.load_checkpoint(str(model_path))
        ms.load_param_into_net(net, param_dict)
        net.set_train(False)
        print(f"  ✅ 模型已加载: {model_path}")
    else:
        print(f"  ❌ 未找到模型文件 {model_path}，请先运行 train.py")
    return net


def preprocess_image(image_path):
    """预处理输入图片"""
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # (H, W, C) uint8

    mean_rgb = [m * 255 for m in [0.485, 0.456, 0.406]]
    std_rgb = [s * 255 for s in [0.229, 0.224, 0.225]]

    transform = transforms.Compose([
        vision.Resize(256),
        vision.CenterCrop(IMAGE_SIZE),
        vision.Normalize(mean=mean_rgb, std=std_rgb),  # 标准化 (HWC), uint8→float32
        vision.HWC2CHW(),                              # → (C,H,W) float32
    ])

    img_tensor = transform(img_np)
    return ms.Tensor(img_tensor).expand_dims(0)  # (1, 3, 224, 224)


def visualize_feature_maps(net, image_path, save_dir=None):
    """
    可视化 CNN 各层的特征图。

    对每一层卷积的输出：
    1. 取前 16 个通道（或全部，取 min）
    2. 绘制为网格图
    3. 每个子图是一个通道的特征图 → 展示 AI "关注"了图片的哪些特征
    """
    if save_dir is None:
        save_dir = RESULT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  🔬 开始特征图可视化...")

    # 预处理
    input_tensor = preprocess_image(image_path)

    # 前向传播到各层
    # Conv1
    x = net.conv1(input_tensor)
    x = net.bn1(x)
    x = net.relu1(x)
    conv1_out = x

    # Conv2
    x = net.pool1(x)
    x = net.conv2(x)
    x = net.bn2(x)
    x = net.relu2(x)
    conv2_out = x

    # Conv3
    x = net.pool2(x)
    x = net.conv3(x)
    x = net.bn3(x)
    x = net.relu3(x)
    conv3_out = x

    # Conv4
    x = net.pool3(x)
    x = net.conv4(x)
    x = net.bn4(x)
    x = net.relu4(x)
    conv4_out = x

    # 收集各层输出
    layer_outputs = [
        ("Conv1 (边缘/颜色检测)", conv1_out, 32),
        ("Conv2 (纹理/形状检测)", conv2_out, 64),
        ("Conv3 (花瓣/局部特征)", conv3_out, 128),
        ("Conv4 (花朵/语义特征)", conv4_out, 256),
    ]

    saved_paths = []

    for layer_name, output, num_channels in layer_outputs:
        # 取特征图数据: (1, C, H, W) → (C, H, W)
        feature_maps = output.asnumpy()[0]

        # 显示前 16 个通道
        num_show = min(16, num_channels)
        cols = 4
        rows = (num_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if isinstance(axes, plt.Axes) else axes.flatten()

        for i in range(num_show):
            fm = feature_maps[i]
            # 归一化到 [0, 1]
            fm_min, fm_max = fm.min(), fm.max()
            if fm_max > fm_min:
                fm = (fm - fm_min) / (fm_max - fm_min)
            axes[i].imshow(fm, cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"Ch.{i+1}", fontsize=8)

        # 隐藏多余的子图
        for i in range(num_show, len(axes)):
            axes[i].axis("off")

        fig.suptitle(f"CNN 特征图可视化 —— {layer_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        safe_name = layer_name.split("(")[0].strip().lower().replace(" ", "_")
        save_path = save_dir / f"feature_map_{safe_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(str(save_path))
        print(f"    📸 {layer_name}: {save_path}")

    print(f"\n  ✅ 特征图可视化完成！共生成 {len(saved_paths)} 张图片")
    return saved_paths


def generate_all_visualizations(image_path=None):
    """
    生成完整的可视化报告：
    1. 原始图片
    2. 各层特征图
    3. 合成一张汇总图

    如果未指定图片路径，使用 data 目录下第一张图片。
    """
    # 加载模型
    net = load_model()

    # 找一张测试图片
    if image_path is None:
        data_dir = PROJECT_DIR / "data" / "flowers"
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        image_path = str(img_file)
                        break
            if image_path:
                break

    if image_path is None or not Path(image_path).exists():
        print("  ❌ 未找到测试图片")
        return []

    print(f"\n  🖼️  测试图片: {image_path}")
    return visualize_feature_maps(net, image_path)


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       🔬 CNN 特征图可视化 —— 看 AI 如何「看」世界         ║")
    print("╚══════════════════════════════════════════════════════╝")
    generate_all_visualizations()
