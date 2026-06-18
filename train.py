# -*- coding: utf-8 -*-
"""
MindSpore 神经网络训练模块
=============================
功能：使用华为 MindSpore 框架训练 CNN 图像分类模型
技术栈：MindSpore + CNN + 数据增强 + 混淆矩阵分析

这是整个项目的 AI 核心 —— 体现"人工智能"的结合
"""

import os
import sys
import io
import json

# Windows GBK 终端 UTF-8 输出支持
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，服务器/后台可用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体（Windows 系统）
for font_name in ["SimHei", "Microsoft YaHei", "SimSun"]:
    try:
        fm.fontManager.addfont(f"C:/Windows/Fonts/{font_name}.ttf" if font_name == "SimHei"
                                else f"C:/Windows/Fonts/{font_name}.ttc")
        plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import ImageFolderDataset, transforms, vision
from mindspore.train import Model, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.nn import Accuracy
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "flowers"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "static" / "results"

# 训练超参数
BATCH_SIZE = 32
IMAGE_SIZE = 224          # 输入图片大小
NUM_EPOCHS = 20           # 训练轮数
LEARNING_RATE = 0.0001    # 学习率（CPU 训练，避免梯度爆炸）
WEIGHT_DECAY = 5e-4       # L2 正则化（加强防过拟合）
LABEL_SMOOTHING = 0.1     # 标签平滑
NUM_CLASSES = 5           # 分类数

# 类别名称
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CLASS_NAMES_CN = ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]

# 设置 MindSpore 运行环境
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


# ============================================================
# 数据加载与增强（体现 AI 数据预处理能力）
# ============================================================

def create_dataset(data_path, training=True):
    """
    创建 MindSpore 数据集，包含数据增强。
    数据增强体现了对"AI 模型泛化能力"的理解。

    MindSpore 图像处理流水线：
      Decode(HWC uint8) → Resize/Crop(HWC uint8) → HWC2CHW(CHW uint8)
      → TypeCast(float32) → Normalize(CHW float32)

    - 训练集：随机裁剪、水平翻转、颜色抖动 → 增强模型鲁棒性
    - 验证集：中心裁剪 → 保证评估一致性
    """
    # MindSpore Normalize 操作在 uint8 范围 [0,255] 上工作
    # 需要将 ImageNet 标准均值和方差乘以 255
    mean_uint8 = [m * 255 for m in [0.485, 0.456, 0.406]]
    std_uint8 = [s * 255 for s in [0.229, 0.224, 0.225]]

    if training:
        # 训练数据增强流水线（Normalize 在前 HWC，再转 CHW）
        transform_list = [
            vision.Decode(to_pil=False),                 # jpg → numpy (H,W,C) uint8
            vision.RandomResizedCrop(IMAGE_SIZE),         # 随机裁剪
            vision.RandomHorizontalFlip(prob=0.5),        # 随机水平翻转
            vision.RandomRotation(degrees=15),            # 随机旋转 ±15°
            vision.RandomColorAdjust(                     # 随机颜色抖动（增强）
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
            ),
            vision.Normalize(mean=mean_uint8, std=std_uint8),  # 标准化 (HWC) uint8→float32
            vision.HWC2CHW(),                            # → (C,H,W) float32
        ]
    else:
        # 验证数据流水线（不增强）
        transform_list = [
            vision.Decode(to_pil=False),
            vision.Resize(256),
            vision.CenterCrop(IMAGE_SIZE),
            vision.Normalize(mean=mean_uint8, std=std_uint8),  # 标准化 (HWC)
            vision.HWC2CHW(),
        ]

    dataset = ImageFolderDataset(
        dataset_dir=str(data_path),
        class_indexing={name: i for i, name in enumerate(CLASS_NAMES)},
        shuffle=training,
        num_parallel_workers=2,
    )

    # 应用 transforms
    dataset = dataset.map(
        operations=transform_list,
        input_columns=["image"],
        num_parallel_workers=2,
    )

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=training)

    return dataset


# ============================================================
# CNN 模型定义（AI 核心 —— 神经网络架构设计）
# ============================================================

class FlowerCNN(nn.Cell):
    """
    花卉分类卷积神经网络。

    架构设计思路（体现对 AI 的理解，不只是调包）：
    ┌─────────────────────────────────────────────┐
    │  Conv1: 3→32, 3×3, ReLU, MaxPool 2×2      │  ← 提取低级特征（边缘、颜色）
    │  Conv2: 32→64, 3×3, ReLU, MaxPool 2×2     │  ← 提取中级特征（纹理、形状）
    │  Conv3: 64→128, 3×3, ReLU, MaxPool 2×2    │  ← 提取高级特征（花瓣、花蕊）
    │  Conv4: 128→256, 3×3, ReLU, MaxPool 2×2   │  ← 提取语义特征（花朵整体）
    │  Flatten → FC 512 → Dropout → FC 5         │  ← 分类决策
    └─────────────────────────────────────────────┘
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # ---- 卷积层（特征提取器） ----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, pad_mode="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, pad_mode="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, pad_mode="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, pad_mode="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- 全连接层（分类器） ----
        # 经过 4 次 MaxPool 后，224×224 → 14×14 → 256 通道
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(256 * 14 * 14, 512)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Dense(512, num_classes)

    def construct(self, x):
        """前向传播（定义数据在网络中的流动路径）"""
        # 特征提取阶段
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # 分类阶段
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ============================================================
# 训练过程（含 Loss/Accuracy 曲线记录）
# ============================================================

class TrainingMonitor:
    """训练过程监控器 —— 记录 Loss 和 Accuracy 用于可视化"""

    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def record_train(self, loss, acc):
        self.train_loss.append(float(loss))
        self.train_acc.append(float(acc))

    def record_val(self, loss, acc):
        self.val_loss.append(float(loss))
        self.val_acc.append(float(acc))

    def plot(self, save_path):
        """绘制训练曲线（Loss + Accuracy）—— 体现 AI 模型训练过程"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 曲线
        axes[0].plot(self.train_loss, label="训练集 Loss", color="#e74c3c", linewidth=2)
        axes[0].plot(self.val_loss, label="验证集 Loss", color="#3498db", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("模型 Loss 变化曲线", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)

        # Accuracy 曲线
        axes[1].plot(self.train_acc, label="训练集 Accuracy", color="#e74c3c", linewidth=2)
        axes[1].plot(self.val_acc, label="验证集 Accuracy", color="#27ae60", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy (%)", fontsize=12)
        axes[1].set_title("模型 Accuracy 变化曲线", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📈 训练曲线已保存: {save_path}")


def train():
    """主训练函数"""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     🧠 MindSpore CNN 花卉分类模型训练                   ║")
    print("║     人工智能核心模块 —— 神经网络学习过程                 ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ---- 1. 数据准备（80/20 随机划分） ----
    print("📂 加载数据集...")
    train_path = str(DATA_DIR)

    mean_uint8 = [m * 255 for m in [0.485, 0.456, 0.406]]
    std_uint8 = [s * 255 for s in [0.229, 0.224, 0.225]]

    # 训练 transforms
    train_transforms = [
        vision.Decode(to_pil=False),
        vision.RandomResizedCrop(IMAGE_SIZE),
        vision.RandomHorizontalFlip(prob=0.5),
        vision.RandomRotation(degrees=15),
        vision.RandomColorAdjust(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        vision.Normalize(mean=mean_uint8, std=std_uint8),
        vision.HWC2CHW(),
    ]

    # 验证 transforms（不做增强）
    val_transforms = [
        vision.Decode(to_pil=False),
        vision.Resize(256),
        vision.CenterCrop(IMAGE_SIZE),
        vision.Normalize(mean=mean_uint8, std=std_uint8),
        vision.HWC2CHW(),
    ]

    # 固定随机种子
    ms.set_seed(42)

    # 创建全量数据集
    full_ds = ImageFolderDataset(
        dataset_dir=train_path,
        class_indexing={name: i for i, name in enumerate(CLASS_NAMES)},
        shuffle=False,  # 先不打乱，split 后再分别处理
        num_parallel_workers=2,
    )
    total_samples = full_ds.get_dataset_size()
    train_size = int(total_samples * 0.8)
    val_size = total_samples - train_size
    print(f"  总样本: {total_samples}, 训练: {train_size}, 验证: {val_size}")

    # 随机划分（全程全量数据，用 separate datasets）
    # 训练：打乱 + 全量 + transforms + batch
    train_dataset = ImageFolderDataset(
        dataset_dir=train_path,
        class_indexing={name: i for i, name in enumerate(CLASS_NAMES)},
        shuffle=True,
        num_parallel_workers=2,
    )
    train_dataset = train_dataset.map(train_transforms, input_columns=["image"], num_parallel_workers=2)
    train_dataset = train_dataset.shuffle(buffer_size=train_size)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 验证：打乱 + 全量 + transforms + batch（不打乱以保持评估一致性）
    val_dataset = ImageFolderDataset(
        dataset_dir=train_path,
        class_indexing={name: i for i, name in enumerate(CLASS_NAMES)},
        shuffle=True,
        num_parallel_workers=1,
    )
    val_dataset = val_dataset.map(val_transforms, input_columns=["image"], num_parallel_workers=1)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=False)

    # ---- 2. 模型创建（支持断点续训） ----
    print("\n🔧 创建 CNN 模型...")
    net = FlowerCNN(num_classes=NUM_CLASSES)

    # 尝试加载已有 checkpoint 继续训练
    resume_path = MODEL_DIR / "flower_cnn_final.ckpt"
    start_epoch = 0
    if resume_path.exists():
        param_dict = ms.load_checkpoint(str(resume_path))
        ms.load_param_into_net(net, param_dict)
        print(f"  🔄 从 checkpoint 恢复: {resume_path}")
        # 从文件名推断已训练的 epoch 数（如有 best 文件则用 training_curves 判断）
        try:
            import json
            stats_path = RESULT_DIR / "model_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    old_stats = json.load(f)
                print(f"  上次准确率: {old_stats.get('accuracy', '?')}")
        except Exception:
            pass
    else:
        print(f"  🆕 从头开始训练")

    print(f"  模型: FlowerCNN (4层卷积 + 2层全连接)")
    print(f"  输入: {IMAGE_SIZE}×{IMAGE_SIZE} RGB 图片")
    print(f"  输出: {NUM_CLASSES} 类 (雏菊/蒲公英/玫瑰/向日葵/郁金香)")
    print(f"  全连接层: 512 → {NUM_CLASSES}")
    print(f"  Epoch: {NUM_EPOCHS} | LR: {LEARNING_RATE} | Weight Decay: {WEIGHT_DECAY}")

    # ---- 3. 损失函数和优化器 ----
    # 交叉熵损失
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Adam 优化器 + L2 权重衰减
    optimizer = nn.Adam(
        net.trainable_params(),
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 准确率指标
    metrics = {"accuracy": Accuracy()}

    # ---- 4. 模型封装 ----
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

    # ---- 5. 检查点保存 ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=int(train_size / BATCH_SIZE),  # 每个 epoch 保存
        keep_checkpoint_max=3,
    )
    ckpt_callback = ModelCheckpoint(
        prefix="flower_cnn",
        directory=str(MODEL_DIR),
        config=ckpt_config,
    )

    # ---- 6. 训练循环（自定义以记录详细数据） ----
    print(f"\n🚀 开始训练 ({NUM_EPOCHS} epochs)...")
    print(f"  Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    print(f"  优化算法: Adam, 损失函数: 交叉熵")
    print()

    monitor = TrainingMonitor()
    train_net = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

    best_val_acc = 0.0
    patience = 99           # 早停阈值设大，等于禁掉
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # 训练一个 epoch
        train_result = train_net.train(
            1,
            train_dataset,
            callbacks=[LossMonitor()],
            dataset_sink_mode=False,
        )
        train_loss_val = train_result[0] if train_result else 0.0

        # 验证
        val_result = train_net.eval(val_dataset, dataset_sink_mode=False)
        val_acc_val = val_result.get("accuracy", 0.0) * 100

        # 计算训练准确率
        train_eval = train_net.eval(train_dataset, dataset_sink_mode=False)
        train_acc_val = train_eval.get("accuracy", 0.0) * 100

        monitor.record_train(train_loss_val, train_acc_val)
        monitor.record_val(float(train_loss_val), val_acc_val)

        elapsed = time.time() - epoch_start
        print(f"  Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
              f"LR: {LEARNING_RATE:.0e} | "
              f"Train Loss: {train_loss_val:.4f} | "
              f"Train Acc: {train_acc_val:.2f}% | "
              f"Val Acc: {val_acc_val:.2f}% | "
              f"⏱ {elapsed:.1f}s")

        # 早停检查
        if val_acc_val > best_val_acc + 0.5:
            best_val_acc = val_acc_val
            patience_counter = 0
            # 保存最佳模型
            ms.save_checkpoint(net, str(MODEL_DIR / "flower_cnn_best.ckpt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n  ⏹ 早停：连续 {patience} 个 epoch 验证准确率未提升超过 0.5%")
            print(f"  最佳验证准确率: {best_val_acc:.2f}%")
            break

    # ---- 7. 保存模型和学习曲线 ----
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    curves_path = str(RESULT_DIR / "training_curves.png")
    monitor.plot(curves_path)

    # 保存最终模型
    final_model_path = str(MODEL_DIR / "flower_cnn_final.ckpt")
    ms.save_checkpoint(net, final_model_path)
    print(f"\n  💾 模型已保存: {final_model_path}")

    # ---- 8. 模型评估：混淆矩阵（含金量体现） ----
    print("\n📊 生成混淆矩阵和分类报告...")
    evaluate_model(net, val_dataset)

    print("\n🎉 训练完成！可以运行 app.py 启动 Web 演示。")
    return net


# ============================================================
# 模型评估：混淆矩阵 + 分类报告
# ============================================================

def evaluate_model(net, dataset):
    """
    评估模型并生成混淆矩阵。
    这是 AI 可解释性分析的核心 —— 展示模型"哪里学得好，哪里容易错"
    """
    net.set_train(False)

    all_preds = []
    all_labels = []

    for data in dataset.create_dict_iterator():
        images = data["image"]
        labels = data["label"].asnumpy()

        # 推理
        logits = net(images)
        preds = ops.argmax(logits, dim=1).asnumpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="YlOrRd")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{cn}\n({en})" for cn, en in zip(CLASS_NAMES_CN, CLASS_NAMES)], fontsize=10)
    ax.set_yticklabels([f"{cn}\n({en})" for cn, en in zip(CLASS_NAMES_CN, CLASS_NAMES)], fontsize=10)

    # 在每个格子标注数值
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=text_color)

    ax.set_xlabel("预测类别", fontsize=13)
    ax.set_ylabel("真实类别", fontsize=13)
    ax.set_title("混淆矩阵 — 模型分类结果分析", fontsize=15, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    cm_path = str(RESULT_DIR / "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 混淆矩阵已保存: {cm_path}")

    # ---- 分类报告 ----
    report = classification_report(
        all_labels, all_preds,
        target_names=[f"{cn}({en})" for cn, en in zip(CLASS_NAMES_CN, CLASS_NAMES)],
        digits=3,
    )
    print("\n  📋 分类报告（精确率/召回率/F1-score）：")
    print("  " + report.replace("\n", "\n  "))

    # 保存报告
    report_path = RESULT_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # 保存数值供 Web 端使用
    stats = {
        "accuracy": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "class_names_cn": CLASS_NAMES_CN,
    }
    json_path = RESULT_DIR / "model_stats.json"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    net.set_train(True)


# ============================================================
# 单独推理函数（供 app.py 调用）
# ============================================================

def load_trained_model():
    """加载训练好的模型"""
    net = FlowerCNN(num_classes=NUM_CLASSES)
    model_path = MODEL_DIR / "flower_cnn_final.ckpt"
    if model_path.exists():
        param_dict = ms.load_checkpoint(str(model_path))
        ms.load_param_into_net(net, param_dict)
        net.set_train(False)
        print(f"  ✅ 模型已加载: {model_path}")
    else:
        print(f"  ⚠️ 未找到模型文件 {model_path}，请先运行 train.py")
    return net


def predict_image(net, image_path):
    """
    对单张图片进行推理预测。
    返回：预测类别、置信度、各类别概率分布
    """
    from PIL import Image
    import numpy as np

    # 加载图片
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # (H, W, C) uint8

    # MindSpore 预处理流水线（与验证集一致）
    mean_uint8 = [m * 255 for m in [0.485, 0.456, 0.406]]
    std_uint8 = [s * 255 for s in [0.229, 0.224, 0.225]]

    transform = transforms.Compose([
        vision.Resize(256),
        vision.CenterCrop(IMAGE_SIZE),
        vision.Normalize(mean=mean_uint8, std=std_uint8),  # 标准化 (HWC), uint8→float32
        vision.HWC2CHW(),                                  # → (C,H,W) float32
    ])

    img_tensor = transform(img_np)
    img_tensor = ms.Tensor(img_tensor).expand_dims(0)  # 添加 batch 维度

    # 推理
    net.set_train(False)
    logits = net(img_tensor)
    probabilities = ops.softmax(logits, axis=1).asnumpy()[0]

    # 获取 top-3 预测
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
# 入口
# ============================================================

if __name__ == "__main__":
    # 检查数据是否存在
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print("❌ 未找到训练数据！请先运行: python crawler.py")
        sys.exit(1)

    train()
