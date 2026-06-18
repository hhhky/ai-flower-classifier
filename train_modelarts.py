# -*- coding: utf-8 -*-
"""
ModelArts GPU 训练版本 — 花卉分类 CNN
运行时会自动下载 TensorFlow Flower Photos 数据集
"""
import os, sys, io, tarfile, urllib.request, time, json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体（ModelArts 节点上可能没有，回退到英文）
try:
    for fn in ["SimHei", "Microsoft YaHei"]:
        try:
            fm.fontManager.addfont(f"C:/Windows/Fonts/{fn}.ttf")
            plt.rcParams["font.sans-serif"] = [fn] + plt.rcParams["font.sans-serif"]
            break
        except Exception:
            continue
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import ImageFolderDataset, transforms, vision
from mindspore.train import Model, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.nn import Accuracy
from sklearn.metrics import confusion_matrix, classification_report

# ════════════════════════════════════════════════════
# 配置 — GPU 优化版
# ════════════════════════════════════════════════════

# ModelArts 路径
CACHE_DIR = Path("/cache")
DATA_DIR = CACHE_DIR / "flower_photos"
OUTPUT_DIR = CACHE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU 超参数（比 CPU 更大更快）
BATCH_SIZE = 64
IMAGE_SIZE = 224
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 5

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CLASS_NAMES_CN = ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]

# GPU 模式
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


# ════════════════════════════════════════════════════
# 数据集自动下载
# ════════════════════════════════════════════════════

def download_dataset():
    """若数据不存在则从 TensorFlow 官方 URL 下载（~220MB）"""
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"✅ 数据已存在: {DATA_DIR}")
        return

    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    tgz_path = CACHE_DIR / "flower_photos.tgz"

    print(f"⬇️  下载数据集 (~220MB)...")
    print(f"   URL: {url}")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, str(tgz_path))
            print("   下载完成")
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"   第 {attempt+1} 次失败: {e}，重试...")
            time.sleep(5)

    print("📦 解压中...")
    with tarfile.open(str(tgz_path), "r:gz") as tar:
        tar.extractall(path=str(CACHE_DIR), filter='data')
    tgz_path.unlink()  # 删掉压缩包节省空间
    print(f"✅ 数据就绪: {DATA_DIR}")


# ════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════

def create_dataset(data_path, training=True):
    mean_uint8 = [m * 255 for m in [0.485, 0.456, 0.406]]
    std_uint8 = [s * 255 for s in [0.229, 0.224, 0.225]]

    if training:
        transform_list = [
            vision.Decode(to_pil=False),
            vision.RandomResizedCrop(IMAGE_SIZE),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomRotation(degrees=15),
            vision.RandomColorAdjust(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            vision.Normalize(mean=mean_uint8, std=std_uint8),
            vision.HWC2CHW(),
        ]
    else:
        transform_list = [
            vision.Decode(to_pil=False),
            vision.Resize(256),
            vision.CenterCrop(IMAGE_SIZE),
            vision.Normalize(mean=mean_uint8, std=std_uint8),
            vision.HWC2CHW(),
        ]

    dataset = ImageFolderDataset(
        dataset_dir=str(data_path),
        class_indexing={name: i for i, name in enumerate(CLASS_NAMES)},
        shuffle=training,
        num_parallel_workers=4,
    )
    dataset = dataset.map(operations=transform_list, input_columns=["image"], num_parallel_workers=4)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=training)
    return dataset


# ════════════════════════════════════════════════════
# CNN 模型
# ════════════════════════════════════════════════════

class FlowerCNN(nn.Cell):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
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

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(256 * 14 * 14, 512)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Dense(512, num_classes)

    def construct(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ════════════════════════════════════════════════════
# 训练
# ════════════════════════════════════════════════════

def train():
    download_dataset()

    print("\n" + "=" * 50)
    print("📂 加载数据...")
    train_ds = create_dataset(DATA_DIR, training=True)
    val_ds = create_dataset(DATA_DIR, training=False)
    print(f"   训练集: {train_ds.get_dataset_size()} batches")
    print(f"   验证集: {val_ds.get_dataset_size()} batches")

    net = FlowerCNN(num_classes=NUM_CLASSES)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY)
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer,
                  metrics={"acc": Accuracy()})

    # 回调
    ckpt_config = CheckpointConfig(save_checkpoint_steps=val_ds.get_dataset_size(),
                                   keep_checkpoint_max=3)
    ckpt_cb = ModelCheckpoint(prefix="flower_cnn", directory=str(OUTPUT_DIR),
                              config=ckpt_config)
    loss_cb = LossMonitor(per_print_times=10)

    print("\n" + "=" * 50)
    print("🚀 GPU 训练开始")
    print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"   设备: GPU | 模式: GRAPH_MODE")
    print("=" * 50)

    model.train(NUM_EPOCHS, train_ds, callbacks=[loss_cb, ckpt_cb],
                dataset_sink_mode=True)

    # 保存最终模型
    final_path = OUTPUT_DIR / "flower_cnn_final.ckpt"
    ms.save_checkpoint(net, str(final_path))
    print(f"\n✅ 模型已保存: {final_path}")

    # 评估
    print("\n📊 评估中...")
    net.set_train(False)
    all_preds, all_labels = [], []
    for data in val_ds.create_dict_iterator():
        logits = net(data["image"])
        preds = ops.argmax(logits, dim=1).asnumpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(data["label"].asnumpy().tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🎯 最终准确率: {acc * 100:.2f}%")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="YlOrRd")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES_CN)
    ax.set_yticklabels(CLASS_NAMES_CN)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Acc: {acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "confusion_matrix.png"), dpi=150)
    plt.close()

    # 分类报告
    report = classification_report(
        all_labels, all_preds,
        target_names=[f"{cn}({en})" for cn, en in zip(CLASS_NAMES_CN, CLASS_NAMES)],
        digits=3)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    # model_stats.json
    stats = {"accuracy": float(acc), "confusion_matrix": cm.tolist(),
             "class_names": CLASS_NAMES, "class_names_cn": CLASS_NAMES_CN}
    (OUTPUT_DIR / "model_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"\n✅ 训练完成！所有输出保存在: {OUTPUT_DIR}")
    print(f"   模型: {OUTPUT_DIR / 'flower_cnn_final.ckpt'}")
    print(f"   混淆矩阵: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"   分类报告: {OUTPUT_DIR / 'classification_report.txt'}")


if __name__ == "__main__":
    train()
