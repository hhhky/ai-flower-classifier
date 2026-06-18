# -*- coding: utf-8 -*-
"""
迁移学习训练 — mindcv ResNet-50 预训练 + 微调
目标准确率 85%+，CPU 约 10 epochs
"""
import os, sys, json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for fn in ["SimHei", "Microsoft YaHei"]:
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(f"C:/Windows/Fonts/{fn}.ttf")
        plt.rcParams["font.sans-serif"] = [fn] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

import mindspore as ms
from mindspore import nn, ops, load_checkpoint, load_param_into_net
from mindspore.dataset import ImageFolderDataset, transforms, vision
from mindspore.train import Model, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.nn import Accuracy
from sklearn.metrics import confusion_matrix, classification_report

# mindcv 自动下载预训练权重
import mindcv

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "flowers"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "static" / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16          # ResNet-50 大，CPU batch 减小
IMAGE_SIZE = 224
EPOCHS_HEAD = 3          # 阶段一：只训分类头
EPOCHS_FINETUNE = 7      # 阶段二：全网络微调
LR_HEAD = 0.001
LR_FINETUNE = 0.0001
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 5

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CLASS_NAMES_CN = ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]

ms.set_device("CPU")


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
        class_indexing={n: i for i, n in enumerate(CLASS_NAMES)},
        shuffle=training, num_parallel_workers=2,
    )
    dataset = dataset.map(operations=transform_list, input_columns=["image"], num_parallel_workers=2)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=training)
    return dataset


class TransferNet(nn.Cell):
    """ResNet-50 直接 5 类输出，预训练 backbone + 新分类头"""
    def __init__(self):
        super().__init__()
        self.network = mindcv.create_model(
            'resnet50', pretrained=True,
            num_classes=NUM_CLASSES,
            in_channels=3,
        )

    def construct(self, x):
        return self.network(x)


def evaluate(net, dataset):
    net.set_train(False)
    all_preds, all_labels = [], []
    for data in dataset.create_dict_iterator():
        logits = net(data["image"])
        preds = ops.argmax(logits, dim=1).asnumpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(data["label"].asnumpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def train():
    print("=" * 60)
    print("🚀 迁移学习训练 — mindcv ResNet-50 + 微调")
    print("=" * 60)

    print("\n📂 加载数据...")
    train_ds = create_dataset(DATA_DIR, training=True)
    val_ds = create_dataset(DATA_DIR, training=False)
    print(f"   训练 batches: {train_ds.get_dataset_size()}")
    print(f"   验证 batches: {val_ds.get_dataset_size()}")

    # 创建模型（mindcv 自动下载预训练权重）
    print("\n📥 创建 ResNet-50，下载预训练权重...")
    net = TransferNet()

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # ── 阶段一：冻结 backbone，只训分类头 ──
    print("\n" + "-" * 40)
    print(f"📌 阶段一：冻结 Backbone，训练分类头 ({EPOCHS_HEAD} epochs)")
    print("-" * 40)
    # 只训练 classifier 参数（按名字匹配，避免 MindSpore tensor 比较）
    head_names = {p.name for p in net.network.classifier.trainable_params()}
    head_params = []
    for p in net.trainable_params():
        if p.name in head_names:
            p.requires_grad = True
            head_params.append(p)
        else:
            p.requires_grad = False
    print(f"   分类头参数: {len(head_params)}")

    opt_head = nn.Adam(
        params=head_params,
        learning_rate=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )
    model = Model(net, loss_fn=loss_fn, optimizer=opt_head, metrics={"acc": Accuracy()})
    model.train(EPOCHS_HEAD, train_ds,
                callbacks=[LossMonitor(per_print_times=5)],
                dataset_sink_mode=False)

    # ── 阶段二：全网络微调 ──
    # 重建数据集（MindSpore 不允许复用）
    del train_ds, val_ds
    train_ds = create_dataset(DATA_DIR, training=True)
    val_ds = create_dataset(DATA_DIR, training=False)
    print(f"   训练 batches: {train_ds.get_dataset_size()}")

    print("\n" + "-" * 40)
    print(f"📌 阶段二：全网络微调 ({EPOCHS_FINETUNE} epochs, lr={LR_FINETUNE})")
    print("-" * 40)
    for p in net.trainable_params():
        p.requires_grad = True

    ckpt_cb = ModelCheckpoint(
        prefix="flower_transfer", directory=str(MODEL_DIR),
        config=CheckpointConfig(save_checkpoint_steps=val_ds.get_dataset_size(),
                                keep_checkpoint_max=2),
    )
    opt_all = nn.Adam(
        params=net.trainable_params(),
        learning_rate=LR_FINETUNE, weight_decay=WEIGHT_DECAY,
    )
    model = Model(net, loss_fn=loss_fn, optimizer=opt_all, metrics={"acc": Accuracy()})
    model.train(EPOCHS_FINETUNE, train_ds,
                callbacks=[LossMonitor(per_print_times=5), ckpt_cb],
                dataset_sink_mode=False)

    # 保存
    final_path = MODEL_DIR / "flower_cnn_final.ckpt"
    ms.save_checkpoint(net, str(final_path))
    print(f"\n✅ 模型已保存: {final_path}")

    # 评估
    print("\n📊 评估中...")
    preds, labels = evaluate(net, val_ds)
    acc = np.mean(preds == labels)
    print(f"\n🎯 最终准确率: {acc * 100:.2f}%")

    # 分类报告
    report = classification_report(
        labels, preds,
        target_names=[f"{cn}({en})" for cn, en in zip(CLASS_NAMES_CN, CLASS_NAMES)],
        digits=3,
    )
    print("\n" + report)
    (RESULT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="YlOrRd")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES_CN); ax.set_yticklabels(CLASS_NAMES_CN)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Transfer Learning) — Acc: {acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig(str(RESULT_DIR / "confusion_matrix.png"), dpi=150)
    plt.close()

    stats = {"accuracy": float(acc), "confusion_matrix": cm.tolist(),
             "class_names": CLASS_NAMES, "class_names_cn": CLASS_NAMES_CN}
    (RESULT_DIR / "model_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"\n✅ 训练完成！准确率: {acc*100:.1f}%")


if __name__ == "__main__":
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print("❌ 未找到训练数据！")
        sys.exit(1)
    train()
