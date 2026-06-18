# -*- coding: utf-8 -*-
"""
Flask Web 应用 —— AI 图像智能识别系统
========================================
功能：提供 Web 界面，用户上传图片 → AI 推理 → 展示结果
这是"作品"的外壳 —— 让 AI 模型变成一个可交互的产品

启动方式：
    python app.py
    浏览器访问 http://localhost:5000
"""

import os
import sys
import io

# Windows GBK 终端 UTF-8 输出支持
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import base64
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore import ops, context
from mindspore.dataset import vision, transforms

# 设置 MindSpore 环境
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

# 导入模型模块（ResNet-50 迁移学习，86.24% 准确率）
from model import (
    TransferNet, load_trained_model, predict_image, extract_feature_maps,
    CLASS_NAMES, CLASS_NAMES_CN, IMAGE_SIZE, NUM_CLASSES,
)

# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 最大上传 16MB

PROJECT_DIR = Path(__file__).parent
UPLOAD_DIR = PROJECT_DIR / "static" / "uploads"
RESULT_DIR = PROJECT_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 全局模型实例（延迟加载）
_model = None


def get_model():
    """获取或加载模型（单例模式）"""
    global _model
    if _model is None:
        _model = load_trained_model()
    return _model


# ============================================================
# 特征图生成（Web 端调用）
# ============================================================

def generate_feature_maps_for_web(image_path):
    """为 Web 端生成特征图（Base64 返回），展示 ResNet-50 各层"看到了什么"."""
    net = get_model()
    return extract_feature_maps(net, image_path)


# ============================================================
# 路由
# ============================================================

@app.route("/")
def index():
    """首页"""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    AI 预测 API
    接收上传的图片 → 模型推理 → 返回分类结果
    """
    if "image" not in request.files:
        return jsonify({"error": "请上传图片文件"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 保存上传文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}.jpg"
    save_path = UPLOAD_DIR / filename
    file.save(str(save_path))

    try:
        # 模型推理
        net = get_model()
        net.set_train(False)

        results = predict_image(net, str(save_path))

        # 生成特征图预览
        try:
            feature_maps = generate_feature_maps_for_web(str(save_path))
        except Exception as e:
            print(f"  特征图生成失败: {e}")
            feature_maps = []

        # 读取上传图片的 base64 用于前端显示
        with open(save_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_b64}",
            "predictions": results,
            "feature_maps": feature_maps,
            "top_result": results[0] if results else None,
        })

    except Exception as e:
        return jsonify({"error": f"推理失败: {str(e)}"}), 500


@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    """获取模型信息"""
    # 读取训练统计
    stats_path = RESULT_DIR / "model_stats.json"
    model_stats = None
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            model_stats = json.load(f)

    return jsonify({
        "model_name": "ResNet-50 (Transfer Learning)",
        "framework": "华为 MindSpore + mindcv",
        "architecture": "ResNet-50 backbone + 5类分类头 (ImageNet预训练 → 微调)",
        "input_size": f"{IMAGE_SIZE}×{IMAGE_SIZE}",
        "num_classes": NUM_CLASSES,
        "classes": [
            {"en": en, "cn": cn} for en, cn in zip(CLASS_NAMES, CLASS_NAMES_CN)
        ],
        "stats": model_stats,
        "accuracy": "86.24%",
        "optimizer": "Adam (lr=0.001→0.0001 两阶段)",
        "loss": "交叉熵 (CrossEntropy)",
        "augmentation": "随机裁剪、水平翻转、旋转、颜色抖动",
    })


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║    🌸 AI 图像智能识别系统 —— Web 服务                   ║")
    print("║    基于 MindSpore + Flask                            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  📡 服务地址: http://localhost:5000")
    print(f"  🖼️  上传目录: {UPLOAD_DIR}")
    print(f"  📊 结果目录: {RESULT_DIR}")
    print()
    print("  💡 提示：")
    print("     1. 打开浏览器访问上述地址")
    print("     2. 上传一张花卉照片")
    print("     3. 查看 AI 识别结果和特征图")
    print()

    # 检查模型是否存在
    model_path = PROJECT_DIR / "models" / "flower_cnn_final.ckpt"
    if not model_path.exists():
        print("  ⚠️  未找到训练好的模型！")
        print("     请先运行: python train.py")
        print()

    app.run(host="0.0.0.0", port=5000, debug=True)
