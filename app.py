# -*- coding: utf-8 -*-
"""
Hugging Face Spaces 部署版本 — Gradio 界面
============================================
与 Flask 版本共用 model.py，包装为 Gradio 接口。
Hugging Face 自动从 GitHub 同步，免费部署，16GB RAM。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import gradio as gr
import numpy as np
from PIL import Image

from model import load_trained_model, predict_image, extract_feature_maps, CLASS_NAMES_CN

# 全局模型
_net = None

def get_net():
    global _net
    if _net is None:
        _net = load_trained_model()
    return _net


def classify_image(image):
    """接收 PIL Image，返回预测结果和特征图"""
    if image is None:
        return None, "请上传一张花卉图片"

    # 保存临时文件
    tmp_path = Path("/tmp/upload.jpg")
    image.save(str(tmp_path))

    net = get_net()

    # 预测
    results = predict_image(net, str(tmp_path))

    # 构建文字结果
    text_output = "## 🌸 识别结果\n\n"
    for i, r in enumerate(results):
        emoji = ["🥇", "🥈", "🥉"][i]
        text_output += f"{emoji} **{r['class_name_cn']}** ({r['class_name']})\n"
        text_output += f"   置信度: `{r['confidence']:.1f}%`\n\n"

    text_output += "---\n"
    text_output += "*ResNet-50 迁移学习 | 测试准确率 86.24%*"

    # 特征图
    try:
        fmaps = extract_feature_maps(net, str(tmp_path))
        # 返回第一张特征图作为预览（Layer3 比较有代表性）
        feature_gallery = [(f["image"], f["name"]) for f in fmaps]
    except Exception:
        feature_gallery = []

    return text_output, feature_gallery


# Gradio 界面
with gr.Blocks(title="🌸 AI 花卉识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌸 AI 花卉智能识别系统

    **ResNet-50 迁移学习** | 测试准确率 **86.24%** | 5 类花卉分类

    上传一张花卉照片，AI 会识别它是雏菊、蒲公英、玫瑰、向日葵还是郁金香。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="📷 上传花卉照片")
            submit_btn = gr.Button("🔍 开始识别", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_text = gr.Markdown(label="📊 识别结果")
            feature_gallery = gr.Gallery(
                label="🧠 AI 特征图可视化",
                columns=1, rows=3,
                height=400,
            )

    submit_btn.click(
        fn=classify_image,
        inputs=[input_img],
        outputs=[result_text, feature_gallery],
    )

    gr.Markdown("""
    ---
    ### 关于这个 AI

    - **模型架构**：ResNet-50（ImageNet 预训练） + 5 类分类头
    - **训练方式**：两阶段迁移学习（冻结 backbone → 全网络微调）
    - **框架**：华为 MindSpore + mindcv
    - **测试集准确率**：86.24%

    源代码：[GitHub](https://github.com/hhhky/ai-flower-classifier)
    """)


if __name__ == "__main__":
    # 启动时加载模型
    print("🔄 加载 ResNet-50 模型...")
    get_net()
    print("✅ 模型就绪，启动 Gradio 服务...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
