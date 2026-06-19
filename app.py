# -*- coding: utf-8 -*-
"""
Hugging Face Spaces — Gradio 花卉识别 (PyTorch 版)
===================================================
轻量部署，无需 MindSpore，HF 开箱即用
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import gradio as gr
import numpy as np
from PIL import Image

from model_pt import get_model, predict_image, extract_feature_maps, CLASS_NAMES_CN

_net = None

def get_net():
    global _net
    if _net is None:
        _net = get_model()
    return _net


def classify_image(image):
    if image is None:
        return None, "请上传一张花卉图片"

    tmp_path = Path("/tmp/upload.jpg")
    image.save(str(tmp_path))

    net = get_net()
    results = predict_image(net, str(tmp_path))

    text = "## 🌸 识别结果\n\n"
    for i, r in enumerate(results):
        emoji = ["🥇", "🥈", "🥉"][i]
        text += f"{emoji} **{r['class_name_cn']}** ({r['class_name']})\n"
        text += f"   置信度: `{r['confidence']:.1f}%`\n\n"

    text += "---\n*ResNet-50 迁移学习 | 测试准确率 86.24%*"

    try:
        fmaps = extract_feature_maps(net, str(tmp_path))
        gallery = [(f["image"], f["name"]) for f in fmaps]
    except Exception:
        gallery = []

    return text, gallery


with gr.Blocks(title="🌸 AI 花卉识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌸 AI 花卉智能识别系统

    **ResNet-50 迁移学习** | 测试准确率 **86.24%** | PyTorch 推理

    上传花卉照片，AI 识别它是雏菊、蒲公英、玫瑰、向日葵还是郁金香。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="📷 上传花卉照片")
            submit_btn = gr.Button("🔍 开始识别", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_text = gr.Markdown(label="📊 识别结果")
            feature_gallery = gr.Gallery(
                label="🧠 AI 特征图可视化",
                columns=1, rows=3, height=400,
            )

    submit_btn.click(
        fn=classify_image,
        inputs=[input_img],
        outputs=[result_text, feature_gallery],
    )

    gr.Markdown("""
    ---
    ### 关于这个 AI
    - **模型**：ResNet-50（ImageNet 预训练 → 花卉微调）
    - **准确率**：86.24%（5类花卉）
    - **框架**：PyTorch + torchvision
    - 源代码：[GitHub](https://github.com/hhhky/ai-flower-classifier)
    """)


if __name__ == "__main__":
    print("🔄 加载 PyTorch ResNet-50 模型...")
    get_net()
    print("✅ 模型就绪")
    demo.launch(server_name="0.0.0.0", server_port=7860)
