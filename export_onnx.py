# -*- coding: utf-8 -*-
"""将 MindSpore ResNet-50 模型导出为 ONNX 格式"""
import sys
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore import Tensor

PROJECT_DIR = Path(__file__).parent
CKPT_PATH = PROJECT_DIR / "models" / "flower_cnn_final.ckpt"
ONNX_PATH = PROJECT_DIR / "models" / "flower_model.onnx"

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

# 导入模型结构
from model import TransferNet

print("加载模型...")
net = TransferNet()
param_dict = ms.load_checkpoint(str(CKPT_PATH))
ms.load_param_into_net(net, param_dict)
net.set_train(False)
print("模型已加载")

# 导出 ONNX
dummy_input = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
print("导出 ONNX...")
ms.export(net, dummy_input, file_name=str(ONNX_PATH), file_format="ONNX")
print(f"ONNX 模型已保存: {ONNX_PATH}")
print(f"大小: {ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB")
