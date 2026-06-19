# -*- coding: utf-8 -*-
"""将 MindSpore 模型权重转换为 PyTorch 格式"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

PROJECT_DIR = Path(__file__).parent
CKPT_PATH = PROJECT_DIR / "models" / "flower_cnn_final.ckpt"
PT_PATH = PROJECT_DIR / "models" / "flower_model.pth"
NUM_CLASSES = 5

# 1. 用 mindspore 加载权重，提取 numpy 数组
import mindspore as ms
ms.set_device("CPU")

from model import TransferNet
ms_net = TransferNet()
param_dict = ms.load_checkpoint(str(CKPT_PATH))
ms.load_param_into_net(ms_net, param_dict)

# 2. 提取所有权重为 numpy
ms_params = {}
for p in ms_net.get_parameters():
    name = p.name
    val = p.value().asnumpy()
    ms_params[name] = val
print(f"提取了 {len(ms_params)} 个参数")

# 3. 创建 PyTorch ResNet-50
pt_net = resnet50(weights=None, num_classes=NUM_CLASSES)
pt_state = pt_net.state_dict()

# 4. MindSpore → PyTorch 参数名映射
# mindcv resnet50 使用: network.conv1.weight, network.bn1.gamma, etc.
# torchvision resnet50 使用: conv1.weight, bn1.weight, fc.weight, fc.bias

# 映射规则: 去掉 "network." 前缀，gamma→weight, beta→bias, moving_mean→running_mean, moving_variance→running_var
name_map = {}
for ms_name in ms_params:
    pt_name = ms_name.replace("network.", "")
    pt_name = pt_name.replace(".gamma", ".weight")
    pt_name = pt_name.replace(".beta", ".bias")
    pt_name = pt_name.replace(".moving_mean", ".running_mean")
    pt_name = pt_name.replace(".moving_variance", ".running_var")
    # classifier → fc
    pt_name = pt_name.replace("classifier.weight", "fc.weight")
    pt_name = pt_name.replace("classifier.bias", "fc.bias")
    # mindcv 使用 down_sample (下划线), torchvision 使用 downsample
    pt_name = pt_name.replace("down_sample", "downsample")
    name_map[ms_name] = pt_name

# 5. 复制权重
loaded = 0
for ms_name, pt_name in name_map.items():
    if pt_name in pt_state:
        ms_val = ms_params[ms_name]
        pt_val = pt_state[pt_name]
        # MindSpore Conv2d weight: [out_ch, in_ch, H, W] — 和 PyTorch 一样
        # MindSpore BN gamma: [ch] — 一样
        if ms_val.shape == pt_val.shape:
            pt_state[pt_name] = torch.from_numpy(ms_val.copy())
            loaded += 1
        else:
            print(f"  形状不匹配 {ms_name}: {ms_val.shape} vs {pt_name} {pt_val.shape}")
    else:
        print(f"  未找到 PyTorch 对应: {ms_name} → {pt_name}")

print(f"加载了 {loaded}/{len(name_map)} 个参数")

# 6. 保存 PyTorch 模型
torch.save(pt_state, str(PT_PATH))
print(f"\nPyTorch 模型已保存: {PT_PATH}")
print(f"大小: {PT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# 7. 验证
pt_net.load_state_dict(torch.load(str(PT_PATH)))
pt_net.eval()
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = pt_net(dummy)
print(f"推理测试: {out.shape} → {out.softmax(1).numpy()}")
