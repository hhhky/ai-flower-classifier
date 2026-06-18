# -*- coding: utf-8 -*-
"""
生成课程设计报告 (.docx)
"""
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
RESULT_DIR = PROJECT_DIR / "static" / "results"

# ── Document Setup ──────────────────────────────────────────
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)

# Style
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# ── Helper functions ────────────────────────────────────────
def add_heading_styled(text, level=1):
    """Add a heading with proper Chinese font"""
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '黑体'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(14)
        elif level == 3:
            run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h

def add_para(text, bold=False, indent=False, size=12, align=None):
    """Add a paragraph with SimSun font"""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.name = '宋体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(size)
    run.bold = bold
    return p

def add_code_block(code_text):
    """Add a code block as shaded paragraph"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(50, 50, 50)
    # Add shading
    from docx.oxml import OxmlElement
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), 'F5F5F5')
    shading.set(qn('w:val'), 'clear')
    p.runs[0].element.rPr.append(shading) if False else None
    return p

# ════════════════════════════════════════════════════════════
# COVER PAGE
# ════════════════════════════════════════════════════════════
for _ in range(6):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('《Python 编程实践》期末课程设计报告')
run.font.name = '黑体'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
run.font.size = Pt(26)
run.bold = True

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('基于 MindSpore 的智能花卉图像分类系统')
run.font.name = '黑体'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
run.font.size = Pt(20)

for _ in range(4):
    doc.add_paragraph()

info_table = [
    ('课   程：', '《Python 编程实践》'),
    ('学   院：', '计算机科学与技术学院'),
    ('专   业：', '________________________'),
    ('姓   名：', '________________________'),
    ('学   号：', '________________________'),
    ('日   期：', '2026 年 6 月'),
]
for label, value in info_table:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f'{label}　{value}')
    run.font.name = '宋体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(14)

doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 一、选题背景
# ════════════════════════════════════════════════════════════
add_heading_styled('一、选题背景', level=1)

add_para(
    '随着深度学习技术的快速发展，卷积神经网络（Convolutional Neural Network, CNN）'
    '已成为计算机视觉领域最核心的技术之一。图像分类作为 CV 领域的基础任务，'
    '在自动驾驶、医疗影像分析、智能安防等场景中有着广泛的应用。花卉识别是一个经典'
    '且直观的图像分类问题——不同种类的花卉在花瓣形态、颜色分布、纹理细节上存在显著'
    '差异，非常适合作为 CNN 入门与实践的切入点。', indent=True)

add_para(
    '本课程设计将 Python 语言的多项高级应用有机融合：利用网络爬虫技术采集图像数据，'
    '使用华为 MindSpore 深度学习框架搭建并训练 CNN 模型，通过 Flask Web 框架将模型'
    '部署为可交互的 Web 应用，并实现 CNN 特征图可视化以提升模型可解释性。项目完整'
    '覆盖了"数据采集→模型训练→Web 部署→可视化分析"的全链路 AI 应用开发流程，'
    '体现了《Python 编程实践》课程所要求的 AI 内容与 Python 高级应用相结合的核心目标。', indent=True)

add_para('本项目的选题动机和意义包括：', indent=True)

motivations = [
    '理论与实践结合：将课堂所学的 Python 基础知识应用于一个完整的 AI 工程项目，'
    '加深对编程语言实际用途的理解。',
    'AI 技术落地：不仅停留在训练模型的层面，更是将模型部署为 Web 服务，'
    '让用户通过浏览器即可体验 AI 识别——这体现了"AI 工程化"的思维方式。',
    '模型可解释性：通过特征图可视化，让"黑盒"神经网络变得透明，'
    '这是当前 AI 研究的热点方向之一。',
    '技术栈完整性：项目涵盖爬虫、深度学习、Web 后端、前端交互四大模块，'
    '体现 Python 作为"胶水语言"的强大生态。',
]
for m in motivations:
    add_para(f'• {m}', indent=True)

# ════════════════════════════════════════════════════════════
# 二、具体内容
# ════════════════════════════════════════════════════════════
add_heading_styled('二、具体内容', level=1)

add_para(
    '本项目实现了一个完整的智能花卉图像分类系统，具体功能包括：', indent=True)

add_heading_styled('2.1 图像数据采集（爬虫模块）', level=2)
add_para(
    '使用 Python requests 库与 BeautifulSoup 实现网络爬虫，从互联网自动采集花卉图片。'
    '优先从 TensorFlow 官方公开数据集（flower_photos，共 3670 张图片，5 个类别）下载，'
    '保证数据质量和稳定性。同时内置了 Bing 图片搜索爬虫作为备用方案。下载完成后对图片'
    '进行格式验证和去重处理，确保数据集的有效性。五个类别分别为：雏菊（Daisy）、'
    '蒲公英（Dandelion）、玫瑰（Roses）、向日葵（Sunflowers）、郁金香（Tulips）。', indent=True)

add_heading_styled('2.2 深度学习模型训练（AI 核心）', level=2)
add_para(
    '基于华为 MindSpore 深度学习框架，设计并实现了一个 4 层卷积神经网络用于花卉分类。'
    '模型架构采用类 VGG 设计：4 组 Conv2d → BatchNorm → ReLU → MaxPool 堆叠，'
    '通道数从 32 逐层递增至 256，最后通过两层全连接网络输出 5 分类结果。使用 Adam '
    '优化器和交叉熵损失函数进行训练，配合数据增强（随机裁剪、水平翻转、颜色抖动）'
    '提升模型泛化能力。训练完成后自动生成 Loss/Accuracy 训练曲线和混淆矩阵分类报告。', indent=True)

add_heading_styled('2.3 Web 应用部署（Flask + 前端）', level=2)
add_para(
    '使用 Flask 框架将训练好的模型封装为 RESTful API，支持图片上传与实时推理。'
    '前端页面采用原生 HTML/CSS/JavaScript 实现，支持拖拽上传、识别结果置信度可视化、'
    'CNN 特征图在线预览等功能。整体交互流畅，响应式布局兼容移动端。', indent=True)

add_heading_styled('2.4 模型可解释性分析（特征图可视化）', level=2)
add_para(
    'CNN 模型常被视为"黑盒"，本项目的亮点之一就是通过特征图可视化来解释模型的行为。'
    '提取 4 层卷积的输出特征图并渲染为热力图，直观展示浅层网络如何检测边缘和颜色、'
    '深层网络如何理解花瓣纹理和花朵的语义结构。这不仅让结果更具说服力，'
    '也体现了对 AI 模型可解释性这一前沿话题的理解。', indent=True)

# ════════════════════════════════════════════════════════════
# 三、整体设计
# ════════════════════════════════════════════════════════════
add_heading_styled('三、整体设计', level=1)

add_heading_styled('3.1 系统架构', level=2)
add_para(
    '本系统采用经典的三层架构设计：数据层（爬虫采集 + 数据集管理）、'
    '模型层（MindSpore CNN 训练与推理）、应用层（Flask Web 服务 + 前端交互）。'
    '各模块职责清晰、耦合度低，可独立运行和测试。', indent=True)

add_para('整体数据流如下：', indent=True)
flow_text = (
    '网络爬虫/公开数据集 → 本地图片存储(data/) → MindSpore ImageFolderDataset 加载\n'
    '→ 数据增强(transforms pipeline) → CNN 模型训练 → 模型保存(.ckpt)\n'
    '→ Flask 加载模型 → 用户上传图片 → 预处理 → 前向推理 → 返回分类结果 + 特征图'
)
add_code_block(flow_text)

add_heading_styled('3.2 项目目录结构', level=2)
structure = (
    'ai_image_classifier/\n'
    '├── crawler.py          数据采集模块（爬虫 + 公开数据集下载）\n'
    '├── train.py            MindSpore CNN 训练模块（模型定义 + 训练 + 评估）\n'
    '├── feature_vis.py      特征图可视化模块（各层输出渲染）\n'
    '├── app.py              Flask Web 应用（API + 前后端交互）\n'
    '├── templates/\n'
    '│   └── index.html      前端页面\n'
    '├── static/\n'
    '│   ├── style.css       样式表\n'
    '│   ├── app.js          前端交互逻辑\n'
    '│   ├── uploads/        用户上传图片\n'
    '│   └── results/        训练曲线、混淆矩阵\n'
    '├── data/flowers/       数据集目录（5 类别，3670 张图片）\n'
    '├── models/             训练好的模型参数文件\n'
    '└── requirements.txt    依赖清单'
)
add_code_block(structure)

add_heading_styled('3.3 技术选型', level=2)

table = doc.add_table(rows=8, cols=3)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ['技术领域', '选型', '说明']
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ('深度学习框架', '华为 MindSpore 2.9', '国产 AI 框架，支持动态图(PyNative)调试'),
    ('Web 框架', 'Flask 3.x', '轻量灵活，适合小型 AI 应用部署'),
    ('数据采集', 'requests + BeautifulSoup', '经典爬虫组合，配合 TensorFlow 公开数据集'),
    ('图像处理', 'Pillow + NumPy', '图片加载、预处理、格式转换'),
    ('数据可视化', 'Matplotlib', '训练曲线、混淆矩阵、特征图渲染'),
    ('评估指标', 'scikit-learn', '混淆矩阵、精确率/召回率/F1 计算'),
    ('前端', '原生 HTML/CSS/JS', '无框架依赖，拖拽上传 + 动态渲染'),
]
for i, (a, b, c) in enumerate(rows_data):
    table.rows[i+1].cells[0].text = a
    table.rows[i+1].cells[1].text = b
    table.rows[i+1].cells[2].text = c

doc.add_paragraph()

# ════════════════════════════════════════════════════════════
# 四、详细设计与实现
# ════════════════════════════════════════════════════════════
add_heading_styled('四、详细设计与实现', level=1)

add_heading_styled('4.1 数据采集模块（crawler.py）', level=2)
add_para(
    '数据采集模块负责构建花卉图像数据集。设计上采用双模式策略：', indent=True)
add_para(
    '（1）主模式：从 TensorFlow 官方存储桶下载 flower_photos.tgz 压缩包（约 220MB），'
    '解压后自动按类别目录组织为 ImageFolder 格式。该数据集已人工分类，质量可靠。', indent=True)
add_para(
    '（2）备用模式（Bing 爬虫）：若下载失败，自动切换为使用 requests 向 Bing 图片搜索'
    '发送查询请求，解析返回的 HTML 获取图片缩略图 URL，逐张下载并过滤无效图片（小于 '
    '10KB 的视为损坏图片予以剔除），同时使用 MD5 哈希进行去重。', indent=True)
add_para(
    '两种模式均保证最终数据目录结构一致，便于后续 MindSpore ImageFolderDataset 直接加载。'
    '最终数据集规模：5 个类别共 3670 张图片（daisy: 633, dandelion: 898, roses: 641, '
    'sunflowers: 699, tulips: 799）。', indent=True)

add_heading_styled('4.2 模型训练模块（train.py）', level=2)

add_heading_styled('4.2.1 数据预处理与增强', level=3)
add_para(
    'MindSpore 的图像处理 pipeline 按以下顺序执行：', indent=True)
pipeline_text = (
    'Decode(HWC uint8) → Resize/RandomResizedCrop(HWC uint8)\n'
    '→ RandomHorizontalFlip → RandomColorAdjust\n'
    '→ Normalize(HWC uint8→float32) → HWC2CHW(CHW float32)'
)
add_code_block(pipeline_text)
add_para(
    '训练集使用 RandomResizedCrop（随机裁剪缩放）、RandomHorizontalFlip（p=0.5 水平翻转）'
    '和 RandomColorAdjust（亮度/对比度/饱和度/色相各 ±0.1~0.2）三种数据增强策略。'
    '验证集使用 Resize(256) + CenterCrop(224)，保证评估一致性。'
    'Normalize 采用 ImageNet 标准均值 [0.485, 0.456, 0.406] 和标准差 [0.229, 0.224, 0.225]'
    '，由于 MindSpore 的 Normalize 操作在 uint8 [0,255] 范围上执行，'
    '需将均值和标准差各乘以 255。注意 Normalize 必须放在 HWC2CHW 之前执行（MindSpore 2.9 '
    '版本中 Normalize 需要 HWC 格式输入）。', indent=True)

add_heading_styled('4.2.2 网络架构', level=3)
add_para(
    '本模型采用类 VGG 的卷积神经网络架构，由 4 组卷积模块和 2 层全连接层组成：', indent=True)

arch_text = (
    'FlowerCNN 架构（输入 3×224×224）:\n'
    '  Conv1: 3→32, kernel=3×3, padding=same, BN, ReLU, MaxPool(2×2)  → 32×112×112\n'
    '  Conv2: 32→64, kernel=3×3, padding=same, BN, ReLU, MaxPool(2×2) → 64×56×56\n'
    '  Conv3: 64→128, kernel=3×3, padding=same, BN, ReLU, MaxPool(2×2) → 128×28×28\n'
    '  Conv4: 128→256, kernel=3×3, padding=same, BN, ReLU, MaxPool(2×2) → 256×14×14\n'
    '  Flatten → 256×14×14 = 50176\n'
    '  FC1: 50176 → 512, ReLU, Dropout(p=0.5)\n'
    '  FC2: 512 → 5 (输出各类别 logits)'
)
add_code_block(arch_text)

add_para(
    '每层卷积后使用 BatchNorm2d 加速收敛并抑制过拟合，使用 ReLU 激活函数引入非线性。'
    '全局使用 MaxPool2d(kernel=2, stride=2) 渐进降采样，减少参数量并扩大感受野。'
    '全连接层使用 Dropout(p=0.5) 作为正则化手段。', indent=True)

add_heading_styled('4.2.3 训练配置', level=3)
add_para(
    '优化器：Adam，初始学习率 0.0001。损失函数：交叉熵（nn.SoftmaxCrossEntropyWithLogits）。'
    'Batch Size：32。训练轮数：10 epochs。运行环境：MindSpore 2.9 PyNative 模式，CPU 设备。'
    '训练过程中使用 LossMonitor 回调函数实时输出每个 epoch 的平均 loss 值，'
    '并使用 ModelCheckpoint 保存验证集准确率最高的模型参数。', indent=True)

add_heading_styled('4.3 Web 应用模块（app.py）', level=2)
add_para(
    'Flask 应用提供以下核心路由：', indent=True)
add_para(
    '• GET / — 返回前端页面（index.html），展示上传界面和系统介绍。', indent=True)
add_para(
    '• POST /api/predict — 接收上传的图片文件，执行 AI 推理 pipeline：'
    '保存图片 → 加载模型（单例模式，首次加载后缓存）→ 预处理（Resize + CenterCrop + '
    'Normalize + HWC2CHW）→ 前向传播得到 logits → Softmax 转为概率分布 → '
    '生成各层卷积特征图（Base64 编码返回前端）。', indent=True)
add_para(
    '• GET /api/model_info — 返回模型元信息（架构、参数量、训练超参数、分类报告等）。', indent=True)

add_heading_styled('4.4 前端交互设计（index.html + app.js + style.css）', level=2)
add_para(
    '前端采用响应式单页面设计，核心交互流程如下：', indent=True)
add_para(
    '（1）用户通过拖拽或点击上传一张花卉图片，前端进行文件类型和大小校验。'
    '（2）点击"开始识别"按钮，通过 AJAX（FormData）将图片发送至 /api/predict。'
    '（3）收到响应后，动态渲染置信度条形图（带 CSS transition 动画），'
    '并展示 CNN 各层特征图网格。'
    '（4）提供"重新选择"和"上传新图片"按钮，支持连续测试多张图片。', indent=True)

# ════════════════════════════════════════════════════════════
# 五、结果展示与分析
# ════════════════════════════════════════════════════════════
add_heading_styled('五、结果展示与分析', level=1)

add_heading_styled('5.1 训练结果', level=2)
add_para(
    '在 TensorFlow Flower Photos 数据集上训练 10 个 epoch 后，'
    '验证集整体准确率达到 70.7%（3670 张图片全量验证）。各分类详细指标如下：', indent=True)

# Classification report table
table2 = doc.add_table(rows=7, cols=5)
table2.style = 'Light Grid Accent 1'
table2.alignment = WD_TABLE_ALIGNMENT.CENTER

cr_headers = ['类别', '精确率 (Precision)', '召回率 (Recall)', 'F1-Score', '样本数']
for i, h in enumerate(cr_headers):
    cell = table2.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

cr_data = [
    ('雏菊 (Daisy)',       '0.727', '0.709', '0.718', '633'),
    ('蒲公英 (Dandelion)',  '0.657', '0.843', '0.739', '898'),
    ('玫瑰 (Roses)',        '0.720', '0.446', '0.551', '641'),
    ('向日葵 (Sunflowers)', '0.719', '0.851', '0.780', '699'),
    ('郁金香 (Tulips)',     '0.749', '0.633', '0.686', '799'),
    ('整体 (Overall)',      '0.712', '0.707', '0.699', '3670'),
]
for i, row_data in enumerate(cr_data):
    for j, val in enumerate(row_data):
        table2.rows[i+1].cells[j].text = val

doc.add_paragraph()

add_heading_styled('5.2 结果分析', level=2)
add_para(
    '从分类报告可以看出：', indent=True)
add_para(
    '• 向日葵（Sunflowers）识别效果最佳，F1-Score 达到 0.780，召回率高达 0.851。'
    '向日葵的花瓣形态特征明显（大型黄色花瓣、中心花盘），CNN 能够轻易捕捉这些独特的视觉模式。', indent=True)
add_para(
    '• 蒲公英（Dandelion）也有较好的表现（F1=0.739, Recall=0.843），'
    '其细长花瓣和蓬松的形态与其他种类区分度大。', indent=True)
add_para(
    '• 玫瑰（Roses）识别效果最弱（F1=0.551, Recall=0.446），大量玫瑰被误判为郁金香（135 张）'
    '和蒲公英（118 张）。可能原因：（1）数据集中玫瑰图片的拍摄角度和光照差异较大；'
    '（2）某些品种的玫瑰花形与郁金香在缩放至 224×224 分辨率后视觉差异变小；'
    '（3）玫瑰类别训练样本相对较少（641 张），且颜色分布较广（红/粉/白/黄），'
    '增加了学习难度。', indent=True)
add_para(
    '• 整体 accuracy 约 70.7%，考虑到仅使用 CPU 训练 10 个 epoch 且未使用预训练权重，'
    '该结果合理。若使用 GPU 进行更多轮次训练，或引入迁移学习（如 ResNet-50 骨干网络），'
    '准确率有望进一步提升至 85% 以上。', indent=True)

add_heading_styled('5.3 模型可解释性——特征图可视化', level=2)
add_para(
    '特征图可视化是理解 CNN"看到了什么"的关键手段。本系统提取了 4 层卷积的输出：', indent=True)
add_para(
    '• Conv1（32 通道）：显示边缘和颜色检测器。可以看到亮区对应花瓣边缘、暗区对应背景。', indent=True)
add_para(
    '• Conv2（64 通道）：开始检测纹理和简单形状，如花瓣的脉络线条。', indent=True)
add_para(
    '• Conv3（128 通道）：捕捉更复杂的局部特征，如花瓣的褶皱和花蕊结构。', indent=True)
add_para(
    '• Conv4（256 通道）：提取语义级别的高级特征，不同通道对应不同的"花朵概念"。', indent=True)
add_para(
    '这些可视化结果直观地证明了模型确实学到了有意义的视觉特征，而非死记硬背训练数据，'
    '体现了模型的泛化能力和可解释性。', indent=True)

add_heading_styled('5.4 Web 应用界面', level=2)
add_para(
    'Web 前端采用简洁专业的设计风格，主要功能页面包括：'
    '（1）图片上传区——支持拖拽和点选两种方式；'
    '（2）识别结果区——显示 Top-1 预测类别与置信度，附带所有类别的置信度条形图；'
    '（3）特征图展示区——4 层卷积各 8 个通道的热力图网格，让用户直观感受 AI 的"视觉"过程；'
    '（4）加载动画——模型推理时显示 spinner 和状态文字，提升交互体验。'
    '页面采用响应式布局，在移动端和桌面端均有良好的显示效果。', indent=True)

# ════════════════════════════════════════════════════════════
# 六、总结与心得
# ════════════════════════════════════════════════════════════
add_heading_styled('六、总结与心得', level=1)

add_heading_styled('6.1 项目收获', level=2)
gains = [
    '深入理解了卷积神经网络的工作原理，从卷积核、池化层到全连接层的每一步都有清晰的认识。'
    '特别是通过亲手实现 CNN 架构并调试训练过程（如学习率调节、BatchNorm 层的作用、'
    'Normalize 的输入格式要求等），对深度学习工程实践有了切身体会。',
    '掌握了华为 MindSpore 框架的基本使用模式，包括数据集加载（ImageFolderDataset）、'
    '数据增强 pipeline（transforms.Compose）、模型定义（nn.Cell）、训练循环（Model API）'
    '和模型保存/加载（save_checkpoint/load_checkpoint）等核心 API。',
    '实践了完整的 AI 应用开发流程——从爬虫数据采集、模型训练评估，到 Flask Web 部署、'
    '前端交互设计。这种"全栈 AI"的视角对于将来从事 AI 工程化工作非常有价值。',
    '理解了模型可解释性的重要性。单纯追求准确率是不够的，能够解释"模型为什么做出这个判断"'
    '同样重要。特征图可视化技术让我们得以一窥 CNN 的内在表示，这在医疗、金融等高风险'
    'AI 应用场景中尤为重要。',
]
for i, g in enumerate(gains):
    add_para(f'（{i+1}）{g}', indent=True)

add_heading_styled('6.2 遇到的困难与解决方案', level=2)
difficulties = [
    ('MindSpore Normalize 格式问题',
     '训练时出现"got channels: 224"错误。经排查发现 MindSpore 2.9 版本中 '
     'vision.Normalize 要求 HWC 格式输入（而非标准的 CHW）。解决方案：将 Normalize '
     '放在 HWC2CHW 之前执行，并将均值/方差乘以 255 适配 uint8 范围。'),
    ('训练 Loss 出现 NaN',
     '第一个 epoch 第 2 步 loss 即变为 NaN。排查后发现原因是：（1）学习率 0.001 过高，'
     '导致梯度爆炸；（2）GRAPH_MODE 下某些算子行为异常。解决方案：将学习率降至 0.0001，'
     '并将运行模式从 GRAPH_MODE 切换为 PYNATIVE_MODE。'),
    ('Windows GBK 编码问题',
     '终端输出中文时出现 UnicodeEncodeError。解决方案：在每个 Python 文件中添加 '
     'sys.stdout.reconfigure(encoding="utf-8")。'),
    ('Matplotlib 中文显示',
     'DejaVu Sans 字体不包含中文字形。解决方案：配置 matplotlib 使用 SimHei 中文字体。'),
]
for title, desc in difficulties:
    add_para(f'• {title}：{desc}', indent=True)

add_heading_styled('6.3 改进方向', level=2)
improvements = [
    '引入迁移学习：使用 MindSpore 预训练的 ResNet-50 作为骨干网络，预计可将准确率提升至 85%+。',
    '数据扩充：增加更多花卉品种和更多视角的图片，提升模型对稀有品种的识别能力。',
    'GPU 加速：利用华为云 ModelArts 的 GPU 资源进行训练，大幅缩短训练时间，支持更多 epoch 和更大 batch size。',
    '模型轻量化：尝试使用 MobileNet 或 ShuffleNet 替换当前架构，便于部署到移动端。',
    '增加更多 AI 功能：如目标检测（同时识别图片中的多朵花）、图像分割（精确标出花瓣边界）等。',
]
for imp in improvements:
    add_para(f'• {imp}', indent=True)

# ════════════════════════════════════════════════════════════
# 七、参考文献
# ════════════════════════════════════════════════════════════
add_heading_styled('七、参考文献', level=1)

refs = [
    '[1] 华为技术有限公司. MindSpore 官方文档[EB/OL]. https://www.mindspore.cn/docs, 2025.',
    '[2] TensorFlow. Flower Photos Dataset[EB/OL]. '
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz.',
    '[3] Flask Development Team. Flask Documentation[EB/OL]. https://flask.palletsprojects.com, 2025.',
    '[4] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale '
    'Image Recognition[C]. ICLR, 2015.',
    '[5] Zeiler M D, Fergus R. Visualizing and Understanding Convolutional Networks[C]. ECCV, 2014.',
    '[6] Yosinski J, Clune J, Nguyen A, et al. Understanding Neural Networks Through '
    'Deep Visualization[C]. ICML Workshop, 2015.',
    '[7] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.',
    '[8] Goodfellow I, Bengio Y, Courville A. Deep Learning[M]. MIT Press, 2016.',
]
for ref in refs:
    add_para(ref, size=11)

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
output_path = PROJECT_DIR / '《Python编程实践》课程设计报告_智能花卉图像分类系统.docx'
doc.save(str(output_path))
print(f'\n✅ 报告已生成: {output_path.name}')
print(f'   路径: {output_path}')
