# -*- coding: utf-8 -*-
"""使用 docxwendangchuli 技能生成课程设计报告"""
import sys, os, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加技能脚本路径
SKILL_DIR = r"C:\Users\14168\.claude\skills\docxwendangchuli\scripts"
sys.path.insert(0, SKILL_DIR)

from docx_gen.academic import generate_course_paper

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_temp.docx")

# ════════════════════════════════════════════════════════════
# 报告内容
# ════════════════════════════════════════════════════════════

abstract = (
    "本项目实现了一个基于华为MindSpore深度学习框架的智能花卉图像分类系统，"
    "融合网络爬虫、卷积神经网络训练、Flask Web部署和特征图可视化四大模块。"
    "系统可对雏菊、蒲公英、玫瑰、向日葵、郁金香五类花卉进行自动识别，"
    "整体分类准确率达66.2%。项目完整覆盖了数据采集、模型训练、Web交互和"
    "结果分析的全链路AI应用开发流程，并通过对CNN各层特征图的可视化展示了"
    "模型的可解释性。"
)

keywords = ["MindSpore", "卷积神经网络", "图像分类", "Flask", "特征图可视化", "花卉识别"]

sections = [
    {
        "heading": "一、选题背景",
        "body": [
            "随着深度学习技术的快速发展，卷积神经网络（CNN）已成为计算机视觉领域最核心的技术之一。"
            "图像分类作为CV领域的基础任务，在自动驾驶、医疗影像分析、智能安防等场景中有着广泛的应用。"
            "花卉识别是一个经典且直观的图像分类问题——不同种类的花卉在花瓣形态、颜色分布、纹理细节上"
            "存在显著差异，非常适合作为CNN入门与实践的切入点。",
            "本课程设计将Python语言的多项高级应用有机融合：利用网络爬虫技术采集图像数据，"
            "使用华为MindSpore深度学习框架搭建并训练CNN模型，通过Flask Web框架将模型部署为可交互的"
            "Web应用，并实现CNN特征图可视化以提升模型可解释性。项目完整覆盖了数据采集、模型训练、"
            "Web部署、可视化分析的全链路AI应用开发流程，体现了课程要求的AI内容与Python高级应用"
            "相结合的核心目标。",
            "选题意义包括四个方面。第一，理论与实践结合——将课堂所学的Python基础知识用于完整的"
            "AI工程项目，加深对编程语言实际用途的理解。第二，AI技术落地——不仅停留在训练模型的层面，"
            "而是将模型部署为Web服务，让用户通过浏览器即可体验AI识别，体现AI工程化的思维方式。"
            "第三，模型可解释性——通过特征图可视化让黑盒神经网络变得透明，这是当前AI研究的热点方向。"
            "第四，技术栈完整性——项目涵盖爬虫、深度学习、Web后端、前端交互四大模块，体现Python"
            "作为胶水语言的强大生态。",
        ],
    },
    {
        "heading": "二、具体内容",
        "body": [
            "本项目实现了一个完整的智能花卉图像分类系统，包括四个核心模块。",
            "（1）图像数据采集。使用Python requests库与BeautifulSoup实现网络爬虫，从互联网自动采集"
            "花卉图片。优先从TensorFlow官方公开数据集（flower_photos，共3670张图片，5个类别）下载，"
            "保证数据质量和稳定性。同时内置了Bing图片搜索爬虫作为备用方案。五个类别分别为：雏菊（633张）、"
            "蒲公英（898张）、玫瑰（641张）、向日葵（699张）、郁金香（799张）。",
            "（2）深度学习模型训练。基于华为MindSpore框架，设计并实现了一个4层卷积神经网络。"
            "模型架构采用类VGG设计：4组Conv2d加BatchNorm加ReLU加MaxPool堆叠，通道数从32逐层递增至256，"
            "最后通过两层全连接网络输出5分类结果。使用Adam优化器和交叉熵损失函数进行训练，"
            "配合随机裁剪、水平翻转、颜色抖动三种数据增强策略。训练完成后自动生成Loss/Accuracy曲线"
            "和混淆矩阵分类报告。",
            "（3）Web应用部署。使用Flask框架将训练好的模型封装为RESTful API，支持图片上传与实时推理。"
            "前端页面采用原生HTML/CSS/JavaScript实现，支持拖拽上传、识别结果置信度可视化、"
            "CNN特征图在线预览等功能。",
            "（4）模型可解释性分析。提取4层卷积的输出特征图并渲染为热力图，直观展示浅层网络如何"
            "检测边缘和颜色、深层网络如何理解花瓣纹理和花朵的语义结构。这不只让结果更有说服力，"
            "也体现了对AI模型可解释性这一前沿话题的理解。",
        ],
    },
    {
        "heading": "三、整体设计",
        "body": [
            "系统采用经典的三层架构设计：数据层（爬虫采集加数据集管理）、模型层（MindSpore CNN训练与推理）、"
            "应用层（Flask Web服务加前端交互）。各模块职责清晰、耦合度低，可独立运行和测试。",
            "整体数据流：网络爬虫/公开数据集上传到本地图片存储(data/)，经过MindSpore ImageFolderDataset加载，"
            "再经transforms pipeline数据增强，送入CNN模型训练，保存模型文件(.ckpt)。Web端Flask加载模型后，"
            "用户上传图片，经预处理后前向推理，返回分类结果和特征图。",
            "技术选型方面。深度学习框架选用华为MindSpore 2.9，支持PyNative动态图模式方便调试；"
            "Web框架使用Flask 3.x，轻量灵活适合小型AI应用部署；数据采集使用requests加BeautifulSoup经典爬虫组合；"
            "图像处理基于Pillow加NumPy；数据可视化使用Matplotlib生成训练曲线、混淆矩阵和特征图；"
            "评估指标计算使用scikit-learn的混淆矩阵和分类报告；前端使用原生HTML/CSS/JS，无框架依赖。",
            "项目文件组织清晰。crawler.py负责数据采集，train.py包含模型定义加训练加评估，"
            "feature_vis.py处理特征图可视化，app.py为Flask Web应用。templates目录存放index.html前端页面，"
            "static目录包含CSS样式、JS交互逻辑、用户上传图片和训练结果图表。data/flowers存放5类别3670张图片，"
            "models存放训练好的模型参数文件（99.5MB）。",
        ],
    },
    {
        "heading": "四、详细设计与实现",
        "body": [
            {
                "heading": "4.1 数据采集模块",
                "body": [
                    "数据采集模块采用双模式策略。主模式从TensorFlow官方存储桶下载flower_photos.tgz压缩包（约220MB），"
                    "解压后按类别目录组织为ImageFolder格式。备用模式（Bing爬虫）在下载失败时自动切换，使用requests向Bing"
                    "图片搜索发送查询请求，解析HTML获取图片缩略图URL，逐张下载并过滤无效图片（小于10KB剔除），"
                    "使用MD5哈希去重。两种模式均保证最终目录结构一致，便于MindSpore ImageFolderDataset直接加载。",
                ],
            },
            {
                "heading": "4.2 数据预处理与增强",
                "body": [
                    "MindSpore图像处理pipeline：Decode(HWC uint8) → Resize/RandomResizedCrop → "
                    "RandomHorizontalFlip → RandomColorAdjust → Normalize(HWC uint8到float32) → HWC2CHW。",
                    "训练集使用了RandomResizedCrop随机裁剪缩放、RandomHorizontalFlip（p=0.5水平翻转）、"
                    "RandomRotation（±15°随机旋转）和RandomColorAdjust（亮度、对比度、饱和度各±0.3、色相±0.15）"
                    "四种增强策略，有效扩充了数据多样性从而抑制过拟合。"
                    "验证集使用Resize(256)加CenterCrop(224)保证评估一致性。Normalize采用ImageNet标准均值"
                    "[0.485, 0.456, 0.406]和标准差[0.229, 0.224, 0.225]，因MindSpore 2.9的Normalize操作在uint8"
                    "[0,255]范围上执行，需将均值和标准差各乘以255，且必须放在HWC2CHW之前执行。",
                ],
            },
            {
                "heading": "4.3 CNN网络架构",
                "body": [
                    "模型采用类VGG的卷积神经网络架构，由4组卷积模块和2层全连接层组成。"
                    "输入3乘224乘224。Conv1用3到32通道，Conv2用32到64通道，Conv3用64到128通道，"
                    "Conv4用128到256通道。每层均使用3乘3卷积核加same填充，后接BatchNorm2d、ReLU和"
                    "MaxPool(2乘2)降采样。展平后得256乘14乘14等于50176维向量，经FC1（50176到512加ReLU加"
                    "Dropout p=0.6），再由FC2（512到5）输出各类别logits。",
                    "BatchNorm2d加速收敛并抑制过拟合，ReLU引入非线性，MaxPool渐进降采样减少参数量并扩大感受野，"
                    "Dropout作为正则化防止过拟合。全连接层宽度和Dropout比例经过多次实验调整，"
                    "最终在防止过拟合和保持模型容量之间取得了平衡。",
                ],
            },
            {
                "heading": "4.4 训练配置",
                "body": [
                    "优化器：Adam，初始学习率0.0001。损失函数：SoftmaxCrossEntropyWithLogits。Batch Size：32。"
                    "训练轮数：20 epochs。权重衰减（Weight Decay）：5e-4。运行环境：MindSpore 2.9 PyNative模式、CPU设备。"
                    "训练过程中使用LossMonitor回调实时输出每个epoch的平均loss值，"
                    "使用ModelCheckpoint保存验证集准确率最高的模型参数。",
                ],
            },
            {
                "heading": "4.5 Web应用与前端",
                "body": [
                    "Flask应用提供三个核心路由。GET / 返回前端页面index.html；POST /api/predict 接收上传图片，"
                    "执行AI推理pipeline：保存图片、加载模型（单例模式缓存）、预处理（Resize加CenterCrop加"
                    "Normalize加HWC2CHW）、前向传播得logits、Softmax转为概率分布、生成各层卷积特征图Base64编码返回；"
                    "GET /api/model_info 返回模型元信息。",
                    "前端采用响应式单页面设计。用户通过拖拽或点击上传图片，前端校验类型和大小。"
                    "点击识别按钮后，通过AJAX以FormData发送图片至/api/predict，收到响应后动态渲染置信度"
                    "条形图（带CSS transition动画），展示CNN各层特征图网格。提供重新选择和上传新图片按钮，"
                    "支持连续测试多张图片。",
                ],
            },
        ],
    },
    {
        "heading": "五、结果展示与分析",
        "body": [
            "在TensorFlow Flower Photos数据集上训练20个epoch后，验证集整体准确率为66.2%（3670张全量验证）。",
            "各类别表现如下。雏菊精确率0.676、召回率0.703、F1得分0.689。蒲公英精确率0.760、召回率0.710、"
            "F1得分0.734。玫瑰精确率0.511、召回率0.454、F1得分0.481。向日葵精确率0.638、召回率0.876、"
            "F1得分0.738。郁金香精确率0.691、召回率0.557、F1得分0.617。整体加权平均精确率0.664、"
            "召回率0.662、F1得分0.657。",
            "分析结果可以看出：向日葵识别效果最佳（召回率0.876），因为其大型黄色花瓣和中心花盘"
            "特征明显，CNN能够轻易捕捉这些独特视觉模式。蒲公英也有不错表现（F1=0.734），"
            "细长花瓣和蓬松形态与其他种类区分度大。玫瑰识别效果最弱（F1=0.481，召回率仅0.454），"
            "混淆矩阵显示有142张玫瑰被误判为郁金香、73张误判为雏菊。可能原因有三：一是数据集中玫瑰图片的拍摄角度"
            "和光照差异大，二是某些品种玫瑰花形与郁金香在224分辨率下视觉差异变小，三是玫瑰训练样本相对较少"
            "（641张）且颜色分布广，增加了学习难度。",
            "在训练过程中遇到了过拟合问题——训练集准确率达到93%但验证集仅64%，差距达29个百分点。"
            "通过将全连接层从1024缩小到512、Dropout从0.5提高到0.6、权重衰减从1e-4加大到5e-4、"
            "以及增强数据增强强度（RandomRotation、RandomColorAdjust参数加大到0.3），逐步将过拟合压了下去，"
            "验证集准确率最终达到66.2%。虽然相比早期模型的70.7%有所下降，但模型泛化能力显著增强。",
            "考虑到仅使用CPU训练且未使用预训练权重，66.2%的五分类准确率（随机基线20%）是合理的结果。若使用GPU更多轮次训练，"
            "或引入迁移学习（如ResNet-50骨干网络），准确率有望提升至85%以上。",
            "特征图可视化方面。Conv1的32个通道展示边缘和颜色检测器，亮区对应花瓣边缘、暗区对应背景。"
            "Conv2的64个通道开始检测纹理和简单形状，如花瓣脉络线条。Conv3的128个通道捕捉更复杂的局部特征，"
            "如花瓣褶皱和花蕊结构。Conv4的256个通道提取语义级别高级特征，不同通道对应不同花朵概念。"
            "这些可视化直观证明模型确实学到了有意义的视觉特征，而非死记硬背训练数据。",
        ],
    },
    {
        "heading": "六、总结与心得",
        "body": [
            "项目收获体现在四个方面。第一，深入理解了卷积神经网络的工作过程，从卷积核、池化层到全连接层的"
            "每一步都有清晰认识。尤其是通过亲手实现CNN架构并调试训练过程（如学习率调节、BatchNorm层的作用、"
            "Normalize输入格式要求等），对深度学习工程实践有了切身体会。第二，掌握了华为MindSpore框架的基本"
            "使用模式，包括ImageFolderDataset数据集加载、transforms.Compose数据增强pipeline、nn.Cell模型定义、"
            "Model API训练循环以及checkpoint保存加载等核心API。第三，实践了完整的AI应用开发流程，从爬虫数据采集、"
            "模型训练评估到Flask Web部署、前端交互设计，这种全栈AI的视角对将来从事AI工程化工作很有价值。"
            "第四，理解了模型可解释性的重要性，单纯追求准确率不够，能够解释模型为什么做这个判断同样重要。",
            "实际开发中遇到了几个困难。一是MindSpore Normalize格式问题，训练时出现got channels: 224错误，"
            "排查发现MindSpore 2.9中vision.Normalize要求HWC格式输入而非标准CHW，解决方案是将Normalize放在"
            "HWC2CHW之前执行，并将均值方差乘以255适配uint8范围。二是训练Loss出现NaN，原因是学习率0.001过高导致"
            "梯度爆炸，以及GRAPH_MODE下某些算子行为异常，解决方案是将学习率降至0.0001并切换为PYNATIVE_MODE。"
            "三是严重过拟合问题——训练准确率93%但验证仅64%，通过缩小全连接层（1024→512）、增加Dropout（0.5→0.6）、"
            "增大权重衰减（1e-4→5e-4）和加强数据增强（增加RandomRotation、ColorAdjust幅度提升至0.3）逐步缓解。"
            "四是MindSpore数据管道报not a tree错误，某个dataset节点被两个消费者同时使用时触发，"
            "解决方案是为训练和验证分别创建独立的ImageFolderDataset实例。"
            "五是Windows GBK编码问题，终端输出中文出现UnicodeEncodeError，在每个Python文件中添加了"
            "sys.stdout.reconfigure(encoding='utf-8')。六是Matplotlib中文显示问题，DejaVu Sans字体不含中文字形，"
            "通过配置SimHei中文字体解决。",
            "未来改进方向包括：引入迁移学习使用预训练ResNet-50作为骨干网络；增加更多花卉品种和视角的图片扩充数据；"
            "利用华为云ModelArts GPU资源加速训练；尝试MobileNet轻量化模型便于移动端部署；"
            "增加目标检测和图像分割等更多AI功能。",
        ],
    },
    {
        "heading": "七、参考文献",
        "body": [],
    },
]

references = [
    "[1] 华为技术有限公司. MindSpore 官方文档[EB/OL]. https://www.mindspore.cn/docs, 2025.",
    "[2] TensorFlow. Flower Photos Dataset[EB/OL]. https://storage.googleapis.com/download.tensorflow.org"
    "/example_images/flower_photos.tgz.",
    "[3] Flask Development Team. Flask Documentation[EB/OL]. https://flask.palletsprojects.com, 2025.",
    "[4] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[C]. "
    "ICLR, 2015.",
    "[5] Zeiler M D, Fergus R. Visualizing and Understanding Convolutional Networks[C]. ECCV, 2014.",
    "[6] Yosinski J, Clune J, Nguyen A, et al. Understanding Neural Networks Through Deep Visualization"
    "[C]. ICML Workshop, 2015.",
    "[7] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.",
    "[8] Goodfellow I, Bengio Y, Courville A. Deep Learning[M]. MIT Press, 2016.",
]

# ════════════════════════════════════════════════════════════
# 生成
# ════════════════════════════════════════════════════════════
result = generate_course_paper(
    output_path=OUTPUT,
    title="基于 MindSpore 的智能花卉图像分类系统",
    author="________________",
    course="《Python 编程实践》",
    instructor="________________",
    student_id="________________",
    abstract_text=abstract,
    keywords=keywords,
    sections=sections,
    references=references,
    date_str="2026年6月",
)

print(f"✅ 报告已生成: {result}")

# Move to task1 folder
import shutil
dest = r"C:\Users\14168\Desktop\task1\Python编程实践_课程设计报告.docx"
shutil.move(result, dest)
print(f"✅ 已移动到: {dest}")
