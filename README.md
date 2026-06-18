# 🌸 AI 图像智能识别系统

## 《Python编程实践》期末课程设计

基于华为 **MindSpore** 框架的深度学习图像分类应用，融合 **网络爬虫、神经网络训练、Flask Web 交互、特征图可视化** 的完整 AI 项目。

---

## 🎯 项目亮点

| 维度 | 体现 |
|------|------|
| **人工智能** | MindSpore CNN 训练、特征图可视化、混淆矩阵分析 |
| **网络爬虫** | Bing 图片搜索数据采集、图片去重与验证 |
| **Web 交互** | Flask 后端 + 现代前端界面、拖拽上传、实时推理 |
| **可解释性** | 4层卷积特征图展示，揭示 AI "看到了什么" |

## 📁 项目结构

```
ai_image_classifier/
├── crawler.py          # 🕷️ 数据采集模块（爬虫 + 公开数据集）
├── train.py            # 🧠 MindSpore CNN 训练模块（AI 核心）
├── feature_vis.py      # 🔬 特征图可视化模块
├── app.py              # 🌐 Flask Web 应用
├── templates/
│   └── index.html      # 🎨 前端页面
├── static/
│   ├── style.css       # 样式表
│   ├── app.js          # 前端交互逻辑
│   ├── uploads/        # 用户上传图片
│   └── results/        # 训练曲线、混淆矩阵
├── data/               # 数据集目录
├── models/             # 训练好的模型
└── requirements.txt    # 依赖清单
```

## 🚀 使用步骤

### 步骤 1：安装依赖
```bash
pip install -r requirements.txt
```

### 步骤 2：采集数据
```bash
python crawler.py
```
自动下载花卉数据集（5类，约3670张）。如果下载失败会自动切换为 Bing 爬虫模式。

### 步骤 3：训练模型
```bash
python train.py
```
训练完成后会在 `models/` 保存模型，在 `static/results/` 生成训练曲线和混淆矩阵。

### 步骤 4：启动 Web 应用
```bash
python app.py
```
浏览器打开 **http://localhost:5000**，上传图片即可体验 AI 识别。

### 步骤 5（可选）：特征图可视化
```bash
python feature_vis.py
```
生成各层卷积特征图，直观展示 AI 的视觉认知过程。

---

## 🧠 技术详解

### CNN 网络架构
```
输入 (3×224×224)
  ↓
Conv1: 32@3×3 → BN → ReLU → MaxPool(2×2)    [边缘/颜色检测]
  ↓
Conv2: 64@3×3 → BN → ReLU → MaxPool(2×2)    [纹理/形状检测]
  ↓
Conv3: 128@3×3 → BN → ReLU → MaxPool(2×2)   [花瓣/局部特征]
  ↓
Conv4: 256@3×3 → BN → ReLU → MaxPool(2×2)   [花朵/语义理解]
  ↓
Flatten → FC(512) → Dropout(0.5) → FC(5)     [分类决策]
```

### 数据增强策略
- 随机裁剪（RandomResizedCrop）
- 随机水平翻转（RandomHorizontalFlip）
- 颜色抖动（RandomColorAdjust：亮度/对比度/饱和度/色相）

### 评估指标
- 混淆矩阵（Confusion Matrix）
- 精确率/召回率/F1-score（Classification Report）
- Loss/Accuracy 训练曲线

---

## 📊 课程报告对应

| 报告章节 | 对应模块 |
|----------|---------|
| 一、选题背景 | 为什么选择花卉识别 + 爬虫 + AI |
| 二、具体内容 | 项目概述（数据→训练→Web） |
| 三、整体设计 | 技术架构图 + 模块划分 |
| 四、详细设计与实现 | 爬虫实现 + CNN设计 + Flask + 前端 |
| 五、结果展示与分析 | 训练曲线截图 + 混淆矩阵 + Web演示截图 |
| 六、团队组成与分工 | 队员1负责XX，队员2负责XX |
| 七、总结与心得 | 收获、难点、改进方向 |
| 八、参考文献 | MindSpore官方文档等 |
