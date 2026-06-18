/**
 * AI 图像智能识别系统 — 前端交互逻辑
 *
 * 功能：
 *   1. 拖拽/点击上传图片
 *   2. 调用后端 API 进行 AI 识别
 *   3. 展示识别结果（置信度条形图）
 *   4. 展示 CNN 特征图可视化
 */

// ============================================================
// DOM 元素
// ============================================================
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const previewArea = document.getElementById("previewArea");
const previewImg = document.getElementById("previewImg");
const loading = document.getElementById("loading");
const resultSection = document.getElementById("resultSection");
const featureSection = document.getElementById("featureSection");
const btnPredict = document.getElementById("btnPredict");
const toast = document.getElementById("toast");

let uploadedFile = null;

// ============================================================
// 上传区域交互
// ============================================================

// 点击上传
uploadArea.addEventListener("click", () => {
    fileInput.click();
});

// 文件选择
fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
});

// 拖拽上传
uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

// 处理文件
function handleFile(file) {
    if (!file) return;

    // 验证文件类型
    if (!file.type.startsWith("image/")) {
        showToast("请上传图片文件（JPG/PNG/WEBP）", "error");
        return;
    }

    // 验证文件大小（16MB）
    if (file.size > 16 * 1024 * 1024) {
        showToast("图片文件不能超过 16MB", "error");
        return;
    }

    uploadedFile = file;

    // 预览
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadArea.classList.add("hidden");
        previewArea.classList.remove("hidden");
        resultSection.classList.add("hidden");
        featureSection.classList.add("hidden");
    };
    reader.readAsDataURL(file);
}

// ============================================================
// AI 识别
// ============================================================

async function doPredict() {
    if (!uploadedFile) return;

    // 显示加载动画
    btnPredict.disabled = true;
    loading.classList.remove("hidden");
    resultSection.classList.add("hidden");
    featureSection.classList.add("hidden");

    const formData = new FormData();
    formData.append("image", uploadedFile);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            // 显示图片
            document.getElementById("resultImg").src = data.image;

            // 显示预测结果
            document.getElementById("topLabel").textContent =
                data.top_result.class_name_cn;
            document.getElementById("topConfidence").textContent =
                data.top_result.confidence.toFixed(1) + "%";
            document.getElementById("topClassEn").textContent =
                data.top_result.class_name;

            // 置信度条形图
            renderConfidenceBars(data.predictions);

            resultSection.classList.remove("hidden");

            // 特征图
            if (data.feature_maps && data.feature_maps.length > 0) {
                renderFeatureMaps(data.feature_maps);
                featureSection.classList.remove("hidden");
            }

            // 滚动到结果
            resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        } else {
            showToast(data.error || "识别失败", "error");
        }
    } catch (err) {
        showToast("网络错误，请确保服务器正在运行", "error");
        console.error("预测失败:", err);
    } finally {
        btnPredict.disabled = false;
        loading.classList.add("hidden");
    }
}

// 渲染置信度条形图
function renderConfidenceBars(predictions) {
    const container = document.getElementById("confidenceBars");
    // 从主题色渐变到浅灰，表示置信度从高到低
    const colors = [
        "#5b5fc7", "#7678d4", "#9e9fde", "#c5c6e8", "#e0e0e6"
    ];

    container.innerHTML = predictions
        .map(
            (p, i) => `
        <div class="bar-item">
            <span class="bar-label">${p.class_name_cn}</span>
            <div class="bar-track">
                <div class="bar-fill"
                     style="width: 0%; background: ${colors[i]}"
                     data-width="${p.confidence}%">
                    ${p.confidence >= 15 ? p.confidence.toFixed(1) + "%" : ""}
                </div>
            </div>
            <span class="bar-value">${p.confidence.toFixed(1)}%</span>
        </div>
    `
        )
        .join("");

    // 动画效果：延迟后逐步填充条形图
    setTimeout(() => {
        container.querySelectorAll(".bar-fill").forEach((bar) => {
            bar.style.width = bar.dataset.width;
        });
    }, 100);
}

// 渲染特征图
function renderFeatureMaps(featureMaps) {
    const container = document.getElementById("featureGrid");
    container.innerHTML = featureMaps
        .map(
            (fm) => `
        <div class="feature-item">
            <div class="feature-name">${fm.name}</div>
            <img src="${fm.image}" alt="${fm.name}" loading="lazy">
        </div>
    `
        )
        .join("");
}

// ============================================================
// Toast 提示
// ============================================================

function showToast(message, type) {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.remove("hidden");

    setTimeout(() => {
        toast.classList.add("hidden");
    }, 3000);
}

// ============================================================
// 重置表单（重新上传）
// ============================================================

function resetForm() {
    // 清空文件
    uploadedFile = null;
    fileInput.value = "";

    // 隐藏所有结果区域
    previewArea.classList.add("hidden");
    resultSection.classList.add("hidden");
    featureSection.classList.add("hidden");
    loading.classList.add("hidden");

    // 显示上传区域
    uploadArea.classList.remove("hidden");

    // 滚动回顶部
    window.scrollTo({ top: 0, behavior: "smooth" });

    // 重置按钮状态
    btnPredict.disabled = false;
}

// 点击结果图片也可以重新上传
document.getElementById("resultImg")?.addEventListener("click", () => {
    resetForm();
});
