# -*- coding: utf-8 -*-
"""
图片数据采集模块
=================
功能：通过网络爬虫采集图像数据，构建分类数据集
支持两种采集方式：
  1. 从 Bing 图片搜索批量下载（需要网络）
  2. 从公开花卉数据集直接下载（推荐，稳定可靠）

目标类别：雏菊(daisy)、蒲公英(dandelion)、玫瑰(roses)、向日葵(sunflowers)、郁金香(tulips)
"""

import os
import sys
import io
import time

# Windows GBK 终端 UTF-8 输出支持
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import hashlib
import tarfile
import shutil
from pathlib import Path
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent / "data"
DATASET_DIR = DATA_DIR / "flowers"
# TensorFlow 官方花卉数据集（稳定可靠）
FLOWERS_DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/flower_photos.tgz"
)

# 五个花卉类别
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CLASS_CN = {
    "daisy": "雏菊",
    "dandelion": "蒲公英",
    "roses": "玫瑰",
    "sunflowers": "向日葵",
    "tulips": "郁金香",
}

# ============================================================
# 方式一：下载公开花卉数据集（推荐首选）
# ============================================================

def download_flowers_dataset():
    """
    从 TensorFlow 公开存储下载花卉图片数据集。
    包含 5 个类别，共约 3670 张图片，非常适合图像分类入门。
    下载地址稳定，无需担心反爬虫。
    """
    print("=" * 60)
    print("  🌸 下载公开花卉数据集 (TensorFlow Flower Photos)")
    print("=" * 60)
    print(f"  数据集 URL: {FLOWERS_DATASET_URL}")
    print(f"  保存路径: {DATASET_DIR}")
    print()

    # 创建目录
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 下载 tgz 文件
    tgz_path = DATA_DIR / "flower_photos.tgz"
    if not tgz_path.exists():
        print("  📥 正在下载数据集（约 220MB）...")
        try:
            response = requests.get(FLOWERS_DATASET_URL, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(tgz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        print(f"\r  进度: {downloaded / 1024 / 1024:.1f}MB / "
                              f"{total_size / 1024 / 1024:.1f}MB ({pct:.0f}%)", end="")
            print("\n  ✅ 下载完成！")
        except Exception as e:
            print(f"\n  ❌ 下载失败: {e}")
            if tgz_path.exists():
                tgz_path.unlink()
            return False
    else:
        print("  📦 发现已有压缩包，跳过下载")

    # 解压
    extract_dir = DATA_DIR / "flower_photos"
    if not extract_dir.exists():
        print("  📂 正在解压...")
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=DATA_DIR, filter='data')
            print("  ✅ 解压完成！")
        except Exception as e:
            # Python 旧版本不支持 filter 参数
            print(f"  尝试兼容模式解压...")
            try:
                with tarfile.open(tgz_path, "r:gz") as tar:
                    tar.extractall(path=DATA_DIR)
                print("  ✅ 解压完成！")
            except Exception as e2:
                print(f"  ❌ 解压失败: {e2}")
                return False
    else:
        print("  📂 发现已有解压目录，跳过解压")

    # 整理为标准目录结构: data/flowers/{class_name}/*.jpg
    print("  🔧 正在整理目录结构...")
    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # 原始数据在这个目录下
        src_dir = extract_dir / class_name
        if src_dir.exists():
            for img_file in src_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    dest = class_dir / img_file.name
                    if not dest.exists():
                        shutil.copy2(img_file, dest)

    # 统计
    print("\n  📊 数据集统计：")
    total = 0
    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.jpeg"))) + len(list(class_dir.glob("*.png")))
        print(f"     {CLASS_CN[class_name]:6s} ({class_name:12s}): {count:4d} 张")
        total += count
    print(f"     {'─' * 30}")
    print(f"     总计:                 {total:4d} 张")
    print()

    # 清理
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if tgz_path.exists():
        tgz_path.unlink()
    DATA_DIR.joinpath("LICENSE.txt").unlink(missing_ok=True)

    return total > 0


# ============================================================
# 方式二：Bing 图片搜索爬虫（含金量体现）
# ============================================================

def search_bing_images(query, num_images=100):
    """
    从 Bing 图片搜索获取图片 URL 列表。
    这是"网络爬虫+信息提取"的核心体现。

    参数:
        query: 搜索关键词
        num_images: 需要的图片数量

    返回:
        list: 图片 URL 列表
    """
    print(f"  🔍 搜索 Bing 图片: '{query}' (目标 {num_images} 张)...")

    image_urls = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # Bing 图片搜索每页约 35 张
    pages_needed = max(1, num_images // 35 + 1)

    for page in range(pages_needed):
        if len(image_urls) >= num_images:
            break

        # Bing 图片搜索 URL（使用 &first 分页）
        offset = page * 35
        search_url = (
            f"https://www.bing.com/images/search?"
            f"q={requests.utils.quote(query)}&first={offset}&form=HDRSC2"
        )

        try:
            resp = requests.get(search_url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Bing 将图片 URL 存在 <a class="iusc"> 的 m 属性中
            for a_tag in soup.find_all("a", class_="iusc"):
                m_attr = a_tag.get("m")
                if m_attr:
                    import json
                    try:
                        meta = json.loads(m_attr)
                        img_url = meta.get("murl") or meta.get("turl")
                        if img_url and img_url.startswith("http"):
                            image_urls.append(img_url)
                    except json.JSONDecodeError:
                        continue

            # 备用解析方式：找 <img> 标签
            if not image_urls:
                for img_tag in soup.find_all("img"):
                    src = img_tag.get("src") or img_tag.get("data-src")
                    if src and src.startswith("http") and "bing" not in src:
                        image_urls.append(src)

            time.sleep(0.5)  # 礼貌爬取

        except Exception as e:
            print(f"     ⚠️ 第 {page + 1} 页搜索失败: {e}")
            continue

    print(f"     获取到 {len(image_urls)} 个图片 URL")
    return image_urls[:num_images]


def download_images(urls, save_dir, class_name, max_images=150):
    """
    下载图片到指定目录，含去重和验证。

    参数:
        urls: 图片 URL 列表
        save_dir: 保存目录
        class_name: 类别名称
        max_images: 最多下载数量
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for i, url in enumerate(urls):
        if downloaded >= max_images:
            break

        try:
            # 用 URL 的 hash 做文件名，避免重复
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = ".jpg"
            if ".png" in url.lower():
                ext = ".png"
            elif ".webp" in url.lower():
                ext = ".webp"

            filename = f"{class_name}_{url_hash}{ext}"
            filepath = save_dir / filename

            if filepath.exists():
                skipped += 1
                continue

            # 下载
            resp = requests.get(url, headers=headers, timeout=10, stream=True)
            resp.raise_for_status()

            # 验证是否为有效图片
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                failed += 1
                continue

            # 保存并验证
            img_data = resp.content
            if len(img_data) < 1024:  # 小于 1KB 的忽略
                failed += 1
                continue

            try:
                img = Image.open(BytesIO(img_data))
                img.verify()  # 验证图片完整性
                # 重新打开（verify 后需要重新加载）
                img = Image.open(BytesIO(img_data))
                # 转换为 RGB（处理 RGBA/灰度等）
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                # 过滤太小的图片
                if img.width < 100 or img.height < 100:
                    failed += 1
                    continue
                # 保存为 JPEG
                img.save(filepath, "JPEG", quality=90)
                downloaded += 1
            except Exception:
                failed += 1
                if filepath.exists():
                    filepath.unlink()
                continue

            if (downloaded + failed + skipped) % 20 == 0:
                print(f"\r    进度: ✓{downloaded} ✗{failed} →{skipped}", end="")

        except Exception:
            failed += 1
            continue

    print(f"\r    ✅ {class_name}: 下载 {downloaded} 张, 失败 {failed} 张, 跳过 {skipped} 张")
    return downloaded


def crawl_images_via_bing():
    """
    通过 Bing 图片搜索爬取花卉图片。
    适用于公开数据集不可用的情况，也体现"爬虫"技术含量。
    """
    print("=" * 60)
    print("  🕷️ Bing 图片搜索爬虫模式")
    print("=" * 60)

    search_queries = {
        "daisy": "daisy flower close-up",
        "dandelion": "dandelion flower close-up",
        "roses": "roses flower close-up",
        "sunflowers": "sunflower close-up",
        "tulips": "tulips flower close-up",
    }

    total_downloaded = 0
    for class_name, query in search_queries.items():
        print(f"\n  📸 采集类别: {CLASS_CN[class_name]} ({class_name})")
        urls = search_bing_images(query, num_images=150)
        if urls:
            count = download_images(
                urls,
                save_dir=DATASET_DIR / class_name,
                class_name=class_name,
                max_images=120,
            )
            total_downloaded += count
        else:
            print(f"     ⚠️ 未获取到图片 URL，请检查网络连接")

    print(f"\n  📊 爬虫采集总计: {total_downloaded} 张有效图片")
    return total_downloaded > 0


# ============================================================
# 统一入口
# ============================================================

def main():
    """主函数：智能选择数据采集方式"""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       🌸 AI 图像分类 — 数据采集模块                    ║")
    print("║       网络爬虫 + 公开数据集 双模式                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 检查是否已有数据
    existing_data = True
    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists() or len(list(class_dir.glob("*"))) < 10:
            existing_data = False
            break

    if existing_data:
        print("  ℹ️  检测到已有数据集，跳过采集。")
        print("     如需重新采集，请删除 data/flowers 目录后重试。")
        return True

    # 优先使用公开数据集（稳定可靠）
    print("  📋 方案一：尝试下载公开花卉数据集...")
    success = download_flowers_dataset()

    # 如果公开数据集失败，使用爬虫
    if not success:
        print("\n  ⚠️  公开数据集下载失败，切换到方案二：Bing 爬虫")
        print("  📋 方案二：通过 Bing 图片搜索爬取...")
        success = crawl_images_via_bing()

    if success:
        print("\n  🎉 数据采集完成！可以运行 train.py 开始训练。")
    else:
        print("\n  ❌ 数据采集失败，请检查网络连接。")
        print("     你也可以手动下载图片放到 data/flowers/ 目录下：")
        print("     data/flowers/daisy/       ← 雏菊图片")
        print("     data/flowers/dandelion/   ← 蒲公英图片")
        print("     data/flowers/roses/       ← 玫瑰图片")
        print("     data/flowers/sunflowers/  ← 向日葵图片")
        print("     data/flowers/tulips/      ← 郁金香图片")

    return success


if __name__ == "__main__":
    main()
