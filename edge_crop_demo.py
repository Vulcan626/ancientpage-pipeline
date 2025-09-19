#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
from pathlib import Path

# ------ Configs ------
INPUT_PATH = Path("demo/data/20250919-111130.jpg")
OUTPUT_DIR = Path("demo/outputs")

# Canny 边缘参数
CANNY_SIGMA = 0.33
COL_THRESHOLD_RATIO = 0.45  # 列投影能量阈值
MIN_PAGE_WIDTH_RATIO = 0.15 # 页面最小宽度占整图比例

# ------ Functions ------
def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def process_image(img_path: Path, out_dir: Path):
    name = img_path.stem
    save_dir = out_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 读取原图
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"[ERROR] 图像读取失败: {img_path}")
    H, W = img.shape[:2]

    # 生成边缘图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray, sigma=CANNY_SIGMA)

    edge_path = save_dir / f"edge_map_{name}.jpg"
    cv2.imwrite(str(edge_path), edges)

    # 列投影能量
    col_sum = edges.sum(axis=0).astype(np.float32)
    thr = col_sum.max() * COL_THRESHOLD_RATIO
    strong_cols = np.where(col_sum > thr)[0]
    if strong_cols.size < 2:
        raise RuntimeError(f"[ERROR] 未检测到明显的左右边界: {img_path}")

    xmin, xmax = int(strong_cols[0]), int(strong_cols[-1])
    ymin, ymax = 0, H-1

    # 宽度保护
    min_w = int(W * MIN_PAGE_WIDTH_RATIO)
    if (xmax - xmin) < min_w:
        delta = (min_w - (xmax - xmin)) // 2 + 1
        xmin = max(0, xmin - delta)
        xmax = min(W-1, xmax + delta)

    # 裁剪
    crop = img[ymin:ymax, xmin:xmax]
    crop_path = save_dir / f"page_crop_{name}.jpg"
    cv2.imwrite(str(crop_path), crop)

    # 带框调试图
    debug = img.copy()
    cv2.rectangle(debug, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    debug_path = save_dir / f"page_crop_debug_{name}.jpg"
    cv2.imwrite(str(debug_path), debug)

    # JSON 边框描述
    bbox = {
        "drawType": "RECTANGLE",
        "points": [float(xmin), float(ymin), float(xmax), float(ymax)]
    }
    json_path = save_dir / f"bbox_{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bbox, f, indent=2, ensure_ascii=False)

    print(f"[COMPLETED] 处理完成: {name}")
    print(f" - 裁剪结果: {crop_path}")
    print(f" - 边缘图: {edge_path}")
    print(f" - 调试图: {debug_path}")
    print(f" - 边框JSON: {json_path}")

# ------ Main ------
if __name__ == "__main__":
    process_image(INPUT_PATH, OUTPUT_DIR)