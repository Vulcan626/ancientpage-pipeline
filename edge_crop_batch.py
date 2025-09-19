#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量边缘检测与裁剪
支持多线程并发处理

e.g.
python edge_crop_batch.py --i demo/data --o demo/outputs --workers 4 --mode all
--i: 输入路径（文件夹，包含图片）
--o: 输出路径（文件夹）
--canny_sigma: Canny 边缘检测的 sigma 参数，默认 0.33
--canny_col_threshold_ratio: 列投影能量阈值比例，默认 0.45
--min_page_width_ratio: 页面最小宽度占整图比例，默认 0.15
--workers: 并发线程数，默认 4
--window_lines: 每个线程日志窗口行数，默认 8
--mode: 输出模式，all=完整生成物; result=仅裁剪结果，默认 all
"""

import cv2, os, sys, json, argparse, math
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, queue, time
from tqdm import tqdm

# ------ Default Configs ------
DEF_IN = 'demo/data'
DEF_OUT = 'demo/outputs'
DEF_CANNY_SIGMA = 0.33
DEF_COL_THR = 0.45
DEF_MIN_W_RATIO = 0.15
DEF_WORKERS = 4
DEF_WINDOW_LINES = 8

# 日志队列（线程安全）
log_queues = {}

# ------ Logging ------
def init_worker_log(worker_id, log_dir, window_lines):
    """
    初始化每个线程的日志文件和队列

    in: 
        worker_id: 线程ID
        log_dir: 日志目录
        window_lines: 日志窗口行数

    out: 
        None
    """
    log_path = log_dir / f"w{worker_id:02d}.log"
    log_queues[worker_id] = {"path": log_path, "fh": open(log_path, "w", encoding="utf-8"), "lines": queue.deque(maxlen=window_lines)}

def log_msg(worker_id, msg):
    """
    记录日志消息到对应线程的日志文件和队列

    in: 
        worker_id: 线程ID
        msg: 日志消息

    out: 
        None
    """
    if worker_id not in log_queues:
        print(msg); return
    entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    q = log_queues[worker_id]["lines"]
    q.append(entry)
    fh = log_queues[worker_id]["fh"]
    fh.write(entry + "\n"); fh.flush()
    # 同步到终端（仅当前窗口的几行）
    sys.stdout.write("\033[2K\r")  # 清行
    for line in list(q):
        print(line)
    sys.stdout.flush()

def close_logs():
    """
    关闭所有线程的日志文件

    in: 
        None
    out: 
        None
    """
    for v in log_queues.values():
        v["fh"].close()


# ------ Functions ------
def auto_canny(gray, sigma=DEF_CANNY_SIGMA):
    """
    计算 Canny 边缘检测的阈值

    in: 
        gray: 灰度图像
        sigma: 控制阈值范围的参数, sigma 越大，边缘越少, 默认 0.33
    out: 
        edges: 边缘图像
    """
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def process_single(img_path: Path, out_dir: Path, mode: str,
                   canny_sigma, col_thr, min_w_ratio,
                   worker_id=0):
    """
    处理单张图片，生成边缘图、裁剪图、调试图和 JSON 边框描述

    in: 
        img_path: 图片路径
        out_dir: 输出目录
        mode: 输出模式, all=完整生成物; result=仅裁剪结果
        canny_sigma: Canny 边缘检测的 sigma 参数
        col_thr: 列投影能量阈值比例, col_thr越大, 边界越靠近中间
        min_w_ratio: 页面最小宽度占整图比例, 用于宽度保护
        worker_id: 线程ID
    out: 
        success: 处理是否成功 (bool)
    """
    name = img_path.stem
    log_msg(worker_id, f"[INFO] 开始处理 {name}")

    img = cv2.imread(str(img_path))
    if img is None:
        log_msg(worker_id, f"[WARNING] 读取失败: {img_path}")
        return False

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray, sigma=canny_sigma)

    # 列投影
    col_sum = edges.sum(axis=0).astype(np.float32)
    thr = col_sum.max() * col_thr
    strong_cols = np.where(col_sum > thr)[0]
    if strong_cols.size < 2:
        log_msg(worker_id, f"[WARNING] 边界检测失败: {name}")
        return False
    xmin, xmax = int(strong_cols[0]), int(strong_cols[-1])
    ymin, ymax = 0, H-1

    # 宽度保护
    min_w = int(W * min_w_ratio)
    if (xmax - xmin) < min_w:
        delta = (min_w - (xmax - xmin)) // 2 + 1
        xmin = max(0, xmin - delta)
        xmax = min(W-1, xmax + delta)

    # 输出
    if mode == "all":
        save_dir = out_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        edge_path = save_dir / f"edge_map_{name}.jpg"
        crop_path = save_dir / f"page_crop_{name}.jpg"
        debug_path = save_dir / f"page_crop_debug_{name}.jpg"
        json_path = save_dir / f"bbox_{name}.json"
    else:  # result
        out_dir.mkdir(parents=True, exist_ok=True)
        edge_path = None
        debug_path = None
        json_path = None
        crop_path = out_dir / f"page_crop_{name}.jpg"

    # 保存裁剪
    crop = img[ymin:ymax, xmin:xmax]
    cv2.imwrite(str(crop_path), crop)

    if mode == "all":
        cv2.imwrite(str(edge_path), edges)
        debug = img.copy()
        cv2.rectangle(debug, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.imwrite(str(debug_path), debug)
        bbox = {
            "drawType": "RECTANGLE",
            "points": [float(xmin), float(ymin), float(xmax), float(ymax)]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bbox, f, indent=2, ensure_ascii=False)

    log_msg(worker_id, f"[COMPLETED] 处理完成 {name}")
    return True

# ------ Main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", required=True, help="输入路径", default=DEF_IN)
    ap.add_argument("--o", required=True, help="输出路径", default=DEF_OUT)
    ap.add_argument("--canny_sigma", type=float, default=DEF_CANNY_SIGMA)
    ap.add_argument("--canny_col_threshold_ratio", type=float, default=DEF_COL_THR)
    ap.add_argument("--min_page_width_ratio", type=float, default=DEF_MIN_W_RATIO)
    ap.add_argument("--workers", type=int, default=DEF_WORKERS)
    ap.add_argument("--window_lines", type=int, default=DEF_WINDOW_LINES)
    ap.add_argument("--mode", choices=["all", "result"], default="all",
                    help="输出模式: all=完整生成物; result=仅裁剪结果")
    args = ap.parse_args()

    in_dir = Path(args.i)
    out_dir = Path(args.o)
    log_dir = out_dir / "_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not imgs:
        print("[ERROR] 输入目录下没有图片")
        return

    # 初始化日志
    for wid in range(args.workers):
        init_worker_log(wid, log_dir, args.window_lines)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as exe, tqdm(total=len(imgs), desc="[INFO] 批处理进度", ncols=100) as pbar:
        fut2name = {
            exe.submit(process_single, img, out_dir, args.mode,
                       args.canny_sigma, args.canny_col_threshold_ratio, args.min_page_width_ratio,
                       wid % args.workers): img.name
            for wid, img in enumerate(imgs)
        }
        for fut in as_completed(fut2name):
            name = fut2name[fut]
            try:
                ok = fut.result()
                results.append((name, ok))
            except Exception as e:
                results.append((name, False))
                log_msg(0, f"[ERROR] {name}: {e}")
            finally:
                pbar.update(1)   # 更新进度条

    # 汇总日志
    final_path = log_dir / "final.log"
    with open(final_path, "w", encoding="utf-8") as f:
        total = len(results)
        success = sum(1 for _, ok in results if ok)
        fail = total - success
        f.write(f"[INFO] 总计: {total}, 成功: {success}, 失败: {fail}\n")
        for name, ok in results:
            f.write(f"{name}\t{'OK' if ok else 'FAIL'}\n")

    close_logs()
    print(f"[COMPLETED] 批处理完成，总计 {len(results)} 张，详情见 {final_path}")

if __name__ == "__main__":
    main()