#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量边缘检测与裁剪
支持多线程并发处理，支持输入目录递归扫描并在输出路径下镜像结构

e.g.
python edge_crop_batch.py --i demo/data --o demo/outputs --workers 4 --mode all
python edge_crop_batch.py --i /Volumes/staff/development/ccf/ancient_crawl/content/screenshot --o /Volumes/staff/development/hzb/work/datasets/ancient_crawl/outputs-result --workers 8 --mode result
--i: 输入路径（文件夹，包含图片）
--o: 输出路径（文件夹）
--canny_sigma: Canny 边缘检测的 sigma 参数，默认 0.33
--canny_col_threshold_ratio: 列投影能量阈值比例，默认 0.45
--min_page_width_ratio: 页面最小宽度占整图比例，默认 0.15
--workers: 并发线程数，默认 4
--window_lines: 每个线程日志窗口行数，默认 8
--mode: 输出模式，all=完整生成物; result=仅裁剪结果，默认 all
--resume: 开启断点恢复，完整输出则跳过，残缺则清理后重做
"""

import cv2, sys, json, argparse, time, os, shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
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
    log_path = log_dir / f"w{worker_id:02d}.log"
    log_queues[worker_id] = {
        "path": log_path,
        "fh": open(log_path, "w", encoding="utf-8"),
        "lines": queue.deque(maxlen=window_lines)
    }

def log_msg(worker_id, msg):
    if worker_id not in log_queues:
        print(msg); return
    entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    fh = log_queues[worker_id]["fh"]
    fh.write(entry + "\n"); fh.flush()

def close_logs():
    for v in log_queues.values():
        v["fh"].close()

# ------ Resume Helpers ------

def exists_nonempty(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def expected_outputs(base_out: Path, name: str, mode: str):
    """Return a dict of expected output paths based on mode."""
    if mode == "all":
        save_dir = base_out / name
        return {
            "save_dir": save_dir,
            "edge": save_dir / f"edge_map_{name}.jpg",
            "crop": save_dir / f"page_crop_{name}.jpg",
            "debug": save_dir / f"page_crop_debug_{name}.jpg",
            "bbox": save_dir / f"bbox_{name}.json",
        }
    else:  # result
        return {
            "save_dir": base_out,
            "crop": base_out / f"page_crop_{name}.jpg",
        }


def is_complete(outputs: dict, mode: str) -> bool:
    if mode == "all":
        need = [outputs["edge"], outputs["crop"], outputs["debug"], outputs["bbox"]]
        if not all(exists_nonempty(p) for p in need):
            return False
        # lightweight JSON sanity check
        try:
            with open(outputs["bbox"], "r", encoding="utf-8") as f:
                data = json.load(f)
            ok = isinstance(data, dict) and data.get("drawType") == "RECTANGLE" and \
                 isinstance(data.get("points"), list) and len(data.get("points")) == 4
            return bool(ok)
        except Exception:
            return False
    else:  # result
        return exists_nonempty(outputs["crop"])


def cleanup_incomplete(outputs: dict, mode: str):
    """Remove broken outputs so we can regenerate cleanly."""
    try:
        if mode == "all":
            # remove the whole subdir for this sample
            save_dir = outputs["save_dir"]
            if save_dir.exists():
                shutil.rmtree(save_dir, ignore_errors=True)
        else:
            p = outputs.get("crop")
            if p and p.exists():
                p.unlink(missing_ok=True)
    except Exception:
        # best-effort cleanup; continue
        pass

# ------ Functions ------
def auto_canny(gray, sigma=DEF_CANNY_SIGMA):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def process_single(img_path: Path, in_dir: Path, out_dir: Path, mode: str,
                   canny_sigma, col_thr, min_w_ratio, worker_id=0, resume=False):
    name = img_path.stem
    rel = img_path.parent.relative_to(in_dir)  # 输入路径下的相对子目录
    base_out = out_dir / rel

    outs = expected_outputs(base_out, name, mode)

    # Resume / skip logic
    if resume and is_complete(outs, mode):
        log_msg(worker_id, f"[INFO] 已存在完整生成物，跳过 {rel}/{name}")
        return True
    if resume and not is_complete(outs, mode):
        log_msg(worker_id, f"[INFO] 检测到不完整生成物，清理后重试 {rel}/{name}")
        cleanup_incomplete(outs, mode)

    base_out.mkdir(parents=True, exist_ok=True)

    log_msg(worker_id, f"[INFO] 开始处理 {rel}/{name}")

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

    # 输出路径（基于期望清单创建目录）
    save_dir = outs["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    edge_path = outs.get("edge")
    crop_path = outs.get("crop")
    debug_path = outs.get("debug")
    json_path = outs.get("bbox")

    # 保存裁剪结果
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

    log_msg(worker_id, f"[COMPLETED] 处理完成 {rel}/{name}")
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
    ap.add_argument("--resume", action="store_true", help="开启断点恢复：完整输出则跳过，残缺则清理后重做")
    args = ap.parse_args()

    in_dir = Path(args.i)
    out_dir = Path(args.o)
    log_dir = out_dir / "_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 递归查找所有图片
    imgs = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not imgs:
        print("[ERROR] 输入目录下没有图片")
        return

    # 初始化日志
    for wid in range(args.workers):
        init_worker_log(wid, log_dir, args.window_lines)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as exe, tqdm(total=len(imgs), desc="[INFO] 批处理进度", ncols=100) as pbar:
        fut2name = {
            exe.submit(process_single, img, in_dir, out_dir, args.mode,
                       args.canny_sigma, args.canny_col_threshold_ratio, args.min_page_width_ratio,
                       wid % args.workers, args.resume): img.name
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
                pbar.update(1)

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