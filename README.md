# ANCIENTPAGE-PIPELINE
> Created by ZhanboHua on 2025/09/19


## 项目简介
ANCIENTPAGE-PIPELINE 是一个用于中华古籍图像页面裁剪与边缘检测的处理工具，旨在自动化批量处理古籍扫描图片。它能够高效、准确地检测页面边缘并进行裁剪，适用于数字化古籍整理、图像预处理等场景。

## 功能特性
- 支持单张图片和批量图片的页面边缘检测与自动裁剪
- 可灵活调整边缘检测与裁剪参数
- 支持日志输出，便于追踪处理过程
- 输出标准化裁剪图片及处理报告

## 安装步骤
1. 克隆本项目：
   ```bash
   git clone https://github.com/Vulcan626/ancientpage-pipeline.git
   cd ancientpage-pipeline
   ```
2. 安装依赖（推荐使用conda环境以及 Python 3.9及以上）：
   ```bash
   conda create -n ancientpage python=3.9
   conda activate ancientpage
   pip install -r requirements.txt
   ```

## 使用说明
本项目主要包含两个入口脚本：`edge_crop_demo.py`（单张图片演示）和 `edge_crop_batch.py`（批量处理）。

### 1. 单张图片裁剪示例（在代码中修改配置）
```bash
python edge_crop_demo.py
```

### 2. 批量图片裁剪示例
```bash
python edge_crop_batch.py --i demo/data --o demo/outputs --workers 4 --mode all
```

## 参数说明
```
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
```

## 输出结构示例
- result
处理完成后，输出文件夹结构如下：
```
output/
├── page1_cropped.jpg
├── page2_cropped.jpg
├── ...
├── _log
```
每张图片对应一个裁剪后的输出文件，`_log` 记录处理日志。

- all
处理完成后，输出文件夹结构如下：
```
output/
├── page1_cropped
    ├── bbox_page1_cropped.json
    ├── edge_map_page1_cropped.jpg
    ├── page_crop_debug_page1_cropped.jpg
    ├── page_crop_page1_cropped.jpg(result)
├── page2_cropped
    ├── ...
├── ...
├── _log
```

## 日志说明
处理过程中会自动生成 `_log` 日志文件夹，包含每个worker的日志文件(例如 `w00.log`、`w01.log` 等)和 `final.log`，记录每个worker的处理状态和错误信息，便于排查问题。
## 联系方式
如有问题或建议，请联系@Zhanbo Hua