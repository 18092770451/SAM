import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import time
import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)   # 输出到终端
        self.log.write(message)        # 写入txt

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ====================== 1. 路径配置 ======================

INPUT_FOLDER = "/home/zhao_ziyi/program/segment-anything-main/images"

OUTPUT_FOLDER = "/home/zhao_ziyi/program/segment-anything-main/sam_results10_new"

#保存打印内容
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

log_path = os.path.join(OUTPUT_FOLDER, "run_log.txt")
sys.stdout = Logger(log_path)
IMG_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

subfolders = [
    'red_contour',
    'color_vis',
    'mask',
    'single_cells',
    'sam_masks',
    'filtered_masks'
]

# ====================== 2. 单细胞参数 ======================

OUTPUT_SIZE = 256       # 最终输出尺寸
MARGIN_RATIO = 0.25     # 细胞周围留25%边距

DOWNSAMPLE = 1  # 下采样比例，越小速度越快，但可能漏检小细胞

MIN_AREA = int(3500 * DOWNSAMPLE * DOWNSAMPLE)          # 过滤小碎片
MAX_AREA = int(30000 * DOWNSAMPLE * DOWNSAMPLE)   # 过滤太大的

SKIP_BORDER_OBJECT = True  # 是否过滤贴边目标

USE_FINETUNE = False  # True使用微调权重，False只使用官方SAM权重

FILTER_BY_COLOR = True
MIN_MEAN_GRAY = 15       # mask内平均灰度过低时过滤
MAX_MEAN_GRAY = 245      # mask内平均灰度过高时过滤
MIN_COLOR_STD = 5        # mask内颜色变化过小时过滤
NUCLEUS_GRAY_SCALE = 0.75       # 细胞核深色阈值 = 全局Otsu阈值 * 该系数
MIN_NUCLEUS_DARK_RATIO = 0.05  # mask内深染像素的最低占比

FILTER=True # 是否使用过滤条件

# ====================== 3. 加载SAM ======================

print("正在加载 SAM 模型...")

BASE_CHECKPOINT = "/home/zhao_ziyi/program/segment-anything-main/sam_vit_h_4b8939.pth"
FINETUNE_CHECKPOINT = "/home/zhao_ziyi/program/segment-anything-main/sam_finetune_outputs/last_decoder_finetune.pth"

sam = sam_model_registry["vit_h"](
    checkpoint=BASE_CHECKPOINT
)

if USE_FINETUNE and os.path.exists(FINETUNE_CHECKPOINT):
    finetune_ckpt = torch.load(
        FINETUNE_CHECKPOINT,
        map_location="cpu"
    )

    if "mask_decoder_state_dict" in finetune_ckpt:
        sam.mask_decoder.load_state_dict(
            finetune_ckpt["mask_decoder_state_dict"]
        )
        print("已加载微调 mask decoder 权重")
    elif "sam_state_dict" in finetune_ckpt:
        sam.load_state_dict(
            finetune_ckpt["sam_state_dict"]
        )
        print("已加载微调整体 SAM 权重")
    else:
        sam.load_state_dict(finetune_ckpt)
        print("已加载微调权重")
else:
    if USE_FINETUNE:
        print("未找到微调权重，使用官方 SAM 权重")
    else:
        print("已选择不使用微调权重，使用官方 SAM 权重")

device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    min_mask_region_area=MIN_AREA# 小于MIN_AREA像素的区域自动过滤
)

print("模型加载完成！")

# ====================== 4. 创建输出目录 ======================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

subfolders = [
    'red_contour',
    'color_vis',
    'mask',
    'single_cells',
    'sam_masks',
    'filtered_masks'
]

for sf in subfolders:
    os.makedirs(
        os.path.join(OUTPUT_FOLDER, sf),
        exist_ok=True
    )

# ====================== 5. 获取图片 ======================

img_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if os.path.splitext(f)[-1].lower() in IMG_FORMATS
]

print(f"找到 {len(img_files)} 张图片")

# ====================== 6. 单细胞裁剪函数 ======================

def crop_and_resize_cell(
        image,
        mask,
        output_size=256,
        margin_ratio=0.3):
    """
    
    output_size : int
        输出尺寸
    margin_ratio : float
        裁剪边缘扩张比例，例如0.3表示在细胞边界基础上增加30%的边距
    Returns
    -------
    crop : ndarray
        裁剪后的单细胞图像
    """
    ys, xs = np.where(mask)# 找到mask所有前景像素坐标

    if len(xs) == 0:
        return None

    x_min = xs.min() # 获取mask外接矩形
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    w = x_max - x_min + 1 # 外接矩形宽高
    h = y_max - y_min + 1

    side = max(w, h)

    crop_size = int(side * (1 + margin_ratio))# 增加额外边缘

    cx = (x_min + x_max) // 2# 当前mask中心
    cy = (y_min + y_max) // 2

    half = crop_size // 2

    x1 = cx - half
    x2 = cx + half

    y1 = cy - half
    y2 = cy + half

    H, W = image.shape[:2]

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)

    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    x1 = max(0, x1)# 限制裁剪框在图像内部
    y1 = max(0, y1)

    x2 = min(W, x2)
    y2 = min(H, y2)

    raw_crop = image[y1:y2, x1:x2]

    raw_crop = cv2.copyMakeBorder(
        raw_crop,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    raw_crop = cv2.resize(
        raw_crop,
        (output_size, output_size),
        interpolation=cv2.INTER_CUBIC
    )

    

    # 只保留当前细胞
    #cell_img = image.copy()
    #cell_img[~mask] = 0

    #crop = cell_img[y1:y2, x1:x2]
    #crop = image[y1:y2, x1:x2]

    # ======================
    # 圆形保留区域
    # ======================

    area = np.sum(mask) # 当前mask面积

    equivalent_radius = np.sqrt(area / np.pi)
    # ======================================================
    # 等效圆半径
    # r = sqrt(area/pi)
    # 假设mask面积对应一个圆
    # 求出这个圆的半径
    # ======================================================
    circle_radius = int(equivalent_radius * 1.8)

    circle_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )   # 创建全黑mask

    cv2.circle(
        circle_mask,
        (cx, cy),
        circle_radius,
        255,
        -1
    )# 在细胞中心画圆

    masked_img = image.copy()

    masked_img[circle_mask == 0] = 0

    crop = masked_img[y1:y2, x1:x2]

    crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    crop = cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=cv2.INTER_CUBIC
    )

    return crop,raw_crop

# ====================== 7. mask过滤 ======================

def is_valid_mask(mask_dict, image,global_otsu, return_reason=False):
    """判断当前SAM mask是否有效；return_reason=True时返回筛选理由和指标。"""
    info = {}

    def finish(valid, reason):
        if return_reason:
            return valid, reason, info
        return valid

    H, W = image.shape[:2]

    area = mask_dict['area']
    info['area'] = int(area)

    if area < MIN_AREA:
        return finish(False, f"area {area} < MIN_AREA {MIN_AREA}")

    #if area > MAX_AREA:
        #return finish(False, f"area {area} > MAX_AREA {MAX_AREA}")

    mask = mask_dict['segmentation']

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return finish(False, "empty mask")

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()
    info['bbox'] = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

    #过滤贴边目标
    if SKIP_BORDER_OBJECT:

        if (
            x_min <= 0 or
            y_min <= 0 or
            x_max >= W - 1 or
            y_max >= H - 1
        ):
            return finish(False, f"border object bbox={info['bbox']}")

    pred_iou = mask_dict['predicted_iou']
    info['pred_iou'] = float(pred_iou)

    if pred_iou < 0.88:
        return finish(False, f"predicted_iou {pred_iou:.3f} < 0.88")

    stability = mask_dict['stability_score']
    info['stability'] = float(stability)

    if stability < 0.92:
        return finish(False, f"stability_score {stability:.3f} < 0.92")

    #颜色
    if FILTER_BY_COLOR:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        gray_vals = gray[mask]

        nucleus_gray_thresh = global_otsu * NUCLEUS_GRAY_SCALE
        dark_ratio = (
            gray_vals < global_otsu
        ).mean()

        info['global_otsu'] = float(global_otsu)
        info['nucleus_gray_thresh'] = float(nucleus_gray_thresh)
        info['mean_gray'] = float(gray_vals.mean()) if len(gray_vals) else 0.0
        info['dark_ratio'] = float(dark_ratio)

        if dark_ratio < MIN_NUCLEUS_DARK_RATIO:
            return finish(
                False,
                f"dark_ratio {dark_ratio:.4f} < MIN_NUCLEUS_DARK_RATIO {MIN_NUCLEUS_DARK_RATIO}"
            )

    return finish(True, "pass basic/quality/color filters")

def filter_masks(masks, image,global_otsu, process_log=None):
    """统一筛选mask；process_log不为None时记录每个mask为什么保留/删除。"""
    valid_masks = []

    if process_log is not None:
        process_log.append("\n========== Stage 1: basic / quality / color filter ==========")

    for raw_idx, mask_dict in enumerate(masks):

        valid, reason, info = is_valid_mask(
            mask_dict,
            image,
            global_otsu,
            return_reason=True
        )

        if process_log is not None:
            fields = [
                f"raw_mask={raw_idx}",
                f"decision={'KEEP' if valid else 'DROP'}",
                f"reason={reason}",
                f"area={info.get('area')}",
            ]
            if 'pred_iou' in info:
                fields.append(f"pred_iou={info['pred_iou']:.3f}")
            if 'stability' in info:
                fields.append(f"stability={info['stability']:.3f}")
            if 'mean_gray' in info:
                fields.append(f"mean_gray={info['mean_gray']:.2f}")
            if 'dark_ratio' in info:
                fields.append(f"dark_ratio={info['dark_ratio']:.4f}")
            if 'bbox' in info:
                fields.append(f"bbox={info['bbox']}")

            process_log.append(" | ".join(fields))

        if valid:
            valid_masks.append(mask_dict)

    if process_log is not None:
        process_log.append(f"Stage 1 result: {len(valid_masks)} / {len(masks)} masks kept")

    return valid_masks

# ==========================================================
# 删除被包含的小mask
# ==========================================================
import numpy as np

def filter_masks_by_containment(
        masks,
        image,
        overlap_thresh=0.95,
        area_sum_ratio_thresh=0.7,
        single_area_ratio_thresh=0.3,
        single_mask_gray_diff_thresh=0.08,
        multi_mask_gray_diff_thresh=50,
        process_log=None):
    """
    规则：
    1. 如果一个mask只包含1个mask：
       小mask/大mask面积比 > 70% 时保留大mask；
       否则保留小mask，并将大mask减去小mask的差集作为新mask。

    2. 如果一个mask包含多个mask：
       小mask面积和接近大mask时，若小mask平均灰度差 > 0.02保留大mask；
       否则删除大mask，保留所有小mask

    Parameters
    ----------
    masks : list
        SAM输出mask列表

    overlap_thresh : float
        判定包含关系阈值

    single_area_ratio_thresh : float
        只包含一个mask时的小/大mask面积比阈值

    single_mask_gray_diff_thresh : float
        只包含一个mask时，小mask与大mask差集的平均灰度差阈值（归一化到0–1）。
        超过该阈值时认为两部分颜色差异明显，只保留颜色更深的部分。

    multi_mask_gray_diff_thresh : float
        包含多个mask时，各小mask平均灰度的最大差值阈值（归一化到0–1）

    Returns
    -------
    keep_masks : list
    """

    n = len(masks)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    remove = set()
    new_masks = []

    if process_log is not None:
        process_log.append("\n========== Stage 2: containment filter ==========")
        process_log.append(f"Stage 2 input: {n} masks")

    # 面积从大到小排序
    order = sorted(
        range(n),
        key=lambda i: masks[i]['area'],
        reverse=True
    )

    for i in order:

        if i in remove:
            continue

        big_mask = masks[i]['segmentation']

        contained = []

        for j in range(n):

            if i == j:
                continue

            small_mask = masks[j]['segmentation']

            inter = np.logical_and(
                big_mask,
                small_mask
            ).sum()#两个 mask 重叠区域的像素数量

            small_area = masks[j]['area']

            if small_area == 0:
                continue

            # 小mask有多少比例落在大mask内
            overlap_ratio = inter / small_area

            if overlap_ratio > overlap_thresh:

                # 确保面积更小
                if masks[j]['area'] < masks[i]['area']:
                    contained.append(j)

        # ==========================
        # 情况1：只包含一个mask
        # 面积比 > 30% 保留大mask，否则保留小mask和大mask的差集
        # ==========================
        if len(contained) == 1:
            small_idx = contained[0]
            big_area = masks[i]['area']
            small_area = masks[small_idx]['area']
            area_ratio = small_area / big_area if big_area > 0 else 0

            small_mask = masks[small_idx]['segmentation']
            difference_mask = np.logical_and(big_mask, np.logical_not(small_mask))
            difference_area = int(difference_mask.sum())

            if area_ratio > single_area_ratio_thresh:
                remove.add(small_idx)
                if process_log is not None:
                    process_log.append(
                        f"stage2_mask={i} contains one mask {small_idx}; "
                        f"area_ratio={area_ratio:.4f} > {single_area_ratio_thresh}; "
                        f"keep big mask {i}, remove small mask {small_idx}"
                    )
            else:
                # 原大mask由“小mask + 差集mask”替代
                remove.add(i)
                difference_mask_dict = masks[i].copy()
                difference_mask_dict['segmentation'] = difference_mask
                difference_mask_dict['area'] = difference_area

                ys, xs = np.where(difference_mask)
                if len(xs) > 0:
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    difference_mask_dict['bbox'] = [
                        x_min,
                        y_min,
                        x_max - x_min + 1,
                        y_max - y_min + 1,
                    ]

                new_masks.append(difference_mask_dict)
                if process_log is not None:
                    process_log.append(
                        f"stage2_mask={i} contains one mask {small_idx}; "
                        f"area_ratio={area_ratio:.4f} <= {single_area_ratio_thresh}; "
                        f"remove big mask {i}, keep small mask {small_idx}, add difference mask area={difference_area}"
                    )

        # ==========================
        # 情况2：包含多个mask
        # 删除大mask
        # ==========================
        elif len(contained) > 1:
             # 所有被包含mask面积之和
            contained_area_sum = sum(
                masks[idx]['area']
                for idx in contained
            )

            big_area = masks[i]['area']

    # ----------------------------------
    # 小mask面积和 < 大mask面积的80%
    # 说明大mask明显包含更多区域
    # 保留大mask，删除小mask
    # ----------------------------------
            if contained_area_sum < area_sum_ratio_thresh * big_area:

                for idx in contained:
                    remove.add(idx)
                if process_log is not None:
                    process_log.append(
                        f"stage2_mask={i} contains multiple masks {contained}; "
                        f"contained_area_sum={contained_area_sum} < {area_sum_ratio_thresh}*{big_area}; "
                        f"keep big mask {i}, remove contained masks {contained}"
                    )

    # ----------------------------------
    # 小mask面积和 ≈ 大mask面积
    # 说明大mask基本就是多个mask拼起来的
    # 删除大mask
    # ----------------------------------
            else:
                small_mask_gray_means = [
                    gray[masks[idx]['segmentation']].mean()
                    for idx in contained
                    if masks[idx]['segmentation'].any()
                ]
                gray_diff = (
                    max(small_mask_gray_means) - min(small_mask_gray_means)
                    if len(small_mask_gray_means) > 1 else 0.0
                )

                # 灰度差异明显时，小mask可能属于不同目标，保留大mask并删除其包含的小mask。
                if gray_diff > multi_mask_gray_diff_thresh:
                    for idx in contained:
                        remove.add(idx)
                    if process_log is not None:
                        process_log.append(
                            f"stage2_mask={i} contains multiple masks {contained}; "
                            f"gray_diff={gray_diff:.4f} > {multi_mask_gray_diff_thresh}; "
                            f"keep big mask {i}, remove contained masks {contained}"
                        )
                else:
                    remove.add(i)
                    if process_log is not None:
                        process_log.append(
                            f"stage2_mask={i} contains multiple masks {contained}; "
                            f"gray_diff={gray_diff:.4f} <= {multi_mask_gray_diff_thresh}; "
                            f"remove big mask {i}, keep contained masks"
                        )
            

    keep_masks = [
        masks[i]
        for i in range(n)
        if i not in remove
    ]

    keep_masks.extend(new_masks)

    if process_log is not None:
        process_log.append(f"Stage 2 removed mask indices: {sorted(remove)}")
        process_log.append(f"Stage 2 added difference masks: {len(new_masks)}")
        process_log.append(f"Stage 2 result: {len(keep_masks)} masks kept")

    return keep_masks


#==========================保存轮廓图===============================
def save_contour_vis(
    image,
    masks,
    save_path
):

    vis = image.copy()

    for idx, mask_dict in enumerate(masks):

        mask = mask_dict[
            'segmentation'
        ].astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            vis,
            contours,
            -1,
            (0,0,255),
            2
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        cx = int(xs.mean())
        cy = int(ys.mean())

        cv2.putText(
            vis,
            str(idx),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,0),
            1
        )

    cv2.imwrite(
        save_path,
        vis
    )
    print("轮廓图已保存", save_path)
# ====================== 8. 处理单张图片 ======================

def process_single_image(img_path, save_name):

    #image = cv2.imread(img_path)
    image_original = cv2.imread(img_path)

    if image_original is None:
        return

    image = cv2.resize(
        image_original,
        None,
        fx=DOWNSAMPLE,
        fy=DOWNSAMPLE,
        interpolation=cv2.INTER_AREA
    )

    if image is None:
        return

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    H, W = image.shape[:2]

    tic = time.time()

    process_log = [
        f"image={img_path}",
        f"save_name={save_name}",
        f"DOWNSAMPLE={DOWNSAMPLE}",
        f"MIN_AREA={MIN_AREA}",
        f"MAX_AREA={MAX_AREA}",
        f"SKIP_BORDER_OBJECT={SKIP_BORDER_OBJECT}",
        f"FILTER_BY_COLOR={FILTER_BY_COLOR}",
        f"NUCLEUS_GRAY_SCALE={NUCLEUS_GRAY_SCALE}",
        f"MIN_NUCLEUS_DARK_RATIO={MIN_NUCLEUS_DARK_RATIO}",
    ]

    mask_file = os.path.join(
    OUTPUT_FOLDER,
    "sam_masks",
    f"{save_name}.npz"
    )

    filtered_mask_file = os.path.join(
        OUTPUT_FOLDER,
        "filtered_masks",
        f"{save_name}.npz"
    )

    if os.path.exists(mask_file):

        print(
            f"{save_name}: "
            "加载已有mask"
        )

        data = np.load(
            mask_file,
            allow_pickle=True
        )

        masks = list(
        data["masks"]
        )

    else:

        print(
            f"{save_name}: "
            "SAM生成mask"
        )

        masks = mask_generator.generate(
            image_rgb
        )

        np.savez_compressed(
            mask_file,
            masks=np.array(
                masks,
                dtype=object
            )
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 只使用所有SAM mask覆盖到的区域计算Otsu阈值，避免大面积背景影响颜色筛选。
    all_mask = np.zeros(gray.shape, dtype=bool)

    for mask_dict in masks:
        all_mask |= mask_dict['segmentation'].astype(bool)

    mask_pixel_count = int(all_mask.sum())
    total_pixel_count = int(all_mask.size)
    mask_coverage = mask_pixel_count / total_pixel_count if total_pixel_count > 0 else 0.0

    if all_mask.any():
        masked_gray_vals = gray[all_mask].astype(np.uint8)
        global_otsu = cv2.threshold(
            masked_gray_vals.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[0]
        print("Mask-region Otsu:", global_otsu)
        process_log.append(f"Mask-region Otsu={global_otsu}")
        process_log.append(
            f"Otsu detail: mask_pixel_count={mask_pixel_count}, "
            f"total_pixel_count={total_pixel_count}, mask_coverage={mask_coverage:.4f}, "
            f"reason=exclude non-mask background pixels from threshold calculation"
        )
    else:
        global_otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[0]
        print("Global Otsu fallback:", global_otsu)
        process_log.append(f"Global Otsu fallback={global_otsu}")
        process_log.append(
            f"Otsu detail: mask_pixel_count=0, reason=no mask pixels available"
        )

    raw_masks = masks.copy()
    save_contour_vis(
        image,
        raw_masks,
        os.path.join(
            OUTPUT_FOLDER,
            "debug",
            f"{save_name}_01_raw.png"
        )
    )


    raw_num = len(masks)
    
    print(
        f"{save_name}: "
        f"{raw_num} raw masks, "
        f"time={time.time()-tic:.2f}s"
    )

    if FILTER:
        #筛掉红细胞
        valid_masks = filter_masks(
            masks,
            image,
            global_otsu,
            process_log=process_log
        )
        save_contour_vis(
            image,
            valid_masks,
            os.path.join(
                OUTPUT_FOLDER,
                "debug",
                f"{save_name}_02_filtered.png"
            )
        )

        print(
            f"{save_name}: "
            f"筛掉红细胞与贴边细胞后保留 {len(valid_masks)} masks  "
            f"time={time.time()-tic:.2f}s"
        )

        valid_masks=filter_masks_by_containment(
            valid_masks,
            image,
            process_log=process_log
        )

        save_contour_vis(
            image,
            valid_masks,
            os.path.join(
                OUTPUT_FOLDER,
                "debug",
                f"{save_name}_03_contained.png"
            )
        )
        print(
            f"{save_name}: "
            f"筛掉错误分割细胞后保留 {len(valid_masks)} masks  "
            f"time={time.time()-tic:.2f}s"
        )
        
    else:
        valid_masks = masks
        print("未使用过滤条件，保留全部mask")

    np.savez_compressed(
        filtered_mask_file,
        masks=np.array(
            valid_masks,
            dtype=object
        )
    )

    process_log.append("\n========== Final result ==========")
    process_log.append(f"raw_masks={len(masks)}")
    process_log.append(f"final_valid_masks={len(valid_masks)}")

    process_log_path = os.path.join(
        OUTPUT_FOLDER,
        "debug",
        f"{save_name}_mask_process_log.txt"
    )
    with open(process_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(process_log))

    print("mask处理日志已保存", process_log_path)

    

    # ======================
    # 彩色可视化
    # ======================

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    def show_anns(anns):

        if len(anns) == 0:
            return

        anns = sorted(
            anns,
            key=lambda x: x['area'],
            reverse=True
        )

        ax = plt.gca()

        ax.set_autoscale_on(False)

        img = np.ones(
            (
                anns[0]['segmentation'].shape[0],
                anns[0]['segmentation'].shape[1],
                4
            )
        )

        img[:, :, 3] = 0

        for ann in anns:

            m = ann['segmentation']

            color_mask = np.concatenate(
                [np.random.random(3), [0.5]]
            )

            img[m] = color_mask

        ax.imshow(img)

    show_anns(valid_masks)

    plt.axis('off')

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            'color_vis',
            f'{save_name}.png'
        ),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close()

    # ======================
    # 红色轮廓
    # ======================

    result_img = image.copy()

    cell_id = 0

    for mask_dict in valid_masks:

        mask = mask_dict['segmentation'].astype(
            np.uint8
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            result_img,
            contours,
            -1,
            (0, 0, 255),
            2
        )

        ys, xs = np.where(mask)

        cx = int(xs.mean())
        cy = int(ys.mean())

        cv2.putText(
            result_img,
            str(cell_id),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        cell_id += 1




    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            'red_contour',
            f'{save_name}.png'
        ),
        result_img
    )

    # ======================
    # 合并mask
    # ======================

    instance_mask = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    for idx, mask_dict in enumerate(valid_masks):

        mask = mask_dict['segmentation']

        instance_mask[mask] = idx + 1

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            'mask',
            f'{save_name}.png'
        ),
        instance_mask
    )

    # ======================
    # 单细胞保存
    # ======================

    cell_folder = os.path.join(
        OUTPUT_FOLDER,
        'single_cells',
        save_name
    )

    os.makedirs(
        cell_folder,
        exist_ok=True
    )

    saved_num = 0

    for idx, mask_dict in enumerate(valid_masks):
        """
        mask = mask_dict[
            'segmentation'
        ]
        """
        #还原maskk
        mask = cv2.resize(
            mask_dict["segmentation"].astype(np.uint8),
            (
                image_original.shape[1],
                image_original.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        crop, raw_crop = crop_and_resize_cell(
            image=image_original,#image
            mask=mask,
            output_size=OUTPUT_SIZE,
            margin_ratio=MARGIN_RATIO
        )

        if crop is None:
            continue

        cv2.imwrite(
            os.path.join(
                cell_folder,
                f'cell_{idx:04d}.png'
            ),
            crop
        )
        
        cv2.imwrite(
            os.path.join(
                cell_folder,
                f'cell_{idx:04d}_raw.png'
            ),
            raw_crop
        )

        saved_num += 1

    print(
        f"{save_name}: "
        f"保存 {saved_num} 个细胞"
    )

# ====================== 9. 批量执行 ======================

for img_file in img_files:

    img_path = os.path.join(
        INPUT_FOLDER,
        img_file
    )

    save_name = os.path.splitext(
        img_file
    )[0]

    process_single_image(
        img_path,
        save_name
    )

print("\n全部处理完成！")
print(f"结果保存在: {OUTPUT_FOLDER}")