import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import SamPredictor, sam_model_registry
import os
import time
import sys
import os
import contextlib

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

OUTPUT_FOLDER = "/home/zhao_ziyi/program/segment-anything-main/sam_results_point_prompt"

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
    'filtered_masks',
    'debug'
]

# ====================== 2. 单细胞参数 ======================

OUTPUT_SIZE = 256       # 最终输出尺寸
MARGIN_RATIO = 0.25     # 细胞周围留25%边距

DOWNSAMPLE = 1 # SAM和候选点检测都在降采样图上运行

# ====================== 点提示高速流程参数 ======================
USE_POINT_PROMPT = True
CENTER_METHOD = "hybrid"       # hybrid: 连通域主检测 + 多尺度峰值补点
CENTER_REFERENCE_DOWNSAMPLE = 1 / 8  # 下列检测参数是在该尺度下定义的
POINT_MIN_DISTANCE = 6         # 参考尺度上的候选点最小距离
POINT_BORDER_MARGIN = 2        # 参考尺度上的边缘距离
POINT_MIN_COMPONENT_AREA = 6   # 参考尺度上的连通域最小面积
CENTER_FOREGROUND_SIGMA = 1.0  # 参考尺度上的前景平滑sigma
CENTER_MORPH_KERNEL = 3        # 参考尺度上的形态学核
CENTER_MULTISCALE_SIGMAS = (1.2, 2.0, 3.2)
POINT_MAX_COMPONENT_RATIO = 0.12
POINT_MAX_CANDIDATES = 128     # 防止异常图像产生过多decoder调用
POINT_BATCH_SIZE = 32          # decoder批大小；显存不足时调小
POINT_USE_MULTISCALE = True    # 浅染白细胞漏检时用多尺度局部峰值补点
POINT_MASK_NMS_THRESH = 0.90   # 删除不同点产生的近重复mask
POINT_DEBUG = True
USE_AMP = True
WARMUP_SAM = True             # 模型加载后预热一次，避免首张图承担CUDA初始化
MASK_CACHE_VERSION = 3         # 缓存结构/生成流程版本


def amp_autocast(enabled):
    """兼容新旧PyTorch的CUDA FP16上下文。"""
    if not enabled:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(enabled=True)

MIN_AREA = int(3500 * DOWNSAMPLE * DOWNSAMPLE)          # 过滤小碎片
MAX_AREA = int(30000 * DOWNSAMPLE * DOWNSAMPLE)   # 过滤太大的
MAX_AREA_RATIO = 0.5  # 超过整张图该比例的mask不参与颜色阈值计算，也会被过滤

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

BASE_CHECKPOINT = "/home/zhao_ziyi/program/segment-anything-main/sam_vit_b_01ec64.pth"
FINETUNE_CHECKPOINT = "/home/zhao_ziyi/program/segment-anything-main/sam_finetune_outputs/last_decoder_finetune.pth"

sam = sam_model_registry["vit_b"](
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
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
sam.to(device)

sam.eval()
predictor = SamPredictor(sam)

# 仅在加载阶段预热一次，避免第一张图的分割计时包含CUDA上下文/算子初始化。
if WARMUP_SAM and device == "cuda":
    warmup_tic = time.perf_counter()
    warmup_image = np.zeros((64, 64, 3), dtype=np.uint8)
    with torch.no_grad(), amp_autocast(USE_AMP):
        predictor.set_image(warmup_image)
        predictor.predict(
            point_coords=np.array([[32, 32]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
    torch.cuda.synchronize()
    predictor.reset_image()
    print(f"SAM GPU warm-up: {time.perf_counter() - warmup_tic:.3f}s")

print("模型加载完成！")

# ====================== 4. 创建输出目录 ======================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

subfolders = [
    'red_contour',
    'color_vis',
    'mask',
    'single_cells',
    'sam_masks',
    'filtered_masks',
    'debug'
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

    max_area_by_ratio = int(H * W * MAX_AREA_RATIO)
    info['max_area_by_ratio'] = max_area_by_ratio

    if area > max_area_by_ratio:
        return finish(False, f"area {area} > image_area*MAX_AREA_RATIO {max_area_by_ratio}")

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
    '''
    pred_iou = mask_dict['predicted_iou']
    info['pred_iou'] = float(pred_iou)

    if pred_iou < 0.88:
        return finish(False, f"predicted_iou {pred_iou:.3f} < 0.88")

    stability = mask_dict['stability_score']
    info['stability'] = float(stability)

    if stability < 0.92:
        return finish(False, f"stability_score {stability:.3f} < 0.92")
    '''
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
        overlap_thresh=0.9,
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
# ====================== 8. 候选中心与点提示SAM ======================

def _normalize_u8(feature):
    """用百分位数自适应归一化，避免固定颜色阈值。"""
    low, high = np.percentile(feature, (2.0, 98.0))
    if high <= low + 1e-6:
        return np.zeros(feature.shape, dtype=np.uint8)
    return np.clip((feature - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _nms_points(candidates, min_distance, max_points):
    """按显著性从高到低做距离抑制。candidates元素为(score, x, y)。"""
    selected = []
    min_distance_sq = float(min_distance * min_distance)
    for score, x, y in sorted(candidates, reverse=True):
        if all((x - px) ** 2 + (y - py) ** 2 >= min_distance_sq
               for px, py in selected):
            selected.append((int(x), int(y)))
            if len(selected) >= max_points:
                break
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def _scaled_odd_kernel(base_size, linear_scale):
    size = max(3, int(round(base_size * linear_scale)))
    return size if size % 2 == 1 else size + 1


def get_center_detector_params():
    """把1/8参考尺度参数换算到当前DOWNSAMPLE对应的像素尺度。"""
    if DOWNSAMPLE <= 0 or CENTER_REFERENCE_DOWNSAMPLE <= 0:
        raise ValueError("DOWNSAMPLE和CENTER_REFERENCE_DOWNSAMPLE必须大于0")
    linear_scale = DOWNSAMPLE / CENTER_REFERENCE_DOWNSAMPLE
    return {
        "linear_scale": linear_scale,
        "foreground_sigma": max(0.5, CENTER_FOREGROUND_SIGMA * linear_scale),
        "morph_kernel": _scaled_odd_kernel(CENTER_MORPH_KERNEL, linear_scale),
        "min_distance": max(1.0, POINT_MIN_DISTANCE * linear_scale),
        "border_margin": max(1, int(round(POINT_BORDER_MARGIN * linear_scale))),
        "min_component_area": max(1, int(round(POINT_MIN_COMPONENT_AREA * linear_scale ** 2))),
        "multiscale_sigmas": tuple(max(0.5, sigma * linear_scale)
                                   for sigma in CENTER_MULTISCALE_SIGMAS),
    }


def detect_cell_centers(image_bgr):
    """
    尺度自适应的白细胞优先候选检测。
    暗度和饱和度通过图像自身百分位/Otsu自适应；所有以像素为单位的
    模糊、形态学、面积、边距及NMS参数随DOWNSAMPLE自动换算。
    """
    h, w = image_bgr.shape[:2]
    params = get_center_detector_params()
    foreground_sigma = params["foreground_sigma"]
    border_margin = params["border_margin"]
    min_component_area = params["min_component_area"]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(gray, (0, 0), foreground_sigma)

    # 背景尺度由图像尺寸决定，本身已经随输入分辨率变化。
    sigma_bg = max(3.0 * params["linear_scale"], min(h, w) / 24.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigma_bg)
    darkness = cv2.subtract(background, gray)
    dark_u8 = _normalize_u8(darkness.astype(np.float32))
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (0, 0), foreground_sigma)
    sat_u8 = _normalize_u8(saturation.astype(np.float32))
    saliency = cv2.addWeighted(dark_u8, 0.70, sat_u8, 0.30, 0)
    saliency = cv2.GaussianBlur(saliency, (0, 0), foreground_sigma)

    _, binary = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = params["morph_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidates = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_component_area = max(
        min_component_area + 1,
        int(h * w * POINT_MAX_COMPONENT_RATIO),
    )
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area or area > max_component_area:
            continue
        component = labels == label
        ys, xs = np.where(component)
        weights = saliency[ys, xs].astype(np.float32) + 1.0
        x = int(np.average(xs, weights=weights))
        y = int(np.average(ys, weights=weights))
        if border_margin <= x < w - border_margin and border_margin <= y < h - border_margin:
            # 连通域候选略加优先级，避免补点抢占候选上限。
            score = float(saliency[y, x]) + np.log1p(area) + 16.0
            candidates.append((score, x, y))

    # 多尺度局部极大值补充浅染或粘连目标；尺度与当前分辨率同步。
    if CENTER_METHOD.lower() in ("hybrid", "log") and POINT_USE_MULTISCALE:
        otsu_value = cv2.threshold(
            saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[0]
        for sigma in params["multiscale_sigmas"]:
            response = cv2.GaussianBlur(saliency, (0, 0), sigma)
            window = _scaled_odd_kernel(4.0 * sigma, 1.0)
            local_max = response == cv2.dilate(
                response,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window)),
            )
            ys, xs = np.where(local_max & (response >= 0.75 * otsu_value))
            for x, y in zip(xs, ys):
                if border_margin <= x < w - border_margin and border_margin <= y < h - border_margin:
                    candidates.append((float(response[y, x]), int(x), int(y)))

    points = _nms_points(
        candidates,
        params["min_distance"],
        POINT_MAX_CANDIDATES,
    )
    return points, saliency, binary


def _mask_to_sam_dict(mask, score, point):
    mask = mask.astype(bool)
    area = int(mask.sum())
    ys, xs = np.where(mask)
    bbox = [0, 0, 0, 0] if area == 0 else [
        int(xs.min()), int(ys.min()),
        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    ]
    return {
        "segmentation": mask,
        "area": area,
        "bbox": bbox,
        "predicted_iou": float(score),
        "point_coords": [point.tolist()],
        "point_coords_original": [(point / DOWNSAMPLE).tolist()],
    }


def _deduplicate_prompt_masks(masks, iou_thresh):
    """模拟AutomaticMaskGenerator的跨提示NMS，优先保留predicted_iou高的mask。"""
    kept = []
    for item in sorted(masks, key=lambda x: x["predicted_iou"], reverse=True):
        mask = item["segmentation"]
        duplicate = False
        for old in kept:
            inter = np.logical_and(mask, old["segmentation"]).sum()
            union = item["area"] + old["area"] - inter
            if union > 0 and inter / union >= iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def generate_point_prompt_masks(image_rgb, points):
    """一次encoder，点提示分批并行decoder；输出兼容AutomaticMaskGenerator。"""
    timings = {"encoder": 0.0, "prediction": 0.0}
    if len(points) == 0:
        return [], timings

    use_amp = bool(USE_AMP and device == "cuda")
    if device == "cuda":
        torch.cuda.synchronize()
    encoder_tic = time.perf_counter()
    with torch.no_grad(), amp_autocast(use_amp):
        predictor.set_image(image_rgb)
    if device == "cuda":
        torch.cuda.synchronize()
    timings["encoder"] = time.perf_counter() - encoder_tic

    prediction_tic = time.perf_counter()
    output = []
    with torch.no_grad(), amp_autocast(use_amp):
        for start in range(0, len(points), POINT_BATCH_SIZE):
            batch_np = points[start:start + POINT_BATCH_SIZE]
            coords = torch.as_tensor(batch_np[:, None, :], dtype=torch.float32, device=device)
            coords = predictor.transform.apply_coords_torch(coords, image_rgb.shape[:2])
            labels = torch.ones((coords.shape[0], 1), dtype=torch.int64, device=device)
            batch_masks, batch_scores, _ = predictor.predict_torch(
                point_coords=coords,
                point_labels=labels,
                boxes=None,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
            )
            best = batch_scores.argmax(dim=1)
            rows = torch.arange(batch_scores.shape[0], device=device)
            best_masks = batch_masks[rows, best].detach().cpu().numpy()
            best_scores = batch_scores[rows, best].detach().float().cpu().numpy()
            for local_idx, (mask, score) in enumerate(zip(best_masks, best_scores)):
                output.append(_mask_to_sam_dict(mask, score, batch_np[local_idx]))
    if device == "cuda":
        torch.cuda.synchronize()
    timings["prediction"] = time.perf_counter() - prediction_tic
    return _deduplicate_prompt_masks(output, POINT_MASK_NMS_THRESH), timings


def save_center_debug(image, points, saliency, binary, save_name):
    if not POINT_DEBUG:
        return
    vis = image.copy()
    for idx, (x, y) in enumerate(points.astype(int)):
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(vis, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "debug", f"{save_name}_00_centers.png"), vis)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "debug", f"{save_name}_00_saliency.png"), saliency)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "debug", f"{save_name}_00_binary.png"), binary)


# ====================== 9. 处理单张图片 ======================


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
        f"MAX_AREA_RATIO={MAX_AREA_RATIO}",
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

    center_detection_time = 0.0
    encoder_time = 0.0
    prediction_time = 0.0
    cache_valid = False
    masks = []

    if os.path.exists(mask_file):
        try:
            data = np.load(mask_file, allow_pickle=True)
            cached_masks = list(data["masks"])
            cached_shapes = {
                tuple(mask_dict["segmentation"].shape)
                for mask_dict in cached_masks
            }
            shape_matches = not cached_masks or cached_shapes == {(H, W)}
            version_matches = (
                "cache_version" in data.files and
                int(np.asarray(data["cache_version"]).item()) == MASK_CACHE_VERSION
            )
            if shape_matches and version_matches:
                masks = cached_masks
                cache_valid = True
                mask_source = "cache"
                print(f"{save_name}: 加载已有点提示mask")
                process_log.append(f"mask_source={mask_source}")
                process_log.append("timing=cache_reused")
            else:
                print(
                    f"{save_name}: 缓存失效，当前图像shape={(H, W)}，"
                    f"缓存mask shape={sorted(cached_shapes)}，将重新生成"
                )
                process_log.append(
                    f"cache_invalid: image_shape={(H, W)}, "
                    f"mask_shapes={sorted(cached_shapes)}, "
                    f"version_matches={version_matches}"
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"{save_name}: mask缓存读取失败，将重新生成: {exc}")
            process_log.append(f"cache_read_failed={exc!r}")

    if not cache_valid:
        mask_source = "point_prompt"
        center_tic = time.perf_counter()
        points, center_saliency, center_binary = detect_cell_centers(image)
        center_detection_time = time.perf_counter() - center_tic
        save_center_debug(image, points, center_saliency, center_binary, save_name)
        print(f"{save_name}: 检测到 {len(points)} 个候选中心")

        masks, sam_timings = generate_point_prompt_masks(image_rgb, points)
        encoder_time = sam_timings["encoder"]
        prediction_time = sam_timings["prediction"]
        process_log.extend([
            f"mask_source={mask_source}",
            f"center_method={CENTER_METHOD}",
            f"center_detector_params={get_center_detector_params()}",
            f"candidate_points={len(points)}",
            f"center_detection_time={center_detection_time:.4f}s",
            f"sam_encoder_time={encoder_time:.4f}s",
            f"sam_prediction_time={prediction_time:.4f}s",
        ])
        np.savez_compressed(
            mask_file,
            masks=np.array(masks, dtype=object),
            image_shape=np.asarray([H, W], dtype=np.int32),
            downsample=np.asarray(DOWNSAMPLE, dtype=np.float32),
            cache_version=np.asarray(MASK_CACHE_VERSION, dtype=np.int32),
        )

    post_tic = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 先排除接近整图的巨大mask，再计算Otsu阈值，避免颜色阈值被异常大mask带偏。
    max_area_for_otsu = int(H * W * MAX_AREA_RATIO)
    masks_for_otsu = [
        mask_dict
        for mask_dict in masks
        if mask_dict['area'] <= max_area_for_otsu
    ]
    removed_large_for_otsu = len(masks) - len(masks_for_otsu)

    all_mask = np.zeros(gray.shape, dtype=bool)

    for mask_dict in masks_for_otsu:
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
            f"removed_large_for_otsu={removed_large_for_otsu}, "
            f"max_area_for_otsu={max_area_for_otsu}, "
            f"reason=exclude non-mask background pixels and oversized masks from threshold calculation"
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
            f"Otsu detail: mask_pixel_count=0, "
            f"removed_large_for_otsu={removed_large_for_otsu}, "
            f"max_area_for_otsu={max_area_for_otsu}, "
            f"reason=no mask pixels available after oversized-mask prefilter"
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

    post_processing_time = time.perf_counter() - post_tic
    total_time = time.time() - tic
    process_log.append("\n========== Final result ==========")
    process_log.append(f"mask_source={mask_source}")
    process_log.append(f"center_detection_time={center_detection_time:.4f}s")
    process_log.append(f"sam_encoder_time={encoder_time:.4f}s")
    process_log.append(f"sam_prediction_time={prediction_time:.4f}s")
    process_log.append(f"post_processing_time={post_processing_time:.4f}s")
    process_log.append(f"total_time={total_time:.4f}s")
    process_log.append(f"raw_masks={len(masks)}")
    process_log.append(f"final_valid_masks={len(valid_masks)}")
    print(f"{save_name} timing:")
    print(f"  center detection: {center_detection_time:.3f}s")
    print(f"  SAM encoder:      {encoder_time:.3f}s")
    print(f"  SAM prediction:   {prediction_time:.3f}s")
    print(f"  post processing:  {post_processing_time:.3f}s")
    print(f"  total:            {total_time:.3f}s")

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

# ====================== 10. 批量执行 ======================

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