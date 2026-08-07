import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "未找到 ultralytics。请先在 sam 环境中安装：\n"
        "/home/zhao_ziyi/.conda/envs/sam/bin/python -m pip install ultralytics"
    ) from exc

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:
    raise SystemExit("未找到 segment_anything，请在 segment-anything-main 项目中运行本脚本。") from exc


IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_INPUT_FOLDER = "/home/zhao_ziyi/program/segment-anything-main/images"
DEFAULT_OUTPUT_FOLDER = "/home/zhao_ziyi/program/segment-anything-main/fastsam_refine_fliter_8downsample"
DEFAULT_MODEL = "/home/zhao_ziyi/program/segment-anything-main/FastSAM-s.pt"
DEFAULT_SAM_CHECKPOINT = "/home/zhao_ziyi/program/segment-anything-main/sam_vit_b_01ec64.pth"


def print_elapsed(step_name, start_time, timings=None, prefix=""):
    """输出步骤耗时，并按需保存供最终汇总。"""
    elapsed = time.perf_counter() - start_time
    if timings is not None:
        timings[step_name] = elapsed
    prefix = f"{prefix} | " if prefix else ""
    print(f"{prefix}{step_name}: {elapsed:.3f}s")
    return elapsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="FastSAM batch cell segmentation scaffold. 默认只检查配置，不执行推理。"
    )
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="FastSAM 权重路径，例如 FastSAM-s.pt")
    parser.add_argument("--device", default="0", help="GPU id，如 0；CPU 用 cpu")
    parser.add_argument("--imgsz", type=int, default=1024, help="降采样前对应的 FastSAM 输入尺寸")
    parser.add_argument(
        "--downsample", type=float, default=1/8,
        help="处理图像的宽高缩放比例，默认 0.25（即 1/4）",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，越低召回越多")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--retina-masks", action="store_true", help="保存更高分辨率 mask，较慢但边界更细")
    parser.add_argument("--save-npz", action="store_true", help="保存每张图的实例 mask npz")
    parser.add_argument("--run", action="store_true", help="显式执行推理；不加该参数只做 dry-run")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 张图，便于试跑")
    parser.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CHECKPOINT, help="用于补漏的 SAM 权重")
    parser.add_argument("--sam-model-type", default="vit_b", help="用于补漏的 SAM 类型，如 vit_b")
    parser.add_argument("--no-sam-refine", action="store_true", help="关闭深染漏检区域的 SAM Predictor 补分割")
    parser.add_argument("--dark-scale", type=float, default=0.75, help="深染阈值 = 灰度 Otsu * dark_scale")
    parser.add_argument("--purple-saturation-min", type=int, default=30, help="紫色/深染候选的最低饱和度")
    parser.add_argument("--miss-coverage-thresh", type=float, default=0.25, help="候选深染区域被 FastSAM 覆盖低于该比例时认为漏检")
    parser.add_argument("--min-missing-area", type=int, default=80, help="漏检深染连通域最小面积")
    parser.add_argument("--refine-box-expand", type=float, default=3.0, help="漏检候选框扩张倍数")
    parser.add_argument("--max-refine-boxes", type=int, default=80, help="每张图最多补分割的候选框数量")
    parser.add_argument("--refine-overlap-drop", type=float, default=0.75, help="补出的SAM mask与已有mask重叠超过该比例才丢弃")
    parser.add_argument("--no-filter", action="store_true", help="关闭与 test_mask_autother.py 相同的两阶段筛选")
    parser.add_argument("--min-area", type=int, default=3500, help="筛选保留的最小 mask 面积")
    parser.add_argument("--max-area-ratio", type=float, default=0.5, help="mask 最大面积占整图比例")
    parser.add_argument("--keep-border", action="store_true", help="保留接触图像边缘的目标")
    parser.add_argument("--min-nucleus-dark-ratio", type=float, default=0.05, help="mask 内低于 Otsu 阈值的像素最低占比")
    parser.add_argument("--containment-overlap", type=float, default=0.9, help="判定包含关系的重叠比例")
    parser.add_argument("--containment-area-sum-ratio", type=float, default=0.7, help="多个小 mask 的面积和阈值")
    parser.add_argument("--containment-single-area-ratio", type=float, default=0.3, help="单个小 mask 与大 mask 面积比阈值")
    parser.add_argument("--existing-mask-folder", default=None, help="只筛选已有实例 mask PNG；设置后不加载 FastSAM/SAM")
    return parser.parse_args()


def ensure_output_dirs(output_folder):
    for name in ["mask", "red_contour", "color_vis", "fastsam_masks", "filtered_masks", "debug"]:
        os.makedirs(os.path.join(output_folder, name), exist_ok=True)


def list_images(input_folder, limit=None):
    input_path = Path(input_folder)
    images = [
        p for p in sorted(input_path.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_FORMATS
    ]
    if limit is not None:
        images = images[:limit]
    return images



def load_sam_predictor(args):
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def make_union_mask(masks, height, width):
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    return union


def find_missing_dark_boxes(image, fastsam_masks, args):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]

    dark = gray < otsu * args.dark_scale
    purple = ((hue >= 120) & (hue <= 170) & (saturation >= args.purple_saturation_min))
    candidate = (dark | purple)

    height, width = image.shape[:2]
    covered = make_union_mask(fastsam_masks, height, width)
    missing_candidate = np.logical_and(candidate, np.logical_not(covered)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    missing_candidate = cv2.morphologyEx(missing_candidate, cv2.MORPH_OPEN, kernel)
    missing_candidate = cv2.morphologyEx(missing_candidate, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(missing_candidate, connectivity=8)

    boxes = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < args.min_missing_area:
            continue

        original_component = candidate[y:y + h, x:x + w]
        original_area = int(original_component.sum())
        missing_area = int(area)
        missing_ratio = missing_area / original_area if original_area > 0 else 0.0

        cx, cy = centroids[label_idx]
        side = max(w, h)
        min_expand_pixels = max(1, int(round(12 * args.downsample)))
        expanded = max(side * args.refine_box_expand, side + min_expand_pixels)
        half = expanded / 2.0
        x1 = max(0, int(round(cx - half)))
        y1 = max(0, int(round(cy - half)))
        x2 = min(width - 1, int(round(cx + half)))
        y2 = min(height - 1, int(round(cy + half)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, float(missing_ratio), int(missing_area), int(cx), int(cy)])

    boxes.sort(key=lambda item: item[5], reverse=True)
    return boxes[:args.max_refine_boxes]


def refine_missing_with_sam(predictor, image, fastsam_masks, args):
    boxes = find_missing_dark_boxes(image, fastsam_masks, args)
    if not boxes:
        return [], 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    box_tensor = torch.as_tensor([box[:4] for box in boxes], dtype=torch.float, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(box_tensor, image_rgb.shape[:2])
    point_coords = torch.as_tensor([[[box[6], box[7]]] for box in boxes], dtype=torch.float, device=predictor.device)
    transformed_points = predictor.transform.apply_coords_torch(point_coords, image_rgb.shape[:2])
    point_labels = torch.ones((len(boxes), 1), dtype=torch.int, device=predictor.device)

    with torch.inference_mode():
        pred_masks, scores, _ = predictor.predict_torch(
            point_coords=transformed_points,
            point_labels=point_labels,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    existing = make_union_mask(fastsam_masks, image.shape[0], image.shape[1])
    refined_masks = []
    for mask, score in zip(pred_masks[:, 0].detach().cpu().numpy().astype(bool), scores[:, 0].detach().cpu().numpy()):
        area = int(mask.sum())
        if area == 0:
            continue
        overlap = np.logical_and(mask, existing).sum() / area
        if overlap > args.refine_overlap_drop:
            continue
        refined_masks.append(mask)
        existing |= mask

    return refined_masks, len(boxes)

def masks_to_instance_mask(masks, height, width):
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    for idx, mask in enumerate(masks, start=1):
        instance_mask[mask.astype(bool)] = idx
    return instance_mask



def masks_to_dicts(masks):
    """构建筛选所需信息，复用已有布尔 mask，避免全尺寸数组重复复制。"""
    mask_dicts = []
    for mask in masks:
        segmentation = (
            mask if mask.dtype == np.bool_
            else mask.astype(bool, copy=False)
        )
        mask_dicts.append({
            "segmentation": segmentation,
            "area": int(np.count_nonzero(segmentation)),
        })
    return mask_dicts


def calculate_mask_region_otsu(image, mask_dicts, args, process_log):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_area = int(gray.size * args.max_area_ratio)
    usable = [item for item in mask_dicts if item["area"] <= max_area]
    union = np.zeros(gray.shape, dtype=bool)
    for item in usable:
        union |= item["segmentation"]
    values = gray[union] if union.any() else gray.reshape(-1)
    otsu = cv2.threshold(values.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    process_log.append(f"Otsu={otsu:.2f} | mask_pixels={int(union.sum())} | removed_oversized={len(mask_dicts)-len(usable)}")
    return otsu


def filter_masks_basic(mask_dicts, image, global_otsu, args, process_log):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_area = int(height * width * args.max_area_ratio)
    valid = []
    process_log.append("\n========== Stage 1: basic / color filter ==========")
    for idx, item in enumerate(mask_dicts):
        mask, area = item["segmentation"], item["area"]
        ys, xs = np.where(mask)
        keep, reason, dark_ratio, bbox = True, "pass", None, None
        if area < args.min_area:
            keep, reason = False, f"area {area} < {args.min_area}"
        elif area > max_area:
            keep, reason = False, f"area {area} > {max_area}"
        elif not len(xs):
            keep, reason = False, "empty mask"
        else:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
            border = xs.min() <= 0 or ys.min() <= 0 or xs.max() >= width-1 or ys.max() >= height-1
            if border and not args.keep_border:
                keep, reason = False, f"border object bbox={bbox}"
            else:
                dark_ratio = float((gray[mask] < global_otsu).mean())
                if dark_ratio < args.min_nucleus_dark_ratio:
                    keep, reason = False, f"dark_ratio {dark_ratio:.4f} < {args.min_nucleus_dark_ratio}"
        process_log.append(f"raw_mask={idx} | decision={'KEEP' if keep else 'DROP'} | reason={reason} | area={area} | dark_ratio={dark_ratio} | bbox={bbox}")
        if keep:
            valid.append(item)
    process_log.append(f"Stage 1 result: {len(valid)} / {len(mask_dicts)} masks kept")
    return valid


def filter_masks_by_containment(mask_dicts, image, args, process_log):
    n, remove, additions = len(mask_dicts), set(), []
    process_log.append("\n========== Stage 2: containment filter ==========")
    for i in sorted(range(n), key=lambda k: mask_dicts[k]["area"], reverse=True):
        if i in remove:
            continue
        big = mask_dicts[i]["segmentation"]
        contained = [j for j in range(n) if j != i and mask_dicts[j]["area"] < mask_dicts[i]["area"] and np.logical_and(big, mask_dicts[j]["segmentation"]).sum()/max(mask_dicts[j]["area"], 1) > args.containment_overlap]
        if len(contained) == 1:
            j = contained[0]
            ratio = mask_dicts[j]["area"] / mask_dicts[i]["area"]
            if ratio > args.containment_single_area_ratio:
                remove.add(j)
                process_log.append(f"mask={i} contains {j}; keep big, remove small; ratio={ratio:.4f}")
            else:
                difference = np.logical_and(big, ~mask_dicts[j]["segmentation"])
                remove.add(i)
                if difference.any(): additions.append({"segmentation": difference, "area": int(difference.sum())})
                process_log.append(f"mask={i} contains {j}; keep small and difference; ratio={ratio:.4f}")
        elif len(contained) > 1:
            area_sum = sum(mask_dicts[j]["area"] for j in contained)
            if area_sum < args.containment_area_sum_ratio * mask_dicts[i]["area"]:
                remove.update(contained)
                process_log.append(f"mask={i} contains {contained}; keep big")
            else:
                remove.add(i)
                process_log.append(f"mask={i} contains {contained}; remove big")
    kept = [item for i, item in enumerate(mask_dicts) if i not in remove] + additions
    process_log.append(f"Stage 2 removed={sorted(remove)} | added={len(additions)} | kept={len(kept)}")
    return kept


def save_color_visualization(image, masks, save_path):
    overlay = image.copy()
    rng = np.random.default_rng(42)
    for mask in masks:
        overlay[mask.astype(bool)] = rng.integers(0, 256, 3, dtype=np.uint8)
    cv2.imwrite(save_path, cv2.addWeighted(image, 0.5, overlay, 0.5, 0))

def save_contour_image(image, masks, save_path):
    vis = image.copy()
    for idx, mask in enumerate(masks):
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            continue
        cv2.putText(
            vis,
            str(idx),
            (int(xs.mean()), int(ys.mean())),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    cv2.imwrite(save_path, vis)


def run_one_image(model, sam_predictor, image_path, output_folder, args):
    total_tic = time.perf_counter()
    step_times = {}
    step_tic = time.perf_counter()
    image_original = cv2.imread(str(image_path))
    if image_original is None:
        print(f"跳过无法读取的图片: {image_path}")
        return

    original_height, original_width = image_original.shape[:2]
    if args.downsample == 1.0:
        image = image_original
    else:
        target_width = max(1, int(round(original_width * args.downsample)))
        target_height = max(1, int(round(original_height * args.downsample)))
        image = cv2.resize(
            image_original,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    height, width = image.shape[:2]
    save_name = image_path.stem
    print_elapsed("读取并降采样图片", step_tic, step_times, save_name)
    print(
        f"{save_name}: 原图={original_width}x{original_height}, "
        f"处理图={width}x{height}, downsample={args.downsample}"
    )

    masks = []
    if args.existing_mask_folder:
        step_tic = time.perf_counter()
        existing_path = Path(args.existing_mask_folder) / f"{save_name}.png"
        instance_mask = cv2.imread(str(existing_path), cv2.IMREAD_UNCHANGED)
        if instance_mask is None:
            print(f"跳过：找不到已有 mask: {existing_path}")
            return
        if instance_mask.ndim == 3:
            instance_mask = instance_mask[:, :, 0]
        if instance_mask.shape != (height, width):
            instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        labels = np.unique(instance_mask)
        masks = [(instance_mask == label) for label in labels if label != 0]
        inference_time = 0.0
        print_elapsed("读取并解析已有 mask", step_tic, step_times, save_name)
    else:
        step_tic = time.perf_counter()
        results = model.predict(
            source=image, device=args.device, imgsz=args.effective_imgsz,
            conf=args.conf, iou=args.iou, retina_masks=args.retina_masks, verbose=False,
        )
        inference_time = print_elapsed("FastSAM 推理", step_tic, step_times, save_name)
        step_tic = time.perf_counter()
        result = results[0]
        if result.masks is not None:
            mask_array = result.masks.data.detach().cpu().numpy()
            for mask in mask_array:
                mask_u8 = (mask > 0.5).astype(np.uint8)
                if mask_u8.shape != (height, width):
                    mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
                masks.append(mask_u8.astype(bool))
        print_elapsed("FastSAM mask 转换", step_tic, step_times, save_name)

    refine_time = 0.0
    refine_candidates = 0
    refined_masks = []
    if sam_predictor is not None and not args.existing_mask_folder:
        step_tic = time.perf_counter()
        refined_masks, refine_candidates = refine_missing_with_sam(sam_predictor, image, masks, args)
        refine_time = print_elapsed("SAM 漏检补分割", step_tic, step_times, save_name)
        masks.extend(refined_masks)

    if args.save_npz:
        step_tic = time.perf_counter()
        np.savez(
            os.path.join(output_folder, "fastsam_masks", f"{save_name}.npz"),
            masks=np.array(masks, dtype=object),
        )
        print_elapsed("保存原始 mask NPZ", step_tic, step_times, save_name)

    process_log = [f"image={image_path}", f"min_area={args.min_area}", f"max_area_ratio={args.max_area_ratio}", f"skip_border={not args.keep_border}", f"min_nucleus_dark_ratio={args.min_nucleus_dark_ratio}"]
    step_tic = time.perf_counter()
    save_contour_image(image, masks, os.path.join(output_folder, "debug", f"{save_name}_01_raw.png"))
    print_elapsed("保存原始轮廓调试图", step_tic, step_times, save_name)
    raw_mask_count = len(masks)
    if not args.no_filter:
        step_tic = time.perf_counter()
        mask_dicts = masks_to_dicts(masks)
        print_elapsed("构建 mask 信息", step_tic, step_times, save_name)
        step_tic = time.perf_counter()
        global_otsu = calculate_mask_region_otsu(image, mask_dicts, args, process_log)
        print_elapsed("计算区域 Otsu 阈值", step_tic, step_times, save_name)
        step_tic = time.perf_counter()
        mask_dicts = filter_masks_basic(mask_dicts, image, global_otsu, args, process_log)
        print_elapsed("Stage 1 基础/颜色筛选", step_tic, step_times, save_name)
        step_tic = time.perf_counter()
        save_contour_image(image, [item["segmentation"] for item in mask_dicts], os.path.join(output_folder, "debug", f"{save_name}_02_filtered.png"))
        print_elapsed("保存 Stage 1 调试图", step_tic, step_times, save_name)
        step_tic = time.perf_counter()
        mask_dicts = filter_masks_by_containment(mask_dicts, image, args, process_log)
        print_elapsed("Stage 2 包含关系筛选", step_tic, step_times, save_name)
        masks = [item["segmentation"] for item in mask_dicts]
        step_tic = time.perf_counter()
        save_contour_image(image, masks, os.path.join(output_folder, "debug", f"{save_name}_03_contained.png"))
        print_elapsed("保存 Stage 2 调试图", step_tic, step_times, save_name)
    else:
        process_log.append("filter disabled; all masks kept")
    step_tic = time.perf_counter()
    np.savez_compressed(os.path.join(output_folder, "filtered_masks", f"{save_name}.npz"), masks=np.array(masks, dtype=object))
    print_elapsed("保存筛选后 mask NPZ", step_tic, step_times, save_name)
    process_log.extend(["\n========== Final result ==========", f"raw_masks={raw_mask_count}", f"final_valid_masks={len(masks)}"])
    step_tic = time.perf_counter()
    with open(os.path.join(output_folder, "debug", f"{save_name}_mask_process_log.txt"), "w", encoding="utf-8") as log:
        log.write("\n".join(process_log))
    print_elapsed("保存 mask 处理日志", step_tic, step_times, save_name)

    step_tic = time.perf_counter()
    instance_mask = masks_to_instance_mask(masks, height, width)
    cv2.imwrite(os.path.join(output_folder, "mask", f"{save_name}.png"), instance_mask)
    print_elapsed("生成并保存实例 mask PNG", step_tic, step_times, save_name)
    step_tic = time.perf_counter()
    save_contour_image(image, masks, os.path.join(output_folder, "red_contour", f"{save_name}.png"))
    print_elapsed("保存红色轮廓图", step_tic, step_times, save_name)
    step_tic = time.perf_counter()
    save_color_visualization(image, masks, os.path.join(output_folder, "color_vis", f"{save_name}.png"))
    print_elapsed("保存彩色可视化图", step_tic, step_times, save_name)

    total_time = time.perf_counter() - total_tic
    postprocess_time = total_time - step_times["读取并降采样图片"] - inference_time

    print(
        f"{save_name}: {raw_mask_count} raw masks, {len(masks)} filtered masks, "
        f"inference_time={inference_time:.3f}s, "
        f"postprocess_time={postprocess_time:.3f}s, "
        f"sam_refine_time={refine_time:.3f}s, "
        f"refine_candidates={refine_candidates}, "
        f"refined_masks={len(refined_masks)}, "
        f"total_time={total_time:.3f}s"
    )

    return {
        "image": save_name,
        "masks": len(masks),
        "inference_time": inference_time,
        "postprocess_time": postprocess_time,
        "sam_refine_time": refine_time,
        "refine_candidates": refine_candidates,
        "refined_masks": len(refined_masks),
        "total_time": total_time,
        "step_times": step_times,
    }


def main():
    program_tic = time.perf_counter()
    step_tic = time.perf_counter()
    args = parse_args()
    if not 0 < args.downsample <= 1:
        raise SystemExit("--downsample 必须大于 0 且不超过 1。")

    area_scale = args.downsample ** 2
    args.original_min_area = args.min_area
    args.original_min_missing_area = args.min_missing_area
    args.min_area = max(1, int(round(args.min_area * area_scale)))
    args.min_missing_area = max(1, int(round(args.min_missing_area * area_scale)))
    args.effective_imgsz = max(32, int(round(args.imgsz * args.downsample / 32)) * 32)

    images = list_images(args.input_folder, args.limit)
    print_elapsed("解析参数并扫描输入图片", step_tic)

    print("FastSAM 配置检查")
    print(f"input_folder={args.input_folder}")
    print(f"output_folder={args.output_folder}")
    print(f"model={args.model}")
    print(f"device={args.device}")
    print(f"downsample={args.downsample}")
    print(f"base_imgsz={args.imgsz}")
    print(f"effective_imgsz={args.effective_imgsz}")
    print(f"min_area={args.original_min_area} -> effective_min_area={args.min_area}")
    print(
        f"min_missing_area={args.original_min_missing_area} -> "
        f"effective_min_missing_area={args.min_missing_area}"
    )
    print(f"conf={args.conf}")
    print(f"iou={args.iou}")
    print(f"retina_masks={args.retina_masks}")
    print(f"images={len(images)}")
    print(f"sam_refine={not args.no_sam_refine}")
    print(f"sam_checkpoint={args.sam_checkpoint}")
    print(f"miss_coverage_thresh={args.miss_coverage_thresh}")
    print(f"existing_mask_folder={args.existing_mask_folder}")

    if not images:
        raise SystemExit("没有找到可处理图片。")

    if not args.run:
        print("dry-run: 未执行处理。加 --run 后开始运行。")
        if not os.path.exists(args.model):
            print("提示: 当前权重文件不存在。首次运行可使用模型名 FastSAM-s.pt 自动下载，或手动放到上述路径。")
        return

    step_tic = time.perf_counter()
    ensure_output_dirs(args.output_folder)
    print_elapsed("创建输出目录", step_tic)
    if args.existing_mask_folder:
        if Path(args.existing_mask_folder).resolve() == (Path(args.output_folder) / "mask").resolve():
            raise SystemExit("为防止覆盖输入 mask，请将 --output-folder 设置为新的输出目录。")
        model = None
        sam_predictor = None
        print("已有 mask 筛选模式：跳过 FastSAM 和 SAM，不加载模型，仅使用 CPU 后处理。")
    else:
        step_tic = time.perf_counter()
        model = YOLO(args.model)
        print_elapsed("加载 FastSAM 模型", step_tic)
        sam_predictor = None
        if not args.no_sam_refine:
            step_tic = time.perf_counter()
            sam_predictor = load_sam_predictor(args)
            print_elapsed("加载 SAM Predictor", step_tic)

    log_path = os.path.join(args.output_folder, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = TeeStdout(original_stdout, log_file)
        try:
            timings = []
            for image_path in images:
                timing = run_one_image(model, sam_predictor, image_path, args.output_folder, args)
                if timing is not None:
                    timings.append(timing)

            if timings:
                avg_inference = sum(t["inference_time"] for t in timings) / len(timings)
                avg_postprocess = sum(t["postprocess_time"] for t in timings) / len(timings)
                avg_total = sum(t["total_time"] for t in timings) / len(timings)
                avg_refine = sum(t["sam_refine_time"] for t in timings) / len(timings)
                total_refined = sum(t["refined_masks"] for t in timings)
                min_total = min(t["total_time"] for t in timings)
                max_total = max(t["total_time"] for t in timings)

                print("\n========== Time summary ==========")
                print(f"images={len(timings)}")
                print(f"avg_inference_time={avg_inference:.3f}s")
                print(f"avg_postprocess_time={avg_postprocess:.3f}s")
                print(f"avg_sam_refine_time={avg_refine:.3f}s")
                print(f"total_refined_masks={total_refined}")
                print(f"avg_total_time={avg_total:.3f}s")
                print(f"min_total_time={min_total:.3f}s")
                print(f"max_total_time={max_total:.3f}s")

                print("\n========== Average step time ==========")
                step_names = sorted({name for item in timings for name in item["step_times"]})
                for step_name in step_names:
                    values = [item["step_times"][step_name] for item in timings if step_name in item["step_times"]]
                    print(f"{step_name}: avg={sum(values) / len(values):.3f}s, count={len(values)}")

            print("全部处理完成！")
            print(f"结果保存在: {args.output_folder}")
            print(f"整批运行总耗时: {time.perf_counter() - program_tic:.3f}s")
        finally:
            sys.stdout = original_stdout


class TeeStdout:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


if __name__ == "__main__":
    main()
