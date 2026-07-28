# -*- coding: utf-8 -*-
import os
import json
import csv
import random
import time
from datetime import datetime

# （可选）限制底层线程，避免过度并行
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm

# ============== 顶层：Windows 友好 collate_fn（便于 pickle） ==============
def collate_fn(batch):
    return tuple(zip(*batch))

# ============== 小工具 ==============
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def image_open_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ============== 数据集（无数据增强） ==============
class LabelMeDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, categories):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.categories = categories
        # 背景=0；你的唯一类别名是字符串 "1" → id=1
        self.category_to_id = {name: i + 1 for i, name in enumerate(categories)}  # {"1":1}

        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        self.image_filenames = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(exts)])
        if not self.image_filenames:
            raise RuntimeError(f"[ERROR] {images_dir} 下没有图片")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        ann_path = os.path.join(self.annotations_dir, os.path.splitext(img_filename)[0] + ".json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation not found for {img_filename} -> {ann_path}")

        image = F.to_tensor(image_open_rgb(img_path))  # (C,H,W), float32 in [0,1]
        _, h, w = image.shape

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        boxes, labels, masks = [], [], []
        for shape in ann.get("shapes", []):
            pts = shape.get("points", [])
            if len(pts) < 3:
                continue

            lbl_name = shape.get("label", None)
            if lbl_name not in self.category_to_id:
                continue
            label_id = self.category_to_id[lbl_name]  # =1

            pts = np.array(pts, dtype=np.float32)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label_id)

            mask_img = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_img, [pts.astype(np.int32)], 1)
            masks.append(mask_img)

        # 检查目标数量
        if len(boxes) > 200:
            print(f"[WARN] Image {img_filename} has {len(boxes)} targets, exceeding max_detections=200, truncated")
            boxes = boxes[:200]
            labels = labels[:200]
            masks = masks[:200]

        if masks:
            masks_np = np.stack(masks, axis=0).astype(np.uint8, copy=False)  # (N,H,W)
            masks = torch.from_numpy(masks_np)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image, target

# ============== 模型 ==============
def get_maskrcnn_model(num_classes=2, max_detections=200):
    model: MaskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.roi_heads.detections_per_img = max_detections
    return model

# ============== 通用：加载权重（兼容 state_dict 或 {"model": state_dict}） ==============
def load_weights_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        state_dict = obj
    else:
        raise RuntimeError(f"[ERROR] Unsupported checkpoint format: {ckpt_path}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

# ============== 渲染预测（返回 overlay 后的 BGR canvas） ==============
def render_prediction_on_image(
    model,
    img_pil: Image.Image,
    device: torch.device,
    score_thr=0.5,
    mask_thr=0.5,
    draw_masks=True,
    class_id=1
):
    img = F.to_tensor(img_pil).to(device)
    with torch.no_grad():
        pred = model([img])[0]

    boxes = pred.get("boxes", torch.empty((0,4))).detach().cpu()
    scores = pred.get("scores", torch.empty((0,))).detach().cpu()
    labels = pred.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu()
    masks = pred.get("masks", None)

    keep = (scores >= score_thr) & (labels == class_id)
    boxes = boxes[keep].numpy()
    scores = scores[keep].numpy()
    if masks is not None:
        masks = masks[keep].detach().cpu().numpy()  # (N,1,H,W)

    canvas = np.array(img_pil)[:, :, ::-1].copy()  # RGB->BGR

    if draw_masks and masks is not None and len(masks) > 0:
        for m in masks:
            m = (m[0] >= mask_thr).astype(np.uint8)
            colored = np.zeros_like(canvas)
            colored[:, :, 1] = 255  # green
            alpha = 0.4
            canvas[m == 1] = (canvas[m == 1] * (1 - alpha) + colored[m == 1] * alpha).astype(np.uint8)

    for (x1, y1, x2, y2), s in zip(boxes, scores):
        pt1 = (int(x1), int(y1)); pt2 = (int(x2), int(y2))
        cv2.rectangle(canvas, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(canvas, f"{s:.2f}", (pt1[0], max(0, pt1[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return canvas, int(len(boxes))

# ============== 单模型预测可视化（last） ==============
def visualize_predictions(model, val_images_dir, out_dir, device, num_images=6, score_thr=0.5, mask_thr=0.5, class_id=1):
    model.eval()
    vis_dir = ensure_dir(os.path.join(out_dir, "vis"))
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    candidates = [f for f in sorted(os.listdir(val_images_dir)) if f.lower().endswith(exts)]
    if not candidates:
        print("[VIS] val 集合为空，跳过可视化。"); return
    pick = candidates[:num_images]

    for fname in pick:
        path = os.path.join(val_images_dir, fname)
        img_pil = image_open_rgb(path)
        canvas, dets = render_prediction_on_image(
            model, img_pil, device, score_thr=score_thr, mask_thr=mask_thr, draw_masks=True, class_id=class_id
        )
        print(f"[VIS] Image {fname}: {dets} detections after thresholding")
        save_path = os.path.join(vis_dir, os.path.splitext(fname)[0] + "_pred.jpg")
        cv2.imwrite(save_path, canvas)
    print(f"[VIS] 可视化结果已保存到：{vis_dir}")

# ============== 三 checkpoint 对比可视化（epoch1/best/last） ==============
def visualize_compare_checkpoints(
    model_factory_fn,
    ckpt_paths: dict,
    val_images_dir: str,
    out_dir: str,
    device: torch.device,
    num_images=6,
    score_thr=0.5,
    mask_thr=0.5,
    class_id=1
):
    cmp_dir = ensure_dir(os.path.join(out_dir, "vis_compare"))
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    candidates = [f for f in sorted(os.listdir(val_images_dir)) if f.lower().endswith(exts)]
    if not candidates:
        print("[VIS_CMP] val 集合为空，跳过对比可视化。"); return
    pick = candidates[:num_images]

    for k, p in ckpt_paths.items():
        if not os.path.exists(p):
            print(f"[VIS_CMP][WARN] checkpoint not found: {k} -> {p}")

    models = {}
    for tag, p in ckpt_paths.items():
        if not os.path.exists(p):
            continue
        m = model_factory_fn().to(device)
        m = load_weights_into_model(m, p, device)
        models[tag] = m

    required = ["epoch1", "best_by_maskf1", "last"]
    for r in required:
        if r not in models:
            print(f"[VIS_CMP][ERROR] 缺少模型：{r}（请确认 checkpoint 保存是否成功）")
            return

    for fname in pick:
        img_path = os.path.join(val_images_dir, fname)
        img_pil = image_open_rgb(img_path)

        canvases = []
        titles = []
        for tag in required:
            canvas_bgr, dets = render_prediction_on_image(
                models[tag], img_pil, device,
                score_thr=score_thr, mask_thr=mask_thr, draw_masks=True, class_id=class_id
            )
            canvases.append(canvas_bgr[:, :, ::-1])  # BGR->RGB for plt
            titles.append(f"{tag}\n(dets={dets})")

        plt.figure(figsize=(18, 6))
        for i in range(3):
            ax = plt.subplot(1, 3, i+1)
            ax.imshow(canvases[i])
            ax.set_title(titles[i])
            ax.axis("off")
        plt.suptitle(f"Compare checkpoints: {fname} | score_thr={score_thr}, mask_thr={mask_thr}", y=0.98)
        save_path = os.path.join(cmp_dir, os.path.splitext(fname)[0] + "_compare.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[VIS_CMP] 对比可视化已保存到：{cmp_dir}")

# ============== 自定义评估：bbox IoU 矩阵（小） ==============
def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)

# ============== 贪心匹配（pred 逐个匹配最佳 gt） ==============
def greedy_match_by_iou(iou_mat: torch.Tensor, iou_thr: float):
    num_pred, num_gt = iou_mat.shape
    matched_gt = set()
    matches = []

    if num_pred == 0 or num_gt == 0:
        return matches, set(range(num_pred)), set(range(num_gt))

    for pi in range(num_pred):
        ious = iou_mat[pi]
        gi = int(torch.argmax(ious).item())
        best = float(ious[gi].item())
        if best >= iou_thr and gi not in matched_gt:
            matched_gt.add(gi)
            matches.append((pi, gi, best))

    unmatched_pred = set(range(num_pred)) - set([m[0] for m in matches])
    unmatched_gt = set(range(num_gt)) - matched_gt
    return matches, unmatched_pred, unmatched_gt

# ============== 关键：逐对 crop 计算 mask IoU（避免 OOM） ==============
def mask_iou_pair_crop(pm: torch.Tensor, gm: torch.Tensor, pb: torch.Tensor, gb: torch.Tensor) -> float:
    """
    pm, gm: (H,W) uint8/bool on CPU
    pb, gb: (4,) xyxy float on CPU
    在 pred box 与 gt box 的联合区域内计算 mask IoU，降低计算量/内存
    """
    H, W = pm.shape
    x1 = int(max(0, min(pb[0].item(), gb[0].item())))
    y1 = int(max(0, min(pb[1].item(), gb[1].item())))
    x2 = int(min(W, max(pb[2].item(), gb[2].item())))
    y2 = int(min(H, max(pb[3].item(), gb[3].item())))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    pmc = pm[y1:y2, x1:x2].bool()
    gmc = gm[y1:y2, x1:x2].bool()
    inter = (pmc & gmc).sum().item()
    union = (pmc | gmc).sum().item()
    if union == 0:
        return 0.0
    return float(inter / union)

@torch.no_grad()
def evaluate_val_iou_f1(
    model,
    val_loader,
    device,
    score_thr=0.5,
    iou_thr=0.5,
    mask_thr=0.5,
    class_id=1,
    topk=150,   # <<<<<< 默认 150
):
    """
    指标定义：
    - bbox：用 bbox IoU>=iou_thr 的匹配作为 TP；其余 pred 为 FP，未匹配 gt 为 FN
    - mask：先按 bbox 匹配对做候选，再用 mask IoU>=iou_thr 作为 TP（逐对 crop 计算）
    """
    model.eval()

    bbox_TP = bbox_FP = bbox_FN = 0
    mask_TP = mask_FP = mask_FN = 0
    bbox_iou_sum = 0.0
    mask_iou_sum = 0.0
    bbox_match_cnt = 0
    mask_match_cnt = 0
    num_gt_total = 0
    num_pred_total = 0

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        preds = model(images)

        for pred, tgt in zip(preds, targets):
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]
            gt_masks = tgt.get("masks", None)

            gt_keep = (gt_labels == class_id)
            gt_boxes = gt_boxes[gt_keep]
            if gt_masks is not None:
                gt_masks = gt_masks[gt_keep].to(torch.uint8)

            p_boxes = pred.get("boxes", torch.empty((0,4), device=device))
            p_scores = pred.get("scores", torch.empty((0,), device=device))
            p_labels = pred.get("labels", torch.empty((0,), device=device, dtype=torch.int64))
            p_masks = pred.get("masks", None)  # (N,1,H,W)

            keep = (p_scores >= score_thr) & (p_labels == class_id)
            p_boxes = p_boxes[keep]
            p_scores = p_scores[keep]
            if p_masks is not None:
                p_masks = p_masks[keep]

            if p_scores.numel() > 0:
                order = torch.argsort(p_scores, descending=True)
                p_boxes = p_boxes[order]
                p_scores = p_scores[order]
                if p_masks is not None:
                    p_masks = p_masks[order]

            # Top-K 截断（评估用；训练不影响）
            if topk is not None and p_boxes.shape[0] > topk:
                p_boxes = p_boxes[:topk]
                p_scores = p_scores[:topk]
                if p_masks is not None:
                    p_masks = p_masks[:topk]

            num_gt_total += int(gt_boxes.shape[0])
            num_pred_total += int(p_boxes.shape[0])

            # ---------- bbox 匹配 ----------
            iou_b = box_iou_matrix(p_boxes, gt_boxes)
            matches_b, unp_b, ung_b = greedy_match_by_iou(iou_b, iou_thr=iou_thr)

            bbox_TP += len(matches_b)
            bbox_FP += len(unp_b)
            bbox_FN += len(ung_b)
            for _, _, iou_v in matches_b:
                bbox_iou_sum += iou_v
            bbox_match_cnt += len(matches_b)

            # ---------- mask 匹配（OOM 修复版） ----------
            if p_masks is not None and gt_masks is not None and len(matches_b) > 0:
                pm_all = (p_masks[:, 0] >= mask_thr).to(torch.uint8).cpu()
                gm_all = gt_masks.to(torch.uint8).cpu()
                pb_all = p_boxes.detach().cpu()
                gb_all = gt_boxes.detach().cpu()

                mask_matched_gt = set()
                mask_TP_img = 0
                mask_iou_sum_img = 0.0

                for (pi, gi, _) in matches_b:
                    if gi in mask_matched_gt:
                        continue
                    miou = mask_iou_pair_crop(pm_all[pi], gm_all[gi], pb_all[pi], gb_all[gi])
                    if miou >= iou_thr:
                        mask_matched_gt.add(gi)
                        mask_TP_img += 1
                        mask_iou_sum_img += miou

                mask_TP += mask_TP_img
                mask_FP += int(p_boxes.shape[0]) - mask_TP_img
                mask_FN += int(gt_boxes.shape[0]) - mask_TP_img
                mask_iou_sum += mask_iou_sum_img
                mask_match_cnt += mask_TP_img
            else:
                mask_FP += int(p_boxes.shape[0])
                mask_FN += int(gt_boxes.shape[0])

    def prf(tp, fp, fn):
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        return float(p), float(r), float(f1)

    bbox_p, bbox_r, bbox_f1 = prf(bbox_TP, bbox_FP, bbox_FN)
    mask_p, mask_r, mask_f1 = prf(mask_TP, mask_FP, mask_FN)

    return {
        "val_bbox_precision": bbox_p,
        "val_bbox_recall": bbox_r,
        "val_bbox_f1": bbox_f1,
        "val_bbox_miou": float(bbox_iou_sum / max(1, bbox_match_cnt)),
        "val_mask_precision": mask_p,
        "val_mask_recall": mask_r,
        "val_mask_f1": mask_f1,
        "val_mask_miou": float(mask_iou_sum / max(1, mask_match_cnt)),
        "val_num_gt": int(num_gt_total),
        "val_num_pred": int(num_pred_total),
    }

# ============== 训练主函数 ==============
def train_mask_rcnn(
    root_dir="Dataset for Mask-RCNN",
    categories=("1",),
    epochs=15,
    batch_size=2,
    lr=0.005,
    step_size=5,
    gamma=0.1,
    dl_num_workers=0,
    dl_pin_memory=False,
    max_detections=200,
    # 评估超参
    score_thr=0.5,
    iou_thr=0.5,
    mask_thr=0.5,
    eval_class_id=1,
    eval_topk=200,  # <<<<<< 默认 150
    # 对比可视化
    compare_num_images=6,
):
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    train_images = os.path.join(root_dir, "images", "train")
    val_images = os.path.join(root_dir, "images", "val")
    train_anns = os.path.join(root_dir, "annotations", "train")
    val_anns = os.path.join(root_dir, "annotations", "val")
    output_dir = ensure_dir(os.path.join(root_dir, "Results"))

    # 环境 & GPU 信息
    print("torch:", torch.__version__, "| torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"Compute capability: {props.major}.{props.minor}, VRAM: {props.total_memory/1024**3:.1f} GB")
    else:
        print("Using CPU")

    # 保存配置
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "train_images": train_images,
            "val_images": val_images,
            "train_annotations": train_anns,
            "val_annotations": val_anns,
            "output_dir": output_dir
        },
        "categories": list(categories),
        "hyperparams": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "step_size": step_size, "gamma": gamma,
            "optimizer": "SGD(momentum=0.9, weight_decay=5e-4)",
            "scheduler": "StepLR",
            "max_detections": max_detections
        },
        "eval": {
            "score_thr": score_thr,
            "iou_thr": iou_thr,
            "mask_thr": mask_thr,
            "class_id": eval_class_id,
            "topk": eval_topk
        },
        "env": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "seed": 42,
        "dataloader": {"num_workers": dl_num_workers, "pin_memory": dl_pin_memory}
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 数据
    train_ds = LabelMeDataset(train_images, train_anns, categories)
    val_ds = LabelMeDataset(val_images, val_anns, categories)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=dl_num_workers, pin_memory=dl_pin_memory, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        num_workers=dl_num_workers, pin_memory=dl_pin_memory, collate_fn=collate_fn
    )

    # 模型与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_maskrcnn_model(num_classes=2, max_detections=max_detections).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 日志
    log_rows = []
    best_val_loss = float("inf")
    best_mask_f1 = -1.0

    csv_path = os.path.join(output_dir, "training_log.csv")
    xlsx_path = os.path.join(output_dir, "training_log.xlsx")

    header = [
        "epoch", "lr",
        "train_total", "train_loss_classifier", "train_loss_box_reg", "train_loss_mask",
        "train_loss_objectness", "train_loss_rpn_box_reg",
        "val_total", "val_loss_classifier", "val_loss_box_reg", "val_loss_mask",
        "val_loss_objectness", "val_loss_rpn_box_reg",
        # 自定义指标
        "val_bbox_precision", "val_bbox_recall", "val_bbox_f1", "val_bbox_miou",
        "val_mask_precision", "val_mask_recall", "val_mask_f1", "val_mask_miou",
        "val_num_gt", "val_num_pred",
        # 工程信息
        "epoch_time_sec", "gpu_mem_max_mb"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    def forward_one_epoch(loader, is_train=True, epoch_idx=1, total_epochs=1, phase_name="Train"):
        # 训练与（loss用的）验证都需 train() 才会返回 loss 字典
        model.train()
        total, n = 0.0, 0
        comp = {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_mask": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0
        }

        loop = tqdm(loader, total=len(loader), desc=f"{phase_name} [{epoch_idx}/{total_epochs}]", leave=False)
        for images, targets in loop:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == "cuda")):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

            total += float(loss.item())
            n += 1
            for k in comp:
                v = loss_dict.get(k, torch.tensor(0.0))
                comp[k] += float(v.item()) if torch.is_tensor(v) else float(v)

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg = total / max(1, n)
        for k in comp:
            comp[k] /= max(1, n)
        return avg, comp

    # 训练循环
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        current_lr = optimizer.param_groups[0]["lr"]

        train_total, train_comp = forward_one_epoch(
            train_loader, is_train=True, epoch_idx=epoch, total_epochs=epochs, phase_name="Train"
        )
        val_total, val_comp = forward_one_epoch(
            val_loader, is_train=False, epoch_idx=epoch, total_epochs=epochs, phase_name="ValLoss"
        )

        # 自定义指标（eval 推理）
        metrics = evaluate_val_iou_f1(
            model, val_loader, device,
            score_thr=score_thr,
            iou_thr=iou_thr,
            mask_thr=mask_thr,
            class_id=eval_class_id,
            topk=eval_topk
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        gpu_mem_mb = 0.0
        if device.type == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(
            f"Epoch {epoch:02d}/{epochs} | lr {current_lr:.6f} | "
            f"train {train_total:.4f} | valLoss {val_total:.4f} | "
            f"bboxF1 {metrics['val_bbox_f1']:.4f} (mIoU {metrics['val_bbox_miou']:.4f}) | "
            f"maskF1 {metrics['val_mask_f1']:.4f} (mIoU {metrics['val_mask_miou']:.4f}) | "
            f"pred {metrics['val_num_pred']} gt {metrics['val_num_gt']} | "
            f"time {epoch_time:.1f}s mem {gpu_mem_mb:.0f}MB"
        )

        # 写 CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, current_lr,
                train_total, train_comp["loss_classifier"], train_comp["loss_box_reg"], train_comp["loss_mask"],
                train_comp["loss_objectness"], train_comp["loss_rpn_box_reg"],
                val_total, val_comp["loss_classifier"], val_comp["loss_box_reg"], val_comp["loss_mask"],
                val_comp["loss_objectness"], val_comp["loss_rpn_box_reg"],
                metrics["val_bbox_precision"], metrics["val_bbox_recall"], metrics["val_bbox_f1"], metrics["val_bbox_miou"],
                metrics["val_mask_precision"], metrics["val_mask_recall"], metrics["val_mask_f1"], metrics["val_mask_miou"],
                metrics["val_num_gt"], metrics["val_num_pred"],
                epoch_time, gpu_mem_mb
            ])

        # 记录到 df
        row = {
            "epoch": epoch, "lr": current_lr,
            "train_total": train_total, "val_total": val_total,
            "epoch_time_sec": epoch_time, "gpu_mem_max_mb": gpu_mem_mb
        }
        for k, v in train_comp.items():
            row[f"train_{k}"] = v
        for k, v in val_comp.items():
            row[f"val_{k}"] = v
        row.update(metrics)
        log_rows.append(row)

        # -------- checkpoints --------
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict()
        }, os.path.join(output_dir, "last_state.pth"))
        torch.save(model.state_dict(), os.path.join(output_dir, "last.pth"))

        if epoch == 1:
            torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(output_dir, "epoch1_state.pth"))
            torch.save(model.state_dict(), os.path.join(output_dir, "epoch1.pth"))

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(output_dir, "best_by_loss_state.pth"))
            torch.save(model.state_dict(), os.path.join(output_dir, "best_by_loss.pth"))

        if metrics["val_mask_f1"] > best_mask_f1:
            best_mask_f1 = metrics["val_mask_f1"]
            torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(output_dir, "best_by_maskf1_state.pth"))
            torch.save(model.state_dict(), os.path.join(output_dir, "best_by_maskf1.pth"))

    # 导出 Excel
    df = pd.DataFrame(log_rows)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[WARN] 导出 Excel 失败：{e}")

    # 绘图：loss
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_total"], label="Train Loss")
    plt.plot(df["epoch"], df["val_total"], label="Val Loss (train-mode no_grad)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True); plt.legend()
    plt.title("Mask R-CNN Training vs Validation Loss")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # 绘图：F1
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_bbox_f1"], label="Val BBox F1")
    plt.plot(df["epoch"], df["val_mask_f1"], label="Val Mask F1")
    plt.xlabel("Epoch"); plt.ylabel("F1")
    plt.grid(True); plt.legend()
    plt.title(f"Validation F1 (score_thr={score_thr}, iou_thr={iou_thr}, topk={eval_topk})")
    plt.savefig(os.path.join(output_dir, "val_f1_curve.png"), dpi=150)
    plt.close()

    # 绘图：mIoU
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_bbox_miou"], label="Val BBox mIoU (matched)")
    plt.plot(df["epoch"], df["val_mask_miou"], label="Val Mask mIoU (matched)")
    plt.xlabel("Epoch"); plt.ylabel("mIoU")
    plt.grid(True); plt.legend()
    plt.title(f"Validation mIoU (score_thr={score_thr}, iou_thr={iou_thr}, topk={eval_topk})")
    plt.savefig(os.path.join(output_dir, "val_miou_curve.png"), dpi=150)
    plt.close()

    print(f"[PLOT] 已保存：loss_curve.png / val_f1_curve.png / val_miou_curve.png")

    # 单模型可视化：last
    visualize_predictions(model, val_images, output_dir, device, num_images=6,
                          score_thr=score_thr, mask_thr=mask_thr, class_id=eval_class_id)

    # 三 checkpoint 对比可视化：epoch1 / best_by_maskf1 / last
    def model_factory():
        return get_maskrcnn_model(num_classes=2, max_detections=max_detections)

    ckpt_paths = {
        "epoch1": os.path.join(output_dir, "epoch1.pth"),
        "best_by_maskf1": os.path.join(output_dir, "best_by_maskf1.pth"),
        "last": os.path.join(output_dir, "last.pth"),
    }
    visualize_compare_checkpoints(
        model_factory_fn=model_factory,
        ckpt_paths=ckpt_paths,
        val_images_dir=val_images,
        out_dir=output_dir,
        device=device,
        num_images=compare_num_images,
        score_thr=score_thr,
        mask_thr=mask_thr,
        class_id=eval_class_id
    )

# ============== 入口 ==============
if __name__ == "__main__":
    ROOT = "Dataset for Mask-RCNN"  # 改成你的根目录
    CATEGORIES = ("1",)             # 标注类名为字符串 "1"

    train_mask_rcnn(
        root_dir=ROOT,
        categories=CATEGORIES,
        epochs=15,
        batch_size=2,
        lr=0.005,
        step_size=5,
        gamma=0.1,
        dl_num_workers=0,
        dl_pin_memory=False,
        max_detections=200,

        # 自定义评估参数（topk 默认 200）
        score_thr=0.5,
        iou_thr=0.5,
        mask_thr=0.5,
        eval_class_id=1,
        eval_topk=200,

        compare_num_images=6,
    )
