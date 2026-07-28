import sys
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as TF
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QMessageBox, QProgressBar, QGroupBox, QFormLayout, QGridLayout, QToolTip, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
import cv2
from torchvision import transforms
import open3d as o3d
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import defaultdict
import time
import pandas as pd
import traceback

# --- SAM 库导入 ---
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    HAS_SAM = True
except ImportError:
    print("警告: 未找到 segment_anything 库。请安装: pip install segment-anything")
    HAS_SAM = False

# ---------------------------
# 常量定义：粒径区间
# ---------------------------
BIN_LABELS = ['<9.5', '9.5-16', '16-26.5', '26.5-37.5', '37.5-53', '53-75', '>75']
BIN_EDGES = [0, 9.5, 16, 26.5, 37.5, 53, 75, float('inf')]


# ---------------------------
# ClickableLabel 类
# ---------------------------
class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)
    hovered = pyqtSignal(int, int)

    def __init__(self, title=""):
        super().__init__(title)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            self.clicked.emit(x, y)

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.hovered.emit(x, y)


# ---------------------------
# 分布图表画布类
# ---------------------------
class DistributionCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(DistributionCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

    def plot_distribution(self, pred_percents, gt_percents, sam_percents=None):
        self.ax.clear()

        x = np.arange(len(BIN_LABELS))
        # 调整柱状图宽度以适应3组数据
        width = 0.25 if sam_percents else 0.35

        # 绘制 GT 和 Pred
        rects1 = self.ax.bar(x - width, gt_percents, width, label='Annotation (GT)', color='gray', alpha=0.6)
        rects2 = self.ax.bar(x, pred_percents, width, label='PEA-Net Pred', color='#1f77b4', alpha=0.9)

        # 如果有SAM数据，绘制SAM
        if sam_percents:
            rects3 = self.ax.bar(x + width, sam_percents, width, label='SAM Baseline', color='#ff7f0e', alpha=0.8)

        self.ax.set_ylabel('Volume Percentage (%)')
        self.ax.set_title('Particle Size Distribution Comparison')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(BIN_LABELS, rotation=15)
        self.ax.legend()
        self.ax.grid(axis='y', linestyle='--', alpha=0.5)

        self._autolabel(rects1)
        self._autolabel(rects2)
        if sam_percents:
            self._autolabel(rects3)

        self.fig.tight_layout()
        self.draw()

    def _autolabel(self, rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                self.ax.annotate(f'{height:.1f}',
                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=7)


# ---------------------------
# 模型定义 (保持不变)
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OriginalUNetWithCBAMAndLaplacian(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, laplacian_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for f in features:
            self.encoder_layers.append(DoubleConv(in_channels, f))
            in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.cbams = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.cbams.append(CBAM(f))
            self.decoder_layers.append(DoubleConv(f * 2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.laplacian_conv = nn.Sequential(
            nn.Conv2d(laplacian_channels, laplacian_channels, 3, padding=1),
            nn.BatchNorm2d(laplacian_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, laplacian):
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        laplacian = self.laplacian_conv(laplacian)
        skips = skips[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skips[idx]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            lap_resized = F.interpolate(laplacian, size=skip.shape[2:], mode='nearest')
            enhanced_skip = skip + skip * lap_resized
            enhanced_skip = self.cbams[idx](enhanced_skip)
            x = torch.cat([enhanced_skip, x], dim=1)
            x = self.decoder_layers[idx](x)
        return self.final_conv(x)


# ---------------------------
# 预处理转换 & 核心计算函数
# ---------------------------
class ToTensorInference(object):
    def __call__(self, image):
        return TF.to_tensor(image)


class ResizeInference(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return TF.resize(image, self.size)


def process_point_cloud(ply_file, img_size=512, slice_count=100):
    try:
        t_load_start = time.perf_counter()
        pcd = o3d.io.read_point_cloud(ply_file)
        t_load_end = time.perf_counter()
        t_load = t_load_end - t_load_start

        t_proj_start = time.perf_counter()
        if len(pcd.points) < 4: raise ValueError(f"点云文件 {ply_file} 点数不足。")
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        points = np.asarray(pcd.points)
        if len(points) == 0: raise ValueError("下采样后点云为空。")

        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_edges = np.linspace(z_min, z_max, slice_count + 1)
        slice_thickness = (z_max - z_min) / slice_count
        base_plane_z = z_edges[-2] + slice_thickness * 0.5
        distances = np.abs(points[:, 2] - base_plane_z)

        min_bound, max_bound = points.min(axis=0), points.max(axis=0)
        x_min, x_max = min_bound[0], max_bound[0]
        y_min, y_max = min_bound[1], max_bound[1]

        x_bins = np.linspace(x_min, x_max, img_size + 1)
        y_bins = np.linspace(y_min, y_max, img_size + 1)

        x_indices = np.digitize(points[:, 0], bins=x_bins) - 1
        y_indices = np.digitize(points[:, 1], bins=y_bins) - 1
        x_indices = np.clip(x_indices, 0, img_size - 1)
        y_indices = np.clip(y_indices, 0, img_size - 1)
        y_indices_image = img_size - 1 - y_indices

        flatten_indices = y_indices_image * img_size + x_indices
        distance_sums = np.bincount(flatten_indices, weights=distances, minlength=img_size * img_size)
        count_sums = np.bincount(flatten_indices, minlength=img_size * img_size)

        distance_image_raw = distance_sums.reshape((img_size, img_size)) / np.maximum(
            count_sums.reshape((img_size, img_size)), 1)
        distance_image_raw[distance_image_raw == 0] = 0

        sort_idx = np.argsort(flatten_indices)
        sorted_flatten_indices = flatten_indices[sort_idx]
        target_indices = np.arange(img_size * img_size + 1)
        bin_edges = np.searchsorted(sorted_flatten_indices, target_indices)
        pixel_to_point_indices = [
            sort_idx[bin_edges[i]:bin_edges[i + 1]] for i in range(img_size * img_size)
        ]

        distance_image_normalized = (distance_image_raw - distance_image_raw.min()) / (
                distance_image_raw.max() - distance_image_raw.min() + 1e-6)
        distance_image_colored = cm.viridis(distance_image_normalized)[:, :, :3]
        distance_image_colored = (distance_image_colored * 255).astype(np.uint8)
        distance_image_pil = Image.fromarray(distance_image_colored).convert('RGB')

        t_proj_end = time.perf_counter()
        t_proj = t_proj_end - t_proj_start
        return distance_image_pil, pixel_to_point_indices, points, distance_image_raw, t_load, t_proj
    except Exception as e:
        print(f"处理文件 {ply_file} 时出错: {e}")
        raise e


def get_maskrcnn_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# ---------------------------
# Workers (后台线程)
# ---------------------------
class PointCloudWorker(QObject):
    finished = pyqtSignal(Image.Image, list, np.ndarray, float, int, np.ndarray, float, float)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, ply_file, img_size=512):
        super().__init__()
        self.ply_file = ply_file
        self.img_size = img_size

    def run(self):
        try:
            start_time = time.perf_counter()
            self.progress.emit(10)
            depth_image, pixel_to_point_indices, points, raw_height, t_load, t_proj = process_point_cloud(self.ply_file,
                                                                                                          self.img_size)
            self.progress.emit(100)
            total_elapsed = time.perf_counter() - start_time
            self.finished.emit(depth_image, pixel_to_point_indices, points, total_elapsed, len(points), raw_height,
                               t_load, t_proj)
        except Exception as e:
            self.error.emit(str(e))


class MaskedDepthWorker(QObject):
    finished = pyqtSignal(Image.Image, float)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, depth_image, mask_image):
        super().__init__()
        self.depth_image = depth_image
        self.mask_image = mask_image

    def run(self):
        try:
            start_time = time.perf_counter()
            self.progress.emit(10)
            depth_np = np.array(self.depth_image)
            mask_np = np.array(self.mask_image)
            mask_binary = (mask_np > 127).astype(np.uint8)
            masked_depth = depth_np * np.expand_dims(mask_binary, axis=2)
            self.progress.emit(100)
            elapsed_time = time.perf_counter() - start_time
            self.finished.emit(Image.fromarray(masked_depth), elapsed_time)
        except Exception as e:
            self.error.emit(str(e))


class MaskRCNNSegmentationWorker(QObject):
    finished = pyqtSignal(Image.Image, list, float, list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, masked_depth_image, model):
        super().__init__()
        self.masked_depth_image = masked_depth_image
        self.model = model
        self.device = next(model.parameters()).device

    def run(self):
        try:
            start_time = time.perf_counter()
            self.progress.emit(10)
            image_resized = TF.resize(self.masked_depth_image, (512, 512))
            image_tensor = TF.to_tensor(image_resized).to(self.device)
            if self.device.type == 'cuda': image_tensor = image_tensor.half()
            with torch.no_grad():
                predictions = self.model([image_tensor])[0]
            masks = predictions.get("masks", torch.empty((0, 1, 512, 512))).cpu().numpy()
            scores = predictions.get("scores", torch.empty((0,))).cpu().float().numpy()
            keep = scores >= 0.5
            masks = masks[keep]
            mask_list = [{"segmentation": (mask[0] > 0.5).astype(bool)} for mask in masks]
            mask_colors = self._generate_colors(len(masks))
            result_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
            for idx, mask in enumerate(mask_list): result_rgb[mask['segmentation']] = mask_colors[idx]
            self.progress.emit(100)
            elapsed_time = time.perf_counter() - start_time
            self.finished.emit(Image.fromarray(result_rgb), mask_list, elapsed_time, mask_colors)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

    def _generate_colors(self, num):
        np.random.seed(42)
        return [np.random.randint(0, 255, 3).tolist() for _ in range(num)]


class SamSegmentationWorker(QObject):
    finished = pyqtSignal(Image.Image, list, float, list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, raw_depth_image, mask_generator, img_size=(512, 512)):
        super().__init__()
        self.raw_depth_image = raw_depth_image
        self.mask_generator = mask_generator
        self.img_size = img_size

    def run(self):
        try:
            start_time = time.perf_counter()
            self.progress.emit(10)
            img_resized = self.raw_depth_image.resize(self.img_size)
            image_np = np.array(img_resized)

            if image_np.ndim == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif image_np.shape[2] == 1:
                pass

            sam_result = self.mask_generator.generate(image_np)
            self.progress.emit(80)

            mask_list = []
            for item in sam_result:
                mask_list.append({"segmentation": item['segmentation']})

            mask_colors = self._generate_colors(len(mask_list))
            result_rgb = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            sorted_indices = np.argsort([-np.sum(m['segmentation']) for m in mask_list])

            for idx in range(len(mask_list)):
                result_rgb[mask_list[idx]['segmentation']] = mask_colors[idx]

            elapsed_time = time.perf_counter() - start_time
            self.progress.emit(100)
            self.finished.emit(Image.fromarray(result_rgb), mask_list, elapsed_time, mask_colors)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

    def _generate_colors(self, num):
        np.random.seed(123)
        return [np.random.randint(0, 255, 3).tolist() for _ in range(num)]


class DirectAnnotationSegmentationWorker(QObject):
    finished = pyqtSignal(Image.Image, list, float, list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, annotation_data, img_size=(512, 512)):
        super().__init__()
        self.annotation_data = annotation_data
        self.img_size = img_size

    def run(self):
        try:
            start_time = time.perf_counter()
            self.progress.emit(10)
            if 'shapes' not in self.annotation_data: raise ValueError("Annotations 数据无效")
            mask_list = []
            shapes = self.annotation_data['shapes']
            for shape in shapes:
                points = shape.get('points', [])
                if not points: continue
                scaled_points = [(x * 512 / 1024, 511 - y * 512 / 1024) for x, y in points]
                single_mask = Image.new('L', self.img_size, 0)
                ImageDraw.Draw(single_mask).polygon(scaled_points, outline=1, fill=1)
                mask_list.append({"segmentation": np.array(single_mask).astype(bool)})
            mask_colors = self._generate_colors(len(mask_list))
            result_rgb = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            for idx, mask in enumerate(mask_list): result_rgb[mask['segmentation']] = mask_colors[idx]
            self.progress.emit(100)
            elapsed_time = time.perf_counter() - start_time
            self.finished.emit(Image.fromarray(result_rgb), mask_list, elapsed_time, mask_colors)
        except Exception as e:
            self.error.emit(str(e))

    def _generate_colors(self, num):
        np.random.seed(42)
        return [np.random.randint(0, 255, 3).tolist() for _ in range(num)]


class MetricsWorker(QObject):
    finished = pyqtSignal(dict, dict, dict)

    def __init__(self, pred_indices_map, gt_indices_map, iou_threshold=0.5):
        super().__init__()
        self.pred_indices_map = pred_indices_map
        self.gt_indices_map = gt_indices_map
        self.iou_threshold = iou_threshold

    def run(self):
        try:
            start_time = time.time()
            pred_to_gt = {}
            gt_to_pred = {}
            pred_ids = list(self.pred_indices_map.keys())
            gt_ids = list(self.gt_indices_map.keys())
            matches = []
            for p_id in pred_ids:
                p_set = self.pred_indices_map[p_id]
                if not p_set: continue
                for g_id in gt_ids:
                    g_set = self.gt_indices_map[g_id]
                    if not g_set: continue
                    intersection = len(p_set.intersection(g_set))
                    if intersection == 0: continue
                    union = len(p_set.union(g_set))
                    iou = intersection / union if union > 0 else 0
                    if iou >= self.iou_threshold: matches.append((iou, p_id, g_id))
            matches.sort(key=lambda x: x[0], reverse=True)
            used_p, used_g = set(), set()
            for iou, p_id, g_id in matches:
                if p_id not in used_p and g_id not in used_g:
                    pred_to_gt[p_id] = g_id
                    gt_to_pred[g_id] = p_id
                    used_p.add(p_id)
                    used_g.add(g_id)
            tp = len(used_p)
            fp = len(pred_ids) - tp
            fn = len(gt_ids) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics = {"precision": precision, "recall": recall, "f1": f1, "unmatched_pred": fp, "tp": tp,
                       "total_pred": len(pred_ids), "total_gt": len(gt_ids)}
            self.finished.emit(metrics, pred_to_gt, gt_to_pred)
        except Exception as e:
            print(f"指标计算错误: {e}")


# ---------------------------
# 主程序
# ---------------------------
class InferenceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Waste-Net复杂堆叠颗粒点云分割软件 - 匹配与粒径分布分析')
        self.setGeometry(100, 100, 1900, 1000)

        self.pixel_to_point_indices = None
        self.masked_depth_point_indices = set()
        self.pixel_area_physical = 0.0
        self.raw_height_map = None

        self.t_load = 0.0;
        self.t_proj = 0.0;
        self.t_peanet = 0.0
        self.t_mask_depth = 0.0;
        self.t_maskrcnn = 0.0;
        self.t_mapping = 0.0

        # PEA-Net / Mask R-CNN Data
        self.maskrcnn_point_indices = {}
        self.maskrcnn_masks = None
        self.mask_colors = None

        # Annotation Data
        self.annotation_data = None
        self.maskrcnn_annotation_point_indices = {}
        self.maskrcnn_annotation_masks = None
        self.maskrcnn_annotation_colors = None

        # SAM Data
        self.sam_masks = None
        self.sam_point_indices = {}
        self.sam_colors = None
        self.t_sam = 0.0

        # Matches
        self.matches_pred_to_gt = {}
        self.matches_gt_to_pred = {}
        self.points = None
        self.depth_image = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda': torch.backends.cudnn.benchmark = True

        # Load U-Net
        self.model = OriginalUNetWithCBAMAndLaplacian()
        if os.path.exists("best_model.pth"):
            try:
                ckpt = torch.load("best_model.pth", map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
                print("U-Net 模型加载成功")
            except Exception as e:
                print(f"U-Net 加载失败: {e}")
        self.model.to(self.device).eval()
        if self.device.type == 'cuda': self.model.half()

        # Load Mask R-CNN
        self.maskrcnn_model = get_maskrcnn_model(num_classes=2)
        if os.path.exists("mask_rcnn.pth"):
            try:
                ckpt = torch.load("mask_rcnn.pth", map_location=self.device)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    self.maskrcnn_model.load_state_dict(ckpt['model'])
                else:
                    self.maskrcnn_model.load_state_dict(ckpt)
                print("Mask R-CNN 模型预加载成功")
            except Exception as e:
                print(f"Mask R-CNN 加载失败: {e}")
        self.maskrcnn_model.to(self.device).eval()
        if self.device.type == 'cuda': self.maskrcnn_model.half()

        # Load SAM
        self.sam_model = None
        self.sam_mask_generator = None
        if HAS_SAM:
            sam_checkpoint = "vit_b.pth"
            model_type = "vit_b"
            if os.path.exists(sam_checkpoint):
                print(f"Loading SAM model from {sam_checkpoint}...")
                try:
                    self.sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    self.sam_model.to(device=self.device)
                    self.sam_mask_generator = SamAutomaticMaskGenerator(self.sam_model)
                    print("SAM Model Loaded Successfully.")
                except Exception as e:
                    print(f"SAM 加载出错: {e}")
            else:
                print(f"未找到 SAM 权重文件: {sam_checkpoint}")

        self.transform = transforms.Compose([ResizeInference((512, 512)), ToTensorInference()])
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        self.btn_load = QPushButton('选择点云文件')
        self.btn_load.clicked.connect(self.load_point_cloud)
        left_layout.addWidget(self.btn_load)

        self.btn_load_annotations = QPushButton('选择Annotations文件')
        self.btn_load_annotations.clicked.connect(self.load_annotations)
        self.btn_load_annotations.setEnabled(False)
        left_layout.addWidget(self.btn_load_annotations)

        self.btn_save = QPushButton('保存预测结果')
        self.btn_save.clicked.connect(self.save_prediction)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)

        self.btn_show_masked_depth = QPushButton('显示掩码对应深度图像')
        self.btn_show_masked_depth.clicked.connect(self.show_masked_depth)
        self.btn_show_masked_depth.setEnabled(False)
        left_layout.addWidget(self.btn_show_masked_depth)

        # PEA-Net / Mask R-CNN Section
        self.btn_maskrcnn_segment = QPushButton('PEA-Net + Mask R-CNN 分割')
        self.btn_maskrcnn_segment.clicked.connect(self.maskrcnn_segment)
        self.btn_maskrcnn_segment.setEnabled(False)
        left_layout.addWidget(self.btn_maskrcnn_segment)

        self.btn_save_maskrcnn_combined = QPushButton('保存 PEA-Net 整体点云')
        self.btn_save_maskrcnn_combined.clicked.connect(self.save_maskrcnn_combined_point_cloud)
        self.btn_save_maskrcnn_combined.setEnabled(False)
        left_layout.addWidget(self.btn_save_maskrcnn_combined)

        # SAM Section
        sam_group = QGroupBox("SAM Baseline Comparison")
        sam_layout = QVBoxLayout()
        self.btn_sam_segment = QPushButton('运行 SAM 直接分割 (vit_b)')
        self.btn_sam_segment.clicked.connect(self.sam_segment)
        self.btn_sam_segment.setEnabled(False)
        self.btn_sam_segment.setStyleSheet("background-color: #ffecd1; color: black;")
        sam_layout.addWidget(self.btn_sam_segment)

        self.btn_save_sam_combined = QPushButton('保存 SAM 整体点云')
        self.btn_save_sam_combined.clicked.connect(self.save_sam_combined_point_cloud)
        self.btn_save_sam_combined.setEnabled(False)
        sam_layout.addWidget(self.btn_save_sam_combined)
        sam_group.setLayout(sam_layout)
        left_layout.addWidget(sam_group)

        # Annotation Section
        self.btn_maskrcnn_segment_annotation = QPushButton('直接使用 Annotations 分割')
        self.btn_maskrcnn_segment_annotation.clicked.connect(self.maskrcnn_segment_from_annotation)
        self.btn_maskrcnn_segment_annotation.setEnabled(False)
        left_layout.addWidget(self.btn_maskrcnn_segment_annotation)

        self.btn_save_maskrcnn_annotation_combined = QPushButton('保存 (Annotations) 整体点云')
        self.btn_save_maskrcnn_annotation_combined.clicked.connect(self.save_maskrcnn_annotation_combined_point_cloud)
        self.btn_save_maskrcnn_annotation_combined.setEnabled(False)
        left_layout.addWidget(self.btn_save_maskrcnn_annotation_combined)

        self.btn_export_diameter = QPushButton('导出对比分析报告 (Excel)')
        self.btn_export_diameter.clicked.connect(self.export_matched_diameters)
        self.btn_export_diameter.setEnabled(False)
        self.btn_export_diameter.setStyleSheet("background-color: #d1e7dd; color: black; font-weight: bold;")
        left_layout.addWidget(self.btn_export_diameter)

        file_info_group = QGroupBox("当前文件信息")
        file_info_layout = QFormLayout()
        self.lbl_filename = QLabel("None")
        self.lbl_filename.setWordWrap(True)
        self.lbl_point_count = QLabel("0")
        self.lbl_point_count.setStyleSheet("color: darkgreen; font-weight: bold; font-size: 14px;")
        file_info_layout.addRow("Filename:", self.lbl_filename)
        file_info_layout.addRow("Total Points:", self.lbl_point_count)
        file_info_group.setLayout(file_info_layout)
        left_layout.addWidget(file_info_group)

        time_stats_group = QGroupBox("耗时统计")
        time_stats_layout = QFormLayout()
        self.lbl_t_load = QLabel("0.0000 s");
        self.lbl_t_proj = QLabel("0.0000 s")
        self.lbl_t_peanet = QLabel("0.0000 s");
        self.lbl_t_mask_depth = QLabel("0.0000 s")
        self.lbl_t_maskrcnn = QLabel("0.0000 s");
        self.lbl_t_mapping = QLabel("0.0000 s")
        self.lbl_t_total = QLabel("0.0000 s")
        style_sheet = "QLabel { color: blue; font-weight: bold; font-family: Consolas; font-size: 14px; }"
        for lbl in [self.lbl_t_load, self.lbl_t_proj, self.lbl_t_peanet, self.lbl_t_mask_depth, self.lbl_t_maskrcnn,
                    self.lbl_t_mapping, self.lbl_t_total]:
            lbl.setStyleSheet(style_sheet)
        time_stats_layout.addRow("1a. IO加载:", self.lbl_t_load)
        time_stats_layout.addRow("1b. 投影:", self.lbl_t_proj)
        time_stats_layout.addRow("2. Mask提取:", self.lbl_t_peanet)
        time_stats_layout.addRow("3. 深度掩码:", self.lbl_t_mask_depth)
        time_stats_layout.addRow("4. 实例分割:", self.lbl_t_maskrcnn)
        time_stats_layout.addRow("5. 映射:", self.lbl_t_mapping)
        time_stats_layout.addRow("Total:", self.lbl_t_total)
        time_stats_group.setLayout(time_stats_layout)
        left_layout.addWidget(time_stats_group)

        time_group = QGroupBox("处理进度")
        time_layout = QFormLayout()
        self.progress_depth = QProgressBar()
        time_layout.addRow("U-Net:", self.progress_depth)
        self.progress_maskrcnn = QProgressBar()
        time_layout.addRow("R-CNN/SAM:", self.progress_maskrcnn)
        time_group.setLayout(time_layout)
        left_layout.addWidget(time_group)

        metrics_group = QGroupBox("评估指标")
        metrics_layout = QFormLayout()
        self.label_precision = QLabel("Precision: N/A")
        self.label_recall = QLabel("Recall: N/A")
        self.label_f1 = QLabel("F1 Score: N/A")
        self.label_unmatched = QLabel("Unmatched Pred: N/A")
        self.label_tp = QLabel("Matched: N/A")
        font = self.label_precision.font();
        font.setBold(True)
        for lbl in [self.label_precision, self.label_recall, self.label_f1]: lbl.setFont(font)
        metrics_layout.addRow(self.label_precision)
        metrics_layout.addRow(self.label_recall)
        metrics_layout.addRow(self.label_f1)
        metrics_layout.addRow(self.label_unmatched)
        metrics_layout.addRow(self.label_tp)
        metrics_group.setLayout(metrics_layout)
        left_layout.addWidget(metrics_group)

        left_layout.addStretch()
        layout.addLayout(left_layout)

        # --- 右侧布局：图像 + 图表 ---
        right_main_layout = QVBoxLayout()

        grid_layout = QGridLayout()
        labels = []
        for i in range(5): labels.append(self._create_label())

        self.label_input = labels[0]
        grid_layout.addWidget(QLabel('输入深度图像'), 0, 0)
        grid_layout.addWidget(self.label_input, 1, 0)

        self.label_prediction = labels[1]
        grid_layout.addWidget(QLabel('PEA-Net 预测'), 0, 1)
        grid_layout.addWidget(self.label_prediction, 1, 1)

        self.label_masked_depth = ClickableLabel()
        self._setup_label(self.label_masked_depth)
        self.label_masked_depth.clicked.connect(self.on_masked_depth_clicked)
        grid_layout.addWidget(QLabel('掩码对应深度图'), 0, 2)
        grid_layout.addWidget(self.label_masked_depth, 1, 2)

        self.label_maskrcnn_segment = ClickableLabel()
        self._setup_label(self.label_maskrcnn_segment)
        self.label_maskrcnn_segment.clicked.connect(self.on_maskrcnn_segment_clicked)
        self.label_maskrcnn_segment.hovered.connect(self.on_pred_image_hover)
        grid_layout.addWidget(QLabel('Mask R-CNN 预测结果'), 0, 3)
        grid_layout.addWidget(self.label_maskrcnn_segment, 1, 3)

        self.label_annotation_mask = self._create_label()
        grid_layout.addWidget(QLabel('Annotations 掩码'), 2, 0)
        grid_layout.addWidget(self.label_annotation_mask, 3, 0)

        self.label_annotation_depth = self._create_label()
        grid_layout.addWidget(QLabel('Annotations 深度图'), 2, 1)
        grid_layout.addWidget(self.label_annotation_depth, 3, 1)

        self.label_maskrcnn_annotation_segment = ClickableLabel()
        self._setup_label(self.label_maskrcnn_annotation_segment)
        self.label_maskrcnn_annotation_segment.clicked.connect(self.on_maskrcnn_annotation_clicked)
        self.label_maskrcnn_annotation_segment.hovered.connect(self.on_gt_image_hover)
        grid_layout.addWidget(QLabel('Annotations 直接分割结果'), 2, 2)
        grid_layout.addWidget(self.label_maskrcnn_annotation_segment, 3, 2)

        self.label_sam_segment = ClickableLabel()
        self._setup_label(self.label_sam_segment)
        self.label_sam_segment.clicked.connect(self.on_sam_segment_clicked)
        grid_layout.addWidget(QLabel('SAM 直接分割结果'), 2, 3)
        grid_layout.addWidget(self.label_sam_segment, 3, 3)

        right_main_layout.addLayout(grid_layout)

        dist_group = QGroupBox("粒径分布统计 (Distribution Analysis)")
        dist_layout = QVBoxLayout()
        self.dist_canvas = DistributionCanvas(self, width=10, height=3)
        dist_layout.addWidget(self.dist_canvas)
        dist_group.setLayout(dist_layout)
        right_main_layout.addWidget(dist_group)

        layout.addLayout(right_main_layout)
        self.setLayout(layout)

    def _create_label(self):
        lbl = QLabel()
        self._setup_label(lbl)
        return lbl

    def _setup_label(self, lbl):
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(350, 350)
        lbl.setScaledContents(True)
        lbl.setStyleSheet("border: 1px solid black;")

    def _update_time_labels(self):
        total = self.t_load + self.t_proj + self.t_peanet + self.t_mask_depth + self.t_maskrcnn + self.t_mapping
        self.lbl_t_load.setText(f"{self.t_load:.4f} s")
        self.lbl_t_proj.setText(f"{self.t_proj:.4f} s")
        self.lbl_t_peanet.setText(f"{self.t_peanet:.4f} s")
        self.lbl_t_mask_depth.setText(f"{self.t_mask_depth:.4f} s")
        self.lbl_t_maskrcnn.setText(f"{self.t_maskrcnn:.4f} s")
        self.lbl_t_mapping.setText(f"{self.t_mapping:.4f} s")
        self.lbl_t_total.setText(f"{total:.4f} s")

    # ----------------------------------------------------
    # SAM Logic
    # ----------------------------------------------------
    def sam_segment(self):
        if self.sam_mask_generator is None:
            QMessageBox.critical(self, "错误", "SAM 模型未加载 (vit_b.pth 不存在或库缺失)")
            return

        self.thread_sam = QThread()
        self.worker_sam = SamSegmentationWorker(self.depth_image, self.sam_mask_generator)
        self.worker_sam.moveToThread(self.thread_sam)
        self.thread_sam.started.connect(self.worker_sam.run)
        self.worker_sam.finished.connect(self.on_sam_finished)
        self.worker_sam.progress.connect(self.progress_maskrcnn.setValue)
        self.worker_sam.finished.connect(self.thread_sam.quit)
        self.worker_sam.finished.connect(self.worker_sam.deleteLater)
        self.thread_sam.finished.connect(self.thread_sam.deleteLater)
        self.thread_sam.start()

    def on_sam_finished(self, img, masks, t_elapsed, colors):
        self.t_sam = t_elapsed
        self.display_image(img, self.label_sam_segment)
        self.sam_masks = masks
        self.sam_colors = colors

        print("Mapping SAM results to Point Cloud...")
        self.build_mapping_sam(masks)

        self.btn_save_sam_combined.setEnabled(True)
        self.btn_export_diameter.setEnabled(True)

        self.calculate_and_plot_distribution()
        QMessageBox.information(self, "SAM 完成", f"SAM 分割完成，检测到 {len(masks)} 个Mask。\n耗时: {t_elapsed:.4f}s")

    def build_mapping_sam(self, masks):
        mapping = {}
        for idx, m in enumerate(masks):
            mask_bool = m['segmentation']
            if mask_bool.shape != (512, 512):
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(
                    bool)

            indices = set()
            flat_indices = np.where(mask_bool.flatten())[0]
            for fi in flat_indices:
                indices.update(self.pixel_to_point_indices[fi])
            mapping[idx] = indices
        self.sam_point_indices = mapping

    def save_sam_combined_point_cloud(self):
        self._save_combined(self.sam_point_indices, self.sam_colors, "sam_combined.ply")

    def on_sam_segment_clicked(self, x, y):
        idx = self._find_mask(x, y, self.sam_masks, self.label_sam_segment)
        if idx is not None: self._save_pcd(self.sam_point_indices[idx], f"sam_instance_{idx}.ply")

    # ----------------------------------------------------
    # Existing Logic
    # ----------------------------------------------------
    def load_point_cloud(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择点云", "", "PLY (*.ply)")
        if filename:
            self._reset_ui()
            self.lbl_filename.setText(os.path.basename(filename))
            self.thread = QThread()
            self.worker = PointCloudWorker(filename)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_pc_finished)
            self.worker.progress.connect(self.progress_depth.setValue)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()

    def _reset_ui(self):
        self.btn_load.setEnabled(False)
        self.btn_export_diameter.setEnabled(False)
        self.maskrcnn_masks = None
        self.maskrcnn_annotation_masks = None
        self.sam_masks = None
        self.maskrcnn_point_indices = {}
        self.maskrcnn_annotation_point_indices = {}
        self.sam_point_indices = {}
        self.matches_pred_to_gt = {}
        self.matches_gt_to_pred = {}
        self.raw_height_map = None
        self.t_load = 0.0;
        self.t_proj = 0.0;
        self.t_peanet = 0.0
        self.t_mask_depth = 0.0;
        self.t_maskrcnn = 0.0;
        self.t_mapping = 0.0
        self._update_time_labels()
        self.lbl_point_count.setText("0")
        for lbl in [self.label_precision, self.label_recall, self.label_f1]: lbl.setText("N/A")
        if hasattr(self, 'dist_canvas'):
            self.dist_canvas.ax.clear()
            self.dist_canvas.draw()
        self.btn_maskrcnn_segment.setEnabled(False)
        self.btn_sam_segment.setEnabled(False)

    def on_pc_finished(self, depth, indices, points, total_t, count, raw_height_map, t_load, t_proj):
        self.t_load = t_load;
        self.t_proj = t_proj
        self.lbl_point_count.setText(f"{count:,}")
        self.display_image(depth, self.label_input)
        self.pixel_to_point_indices = indices
        self.points = points
        self.depth_image = depth
        self.raw_height_map = raw_height_map
        x_min, x_max = self.points[:, 0].min(), self.points[:, 0].max()
        y_min, y_max = self.points[:, 1].min(), self.points[:, 1].max()
        scale_x = (x_max - x_min) / 512
        scale_y = (y_max - y_min) / 512
        self.pixel_area_physical = scale_x * scale_y
        t_net_start = time.perf_counter()
        inp = self.transform(depth).unsqueeze(0).to(self.device)
        lap = cv2.Laplacian(np.array(depth.convert('L')), cv2.CV_64F)
        lap = np.abs(lap)
        lap = (255 * (lap / np.max(lap))).astype(np.uint8) if np.max(lap) != 0 else lap.astype(np.uint8)
        lap_t = TF.to_tensor(Image.fromarray(lap)).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda': inp = inp.half(); lap_t = lap_t.half()
        with torch.no_grad():
            out = self.model(inp, lap_t)
            pred = (torch.sigmoid(out) > 0.5).float().cpu().numpy()[0, 0]
        self.t_peanet = time.perf_counter() - t_net_start
        pred_img = Image.fromarray((pred * 255).astype(np.uint8)).resize(depth.size).convert('L')
        self.prediction_image = pred_img
        self.display_image(pred_img, self.label_prediction)
        mask_flat = np.array(pred_img).flatten() > 127
        self.masked_depth_point_indices = set()
        flat_indices = np.where(mask_flat)[0]
        for i in flat_indices: self.masked_depth_point_indices.update(self.pixel_to_point_indices[i])
        self.btn_save.setEnabled(True)
        self.btn_show_masked_depth.setEnabled(True)
        self.btn_load_annotations.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_sam_segment.setEnabled(True)
        self._update_time_labels()

    def load_annotations(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择Annotations", "", "JSON (*.json)")
        if filename:
            with open(filename, 'r') as f:
                self.annotation_data = json.load(f)
            mask = Image.new('L', (512, 512), 0)
            draw = ImageDraw.Draw(mask)
            for s in self.annotation_data.get('shapes', []):
                pts = [(x * 512 / 1024, 511 - y * 512 / 1024) for x, y in s['points']]
                draw.polygon(pts, outline=255, fill=255)
            self.display_image(mask, self.label_annotation_mask)
            if self.depth_image:
                d = np.array(self.depth_image)
                m = (np.array(mask) > 127).astype(np.uint8)
                ad = d * np.expand_dims(m, 2)
                self.annotation_depth_image = Image.fromarray(ad)
                self.display_image(self.annotation_depth_image, self.label_annotation_depth)
                self.btn_maskrcnn_segment_annotation.setEnabled(True)

    def show_masked_depth(self):
        self.thread_md = QThread()
        self.worker_md = MaskedDepthWorker(self.depth_image, self.prediction_image)
        self.worker_md.moveToThread(self.thread_md)
        self.thread_md.started.connect(self.worker_md.run)
        self.worker_md.finished.connect(self.on_md_finished)
        self.worker_md.finished.connect(self.thread_md.quit)
        self.worker_md.finished.connect(self.worker_md.deleteLater)
        self.thread_md.finished.connect(self.thread_md.deleteLater)
        self.thread_md.start()

    def on_md_finished(self, res, t_elapsed):
        self.t_mask_depth = t_elapsed
        self._update_time_labels()
        self.display_image(res, self.label_masked_depth)
        self.masked_depth_image = res
        self.btn_maskrcnn_segment.setEnabled(True)

    def maskrcnn_segment(self):
        self.run_seg(self.masked_depth_image, False)

    def maskrcnn_segment_from_annotation(self):
        self.thread_anno = QThread()
        self.worker_anno = DirectAnnotationSegmentationWorker(self.annotation_data)
        self.worker_anno.moveToThread(self.thread_anno)
        self.thread_anno.started.connect(self.worker_anno.run)
        self.worker_anno.finished.connect(self.on_anno_finished)
        self.worker_anno.progress.connect(self.progress_maskrcnn.setValue)
        self.worker_anno.finished.connect(self.thread_anno.quit)
        self.worker_anno.finished.connect(self.worker_anno.deleteLater)
        self.thread_anno.finished.connect(self.thread_anno.deleteLater)
        self.thread_anno.start()

    def run_seg(self, img, is_anno):
        self.thread_seg = QThread()
        self.worker_seg = MaskRCNNSegmentationWorker(img, self.maskrcnn_model)
        self.worker_seg.moveToThread(self.thread_seg)
        self.thread_seg.started.connect(self.worker_seg.run)
        self.worker_seg.finished.connect(self.on_seg_finished)
        self.worker_seg.progress.connect(self.progress_maskrcnn.setValue)
        self.worker_seg.finished.connect(self.thread_seg.quit)
        self.worker_seg.finished.connect(self.worker_seg.deleteLater)
        self.thread_seg.finished.connect(self.thread_seg.deleteLater)
        self.thread_seg.start()

    def on_seg_finished(self, img, masks, t_elapsed, colors):
        self.t_maskrcnn = t_elapsed
        self.display_image(img, self.label_maskrcnn_segment)
        self.maskrcnn_masks = masks
        self.mask_colors = colors
        t_map_start = time.perf_counter()
        self.build_mapping(masks, False)
        self.t_mapping = time.perf_counter() - t_map_start
        self._update_time_labels()
        self.btn_save_maskrcnn_combined.setEnabled(True)
        self.trigger_metrics()

    def on_anno_finished(self, img, masks, t, colors):
        self.display_image(img, self.label_maskrcnn_annotation_segment)
        self.maskrcnn_annotation_masks = masks
        self.maskrcnn_annotation_colors = colors
        self.build_mapping(masks, True)
        self.btn_save_maskrcnn_annotation_combined.setEnabled(True)
        self.trigger_metrics()

    def build_mapping(self, masks, is_anno):
        mapping = {}
        for idx, m in enumerate(masks):
            mr = cv2.resize(m['segmentation'].astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            indices = set()
            flat_indices = np.where(mr.flatten())[0]
            for fi in flat_indices: indices.update(self.pixel_to_point_indices[fi])
            mapping[idx] = indices
        if is_anno:
            self.maskrcnn_annotation_point_indices = mapping
        else:
            self.maskrcnn_point_indices = mapping

    def trigger_metrics(self):
        if self.maskrcnn_point_indices and self.maskrcnn_annotation_point_indices:
            self.thread_met = QThread()
            self.worker_met = MetricsWorker(self.maskrcnn_point_indices, self.maskrcnn_annotation_point_indices)
            self.worker_met.moveToThread(self.thread_met)
            self.thread_met.started.connect(self.worker_met.run)
            self.worker_met.finished.connect(self.on_metrics_done)
            self.worker_met.finished.connect(self.thread_met.quit)
            self.worker_met.finished.connect(self.worker_met.deleteLater)
            self.thread_met.finished.connect(self.thread_met.deleteLater)
            self.thread_met.start()

    def _get_colored_mask_image(self, masks, colors):
        result_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        for idx, mask in enumerate(masks):
            m = mask['segmentation']
            if m.shape != (512, 512):
                m = cv2.resize(m.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
            result_rgb[m] = colors[idx]
        return Image.fromarray(result_rgb)

    def _draw_match_outlines(self, base_img_pil, masks, match_map):
        cv_img = cv2.cvtColor(np.array(base_img_pil), cv2.COLOR_RGB2BGR)
        for idx, mask_info in enumerate(masks):
            if idx in match_map:
                m = mask_info['segmentation'].astype(np.uint8)
                if m.shape != (512, 512): m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(cv_img, contours, -1, (255, 255, 255), 2)
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def on_metrics_done(self, m, p2g, g2p):
        self.label_precision.setText(f"Precision: {m['precision']:.4f}")
        self.label_recall.setText(f"Recall: {m['recall']:.4f}")
        self.label_f1.setText(f"F1 Score: {m['f1']:.4f}")
        self.label_unmatched.setText(f"Unmatched Preds: {m['unmatched_pred']}")
        self.label_tp.setText(f"Matched (TP): {m['tp']} / {m['total_gt']}")
        self.matches_pred_to_gt = p2g
        self.matches_gt_to_pred = g2p
        if self.maskrcnn_masks:
            raw_pred_rgb = self._get_colored_mask_image(self.maskrcnn_masks, self.mask_colors)
            outlined_pred = self._draw_match_outlines(raw_pred_rgb, self.maskrcnn_masks, self.matches_pred_to_gt)
            self.display_image(outlined_pred, self.label_maskrcnn_segment)
        if self.maskrcnn_annotation_masks:
            raw_gt_rgb = self._get_colored_mask_image(self.maskrcnn_annotation_masks, self.maskrcnn_annotation_colors)
            outlined_gt = self._draw_match_outlines(raw_gt_rgb, self.maskrcnn_annotation_masks, self.matches_gt_to_pred)
            self.display_image(outlined_gt, self.label_maskrcnn_annotation_segment)

        self.calculate_and_plot_distribution()
        self.btn_export_diameter.setEnabled(True)
        QMessageBox.information(self, "完成", "基于点云的评估指标计算完成")

    # ---------------------------
    # 分布计算 & Excel 导出
    # ---------------------------
    def calculate_and_plot_distribution(self):
        # 使用新的计算函数，只取 percents 用于绘图
        _, pred_percents = self._calc_dist_data(self.maskrcnn_masks)
        _, gt_percents = self._calc_dist_data(self.maskrcnn_annotation_masks)
        if self.sam_masks:
            _, sam_percents = self._calc_dist_data(self.sam_masks)
        else:
            sam_percents = None

        self.dist_canvas.plot_distribution(pred_percents, gt_percents, sam_percents)

    def _calc_dist_data(self, masks):
        vols = [0.0] * 7
        total_vol = 0.0
        if masks:
            for mask in masks:
                m = mask['segmentation']
                if m.shape != (512, 512):
                    m = cv2.resize(m.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
                vol, d3d = self._calc_volume_and_3d_diameter(m)
                bin_idx = np.digitize([d3d], BIN_EDGES)[0] - 1
                if 0 <= bin_idx < 7:
                    vols[bin_idx] += vol
                    total_vol += vol
        percents = [(v / total_vol * 100) if total_vol > 0 else 0 for v in vols]
        return vols, percents

    def export_matched_diameters(self):
        if not self.matches_gt_to_pred and not self.maskrcnn_masks and not self.sam_masks:
            QMessageBox.warning(self, "错误", "无数据可导出")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "导出Excel", "particle_analysis_full.xlsx",
                                                   "Excel Files (*.xlsx)")
        if not save_path: return

        # --- Sheet 1: PEA-Net 详细匹配数据 ---
        detailed_rows = []
        # ... (Existing PEA-Net matching logic) ...
        for gt_id, pred_id in self.matches_gt_to_pred.items():
            gt_mask = self._resize_mask_if_needed(self.maskrcnn_annotation_masks[gt_id]['segmentation'])
            pred_mask = self._resize_mask_if_needed(self.maskrcnn_masks[pred_id]['segmentation'])
            vol_gt, d_gt_3d = self._calc_volume_and_3d_diameter(gt_mask)
            vol_pred, d_pred_3d = self._calc_volume_and_3d_diameter(pred_mask)
            detailed_rows.append({
                "Type": "Matched", "GT_ID": gt_id, "Pred_ID": pred_id,
                "GT_Diameter(mm)": d_gt_3d, "Pred_Diameter(mm)": d_pred_3d,
                "Diff_Dia(%)": ((d_pred_3d - d_gt_3d) / d_gt_3d * 100) if d_gt_3d else 0,
                "GT_Volume(mm3)": vol_gt, "Pred_Volume(mm3)": vol_pred
            })

        # Unmatched GT
        all_gt_ids = set(range(len(self.maskrcnn_annotation_masks))) if self.maskrcnn_annotation_masks else set()
        matched_gt_ids = set(self.matches_gt_to_pred.keys())
        for gt_id in (all_gt_ids - matched_gt_ids):
            gt_mask = self._resize_mask_if_needed(self.maskrcnn_annotation_masks[gt_id]['segmentation'])
            vol_gt, d_gt_3d = self._calc_volume_and_3d_diameter(gt_mask)
            detailed_rows.append({
                "Type": "Unmatched GT", "GT_ID": gt_id, "Pred_ID": None,
                "GT_Diameter(mm)": d_gt_3d, "Pred_Diameter(mm)": None,
                "Diff_Dia(%)": None, "GT_Volume(mm3)": vol_gt, "Pred_Volume(mm3)": None
            })

        # Unmatched Pred
        all_pred_ids = set(range(len(self.maskrcnn_masks))) if self.maskrcnn_masks else set()
        matched_pred_ids = set(self.matches_pred_to_gt.keys())
        for pred_id in (all_pred_ids - matched_pred_ids):
            pred_mask = self._resize_mask_if_needed(self.maskrcnn_masks[pred_id]['segmentation'])
            vol_pred, d_pred_3d = self._calc_volume_and_3d_diameter(pred_mask)
            detailed_rows.append({
                "Type": "Unmatched Pred", "GT_ID": None, "Pred_ID": pred_id,
                "GT_Diameter(mm)": None, "Pred_Diameter(mm)": d_pred_3d,
                "Diff_Dia(%)": None, "GT_Volume(mm3)": None, "Pred_Volume(mm3)": vol_pred
            })
        df_detail = pd.DataFrame(detailed_rows)

        # --- Sheet 2: SAM 原始分析数据 ---
        sam_rows = []
        if self.sam_masks:
            for idx, mask in enumerate(self.sam_masks):
                m = self._resize_mask_if_needed(mask['segmentation'])
                vol, d3d = self._calc_volume_and_3d_diameter(m)
                sam_rows.append({
                    "SAM_ID": idx,
                    "Diameter(mm)": d3d,
                    "Volume(mm3)": vol,
                    "Pixel_Area": np.sum(m)
                })
        df_sam = pd.DataFrame(sam_rows)

        # --- Sheet 3: 分布对比 (Distribution Summary) ---
        # 计算所有数据的体积和占比
        pred_vols, pred_percents = self._calc_dist_data(self.maskrcnn_masks)
        gt_vols, gt_percents = self._calc_dist_data(self.maskrcnn_annotation_masks)
        if self.sam_masks:
            sam_vols, sam_percents = self._calc_dist_data(self.sam_masks)
        else:
            sam_vols, sam_percents = [0] * 7, [0] * 7

        summary_rows = []
        for i, label in enumerate(BIN_LABELS):
            summary_rows.append({
                "Range": label,
                "GT_Volume_Sum(mm3)": gt_vols[i],
                "GT_Percent(%)": gt_percents[i],
                "PEA_Net_Volume_Sum(mm3)": pred_vols[i],
                "PEA_Net_Percent(%)": pred_percents[i],
                "SAM_Volume_Sum(mm3)": sam_vols[i],
                "SAM_Percent(%)": sam_percents[i],
            })
        df_summary = pd.DataFrame(summary_rows)

        try:
            with pd.ExcelWriter(save_path) as writer:
                df_detail.to_excel(writer, sheet_name='PEA-Net Matched', index=False)
                df_sam.to_excel(writer, sheet_name='SAM Analysis', index=False)
                df_summary.to_excel(writer, sheet_name='Distribution Comparison', index=False)
            QMessageBox.information(self, "成功", f"完整分析报告已导出至 {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def _resize_mask_if_needed(self, mask):
        if mask.shape != (512, 512):
            return cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask

    def _calc_projected_diameter(self, mask_binary):
        if mask_binary is None: return 0.0
        pixel_count = np.sum(mask_binary)
        if pixel_count == 0: return 0.0
        area_physical = pixel_count * self.pixel_area_physical
        return np.sqrt(4 * area_physical / np.pi)

    def _calc_volume_and_3d_diameter(self, mask_binary):
        if mask_binary is None or self.raw_height_map is None: return 0.0, 0.0
        heights = self.raw_height_map[mask_binary > 0]
        if len(heights) == 0: return 0.0, 0.0
        volume = np.sum(heights) * self.pixel_area_physical
        if volume > 0:
            d_3d = np.cbrt(6 * volume / np.pi)
        else:
            d_3d = 0.0
        return volume, d_3d

    def on_pred_image_hover(self, x, y):
        self._handle_hover(x, y, self.maskrcnn_masks, self.label_maskrcnn_segment, True)

    def on_gt_image_hover(self, x, y):
        self._handle_hover(x, y, self.maskrcnn_annotation_masks, self.label_maskrcnn_annotation_segment, False)

    def _handle_hover(self, x, y, masks, lbl, is_pred):
        if masks is None: return
        idx = self._find_mask(x, y, masks, lbl)
        if idx is not None:
            if is_pred:
                match = self.matches_pred_to_gt.get(idx)
                msg = f"<b>Pred: {idx}</b><br><span style='color:{'green' if match is not None else 'red'}'>{'Matched GT: ' + str(match) if match is not None else 'Unmatched'}</span>"
            else:
                match = self.matches_gt_to_pred.get(idx)
                msg = f"<b>GT: {idx}</b><br><span style='color:{'green' if match is not None else 'orange'}'>{'Matched Pred: ' + str(match) if match is not None else 'Not Detected'}</span>"
            QToolTip.showText(QCursor.pos(), msg)
        else:
            QToolTip.hideText()

    def _find_mask(self, x, y, masks, lbl):
        w, h = lbl.width(), lbl.height()
        ix, iy = int(x * 512 / w), int(y * 512 / h)
        for i in range(len(masks) - 1, -1, -1):
            m = cv2.resize(masks[i]['segmentation'].astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            if 0 <= iy < 512 and 0 <= ix < 512 and m[iy, ix]: return i
        return None

    def on_masked_depth_clicked(self, x, y):
        if not self.masked_depth_point_indices: return
        self._save_pcd(self.masked_depth_point_indices, "masked_depth.ply")

    def on_maskrcnn_segment_clicked(self, x, y):
        idx = self._find_mask(x, y, self.maskrcnn_masks, self.label_maskrcnn_segment)
        if idx is not None: self._save_pcd(self.maskrcnn_point_indices[idx], f"pred_instance_{idx}.ply")

    def on_maskrcnn_annotation_clicked(self, x, y):
        idx = self._find_mask(x, y, self.maskrcnn_annotation_masks, self.label_maskrcnn_annotation_segment)
        if idx is not None: self._save_pcd(self.maskrcnn_annotation_point_indices[idx], f"gt_instance_{idx}.ply")

    def _save_pcd(self, indices, default_name):
        if not indices: return
        if isinstance(indices, set):
            indices_list = list(indices)
        else:
            indices_list = indices
        pts = self.points[indices_list]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        fn, _ = QFileDialog.getSaveFileName(self, "保存点云", default_name, "PLY (*.ply)")
        if fn: o3d.io.write_point_cloud(fn, pcd)

    def save_prediction(self):
        fn, _ = QFileDialog.getSaveFileName(self, "保存掩码", "mask.png", "PNG (*.png)")
        if fn: self.prediction_image.save(fn)

    def save_maskrcnn_combined_point_cloud(self):
        self._save_combined(self.maskrcnn_point_indices, self.mask_colors, "pred_combined.ply")

    def save_maskrcnn_annotation_combined_point_cloud(self):
        self._save_combined(self.maskrcnn_annotation_point_indices, self.maskrcnn_annotation_colors, "gt_combined.ply")

    def _save_combined(self, mapping, colors, default_name):
        all_indices = set()
        for idx_list in mapping.values(): all_indices.update(idx_list)
        if not all_indices: return
        all_indices_list = list(all_indices)
        pts = self.points[all_indices_list]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        c_arr = np.zeros((len(pts), 3))
        idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(all_indices_list)}
        for mask_id, p_indices in mapping.items():
            if mask_id < len(colors):
                c = np.array(colors[mask_id]) / 255.0
                for pi in p_indices:
                    if pi in idx_map: c_arr[idx_map[pi]] = c
        pcd.colors = o3d.utility.Vector3dVector(c_arr)
        fn, _ = QFileDialog.getSaveFileName(self, "保存整体点云", default_name, "PLY (*.ply)")
        if fn: o3d.io.write_point_cloud(fn, pcd)

    def display_image(self, img, lbl):
        if img.mode != 'RGB': img = img.convert('RGB')
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InferenceApp()
    window.show()
    sys.exit(app.exec_())