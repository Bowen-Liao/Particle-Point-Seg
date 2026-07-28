import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import InterpolationMode
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging
import traceback
import torch.nn.functional as F
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import (
    InterpolationMode,
    ColorJitter,
    GaussianBlur,           # 必须用 torchvision 的
    RandomApply
)
# 设置 Matplotlib 的字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# ---------------------------
# 设置日志记录
# ---------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------
# 定义 CBAM 模块
# ---------------------------
# 定义 CBAM 模块（通道注意力 + 空间注意力）
# ---------------------------
# Channel Attention Module
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 修改卷积输入通道数为 in_channels
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


# ---------------------------
# Spatial Attention Module
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        out = self.conv(concat)
        return self.sigmoid(out)


# ---------------------------
# CBAM Module
# ---------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


# ---------------------------
# Double Convolution Block
# ---------------------------
class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""

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


# ---------------------------
# U-Net with CBAM and Laplacian Integration
# ---------------------------
class OriginalUNetWithCBAMAndLaplacian(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, laplacian_channels=1, features=[64, 128, 256, 512]):
        super(OriginalUNetWithCBAMAndLaplacian, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        for feature in features:
            self.encoder_layers.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器
        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.cbams = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # 修复：CBAM 的输入通道就是当前层的 skip_connection 通道数 = feature
            self.cbams.append(CBAM(feature, reduction=16, kernel_size=7))
            # decoder 输入是上采样特征（feature） + 增强后的 skip（feature）→ 2*feature
            self.decoder_layers.append(DoubleConv(feature * 2, feature))

        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # 处理 Laplacian 的卷积层
        self.laplacian_conv = nn.Sequential(
            nn.Conv2d(laplacian_channels, laplacian_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(laplacian_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, laplacian):
        skip_connections = []

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # 只处理一次 Laplacian（在最深层处理，然后逐层上采样）
        laplacian = self.laplacian_conv(laplacian)  # [B,1,H,W] → [B,1,H,W]

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)  # 上采样
            skip_connection = skip_connections[idx]

            # 对齐上采样后的 x 和 skip（极少数情况需要）
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # Laplacian 同步放大到当前层尺度
            laplacian_resized = F.interpolate(laplacian, size=skip_connection.shape[2:], mode='nearest')

            # 你最宝贵的残差边缘增强操作（完全保留！）
            edge_guidance = skip_connection * laplacian_resized
            enhanced_skip = skip_connection + edge_guidance  # <--- 这就是你想要的核心！

            # 直接送进 CBAM 做注意力（通道数不变）
            enhanced_skip = self.cbams[idx](enhanced_skip)

            # 直接拼接！不要任何降维！
            x = torch.cat([enhanced_skip, x], dim=1)

            # 标准 DoubleConv
            x = self.decoder_layers[idx](x)

        return self.final_conv(x)



# ---------------------------
# 定义权重初始化函数
# ---------------------------
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class StrongAugment(object):
    def __call__(self, image, mask, laplacian):
        # 用同一个随机参数手动控制所有变换
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            laplacian = TF.hflip(laplacian)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            laplacian = TF.vflip(laplacian)

        if random.random() < 0.7:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            laplacian = TF.rotate(laplacian, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.6:
            scale = random.uniform(0.9, 1.1)
            tx = int(random.uniform(-0.1, 0.1) * image.width)
            ty = int(random.uniform(-0.1, 0.1) * image.height)
            shear = (random.uniform(-10, 10), random.uniform(-10, 10))

            image = TF.affine(image, angle=0, translate=(tx, ty), scale=scale, shear=shear,
                              interpolation=InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=(tx, ty), scale=scale, shear=shear,
                             interpolation=InterpolationMode.NEAREST)
            laplacian = TF.affine(laplacian, angle=0, translate=(tx, ty), scale=scale, shear=shear,
                                  interpolation=InterpolationMode.NEAREST)

        # 颜色增强只对 image
        if random.random() < 0.8:
            image = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)(image)
        # ... 其他颜色增强

        return image, mask, laplacian
# ---------------------------
# 数据加载和预处理
# ---------------------------
class ComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, laplacian):
        for t in self.transforms:
            image, mask, laplacian = t(image, mask, laplacian)
        return image, mask, laplacian


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask, laplacian):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        # 使用最近邻插值缩放掩码和 Laplacian，防止生成伪像素
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        laplacian = TF.resize(laplacian, self.size, interpolation=InterpolationMode.NEAREST)
        return image, mask, laplacian


class ToTensorCustom(object):
    def __call__(self, image, mask, laplacian):
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32))  # 确保为 float32
        mask = (mask > 0).float()  # 确保掩码为 0 和 1
        mask = mask.unsqueeze(0)  # 添加通道维度，形状变为 [1, H, W]
        laplacian = torch.from_numpy(np.array(laplacian, dtype=np.float32))
        laplacian = laplacian.unsqueeze(0)  # 添加通道维度，形状变为 [1, H, W]
        return image, mask, laplacian


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask, laplacian):
        if np.random.rand() < self.prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            laplacian = TF.hflip(laplacian)
        return image, mask, laplacian


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask, laplacian):
        angle = np.random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        laplacian = TF.rotate(laplacian, angle, interpolation=InterpolationMode.NEAREST)
        return image, mask, laplacian


# --------------------------- 最终训练/验证增强（直接替换你原来的） ---------------------------
data_transforms_train = ComposeTransforms([
    Resize((512, 512)),
    StrongAugment(),           # 完美修复版增强
    ToTensorCustom(),
])

data_transforms_val = ComposeTransforms([
    Resize((512, 512)),
    ToTensorCustom(),
])

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 更新类别映射，根据您的实际类别
        self.category_to_id = {
            "background": 0,
            "1": 1,  # 添加您的类别，例如类别标签为 "1"
            # 如果有更多类别，继续添加
        }

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        try:
            img_filename = self.image_filenames[idx]
            img_path = os.path.join(self.images_dir, img_filename)
            annotation_filename = os.path.splitext(img_filename)[0] + '.json'
            annotation_path = os.path.join(self.annotations_dir, annotation_filename)

            image = Image.open(img_path).convert('RGB')

            # 获取图像尺寸
            width, height = image.size

            # 创建空的掩码
            mask = Image.new('L', (width, height), 0)  # PIL 图像的尺寸顺序为 (宽, 高)

            # 读取对应的 JSON 标注文件
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            # 检查标注文件是否包含 'shapes' 键
            if 'shapes' in annotation_data:
                for shape in annotation_data['shapes']:
                    if 'points' in shape and 'label' in shape:
                        category = shape['label']
                        category_id = self.category_to_id.get(category, 0)
                        points = shape['points']
                        # 将 points 转换为多边形坐标
                        poly = [tuple(point) for point in points]
                        # 绘制多边形到掩码上
                        ImageDraw.Draw(mask).polygon(poly, outline=category_id, fill=category_id)
                    else:
                        logger.warning(f"标注缺少 'points' 或 'label' 字段。文件: {annotation_filename}")
            else:
                logger.warning(f"标注文件 {annotation_filename} 中未找到 'shapes' 键。")

            # 计算 Laplacian 图像
            gray_image = image.convert('L')
            gray_np = np.array(gray_image)
            laplacian = cv2.Laplacian(gray_np, cv2.CV_64F)
            laplacian = np.abs(laplacian)
            # 归一化 Laplacian 到 [0, 255]
            if np.max(laplacian) != 0:
                laplacian = (255 * (laplacian / np.max(laplacian))).astype(np.uint8)
            else:
                laplacian = laplacian.astype(np.uint8)
            laplacian = Image.fromarray(laplacian)

            if self.transform:
                image, mask, laplacian = self.transform(image, mask, laplacian)

            return image, mask, laplacian

        except UnicodeDecodeError as e:
            logger.error(f"UnicodeDecodeError 在处理索引 {idx} ({img_filename}) 时: {e}")
            return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512), torch.zeros(1, 512, 512)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError 在处理索引 {idx} ({img_filename}) 时: {e}")
            return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512), torch.zeros(1, 512, 512)
        except Exception as e:
            logger.error(f"在处理索引 {idx} ({img_filename}) 时发生错误: {e}")
            logger.error(traceback.format_exc())
            # 返回全零的掩码作为默认值，避免训练中断
            return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512), torch.zeros(1, 512, 512)


# ---------------------------
# 定义损失函数
# ---------------------------
def dice_loss(pred, target, smooth=1e-6):
    """
    计算 Dice 损失
    """
    pred = torch.sigmoid(pred)  # 将 logits 转换为概率
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def combined_loss(pred, target, alpha=0, beta=1, gamma=0.0, smooth=1e-6):
    """
    组合损失函数：BCE + Dice

    Args:
        pred (torch.Tensor): 预测掩码，未激活。
        target (torch.Tensor): 真实掩码，二值化。
        alpha (float): BCE 损失的权重。
        beta (float): Dice 损失的权重。
        gamma (float): Boundary 损失的权重（未使用）。
        smooth (float): 平滑因子。

    Returns:
        tuple: (total_loss, bce_loss, dice_loss)
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target, smooth)
    total_loss = alpha * bce + beta * dice  # 不使用 gamma * boundary_loss
    return total_loss, bce, dice  # 返回总损失、BCE 损失和 Dice 损失


# ---------------------------
# 定义性能指标计算函数
# ---------------------------
def compute_dsc(pred_mask, true_mask, smooth=1e-6):
    """
    计算 Dice 相似系数 (DSC)
    """
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    dsc = (2. * intersection + smooth) / (pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) + smooth)
    dsc = dsc.mean().item()
    return dsc


def compute_accuracy(pred_mask, true_mask):
    """
    计算准确率 (Accuracy)
    """
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    correct = (pred_mask == true_mask).float()
    acc = correct.sum(dim=(1, 2, 3)) / (correct.size(1) * correct.size(2) * correct.size(3))
    acc = acc.mean().item()
    return acc


def compute_precision(pred_mask, true_mask, smooth=1e-6):
    """
    计算精确率 (Precision)
    """
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    true_positive = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    predicted_positive = pred_mask.sum(dim=(1, 2, 3))

    precision = (true_positive + smooth) / (predicted_positive + smooth)
    precision = precision.mean().item()
    return precision


def compute_recall(pred_mask, true_mask, smooth=1e-6):
    """
    计算召回率 (Recall)
    """
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    true_positive = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    actual_positive = true_mask.sum(dim=(1, 2, 3))

    recall = (true_positive + smooth) / (actual_positive + smooth)
    recall = recall.mean().item()
    return recall


# ---------------------------
# 定义训练和验证循环（包含提前停止和混合精度）
# ---------------------------
def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, patience=5, device='cuda'):
    model = model.to(device)
    model.apply(init_weights)  # 初始化权重
    logger.info("模型已初始化并移动到设备。")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler()
    criterion = combined_loss

    # 初始化用于存储训练和验证指标的列表
    train_losses = []
    train_bce_losses = []
    train_dice_losses = []
    train_dscs = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []

    val_losses = []
    val_bce_losses = []
    val_dice_losses = []
    val_dscs = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []

    best_val_dsc = 0.0  # 用于保存最佳模型
    epochs_no_improve = 0

    # 开始训练循环
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0
        running_dsc = 0.0
        running_acc = 0.0
        running_precision = 0.0
        running_recall = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")

        for i, (images, masks, laplacians) in enumerate(train_loader_tqdm):
            images = images.to(device)
            masks = masks.to(device).float()
            laplacians = laplacians.to(device).float()

            optimizer.zero_grad()

            with autocast():  # 混合精度
                # 前向传播
                outputs = model(images, laplacians)
                loss, bce, dice = criterion(outputs, masks)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_bce_loss += bce.item()
            running_dice_loss += dice.item()

            # 计算指标
            dsc = compute_dsc(outputs, masks)
            acc = compute_accuracy(outputs, masks)
            precision = compute_precision(outputs, masks)
            recall = compute_recall(outputs, masks)

            running_dsc += dsc
            running_acc += acc
            running_precision += precision
            running_recall += recall

            if (i + 1) % 10 == 0:
                train_loader_tqdm.set_postfix({
                    "Loss": loss.item(),
                    "BCE": bce.item(),
                    "Dice": dice.item(),
                    "DSC": dsc,
                    "Accuracy": acc,
                    "Precision": precision,
                    "Recall": recall
                })

        epoch_loss = running_loss / len(train_loader)
        epoch_bce_loss = running_bce_loss / len(train_loader)
        epoch_dice_loss = running_dice_loss / len(train_loader)
        epoch_dsc = running_dsc / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        epoch_precision = running_precision / len(train_loader)
        epoch_recall = running_recall / len(train_loader)
        logger.info(
            f"训练 - 平均损失: {epoch_loss:.4f}, BCE 损失: {epoch_bce_loss:.4f}, Dice 损失: {epoch_dice_loss:.4f}, "
            f"DSC: {epoch_dsc:.4f}, Accuracy: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}"
        )

        # 记录训练指标
        train_losses.append(epoch_loss)
        train_bce_losses.append(epoch_bce_loss)
        train_dice_losses.append(epoch_dice_loss)
        train_dscs.append(epoch_dsc)
        train_accuracies.append(epoch_acc)
        train_precisions.append(epoch_precision)
        train_recalls.append(epoch_recall)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_bce_loss = 0.0
        val_dice_loss = 0.0
        val_dsc = 0.0
        val_acc = 0.0
        val_precision = 0.0
        val_recall = 0.0

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
            for images, masks, laplacians in val_loader_tqdm:
                images = images.to(device)
                masks = masks.to(device).float()
                laplacians = laplacians.to(device).float()

                # 前向传播
                outputs = model(images, laplacians)
                loss, bce, dice = criterion(outputs, masks)
                val_loss += loss.item()
                val_bce_loss += bce.item()
                val_dice_loss += dice.item()

                # 计算指标
                dsc = compute_dsc(outputs, masks)
                acc = compute_accuracy(outputs, masks)
                precision = compute_precision(outputs, masks)
                recall = compute_recall(outputs, masks)

                val_dsc += dsc
                val_acc += acc
                val_precision += precision
                val_recall += recall

        val_loss /= len(val_loader)
        val_bce_loss /= len(val_loader)
        val_dice_loss /= len(val_loader)
        val_dsc /= len(val_loader)
        val_acc /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        logger.info(
            f"验证 - 平均损失: {val_loss:.4f}, BCE 损失: {val_bce_loss:.4f}, Dice 损失: {val_dice_loss:.4f}, "
            f"DSC: {val_dsc:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
        )

        # 记录验证指标
        val_losses.append(val_loss)
        val_bce_losses.append(val_bce_loss)
        val_dice_losses.append(val_dice_loss)
        val_dscs.append(val_dsc)
        val_accuracies.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        # 学习率调度器更新，基于验证 DSC
        scheduler.step(val_dsc)

        # 早停机制
        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_dsc': best_val_dsc,
            }, "best_model.pth")
            logger.info("发现更好的模型，已保存。")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("早停触发，训练结束。")
                break

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_dsc': best_val_dsc,
            }, checkpoint_path)
            logger.info(f"已保存检查点到 {checkpoint_path}")

    # 训练结束后，将指标数据保存为 CSV 文件
    metrics_data = {
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'train_bce_loss': train_bce_losses,
        'train_dice_loss': train_dice_losses,
        'train_dsc': train_dscs,
        'train_accuracy': train_accuracies,
        'train_precision': train_precisions,
        'train_recall': train_recalls,
        'val_loss': val_losses,
        'val_bce_loss': val_bce_losses,
        'val_dice_loss': val_dice_losses,
        'val_dsc': val_dscs,
        'val_accuracy': val_accuracies,
        'val_precision': val_precisions,
        'val_recall': val_recalls,
    }

    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv('training_metrics.csv', index=False)
    logger.info("训练指标已保存到 training_metrics.csv")

    # 使用 Matplotlib 绘制损失和指标曲线，并保存为图像
    # 绘制总损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_loss'], label='训练总损失')
    plt.plot(metrics_data['epoch'], metrics_data['val_loss'], label='验证总损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证总损失曲线')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    logger.info("总损失曲线已保存到 loss_curve.png")

    # 绘制 BCE 损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_bce_loss'], label='训练 BCE 损失')
    plt.plot(metrics_data['epoch'], metrics_data['val_bce_loss'], label='验证 BCE 损失')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('训练和验证 BCE 损失曲线')
    plt.legend()
    plt.savefig('bce_loss_curve.png')
    plt.close()
    logger.info("BCE 损失曲线已保存到 bce_loss_curve.png")

    # 绘制 Dice 损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_dice_loss'], label='训练 Dice 损失')
    plt.plot(metrics_data['epoch'], metrics_data['val_dice_loss'], label='验证 Dice 损失')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('训练和验证 Dice 损失曲线')
    plt.legend()
    plt.savefig('dice_loss_curve.png')
    plt.close()
    logger.info("Dice 损失曲线已保存到 dice_loss_curve.png")

    # 绘制 DSC 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_dsc'], label='训练 DSC')
    plt.plot(metrics_data['epoch'], metrics_data['val_dsc'], label='验证 DSC')
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    plt.title('训练和验证 DSC 曲线')
    plt.legend()
    plt.savefig('dsc_curve.png')
    plt.close()
    logger.info("DSC 曲线已保存到 dsc_curve.png")

    # 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_accuracy'], label='训练 Accuracy')
    plt.plot(metrics_data['epoch'], metrics_data['val_accuracy'], label='验证 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练和验证 Accuracy 曲线')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.close()
    logger.info("Accuracy 曲线已保存到 accuracy_curve.png")

    # 绘制 Precision 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_precision'], label='训练 Precision')
    plt.plot(metrics_data['epoch'], metrics_data['val_precision'], label='验证 Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('训练和验证 Precision 曲线')
    plt.legend()
    plt.savefig('precision_curve.png')
    plt.close()
    logger.info("Precision 曲线已保存到 precision_curve.png")

    # 绘制 Recall 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_recall'], label='训练 Recall')
    plt.plot(metrics_data['epoch'], metrics_data['val_recall'], label='验证 Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('训练和验证 Recall 曲线')
    plt.legend()
    plt.savefig('recall_curve.png')
    plt.close()
    logger.info("Recall 曲线已保存到 recall_curve.png")

    logger.info("训练完成。")
    return model, df_metrics


# ---------------------------
# 主函数
# ---------------------------
if __name__ == "__main__":
    logger.info("初始化训练和验证数据集")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用的设备: {device}")

    # 初始化模型并移动到设备
    model = OriginalUNetWithCBAMAndLaplacian(in_channels=3, out_channels=1, laplacian_channels=1).to(device)
    logger.info("模型初始化完成并已移动到设备。")

    # 测试模型的前向传播，确保没有尺寸不匹配的问题
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    dummy_laplacian = torch.randn(1, 1, 512, 512).to(device)
    try:
        dummy_output = model(dummy_input, dummy_laplacian)
        logger.info(f"Dummy output shape: {dummy_output.shape}")  # 应为 [1, 1, 512, 512]
    except Exception as e:
        logger.error(f"模型前向传播时出错：{e}")
        logger.error(traceback.format_exc())

    # 创建训练和验证数据集
    train_dataset = SegmentationDataset(
        images_dir="dataset/images/train",
        annotations_dir="dataset/annotations/train",
        transform=data_transforms_train
    )

    val_dataset = SegmentationDataset(
        images_dir="dataset/images/val",
        annotations_dir="dataset/annotations/val",
        transform=data_transforms_val  # 验证集不进行数据增强
    )

    # 创建数据加载器，设置合适的 batch_size 和 num_workers
    # 根据您的系统资源，调整 batch_size 和 num_workers
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    # 检查一个批次的数据形状
    try:
        for images, masks, laplacians in train_loader:
            logger.info(f"Batch - Images: {images.shape}, Masks: {masks.shape}, Laplacians: {laplacians.shape}")
            break
    except Exception as e:
        logger.error(f"数据加载时出错：{e}")
        logger.error(traceback.format_exc())

    # 检查数据加载
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 训练模型（包含 Early Stopping 和混合精度）
    trained_model, metrics_df = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        lr=1e-4,
        patience=10,
        device=device
    )

    # ==============================
    # 在训练结束后加载最佳模型并可视化预测结果
    # ==============================

    # 加载保存的最佳模型
    best_model_path = "best_model.pth"
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler = GradScaler()
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_val_dsc = checkpoint['best_val_dsc']
        logger.info("已加载最佳模型权重。")
    else:
        logger.warning(f"未找到最佳模型权重文件 {best_model_path}。")
        scaler = GradScaler()

    # 切换到评估模式
    model.eval()

    # 创建保存预测结果的文件夹
    output_dir = "prediction_results"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化变量以累积指标值
    total_val_dsc = 0.0
    total_val_acc = 0.0
    total_val_precision = 0.0
    total_val_recall = 0.0
    total_samples = len(val_dataset)

    # 在验证集上运行模型，生成预测结果并保存
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="生成预测结果"):
            val_image, val_mask, val_laplacian = val_dataset[idx]
            val_image = val_image.unsqueeze(0).to(device)
            val_mask = val_mask.unsqueeze(0).to(device)
            val_laplacian = val_laplacian.unsqueeze(0).to(device)

            # 前向传播
            with autocast():
                outputs = model(val_image, val_laplacian)

            # 计算指标
            dsc = compute_dsc(outputs, val_mask)
            acc = compute_accuracy(outputs, val_mask)
            precision = compute_precision(outputs, val_mask)
            recall = compute_recall(outputs, val_mask)

            # 累积指标
            total_val_dsc += dsc
            total_val_acc += acc
            total_val_precision += precision
            total_val_recall += recall

            # 将所有结果转换为 CPU 张量
            true_mask_np = val_mask[0, 0].cpu().numpy()  # 真实掩码
            pred_mask_np = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.uint8)  # 预测掩码

            # 调整 pred_mask_np 的形状
            pred_mask_np = pred_mask_np.squeeze()  # 去除多余维度

            # 保存图像和掩码
            plt.figure(figsize=(16, 8))

            plt.subplot(1, 4, 1)
            plt.imshow(val_image[0].cpu().permute(1, 2, 0).numpy())
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(true_mask_np, cmap='gray')
            plt.title('True Mask')
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(pred_mask_np, cmap='gray')  # 修正后的形状
            plt.title(f'Predicted Mask\nDSC: {dsc:.4f}, Accuracy: {acc:.4f}')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(val_laplacian.cpu().squeeze(), cmap='gray')  # 显示 Laplacian 图像
            plt.title('Laplacian Image')
            plt.axis('off')

            # 保存为文件
            output_path = os.path.join(output_dir, f'prediction_{idx + 1}.png')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"结果已保存到 {output_path}")

    # 计算并输出验证集的平均指标
    average_val_dsc = total_val_dsc / total_samples
    average_val_acc = total_val_acc / total_samples
    average_val_precision = total_val_precision / total_samples
    average_val_recall = total_val_recall / total_samples

    logger.info(f"验证集的平均 DSC: {average_val_dsc:.4f}")
    logger.info(f"验证集的平均 Accuracy: {average_val_acc:.4f}")
    logger.info(f"验证集的平均 Precision: {average_val_precision:.4f}")
    logger.info(f"验证集的平均 Recall: {average_val_recall:.4f}")

    # 也在控制台输出
    print(f"验证集的平均 DSC: {average_val_dsc:.4f}")
    print(f"验证集的平均 Accuracy: {average_val_acc:.4f}")
    print(f"验证集的平均 Precision: {average_val_precision:.4f}")
    print(f"验证集的平均 Recall: {average_val_recall:.4f}")

    logger.info("所有预测结果已生成并保存。")
