import sys
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as Fnn   # <<< 新增：用于 interpolate
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QMessageBox

# ---------------------------
# 定义 CBAM 模块（保持和训练代码一致）
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
# 这一段已经改成你训练代码里的那一版结构
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

        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.cbams = nn.ModuleList()
        for feature in reversed(features):
            # 与训练版一致：上采样通道数 feature*2 -> feature
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # CBAM 的输入通道就是 skip_connection 的通道数 = feature
            self.cbams.append(CBAM(feature, reduction=16, kernel_size=7))
            # decoder 输入是 [上采样后的 x (feature)] + [CBAM 后的 skip (feature)] → 2*feature
            self.decoder_layers.append(DoubleConv(feature * 2, feature))

        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Laplacian 卷积处理层
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

        # bottleneck
        x = self.bottleneck(x)

        # Laplacian 只处理一次，然后在 decoder 里按尺度上采样
        laplacian = self.laplacian_conv(laplacian)  # [B,1,H,W]

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.upconvs)):
            # 上采样
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]

            # 对齐 spatial 尺寸（极少数情况下需要）
            if x.shape[2:] != skip_connection.shape[2:]:
                x = Fnn.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # Laplacian 同步到当前层尺度
            laplacian_resized = Fnn.interpolate(laplacian, size=skip_connection.shape[2:], mode='nearest')

            # 核心边缘增强：skip * laplace 然后残差加回去
            edge_guidance = skip_connection * laplacian_resized
            enhanced_skip = skip_connection + edge_guidance

            # CBAM 注意力（不改变通道数）
            enhanced_skip = self.cbams[idx](enhanced_skip)

            # 与上采样后的特征拼接
            x = torch.cat([enhanced_skip, x], dim=1)

            # DoubleConv
            x = self.decoder_layers[idx](x)

        return self.final_conv(x)


# ---------------------------
# 数据预处理
# ---------------------------
class Compose(object):
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
        image = F.resize(image, self.size)
        # 掩码 & Laplacian 使用最近邻，避免插值伪影
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        laplacian = F.resize(laplacian, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask, laplacian


class ToTensor(object):
    def __call__(self, image, mask, laplacian):
        image = F.to_tensor(image)
        # 和训练版保持一致：float32 + 二值化 + 增加通道维
        mask = torch.from_numpy(np.array(mask, dtype=np.float32))
        mask = (mask > 0).float()
        mask = mask.unsqueeze(0)
        laplacian = torch.from_numpy(np.array(laplacian, dtype=np.float32)).unsqueeze(0)
        return image, mask, laplacian


data_transforms = Compose([
    Resize((512, 512)),
    ToTensor(),
])


# ---------------------------
# 数据集类
# ---------------------------
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.category_to_id = {"background": 0, "1": 1}

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        annotation_filename = os.path.splitext(img_filename)[0] + '.json'
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)

        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        mask = Image.new('L', (width, height), 0)

        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)

        if 'shapes' in annotation_data:
            for shape in annotation_data['shapes']:
                if 'points' in shape and 'label' in shape:
                    category = shape['label']
                    category_id = self.category_to_id.get(category, 0)
                    points = shape['points']
                    poly = [tuple(point) for point in points]
                    ImageDraw.Draw(mask).polygon(poly, outline=category_id, fill=category_id)

        gray_image = image.convert('L')
        gray_np = np.array(gray_image)
        laplacian = cv2.Laplacian(gray_np, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        if np.max(laplacian) != 0:
            laplacian = (255 * (laplacian / np.max(laplacian))).astype(np.uint8)
        else:
            laplacian = laplacian.astype(np.uint8)
        laplacian = Image.fromarray(laplacian)

        if self.transform:
            image, mask, laplacian = self.transform(image, mask, laplacian)

        return image, mask, laplacian


# ---------------------------
# 性能评估
# 注意：这里的输入 pred_mask 直接给 logits 更合理，所以 GUI 里会传 outputs 进来
# ---------------------------
def compute_metrics(pred_logits, true_mask, smooth=1e-5):
    # 先做 sigmoid + 阈值
    pred_mask = torch.sigmoid(pred_logits)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    intersection = (pred_mask * true_mask).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + true_mask.sum(dim=(2, 3)) - intersection

    accuracy = (pred_mask == true_mask).sum(dim=(2, 3)) / (pred_mask.size(2) * pred_mask.size(3))
    precision = intersection / (pred_mask.sum(dim=(2, 3)) + smooth)
    recall = intersection / (true_mask.sum(dim=(2, 3)) + smooth)
    dice = (2. * intersection + smooth) / (pred_mask.sum(dim=(2, 3)) + true_mask.sum(dim=(2, 3)) + smooth)
    dsc = (2. * intersection) / (pred_mask.sum(dim=(2, 3)) + true_mask.sum(dim=(2, 3)) + smooth)

    return {
        'accuracy': accuracy.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'dice': dice.mean().item(),
        'dsc': dsc.mean().item()
    }


# ---------------------------
# GUI 界面
# ---------------------------
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Segmentation Validation')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.btn_images = QPushButton('Select Images Directory', self)
        self.btn_images.clicked.connect(self.select_images_dir)
        layout.addWidget(self.btn_images)

        self.btn_annotations = QPushButton('Select Annotations Directory', self)
        self.btn_annotations.clicked.connect(self.select_annotations_dir)
        layout.addWidget(self.btn_annotations)

        self.btn_output = QPushButton('Select Output Directory', self)
        self.btn_output.clicked.connect(self.select_output_dir)
        layout.addWidget(self.btn_output)

        self.btn_validate = QPushButton('Start Validation', self)
        self.btn_validate.clicked.connect(self.start_validation)
        layout.addWidget(self.btn_validate)

        self.text_edit = QTextEdit(self)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

        self.images_dir = ''
        self.annotations_dir = ''
        self.output_dir = ''

    def select_images_dir(self):
        self.images_dir = QFileDialog.getExistingDirectory(self, 'Select Images Directory')
        self.text_edit.append(f'Selected Images Directory: {self.images_dir}')

    def select_annotations_dir(self):
        self.annotations_dir = QFileDialog.getExistingDirectory(self, 'Select Annotations Directory')
        self.text_edit.append(f'Selected Annotations Directory: {self.annotations_dir}')

    def select_output_dir(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        self.text_edit.append(f'Selected Output Directory: {self.output_dir}')

    def start_validation(self):
        try:
            if not self.images_dir or not self.annotations_dir or not self.output_dir:
                QMessageBox.warning(self, "Warning", "Please select all directories.")
                return

            val_dataset = SegmentationDataset(
                images_dir=self.images_dir,
                annotations_dir=self.annotations_dir,
                transform=data_transforms
            )
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

            # 初始化 U-Net 模型（结构已改成训练版本）
            model = OriginalUNetWithCBAMAndLaplacian(
                in_channels=3,
                out_channels=1,
                laplacian_channels=1,
                features=[64, 128, 256, 512]
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            checkpoint_path = "best_model.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.text_edit.append(f"Loaded model weights from {checkpoint_path}")
            else:
                QMessageBox.warning(self, "Warning", f"Checkpoint file {checkpoint_path} not found.")
                return

            val_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'dice': 0.0, 'dsc': 0.0}
            num_samples = 0

            with torch.no_grad():
                for i, (images, masks, laplacians) in enumerate(val_loader):
                    images = images.to(device)
                    masks = masks.to(device).float()
                    laplacians = laplacians.to(device).float()

                    # 模型推理
                    outputs = model(images, laplacians)

                    # 指标用 logits 进 compute_metrics（内部会 sigmoid + 阈值）
                    metrics = compute_metrics(outputs, masks)
                    for key in val_metrics:
                        val_metrics[key] += metrics[key]
                    num_samples += 1

                    # 可视化用阈值后的预测
                    masks_pred = torch.sigmoid(outputs)
                    masks_pred = (masks_pred > 0.5).float()

                    # Save results
                    img_np = images.cpu().numpy().transpose(0, 2, 3, 1)[0]
                    mask_np = masks.cpu().numpy()[0, 0]  # Ground Truth
                    pred_np = masks_pred.cpu().numpy()[0, 0]  # Prediction

                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(img_np)
                    ax[0].set_title('Original Image')
                    ax[1].imshow(mask_np, cmap='gray')
                    ax[1].set_title('Ground Truth')
                    ax[2].imshow(pred_np, cmap='gray')
                    ax[2].set_title(
                        f'Prediction\nACC: {metrics["accuracy"]:.4f}\nPrec: {metrics["precision"]:.4f}\nRecall: {metrics["recall"]:.4f}\nDSC: {metrics["dice"]:.4f}')
                    for a in ax:
                        a.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'result_{i}.png'))
                    plt.close(fig)

            avg_metrics = {key: value / num_samples for key, value in val_metrics.items()}
            self.text_edit.append(
                f'Average Metrics:\n'
                f'DSC: {avg_metrics["dice"]:.4f}\n'
                f'ACC: {avg_metrics["accuracy"]:.4f}\n'
                f'Prec: {avg_metrics["precision"]:.4f}\n'
                f'Recall: {avg_metrics["recall"]:.4f}'
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during validation: {e}")
            self.text_edit.append(f"Error during validation: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
