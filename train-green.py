# train_green_ratio_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.enet_green import ENetGreenRatio
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_green_ratio():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 100
    
    # 修改图像和标签的transform
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 标签transform - 确保与模型输出尺寸匹配
    label_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])
    
    # 首先检查数据集
    print("🔍 检查训练数据集...")
    check_dataset_labels('camvid_binary/train')
    
    print("\n🔍 检查验证数据集...")
    check_dataset_labels('camvid_binary/val')

    # 加载二分类数据集
    train_dataset = BinaryCamVidDataset('camvid_binary/train', 
                                    image_transform=image_transform, 
                                    label_transform=label_transform)
    val_dataset = BinaryCamVidDataset('camvid_binary/val', 
                                    image_transform=image_transform, 
                                    label_transform=label_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 检查第一个batch的数据
    print("\n🔍 检查第一个训练batch...")
    for images, labels in train_loader:
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        
        # 检查标签的唯一值
        unique_vals = torch.unique(labels)
        print(f"batch内标签唯一值: {unique_vals}")
        
        # 统计每个类别的像素数
        for class_val in [0, 1]:
            count = (labels == class_val).sum().item()
            percentage = count / labels.numel() * 100
            print(f"类别 {class_val}: {count} 像素 ({percentage:.2f}%)")
        break  # 只检查第一个batch


    # 创建模型
    model = ENetGreenRatio(num_classes=num_classes, encoder_only=False)
    model.to(device)
    
    # 打印参数量
    print("=" * 50)
    print("模型参数量统计:")
    print("=" * 50)
    total_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    
    # 打印各层参数量
    for name, module in model.named_children():
        module_params = count_parameters(module)
        print(f"{name}: {module_params:,} parameters")
    
    print("=" * 50)
    
    # 损失函数
    criterion_seg = nn.CrossEntropyLoss()  # 高分辨率分割损失
    criterion_green_map = nn.BCELoss()     # 低分辨率绿植概率图损失
    criterion_ratio = nn.MSELoss()        # 绿植比例损失
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_seg_loss = 0
        total_green_loss = 0
        total_ratio_loss = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if model.encoder_only:
                # encoder_only模式返回3个值
                feature_map, green_prob_map, green_ratio = model(images, return_features=True)
                segmentation = None  # 在encoder_only模式下没有分割结果
            else:
                # 完整模式返回4个值
                segmentation, feature_map, green_prob_map, green_ratio = model(
                    images, return_features=True)
            
            # 计算低分辨率标签（用于监督绿植概率图）
            # 特征图尺寸通常是输入尺寸的1/8或1/16
            feature_map_size = green_prob_map.shape[2:]  # [H, W]
            small_labels = F.interpolate(
                labels.unsqueeze(1).float(), 
                size=feature_map_size, 
                mode='nearest'
            ).squeeze(1)
            
            # 将标签转换为二值（绿植=1，非绿植=0）
            green_labels = (small_labels == 1).float()
            
            # 计算绿植比例真值
            green_ratio_gt = torch.mean(green_labels, dim=(1, 2))
            
            # 计算各种损失
            if segmentation is not None:
                seg_loss = criterion_seg(segmentation, labels)  # 高分辨率分割损失
            else:
                seg_loss = 0  # 在encoder_only模式下没有分割损失
            green_map_loss = criterion_green_map(green_prob_map.squeeze(1), green_labels)  # 低分辨率概率图损失
            ratio_loss = criterion_ratio(green_ratio, green_ratio_gt)  # 比例损失
            
            # 组合损失（权重可调整）
            loss = seg_loss + 0.5 * green_map_loss + 0.1 * ratio_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_green_loss += green_map_loss.item()
            total_ratio_loss += ratio_loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                if model.encoder_only:
                    feature_map, green_prob_map, green_ratio = model(images, return_features=True)
                    segmentation = None
                else:
                    segmentation, feature_map, green_prob_map, green_ratio = model(
                        images, return_features=True)
                
                feature_map_size = green_prob_map.shape[2:]
                small_labels = F.interpolate(
                    labels.unsqueeze(1).float(), 
                    size=feature_map_size, 
                    mode='nearest'
                ).squeeze(1)
                green_labels = (small_labels == 1).float()
                green_ratio_gt = torch.mean(green_labels, dim=(1, 2))
                
                if segmentation is not None:
                    seg_loss = criterion_seg(segmentation, labels)  # 高分辨率分割损失
                else:
                    seg_loss = 0  # 在encoder_only模式下没有分割损失
                green_map_loss = criterion_green_map(green_prob_map.squeeze(1), green_labels)
                ratio_loss = criterion_ratio(green_ratio, green_ratio_gt)
                val_loss += (seg_loss + 0.5 * green_map_loss + 0.1 * ratio_loss).item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Total Loss: {total_loss/len(train_loader):.4f}, '
              f'Seg Loss: {total_seg_loss/len(train_loader):.4f}, '
              f'Green Map Loss: {total_green_loss/len(train_loader):.4f}, '
              f'Ratio Loss: {total_ratio_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
    
    # 保存模型
    torch.save({
        'encoder_state_dict': model.get_encoder_params(),
        'epoch': num_epochs
    }, 'enet_green_ratio_encoder.pth')
    
    # 保存完整模型用于debug
    torch.save(model.state_dict(), 'enet_green_ratio_full.pth')
    
    # 打印保存的模型参数量
    print("\n保存的模型参数量:")
    print("=" * 30)
    
    # 加载编码器模型并打印参数量
    encoder_model = ENetGreenRatio(num_classes=2, encoder_only=True)  # 添加num_classes
    checkpoint = torch.load('enet_green_ratio_encoder.pth', map_location='cpu')
    
    # 修复：直接加载到模型
    encoder_model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    encoder_params = count_parameters(encoder_model)
    print(f"编码器模型参数量: {encoder_params:,}")
    
    # 加载完整模型并打印参数量
    full_model = ENetGreenRatio(num_classes=num_classes, encoder_only=False)
    full_model.load_state_dict(torch.load('enet_green_ratio_full.pth', map_location='cpu'))
    full_params = count_parameters(full_model)
    print(f"完整模型参数量: {full_params:,}")
    
    # 在验证集上保存可视化结果
    validate_and_save_visualizations(full_model, val_loader, device, 'validation_results')

class BinaryCamVidDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_transform=None, label_transform=None, debug=False):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.debug = debug  # 新增debug标志
        self.debug_counter = 0  # 限制debug输出数量
        
        self.images = []
        self.labels = []
        
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        # 检查目录是否存在
        if not os.path.exists(image_dir):
            print(f"❌ 图像目录不存在: {image_dir}")
        if not os.path.exists(label_dir):
            print(f"❌ 标签目录不存在: {label_dir}")
        
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                label_path = os.path.join(label_dir, img_name)
                
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.labels.append(label_path)
                else:
                    print(f"⚠️  标签文件不存在: {label_path}")
        
        print(f"📊 数据集统计: {len(self.images)} 个样本")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        
        # Debug: 检查原始标签的像素值
        if self.debug and self.debug_counter < 5:  # 只检查前5个样本
            label_array = np.array(label)
            unique_vals, counts = np.unique(label_array, return_counts=True)
            print(f"🔍 样本 {idx} 原始标签统计:")
            print(f"   唯一值: {unique_vals}")
            print(f"   数量: {counts}")
            print(f"   形状: {label_array.shape}")
            self.debug_counter += 1
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.label_transform:
            label = self.label_transform(label)  # 现在返回的是(H, W)的tensor
        
        # Debug: 检查转换后的标签
        if self.debug and self.debug_counter <= 5:
            label_np = label.numpy()
            unique_vals, counts = np.unique(label_np, return_counts=True)
            print(f"🔍 样本 {idx} 转换后标签统计:")
            print(f"   唯一值: {unique_vals}")
            print(f"   数量: {counts}")
            print(f"   形状: {label_np.shape}")
            print("-" * 50)
        
        return image, label

def check_dataset_labels(data_dir, num_samples=10):
    """检查数据集标签的分布情况"""
    print("=" * 60)
    print("数据集标签检查")
    print("=" * 60)
    
    # 创建简单的transform来检查原始数据
    simple_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # 替换原来的label_transform
    label_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])

    dataset = BinaryCamVidDataset(data_dir, 
                                image_transform=simple_transform, 
                                label_transform=label_transform,
                                debug=True)
    
    if len(dataset) == 0:
        print("❌ 数据集为空！")
        return
    
    # 检查前几个样本
    print(f"\n📋 检查前 {min(num_samples, len(dataset))} 个样本:")
    print("-" * 50)
    
    label_stats = {
        'total_pixels': 0,
        'class_0_pixels': 0,
        'class_1_pixels': 0,
        'samples_with_class_1': 0
    }
    
    for i in range(min(num_samples, len(dataset))):
        image, label = dataset[i]
        
        # 统计标签分布
        label_np = label.numpy()
        unique_vals, counts = np.unique(label_np, return_counts=True)
        
        label_stats['total_pixels'] += label_np.size
        if 0 in unique_vals:
            idx = np.where(unique_vals == 0)[0][0]
            label_stats['class_0_pixels'] += counts[idx]
        if 1 in unique_vals:
            idx = np.where(unique_vals == 1)[0][0]
            label_stats['class_1_pixels'] += counts[idx]
            label_stats['samples_with_class_1'] += 1
        
        print(f"样本 {i}: 唯一值 {unique_vals}, 数量 {counts}")
    
    # 打印统计信息
    print("\n📊 标签分布统计:")
    print(f"总像素数: {label_stats['total_pixels']}")
    print(f"类别0像素数: {label_stats['class_0_pixels']} ({label_stats['class_0_pixels']/label_stats['total_pixels']*100:.2f}%)")
    print(f"类别1像素数: {label_stats['class_1_pixels']} ({label_stats['class_1_pixels']/label_stats['total_pixels']*100:.2f}%)")
    print(f"包含类别1的样本数: {label_stats['samples_with_class_1']}/{min(num_samples, len(dataset))}")
    
    if label_stats['class_1_pixels'] == 0:
        print("❌ 警告: 没有检测到类别1（绿植）像素！")
        print("可能的原因:")
        print("1. 标签文件内容全为0")
        print("2. 标签文件路径不正确")
        print("3. 标签文件格式有问题")
    else:
        print("✅ 标签数据正常")
    
    return label_stats



def validate_and_save_visualizations(model, val_loader, device, output_dir):
    """在验证集上保存原图和小图的绿植区域覆盖，包括GT标签对比"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    original_dir = os.path.join(output_dir, 'original_overlay')
    small_dir = os.path.join(output_dir, 'small_overlay')
    gt_comparison_dir = os.path.join(output_dir, 'gt_comparison')
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(small_dir, exist_ok=True)
    os.makedirs(gt_comparison_dir, exist_ok=True)
    
    # 反归一化转换
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            segmentation, feature_map, green_prob_map, green_ratio = model(
                images, return_features=True)
            
            batch_size = images.size(0)
            
            for i in range(batch_size):
                # 获取当前样本
                img = images[i]  # [3, H, W]
                seg = segmentation[i]  # [num_classes, H, W]
                prob_map = green_prob_map[i]  # [1, H_small, W_small]
                ratio = green_ratio[i].item()  # 绿植比例
                gt_label = labels[i]  # 真实标签 [H, W]
                
                # 反归一化图像
                img_denorm = inv_normalize(img).clamp(0, 1)
                
                # 获取预测的分割结果
                pred_mask = torch.argmax(seg, dim=0)  # [H, W]
                
                # 创建绿植区域掩码（绿色）
                green_mask = (pred_mask == 1).float()  # 绿植区域为1
                gt_green_mask = (gt_label == 1).float()  # 真实绿植区域
                
                # 1. 保存原图覆盖绿植区域（预测+GT）
                img_pil = transforms.ToPILImage()(img_denorm.cpu())
                
                # 创建预测覆盖层（绿色半透明）
                pred_overlay = np.zeros((img_pil.size[1], img_pil.size[0], 4), dtype=np.uint8)
                green_mask_np = green_mask.cpu().numpy()
                pred_overlay[green_mask_np == 1] = [0, 255, 0, 128]  # 半透明绿色
                
                # 创建GT覆盖层（红色半透明）
                gt_overlay = np.zeros((img_pil.size[1], img_pil.size[0], 4), dtype=np.uint8)
                gt_green_mask_np = gt_green_mask.cpu().numpy()
                gt_overlay[gt_green_mask_np == 1] = [255, 0, 0, 128]  # 半透明红色
                
                # 合并原图、预测和GT
                pred_overlay_pil = Image.fromarray(pred_overlay, 'RGBA')
                gt_overlay_pil = Image.fromarray(gt_overlay, 'RGBA')
                
                # 先合并原图和GT
                img_with_gt = Image.alpha_composite(
                    img_pil.convert('RGBA'), gt_overlay_pil)
                # 再合并预测
                img_with_both = Image.alpha_composite(img_with_gt, pred_overlay_pil)
                
                # 保存原图覆盖
                original_filename = f'batch{batch_idx}_img{i}_ratio_{ratio:.4f}.png'
                img_with_both.save(os.path.join(original_dir, original_filename))
                
                # 2. 保存小图覆盖绿植区域
                small_h, small_w = prob_map.shape[1], prob_map.shape[2]
                
                # 将原图下采样到小图尺寸
                small_img = F.interpolate(
                    img_denorm.unsqueeze(0), 
                    size=(small_h, small_w), 
                    mode='bilinear'
                ).squeeze(0)
                
                # 将GT标签下采样到小图尺寸
                small_gt = F.interpolate(
                    gt_green_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(small_h, small_w),
                    mode='nearest'
                ).squeeze().cpu().numpy()
                
                # 将小图转换为PIL
                small_img_pil = transforms.ToPILImage()(small_img.cpu())
                
                # 创建小图预测覆盖层
                small_pred_overlay = np.zeros((small_h, small_w, 4), dtype=np.uint8)
                small_green_mask = (prob_map.squeeze(0) > 0.5).cpu().numpy()  # 阈值化概率图
                small_pred_overlay[small_green_mask] = [0, 255, 0, 128]  # 半透明绿色
                
                # 创建小图GT覆盖层
                small_gt_overlay = np.zeros((small_h, small_w, 4), dtype=np.uint8)
                small_gt_overlay[small_gt == 1] = [255, 0, 0, 128]  # 半透明红色
                
                small_pred_overlay_pil = Image.fromarray(small_pred_overlay, 'RGBA')
                small_gt_overlay_pil = Image.fromarray(small_gt_overlay, 'RGBA')
                
                # 合并小图、GT和预测
                small_with_gt = Image.alpha_composite(
                    small_img_pil.convert('RGBA'), small_gt_overlay_pil)
                small_combined = Image.alpha_composite(small_with_gt, small_pred_overlay_pil)
                
                # 保存小图覆盖
                small_filename = f'batch{batch_idx}_img{i}_ratio_{ratio:.4f}.png'
                small_combined.save(os.path.join(small_dir, small_filename))
                
                # 3. 创建详细的GT对比图
                create_gt_comparison_figure(
                    img_denorm.cpu(), 
                    pred_mask.cpu(), 
                    gt_label.cpu(),
                    prob_map.squeeze(0).cpu(),
                    small_gt,
                    ratio,
                    batch_idx, i, gt_comparison_dir
                )
    
    print(f"可视化结果已保存到: {output_dir}")
    print(f"原图覆盖保存到: {original_dir}")
    print(f"小图覆盖保存到: {small_dir}")
    print(f"GT对比图保存到: {gt_comparison_dir}")


def create_gt_comparison_figure(img, pred_mask, gt_label, prob_map, small_gt, ratio, batch_idx, img_idx, output_dir):
    """创建详细的GT对比图"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 原图
    img_np = img.permute(1, 2, 0).numpy()
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 预测分割结果
    axes[0, 1].imshow(pred_mask.numpy(), cmap='tab20')
    axes[0, 1].set_title('Predicted Segmentation')
    axes[0, 1].axis('off')
    
    # GT分割结果
    axes[0, 2].imshow(gt_label.numpy(), cmap='tab20')
    axes[0, 2].set_title('Ground Truth Segmentation')
    axes[0, 2].axis('off')
    
    # 预测与GT对比（重叠显示）
    overlap = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    # 正确预测的绿植区域（绿色）
    correct_green = (pred_mask == 1) & (gt_label == 1)
    overlap[correct_green.numpy()] = [0, 1, 0]  # 绿色
    # 误报（预测为绿植但实际不是，红色）
    false_positive = (pred_mask == 1) & (gt_label != 1)
    overlap[false_positive.numpy()] = [1, 0, 0]  # 红色
    # 漏报（实际是绿植但未预测到，蓝色）
    false_negative = (pred_mask != 1) & (gt_label == 1)
    overlap[false_negative.numpy()] = [0, 0, 1]  # 蓝色
    
    axes[0, 3].imshow(img_np)
    axes[0, 3].imshow(overlap, alpha=0.5)
    axes[0, 3].set_title('Prediction vs GT\n(Green=Correct, Red=FP, Blue=FN)')
    axes[0, 3].axis('off')
    
    # 小图概率图
    im1 = axes[1, 0].imshow(prob_map.numpy(), cmap='viridis')
    axes[1, 0].set_title('Small Prob Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    # 小图GT
    axes[1, 1].imshow(small_gt, cmap='viridis')
    axes[1, 1].set_title('Small GT')
    axes[1, 1].axis('off')
    
    # 小图预测与GT对比
    small_pred = (prob_map.numpy() > 0.5).astype(np.uint8)
    small_overlap = np.zeros((small_gt.shape[0], small_gt.shape[1], 3))
    small_correct = (small_pred == 1) & (small_gt == 1)
    small_fp = (small_pred == 1) & (small_gt != 1)
    small_fn = (small_pred != 1) & (small_gt == 1)
    small_overlap[small_correct] = [0, 1, 0]  # 绿色
    small_overlap[small_fp] = [1, 0, 0]  # 红色
    small_overlap[small_fn] = [0, 0, 1]  # 蓝色
    
    axes[1, 2].imshow(small_overlap)
    axes[1, 2].set_title('Small Prediction vs GT')
    axes[1, 2].axis('off')
    
    # 统计信息
    axes[1, 3].axis('off')
    total_pixels = pred_mask.numel()
    gt_green_pixels = (gt_label == 1).sum().item()
    pred_green_pixels = (pred_mask == 1).sum().item()
    correct_pixels = correct_green.sum().item()
    fp_pixels = false_positive.sum().item()
    fn_pixels = false_negative.sum().item()
    
    # 计算指标
    precision = correct_pixels / (correct_pixels + fp_pixels) if (correct_pixels + fp_pixels) > 0 else 0
    recall = correct_pixels / (correct_pixels + fn_pixels) if (correct_pixels + fn_pixels) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    textstr = f'Green Ratio: {ratio:.4f}\n\n' \
              f'GT Green Pixels: {gt_green_pixels} ({gt_green_pixels/total_pixels*100:.2f}%)\n' \
              f'Pred Green Pixels: {pred_green_pixels} ({pred_green_pixels/total_pixels*100:.2f}%)\n\n' \
              f'Correct: {correct_pixels} ({correct_pixels/total_pixels*100:.2f}%)\n' \
              f'False Positive: {fp_pixels} ({fp_pixels/total_pixels*100:.2f}%)\n' \
              f'False Negative: {fn_pixels} ({fn_pixels/total_pixels*100:.2f}%)\n\n' \
              f'Precision: {precision:.4f}\n' \
              f'Recall: {recall:.4f}\n' \
              f'F1 Score: {f1:.4f}'
    
    axes[1, 3].text(0.1, 0.9, textstr, transform=axes[1, 3].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Batch {batch_idx}, Image {img_idx} - Green Ratio: {ratio:.4f}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'batch{batch_idx}_img{img_idx}_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def validate_encoder_alignment():
    """验证编码器模型与完整模型的编码部分输出是否对齐"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("验证编码器模型与完整模型的对齐性")
    print("=" * 60)

    # 加载完整模型
    full_model = ENetGreenRatio(num_classes=2, encoder_only=False)
    full_model.load_state_dict(torch.load('enet_green_ratio_full.pth', map_location=device))
    full_model.to(device)
    full_model.eval()
    full_params = count_parameters(full_model)
    print(f"完整模型参数量: {full_params:,}")

    # 加载编码器模型
    encoder_model = ENetGreenRatio(num_classes=2, encoder_only=True)
    checkpoint = torch.load('enet_green_ratio_encoder.pth', map_location=device)
    encoder_params = count_parameters(encoder_model)
    print(f"编码器模型参数量: {encoder_params:,}")

    # 修复：直接加载到模型，而不是encoder属性
    encoder_model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    encoder_model.to(device)
    encoder_model.eval()
    
    # 创建测试数据加载器
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 替换原来的label_transform
    label_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])
    
    val_dataset = BinaryCamVidDataset('camvid_binary/val', 
                                    image_transform=image_transform, 
                                    label_transform=label_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 统计差异
    total_feature_diff = 0
    total_prob_map_diff = 0
    total_ratio_diff = 0
    num_samples = 0
    
    # 反归一化转换
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 创建输出目录
    output_dir = 'encoder_alignment_validation'
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            
            # 完整模型前向传播
            segmentation_full, feature_map_full, green_prob_map_full, green_ratio_full = full_model(
                images, return_features=True)
            
            # 编码器模型前向传播
            feature_map_enc, green_prob_map_enc, green_ratio_enc = encoder_model(
                images, return_features=True)
            
            # 计算差异
            feature_diff = F.mse_loss(feature_map_full, feature_map_enc).item()
            prob_map_diff = F.mse_loss(green_prob_map_full, green_prob_map_enc).item()
            ratio_diff = F.mse_loss(green_ratio_full, green_ratio_enc).item()
            
            total_feature_diff += feature_diff
            total_prob_map_diff += prob_map_diff
            total_ratio_diff += ratio_diff
            num_samples += 1
            
            # 为前几个批次保存可视化对比
            if batch_idx < 3:  # 只保存前3个批次的可视化
                for i in range(images.size(0)):
                    # 反归一化图像
                    img_denorm = inv_normalize(images[i]).clamp(0, 1)
                    img_pil = transforms.ToPILImage()(img_denorm.cpu())
                    
                    # 获取两个模型的绿植概率图
                    prob_full = green_prob_map_full[i].squeeze().cpu().numpy()
                    prob_enc = green_prob_map_enc[i].squeeze().cpu().numpy()
                    
                    # 创建对比图
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # 显示原图
                    axes[0, 0].imshow(img_pil)
                    axes[0, 0].set_title('Original Image')
                    axes[0, 0].axis('off')
                    
                    # 显示完整模型的概率图
                    im1 = axes[0, 1].imshow(prob_full, cmap='viridis')
                    axes[0, 1].set_title(f'Full Model Prob Map\nRatio: {green_ratio_full[i].item():.4f}')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1])
                    
                    # 显示编码器模型的概率图
                    im2 = axes[0, 2].imshow(prob_enc, cmap='viridis')
                    axes[0, 2].set_title(f'Encoder Model Prob Map\nRatio: {green_ratio_enc[i].item():.4f}')
                    axes[0, 2].axis('off')
                    plt.colorbar(im2, ax=axes[0, 2])
                    
                    # 显示差异图
                    diff_map = np.abs(prob_full - prob_enc)
                    im3 = axes[1, 0].imshow(diff_map, cmap='hot')
                    axes[1, 0].set_title(f'Difference Map\nMSE: {diff_map.mean():.6f}')
                    axes[1, 0].axis('off')
                    plt.colorbar(im3, ax=axes[1, 0])
                    
                    # 显示完整模型的分割结果
                    seg_pred_full = torch.argmax(segmentation_full[i], dim=0).cpu().numpy()
                    axes[1, 1].imshow(seg_pred_full, cmap='tab20')
                    axes[1, 1].set_title('Full Model Segmentation')
                    axes[1, 1].axis('off')
                    
                    # 显示比例差异
                    ratio_diff_val = abs(green_ratio_full[i].item() - green_ratio_enc[i].item())
                    axes[1, 2].bar(['Full Model', 'Encoder Model'], 
                                 [green_ratio_full[i].item(), green_ratio_enc[i].item()])
                    axes[1, 2].set_title(f'Ratio Comparison\nDiff: {ratio_diff_val:.6f}')
                    axes[1, 2].set_ylabel('Green Ratio')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 
                                           f'batch{batch_idx}_img{i}_comparison.png'), 
                              dpi=150, bbox_inches='tight')
                    plt.close()
    
    # 打印统计结果
    avg_feature_diff = total_feature_diff / num_samples
    avg_prob_map_diff = total_prob_map_diff / num_samples
    avg_ratio_diff = total_ratio_diff / num_samples
    
    print(f"平均特征图差异 (MSE): {avg_feature_diff:.6f}")
    print(f"平均概率图差异 (MSE): {avg_prob_map_diff:.6f}")
    print(f"平均绿植比例差异 (MSE): {avg_ratio_diff:.6f}")
    
    # 判断对齐程度
    print("\n对齐性评估:")
    print("-" * 30)
    if avg_feature_diff < 1e-6:
        print("✓ 特征图完美对齐")
    elif avg_feature_diff < 1e-4:
        print("✓ 特征图高度对齐")
    elif avg_feature_diff < 1e-2:
        print("○ 特征图基本对齐")
    else:
        print("✗ 特征图存在明显差异")
    
    if avg_prob_map_diff < 1e-6:
        print("✓ 概率图完美对齐")
    elif avg_prob_map_diff < 1e-4:
        print("✓ 概率图高度对齐")
    elif avg_prob_map_diff < 1e-2:
        print("○ 概率图基本对齐")
    else:
        print("✗ 概率图存在明显差异")
    
    if avg_ratio_diff < 1e-6:
        print("✓ 绿植比例完美对齐")
    elif avg_ratio_diff < 1e-4:
        print("✓ 绿植比例高度对齐")
    elif avg_ratio_diff < 1e-2:
        print("○ 绿植比例基本对齐")
    else:
        print("✗ 绿植比例存在明显差异")
    
    print(f"\n详细对比图已保存到: {output_dir}")
    return avg_feature_diff, avg_prob_map_diff, avg_ratio_diff

# 在 main 函数中添加调用
if __name__ == '__main__':

    # print("开始数据检查...")
    # check_dataset_labels('camvid_binary/train', num_samples=5)
    # check_dataset_labels('camvid_binary/val', num_samples=5)

    # 训练模型
    train_green_ratio()

    # 验证编码器对齐性
    # validate_encoder_alignment()
    
    # # 重新加载验证数据集进行可视化
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = ENetGreenRatio(num_classes=2, encoder_only=False)
    # model.load_state_dict(torch.load('enet_green_ratio_full.pth', map_location=device))
    # model.to(device)
    
    # # 重新加载验证数据集
    # image_transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # label_transform = transforms.Compose([
    #     transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    #     transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    # ])
    # val_dataset = BinaryCamVidDataset('camvid_binary/val', 
    #                                 image_transform=image_transform, 
    #                                 label_transform=label_transform)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # # 进行可视化验证
    # validate_and_save_visualizations(model, val_loader, device, 'validation_results')     

