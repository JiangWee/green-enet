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
        transforms.ToTensor()
    ])
    
    # 加载二分类数据集
    train_dataset = BinaryCamVidDataset('camvid_binary/train', 
                                    image_transform=image_transform, 
                                    label_transform=label_transform)
    val_dataset = BinaryCamVidDataset('camvid_binary/val', 
                                    image_transform=image_transform, 
                                    label_transform=label_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = ENetGreenRatio(num_classes=num_classes, encoder_only=False)
    model.to(device)
    
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
            print(f"green_ratio shape: {green_ratio.shape}")  # 应该是 [batch_size]
            print(f"green_ratio_gt shape: {green_ratio_gt.shape}")  # 应该是 [batch_size]
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

class BinaryCamVidDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_transform=None, label_transform=None):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        
        self.images = []
        self.labels = []
        
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                label_path = os.path.join(label_dir, img_name)
                
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.labels.append(label_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.label_transform:
            label = self.label_transform(label)
            label = label.squeeze(0).long()  # 从 [1, H, W] 转换为 [H, W]
        
        return image, label

# 推理脚本
def inference_green_ratio(image_path, model_path):
    """使用训练好的编码器进行绿植比例推理"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = ENetGreenRatio(encoder_only=True)
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        green_ratio = model(input_tensor)
    
    return green_ratio.item()

# Debug脚本：保存小图和重建的分割图
def debug_model(image_path, model_path, output_dir):
    """调试模型，保存小图和重建的分割图"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载完整模型
    model = ENetGreenRatio(num_classes=2, encoder_only=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 获取所有输出
        segmentation, feature_map, green_prob_map, green_ratio = model(
            input_tensor, return_features=True)
        
        # 保存小图（绿植概率图）
        green_prob_np = green_prob_map.squeeze().cpu().numpy()
        green_prob_img = Image.fromarray((green_prob_np * 255).astype(np.uint8))
        green_prob_img.save(os.path.join(output_dir, 'green_prob_map.png'))
        
        # 保存分割图
        seg_pred = torch.argmax(segmentation, dim=1).squeeze().cpu().numpy()
        seg_pred_img = Image.fromarray((seg_pred * 255).astype(np.uint8))
        seg_pred_img.save(os.path.join(output_dir, 'segmentation.png'))
        
        # 保存特征图可视化（第一个通道）
        feature_vis = feature_map[0, 0].cpu().numpy()
        feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min())
        feature_vis_img = Image.fromarray((feature_vis * 255).astype(np.uint8))
        feature_vis_img.save(os.path.join(output_dir, 'feature_map_ch0.png'))
    
    print(f"绿植比例: {green_ratio.item():.4f}")
    print(f"调试图像已保存到: {output_dir}")

if __name__ == '__main__':
    train_green_ratio()