import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from models.enet_green import ENetGreenRatio
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 添加ENet模型路径
sys.path.append('models')  # 假设ENet模型在models目录下
from enet import ENet  # 从文档2中导入ENet模型

def denormalize(tensor):
    """
    将张量准备为适合可视化的格式
    由于您的预处理只是ToTensor()，我们只需要确保张量格式正确
    """
    # 确保是3维张量 [C, H, W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 由于预处理只是ToTensor()，值已经在[0,1]范围内
    # 我们直接返回张量，保持后续处理的灵活性
    return tensor

def validate_models_on_dataset(dataset_path, full_model_path, encoder_model_path, enet_model_path, output_dir):
    """
    在指定数据集上验证完整模型、编码器模型和预训练ENet模型
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    full_output_dir = os.path.join(output_dir, 'full_model_results')
    encoder_output_dir = os.path.join(output_dir, 'encoder_model_results')
    enet_output_dir = os.path.join(output_dir, 'enet_model_results')
    os.makedirs(full_output_dir, exist_ok=True)
    os.makedirs(encoder_output_dir, exist_ok=True)
    os.makedirs(enet_output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    
    # 加载完整模型
    full_model = ENetGreenRatio(num_classes=2, encoder_only=False)
    full_model.load_state_dict(torch.load(full_model_path, map_location=device))
    full_model.to(device)
    full_model.eval()
    
    # 加载编码器模型
    encoder_model = ENetGreenRatio(num_classes=2, encoder_only=True)
    checkpoint = torch.load(encoder_model_path, map_location=device)
    encoder_model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    encoder_model.to(device)
    encoder_model.eval()
    
    # 加载预训练ENet模型
    num_classes_enet = 12
    enet_model = ENet(num_classes_enet).to(device)
    
    # 加载预训练权重（从文档2中的保存路径）
    enet_optimizer = torch.optim.Adam(enet_model.parameters())  # 临时优化器用于加载
    enet_model, _, _, _ = load_checkpoint(enet_model, enet_optimizer, 
                                         os.path.dirname(enet_model_path),
                                         os.path.basename(enet_model_path))
    enet_model.to(device)
    enet_model.eval()
    
    print(f"完整模型参数数量: {sum(p.numel() for p in full_model.parameters())}")
    print(f"编码器模型参数数量: {sum(p.numel() for p in encoder_model.parameters())}")
    print(f"ENet模型参数数量: {sum(p.numel() for p in enet_model.parameters())}")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
    ])
    
    # 修改反归一化定义
    inv_normalize = lambda x: x  # 直接使用恒等映射，因为预处理只是ToTensor()

    # 创建数据集
    class CustomDataset(Dataset):
        def __init__(self, dataset_path, transform=None):
            self.dataset_path = dataset_path
            self.transform = transform
            self.image_files = []
            
            # 查找所有bmp文件
            for file in os.listdir(dataset_path):
                if file.endswith('.bmp') and file.startswith('s') and 'iso' in file:
                    self.image_files.append(file)
            
            self.image_files.sort()
            print(f"找到 {len(self.image_files)} 个图像文件")
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.dataset_path, img_name)
            
            image = Image.open(img_path).convert('RGB')
            original_size = image.size  # (width, height)
            
            if self.transform:
                image_full = self.transform(image)
            else:
                image_full = transforms.ToTensor()(image)

            
            return {
                'image_full': image_full,
                'image_enet': image_full,
                'name': img_name,
                'size': original_size  # 保持为元组 (width, height)
            }
    
    # 自定义collate函数
    def custom_collate_fn(batch):
        images_full = torch.stack([item['image_full'] for item in batch])
        images_enet = torch.stack([item['image_enet'] for item in batch])
        names = [item['name'] for item in batch]
        sizes = [item['size'] for item in batch]
        
        return images_full, images_enet, names, sizes
    
    # 创建数据集和数据加载器
    dataset = CustomDataset(dataset_path, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # 存储结果
    results = []
    
    # CamVid类别编码（从文档2中获取）
    camvid_class_encoding = {
        'sky': (128, 128, 128),
        'building': (128, 0, 0),
        'pole': (192, 192, 128),
        'road': (128, 64, 128),
        'pavement': (60, 40, 222),
        'tree': (128, 128, 0),  # 绿植类别
        'signsymbol': (192, 128, 128),
        'fence': (64, 64, 128),
        'car': (64, 0, 128),
        'pedestrian': (64, 64, 0),
        'bicyclist': (0, 128, 192)
    }

    # 绿植类别索引（在CamVid中是'tree'，索引为5）
    green_class_index = 5
    
    print("开始验证...")
    with torch.no_grad():
        for i, (images_full, images_enet, img_names, original_sizes) in enumerate(dataloader):
            images_full = images_full.to(device)
            images_enet = images_enet.to(device)
            img_name = img_names[0]
            original_size = original_sizes[0]
            
            print(f"处理 {i+1}/{len(dataloader)}: {img_name}")
            print(f"原始尺寸: {original_size}")
            print(f"ENet输入尺寸: {images_enet.shape}")

            # 完整模型推理（使用Full预处理）
            segmentation_full, feature_map_full, green_prob_map_full, green_ratio_full = full_model(
                images_full, return_features=True)
                        
            # 完整模型输出形状检查
            print(f"分割输出形状: {segmentation_full.shape}")  # 应该显示类似 torch.Size([1, 2, 360, 480])
            print(f"特征图形状: {feature_map_full.shape}")
            print(f"绿植概率图形状: {green_prob_map_full.shape}")

            # 编码器模型推理（使用Full预处理）
            feature_map_enc, green_prob_map_enc, green_ratio_enc = encoder_model(
                images_full, return_features=True)

            # 编码器模型输出形状检查
            print(f"特征图形状: {feature_map_enc.shape}")
            print(f"绿植概率图形状: {green_prob_map_enc.shape}")

            # ENet模型推理
            # 注意：ENet模型应该使用[0,1]范围的输入
            enet_input = images_enet  # 已经是[0,1]范围
            
            # 如果ENet模型需要特定的归一化，可以在这里调整
            # enet_input = (images_enet * 255.0).clamp(0, 255) / 255.0  # 确保在[0,1]范围
            print(f"ENet输入范围: [{images_enet.min():.3f}, {images_enet.max():.3f}]")
            print(f"ENet输入均值: {images_enet.mean().item():.3f}")
            print(f"ENet输入标准差: {images_enet.std().item():.3f}")
            enet_output = enet_model(enet_input)
            enet_segmentation = torch.argmax(enet_output, dim=1)
            
            # 计算ENet的绿植比例（统计'tree'类别的像素比例）
            green_mask_enet = (enet_segmentation == green_class_index)
            green_ratio_enet = green_mask_enet.float().mean().item()
            
            # 新增：计算完整模型分割后的绿植比例（统计分割图中类别1的像素比例）
            full_segmentation_mask = torch.argmax(segmentation_full, dim=1)
            green_mask_full_seg = (full_segmentation_mask == 1)  # 假设绿植类别为1
            green_ratio_full_seg = green_mask_full_seg.float().mean().item()
            
            # 获取绿植比例
            full_ratio = green_ratio_full.item()
            full_seg_ratio = green_ratio_full_seg
            enc_ratio = green_ratio_enc.item()
            enet_ratio = green_ratio_enet
            
            print(f"完整模型编码器直出比例: {full_ratio:.4f}")
            print(f"完整模型分割后统计比例: {full_seg_ratio:.4f}")
            print(f"编码器模型比例: {enc_ratio:.4f}")
            print(f"ENet模型比例: {enet_ratio:.4f}")
            
            # 记录结果
            results.append({
                'image_name': img_name,
                'full_ratio': full_ratio,
                'full_seg_ratio': full_seg_ratio,  # 新增：分割后统计比例
                'encoder_ratio': enc_ratio,
                'enet_ratio': enet_ratio
            })
            
            # 为完整模型生成可视化
            create_visualization(
                images_full[0], segmentation_full[0], full_ratio, full_seg_ratio,
                img_name, full_output_dir, 'full', original_size, inv_normalize
            )
            
            # 为编码器模型生成可视化
            create_encoder_visualization(
                images_full[0], green_prob_map_enc[0], enc_ratio,
                img_name, encoder_output_dir, 'encoder', original_size, inv_normalize
            )
            
            # 为ENet模型生成可视化
            create_enet_visualization(
                images_enet[0], enet_segmentation[0], enet_ratio,
                img_name, enet_output_dir, 'enet', original_size, inv_normalize,
                camvid_class_encoding, green_class_index
            )
    
    # 保存汇总结果
    save_summary_results(results, output_dir)
    
    print(f"验证完成！结果保存在: {output_dir}")
    return results

def create_visualization(image_tensor, segmentation, ratio, seg_ratio, img_name, output_dir, model_type, original_size, inv_normalize):
    """为完整模型创建可视化结果"""
    # 统一输入处理
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # 反归一化
    img_denorm = inv_normalize(image_tensor)
    
    # 确保是PyTorch张量
    if not isinstance(img_denorm, torch.Tensor):
        img_denorm = torch.from_numpy(img_denorm)
    
    # 确保值范围正确 [0,1]
    if img_denorm.max() > 1.0:
        img_denorm = img_denorm / 255.0

    # 修复：正确处理分割结果的维度
    if segmentation.dim() == 4:
        segmentation = segmentation.squeeze(0)  # [C, H, W]
    
    # 获取预测的分割结果
    pred_mask = torch.argmax(segmentation, dim=0).cpu().numpy().astype(np.uint8)  # [H, W]
    
    # 上采样到原始尺寸
    img_original = F.interpolate(
        img_denorm.unsqueeze(0),  # [1, C, H, W]
        size=(original_size[1], original_size[0]),
        mode='bilinear'
    ).squeeze(0)  # [C, H, W]
    
    pred_mask_original = F.interpolate(
        torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]
        size=(original_size[1], original_size[0]),
        mode='nearest'
    ).squeeze().cpu().numpy().astype(np.uint8)  # [H, W]
    
    # 转换为PIL图像
    img_pil = transforms.ToPILImage()(img_original.cpu())
    
    # 创建绿植区域掩码（绿色）
    green_mask = (pred_mask_original == 1)
    
    # 创建覆盖层（绿色半透明）
    overlay = np.zeros((img_pil.size[1], img_pil.size[0], 4), dtype=np.uint8)
    overlay[green_mask] = [0, 255, 0, 128]
    
    overlay_pil = Image.fromarray(overlay, 'RGBA')
    
    # 合并原图和覆盖层
    combined = Image.alpha_composite(img_pil.convert('RGBA'), overlay_pil)
    
    # 保存图像
    base_name = os.path.splitext(img_name)[0]
    output_filename = f"{base_name}_{model_type}_enc{ratio:.4f}_seg{seg_ratio:.4f}.png"
    output_path = os.path.join(output_dir, output_filename)
    combined.save(output_path)
    
    # 保存分割图
    seg_img = Image.fromarray(pred_mask_original * 255)  # 直接乘以255，已经是uint8
    seg_filename = f"{base_name}_{model_type}_segmentation.png"
    seg_path = os.path.join(output_dir, seg_filename)
    seg_img.save(seg_path)

def create_encoder_visualization(image_tensor, green_prob_map, ratio, img_name, output_dir, model_type, original_size, inv_normalize):
    """为编码器模型创建可视化结果"""
    # 统一输入处理
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # 反归一化
    img_denorm = inv_normalize(image_tensor)
    
    # 修复：正确处理概率图的维度
    if green_prob_map.dim() == 3:  # [1, H, W] 或 [C, H, W]
        green_prob_map = green_prob_map.squeeze(0)  # 变为 [H, W]
    
    # 将概率图上采样到原始尺寸 - 修复维度
    prob_map_original = F.interpolate(
        green_prob_map.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W] 四维输入
        size=(original_size[1], original_size[0]),
        mode='bilinear'
    ).squeeze().cpu().numpy()  # 变为 [H, W]
    
    # 将图像调整回原始尺寸
    img_original = F.interpolate(
        img_denorm.unsqueeze(0),  # [1, C, H, W]
        size=(original_size[1], original_size[0]),
        mode='bilinear'
    ).squeeze(0)  # [C, H, W]
    
    # 转换为PIL图像
    img_pil = transforms.ToPILImage()(img_original.cpu())
    
    # 创建绿植区域掩码（阈值化概率图）
    green_mask = (prob_map_original > 0.5)
    
    # 创建覆盖层（绿色半透明）
    overlay = np.zeros((img_pil.size[1], img_pil.size[0], 4), dtype=np.uint8)
    overlay[green_mask] = [0, 255, 0, 128]
    
    overlay_pil = Image.fromarray(overlay, 'RGBA')
    
    # 合并原图和覆盖层
    combined = Image.alpha_composite(img_pil.convert('RGBA'), overlay_pil)
    
    # 保存图像
    base_name = os.path.splitext(img_name)[0]
    output_filename = f"{base_name}_{model_type}_ratio_{ratio:.4f}.png"
    output_path = os.path.join(output_dir, output_filename)
    combined.save(output_path)
    
    # 保存概率图
    prob_img = Image.fromarray((prob_map_original * 255).astype(np.uint8))
    prob_filename = f"{base_name}_{model_type}_probability.png"
    prob_path = os.path.join(output_dir, prob_filename)
    prob_img.save(prob_path)

def create_enet_visualization(image_tensor, enet_segmentation, ratio, img_name, output_dir, model_type, original_size, inv_normalize, class_encoding, green_class_index):
    """为ENet模型创建可视化结果"""

    # 添加调试信息
    print(f"ENet分割结果形状: {enet_segmentation.shape}")
    print(f"绿植类别索引: {green_class_index}")
    print(f"绿植像素数量: {(enet_segmentation == green_class_index).sum().item()}")
    print(f"总像素数量: {enet_segmentation.numel()}")
    print(f"计算的比例: {ratio}")
    
    # 检查所有类别的分布
    unique, counts = torch.unique(enet_segmentation, return_counts=True)
    print("类别分布:")
    for cls, count in zip(unique, counts):
        class_name = class_encoding.get(cls.item(), ("未知", (0, 0, 0)))[0]
        print(f"  类别 {cls.item()}({class_name}): {count.item()} 像素")

    # 统一输入处理
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # 反归一化
    img_denorm = inv_normalize(image_tensor)
    
    # 修复：确保分割结果是正确的维度
    if enet_segmentation.dim() == 3:
        enet_segmentation = enet_segmentation.squeeze(0)
    
    # 将分割结果调整回原始尺寸
    seg_original = F.interpolate(
        enet_segmentation.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]
        size=(original_size[1], original_size[0]),
        mode='nearest'
    ).squeeze().cpu().numpy().astype(np.uint8)  # [H, W]
    
    # 将图像调整回原始尺寸
    img_original = F.interpolate(
        img_denorm.unsqueeze(0),  # [1, C, H, W]
        size=(original_size[1], original_size[0]),
        mode='bilinear'
    ).squeeze(0)  # [C, H, W]
    
    # 转换为PIL图像
    img_pil = transforms.ToPILImage()(img_original.cpu())
    
    # 创建绿植区域掩码（绿色类别）
    green_mask = (seg_original == green_class_index)
    
    # 创建覆盖层（绿色半透明）
    overlay = np.zeros((img_pil.size[1], img_pil.size[0], 4), dtype=np.uint8)
    overlay[green_mask] = [0, 255, 0, 128]
    
    overlay_pil = Image.fromarray(overlay, 'RGBA')
    
    # 合并原图和覆盖层
    combined = Image.alpha_composite(img_pil.convert('RGBA'), overlay_pil)
    
    # 保存图像
    base_name = os.path.splitext(img_name)[0]
    output_filename = f"{base_name}_{model_type}_ratio_{ratio:.4f}.png"
    output_path = os.path.join(output_dir, output_filename)
    combined.save(output_path)
    
    # 修复：保存完整的分割图（所有类别）
    seg_rgb = np.zeros((seg_original.shape[0], seg_original.shape[1], 3), dtype=np.uint8)
    for class_idx, (class_name, color) in enumerate(class_encoding.items()):
        mask = seg_original == class_idx
        seg_rgb[mask] = color

    seg_img = Image.fromarray(seg_rgb)
    seg_filename = f"{base_name}_{model_type}_segmentation.png"
    seg_path = os.path.join(output_dir, seg_filename)
    seg_img.save(seg_path)

def save_summary_results(results, output_dir):
    """保存汇总结果到txt文件"""
    summary_path = os.path.join(output_dir, 'green_ratio_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("绿植比例验证结果汇总（四模型比较）\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'图像名称':<20} {'完整编码':<10} {'完整分割':<10} {'编码器':<10} {'ENet':<10} {'编码-分割':<10} {'编码器-分割':<10} {'ENet-分割':<10}\n")
        f.write("-" * 100 + "\n")
        
        full_ratios = []
        full_seg_ratios = []
        encoder_ratios = []
        enet_ratios = []
        diff_full_enc_seg = []  # 完整模型编码器输出与分割统计的差异
        diff_enc_seg = []       # 编码器模型与完整模型分割统计的差异
        diff_enet_seg = []      # ENet模型与完整模型分割统计的差异
        
        for result in results:
            diff_fes = (result['full_ratio'] - result['full_seg_ratio'])
            diff_es = (result['encoder_ratio'] - result['full_seg_ratio'])
            diff_ens = (result['enet_ratio'] - result['full_seg_ratio'])
            
            full_ratios.append(result['full_ratio'])
            full_seg_ratios.append(result['full_seg_ratio'])
            encoder_ratios.append(result['encoder_ratio'])
            enet_ratios.append(result['enet_ratio'])
            diff_full_enc_seg.append(diff_fes)
            diff_enc_seg.append(diff_es)
            diff_enet_seg.append(diff_ens)
            
            f.write(f"{result['image_name']:<20} {result['full_ratio']:<10.4f} {result['full_seg_ratio']:<10.4f} {result['encoder_ratio']:<10.4f} {result['enet_ratio']:<10.4f} {diff_fes:<10.4f} {diff_es:<10.4f} {diff_ens:<10.4f}\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"{'平均值':<20} {np.mean(full_ratios):<10.4f} {np.mean(full_seg_ratios):<10.4f} {np.mean(encoder_ratios):<10.4f} {np.mean(enet_ratios):<10.4f} {np.mean(diff_full_enc_seg):<10.4f} {np.mean(diff_enc_seg):<10.4f} {np.mean(diff_enet_seg):<10.4f}\n")
        f.write(f"{'标准差':<20} {np.std(full_ratios):<10.4f} {np.std(full_seg_ratios):<10.4f} {np.std(encoder_ratios):<10.4f} {np.std(enet_ratios):<10.4f} {np.std(diff_full_enc_seg):<10.4f} {np.std(diff_enc_seg):<10.4f} {np.std(diff_enet_seg):<10.4f}\n")
        
        # 计算相关性
        correlation_full_seg = np.corrcoef(full_ratios, full_seg_ratios)[0, 1]
        correlation_full_enc = np.corrcoef(full_ratios, encoder_ratios)[0, 1]
        correlation_full_enet = np.corrcoef(full_ratios, enet_ratios)[0, 1]
        correlation_enc_enet = np.corrcoef(encoder_ratios, enet_ratios)[0, 1]
        correlation_seg_enc = np.corrcoef(full_seg_ratios, encoder_ratios)[0, 1]
        correlation_seg_enet = np.corrcoef(full_seg_ratios, enet_ratios)[0, 1]
        
        f.write(f"\n相关系数:\n")
        f.write(f"完整模型编码 vs 完整模型分割: {correlation_full_seg:.4f}\n")
        f.write(f"完整模型编码 vs 编码器模型: {correlation_full_enc:.4f}\n")
        f.write(f"完整模型编码 vs ENet模型: {correlation_full_enet:.4f}\n")
        f.write(f"编码器模型 vs ENet模型: {correlation_enc_enet:.4f}\n")
        f.write(f"完整模型分割 vs 编码器模型: {correlation_seg_enc:.4f}\n")
        f.write(f"完整模型分割 vs ENet模型: {correlation_seg_enet:.4f}\n")
    
    # 创建可视化图表
    create_comparison_chart(results, output_dir)
    
    print(f"汇总结果已保存到: {summary_path}")

def create_comparison_chart(results, output_dir):
    """创建四模型比较图表"""
    full_ratios = [r['full_ratio'] for r in results]
    full_seg_ratios = [r['full_seg_ratio'] for r in results]
    encoder_ratios = [r['encoder_ratio'] for r in results]
    enet_ratios = [r['enet_ratio'] for r in results]
    
    # 创建四模型比较图
    plt.figure(figsize=(18, 12))
    
    # 散点图比较：完整模型编码vs分割
    plt.subplot(2, 4, 1)
    plt.scatter(full_ratios, full_seg_ratios, alpha=0.6, label='完整模型分割')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.xlabel('完整模型编码器比例')
    plt.ylabel('完整模型分割比例')
    plt.title('完整模型: 编码器vs分割')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 散点图比较：完整模型分割vs其他模型
    plt.subplot(2, 4, 2)
    plt.scatter(full_seg_ratios, encoder_ratios, alpha=0.6, label='编码器模型')
    plt.scatter(full_seg_ratios, enet_ratios, alpha=0.6, label='ENet模型')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.xlabel('完整模型分割比例')
    plt.ylabel('其他模型比例')
    plt.title('分割比例vs其他模型')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 比例分布直方图
    plt.subplot(2, 4, 3)
    plt.hist(full_ratios, alpha=0.7, label='完整模型编码', bins=20)
    plt.hist(full_seg_ratios, alpha=0.7, label='完整模型分割', bins=20)
    plt.hist(encoder_ratios, alpha=0.7, label='编码器模型', bins=20)
    plt.hist(enet_ratios, alpha=0.7, label='ENet模型', bins=20)

    plt.xlabel('绿植比例')
    plt.ylabel('图像数量')
    plt.title('比例分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 差异分布
    plt.subplot(2, 4, 4)
    diff_full_enc_seg = [abs(r['full_ratio'] - r['full_seg_ratio']) for r in results]
    diff_enc_seg = [abs(r['encoder_ratio'] - r['full_seg_ratio']) for r in results]
    diff_enet_seg = [abs(r['enet_ratio'] - r['full_seg_ratio']) for r in results]
    
    plt.hist(diff_full_enc_seg, bins=20, alpha=0.7, label='完整编码-分割')
    plt.hist(diff_enc_seg, bins=20, alpha=0.7, label='编码器-分割')
    plt.hist(diff_enet_seg, bins=20, alpha=0.7, label='ENet-分割')
    plt.xlabel('比例差异')
    plt.ylabel('图像数量')
    plt.title('模型差异分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 排序后比例曲线
    plt.subplot(2, 4, 5)
    sorted_full = sorted(full_ratios)
    sorted_full_seg = sorted(full_seg_ratios)
    sorted_encoder = sorted(encoder_ratios)
    sorted_enet = sorted(enet_ratios)
    
    plt.plot(range(len(sorted_full)), sorted_full, label='完整编码')
    plt.plot(range(len(sorted_full_seg)), sorted_full_seg, label='完整分割')
    plt.plot(range(len(sorted_encoder)), sorted_encoder, label='编码器')
    plt.plot(range(len(sorted_enet)), sorted_enet, label='ENet')
    plt.xlabel('图像索引（排序后）')
    plt.ylabel('绿植比例')
    plt.title('排序后比例曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 箱线图比较
    plt.subplot(2, 4, 6)
    data_to_plot = [full_ratios, full_seg_ratios, encoder_ratios, enet_ratios]
    plt.boxplot(data_to_plot, tick_labels=['完整编码', '完整分割', '编码器', 'ENet'])
    plt.ylabel('绿植比例')
    plt.title('模型比例箱线图')
    plt.grid(True, alpha=0.3)
    
    # 模型一致性热图（相关系数）
    plt.subplot(2, 4, 7)
    correlation_matrix = np.corrcoef([full_ratios, full_seg_ratios, encoder_ratios, enet_ratios])
    im = plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xticks([0, 1, 2, 3], ['完整编码', '完整分割', '编码器', 'ENet'])
    plt.yticks([0, 1, 2, 3], ['完整编码', '完整分割', '编码器', 'ENet'])
    plt.title('模型相关性热图')
    
    # 添加相关系数值
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f'{correlation_matrix[i, j]:.3f}', 
                    ha='center', va='center', color='white' if correlation_matrix[i, j] < 0.5 else 'black')
    
    # 新增：编码器直出vs分割统计的散点图对比
    plt.subplot(2, 4, 8)
    plt.scatter(full_ratios, full_seg_ratios, alpha=0.6, c='blue', label='完整模型')
    plt.scatter(encoder_ratios, full_seg_ratios, alpha=0.6, c='red', label='编码器模型')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='理想线')
    plt.xlabel('编码器直出比例')
    plt.ylabel('分割统计比例')
    plt.title('编码器直出vs分割统计')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'four_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

# 从文档5中复制的load_checkpoint函数（需要稍作修改以适配当前脚本）
def load_checkpoint(model, optimizer, folder_dir, filename):
    """加载模型检查点（从文档5复制）"""
    assert os.path.isdir(folder_dir), f"目录不存在: {folder_dir}"
    
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(model_path), f"模型文件不存在: {filename}"
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']
    
    return model, optimizer, epoch, miou

def main():
    # 配置参数
    dataset_path = './input/v220-2331'  # 数据集路径
    full_model_path = './output/models/enet_green_ratio_full_officeGreenAndOriInput.pth'  # 完整模型路径
    encoder_model_path = './output/models/enet_green_ratio_encoder_officeGreenAndOriInput.pth'  # 编码器模型路径
    enet_model_path = 'save/ENet_CamVid/ENet'  # ENet模型路径（从文档2中获取）
    output_dir = './output/four_model_officeGreenAndOriInput'  # 输出目录
    
    # 验证四个模型
    results = validate_models_on_dataset(
        dataset_path, 
        full_model_path, 
        encoder_model_path,
        enet_model_path,
        output_dir
    )
    
    print(f"验证完成！共处理 {len(results)} 张图像")
    print(f"完整模型编码器直出平均绿植比例: {np.mean([r['full_ratio'] for r in results]):.4f}")
    print(f"完整模型分割统计平均绿植比例: {np.mean([r['full_seg_ratio'] for r in results]):.4f}")
    print(f"编码器模型平均绿植比例: {np.mean([r['encoder_ratio'] for r in results]):.4f}")
    print(f"ENet模型平均绿植比例: {np.mean([r['enet_ratio'] for r in results]):.4f}")
    
    # 计算编码器直出与分割统计的平均差异
    diff_enc_seg = [abs(r['full_ratio'] - r['full_seg_ratio']) for r in results]
    print(f"编码器直出与分割统计的平均绝对差异: {np.mean(diff_enc_seg):.4f}")

if __name__ == '__main__':
    main()
