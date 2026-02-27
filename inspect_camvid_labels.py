import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def inspect_camvid_labels(camvid_root_dir):
    """检查CamVid标签格式和内容"""
    
    # 找到第一个标签文件
    label_dir = os.path.join(camvid_root_dir, 'trainannot')
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
    
    if not label_files:
        print("未找到标签文件")
        return
    
    first_label_path = os.path.join(label_dir, label_files[0])
    label = Image.open(first_label_path)
    label_array = np.array(label)
    
    print(f"标签形状: {label_array.shape}")
    print(f"标签数据类型: {label_array.dtype}")
    print(f"唯一值: {np.unique(label_array)}")
    
    # 检查是RGB还是索引格式
    if len(label_array.shape) == 3:
        print("检测到RGB格式标签")
        # 显示前几个像素的颜色值
        print("前10个像素的RGB值:")
        print(label_array.reshape(-1, 3)[:10])
    else:
        print("检测到索引格式标签")
    
    # 可视化标签
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(label_array)
    plt.title('标签图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    if len(label_array.shape) == 3:
        # 如果是RGB，转换为灰度来显示值分布
        unique_vals = np.unique(label_array.reshape(-1, 3), axis=0)
        print(f"发现 {len(unique_vals)} 种唯一颜色")
        # 显示颜色映射
        for i, color in enumerate(unique_vals[:10]):  # 只显示前10种
            print(f"颜色 {i}: {color}")
    else:
        plt.hist(label_array.flatten(), bins=50)
        plt.title('像素值分布')
        plt.xlabel('类别索引')
        plt.ylabel('像素数量')
    
    plt.subplot(1, 3, 3)
    # 尝试识别tree类别
    if len(label_array.shape) == 3:
        # RGB格式 - 查找tree的颜色 (128, 128, 0)
        tree_mask = np.all(label_array == [128, 128, 0], axis=-1)
        plt.imshow(tree_mask, cmap='gray')
        plt.title('Tree类别掩码')
        tree_ratio = np.sum(tree_mask) / tree_mask.size
        print(f"Tree类别比例: {tree_ratio:.4f}")
    else:
        # 索引格式 - 尝试不同的索引
        for idx in [6, 7, 8]:  # 常见的tree类别索引
            tree_mask = (label_array == idx)
            tree_ratio = np.sum(tree_mask) / tree_mask.size
            print(f"索引 {idx} 的比例: {tree_ratio:.4f}")
            if tree_ratio > 0.01:  # 如果比例合理，显示它
                plt.imshow(tree_mask, cmap='gray')
                plt.title(f'索引 {idx} 的Tree掩码')
                break
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 使用示例
camvid_root = "CamVid"  # 您的CamVid数据集路径
inspect_camvid_labels(camvid_root)