# check_binary_labels.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 避免显示问题

def check_binary_dataset(binary_dir='camvid_binary'):
    """检查二值化数据集的质量"""
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(binary_dir, split)
        if not os.path.exists(split_dir):
            print(f"跳过：{split_dir} 不存在")
            continue
            
        label_dir = os.path.join(split_dir, 'labels')
        image_dir = os.path.join(split_dir, 'images')
        
        if not os.path.exists(label_dir) or not os.path.exists(image_dir):
            print(f"跳过：{split} 目录不完整")
            continue
            
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
        
        if not label_files:
            print(f"无标签文件在 {label_dir}")
            continue
            
        # 修复这里的语法错误
        print(f"\n{'='*60}")
        print(f"检查 {split} 分割")
        print(f"{'='*60}")

        # 检查前5个文件
        for i, label_file in enumerate(label_files[:5]):
            label_path = os.path.join(label_dir, label_file)
            image_path = os.path.join(image_dir, label_file)
            
            if not os.path.exists(image_path):
                print(f"图片不存在: {image_path}")
                continue
                
            # 加载标签
            label = Image.open(label_path)
            label_array = np.array(label)
            
            # 检查标签
            unique_vals = np.unique(label_array)
            green_pixels = np.sum(label_array == 1)
            total_pixels = label_array.size
            green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
            
            print(f"\n样本 {i+1}: {label_file}")
            print(f"  标签形状: {label_array.shape}")
            print(f"  唯一值: {unique_vals}")
            print(f"  绿植像素: {green_pixels} / {total_pixels} ({green_ratio*100:.2f}%)")
            
            # 验证原始CamVid标签
            original_camvid_dir = "CamVid"
            if os.path.exists(original_camvid_dir):
                # 尝试找到对应的原始标签
                original_label_path = None
                for subdir in ['trainannot', 'valannot', 'testannot']:
                    candidate = os.path.join(original_camvid_dir, subdir, label_file)
                    if os.path.exists(candidate):
                        original_label_path = candidate
                        break
                
                if original_label_path:
                    orig_label = Image.open(original_label_path)
                    orig_array = np.array(orig_label)
                    orig_tree_pixels = np.sum(orig_array == 6)
                    orig_ratio = orig_tree_pixels / orig_array.size
                    print(f"  原始CamVid标签树像素: {orig_tree_pixels} ({orig_ratio*100:.2f}%)")
                    
                    # 比较转换是否正确
                    expected_green = (orig_array == 6).astype(np.uint8)
                    if np.array_equal(label_array, expected_green):
                        print(f"  ✓ 二值化转换正确")
                    else:
                        print(f"  ✗ 二值化转换错误")
                        diff = np.sum(label_array != expected_green)
                        print(f"    差异像素数: {diff}")
            
            # 可视化
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image_array = np.array(image)
                
                # 修复：使用 plt.subplots() 而不是 axes.subplots()
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 原图
                axes[0].imshow(image_array)
                axes[0].set_title(f'Image\n{label_file}')
                axes[0].axis('off')
                
                # 二值标签
                axes[1].imshow(label_array, cmap='gray')
                axes[1].set_title(f'Binary Label\nGreen: {green_ratio*100:.2f}%')
                axes[1].axis('off')
                
                # 原始CamV
                if original_label_path and os.path.exists(original_label_path):
                    axes[2].imshow(orig_array, cmap='tab20')
                    axes[2].set_title(f'Original CamVid Label\nIndex 6: {orig_ratio*100:.2f}%')
                else:
                    axes[2].text(0.5, 0.5, 'Original label\nnot found', 
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=axes[2].transAxes, fontsize=12)
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'check_{split}_{i}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        # 统计整个分割
        all_green_ratios = []
        for label_file in label_files[:20]:  # 只检查前20个以节省时间
            label_path = os.path.join(label_dir, label_file)
            label = Image.open(label_path)
            label_array = np.array(label)
            
            green_ratio = np.sum(label_array == 1) / label_array.size
            all_green_ratios.append(green_ratio)
        
        if all_green_ratios:
            avg_ratio = np.mean(all_green_ratios)
            print(f"\n{split} 平均绿植比例: {avg_ratio*100:.4f}%")
            print(f"最小绿植比例: {np.min(all_green_ratios)*100:.4f}%")
            print(f"最大绿植比例: {np.max(all_green_ratios)*100:.4f}%")
            
            # 绘制分布
            plt.figure(figsize=(10, 6))
            plt.hist(all_green_ratios, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Green Ratio')
            plt.ylabel('Frequency')
            plt.title(f'{split} Split - Green Ratio Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'green_ratio_distribution_{split}.png', dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    check_binary_dataset()