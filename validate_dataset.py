import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def validate_binary_dataset(data_dir, output_dir, num_samples=50):
    """
    验证二分类数据集是否正确
    
    Args:
        data_dir: 数据集根目录 (camvid_binary)
        output_dir: 输出目录，用于保存可视化结果
        num_samples: 每个分割要检查的样本数量
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义分割类型
    splits = ['train', 'val', 'test']
    
    # 统计信息
    stats = {}
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"警告: {split_dir} 不存在，跳过")
            continue
            
        # 创建分割的输出目录
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # 获取图像和标签路径
        image_dir = os.path.join(split_dir, 'images')
        label_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"警告: {image_dir} 或 {label_dir} 不存在，跳过 {split}")
            continue
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(image_files) == 0:
            print(f"警告: {image_dir} 中没有图像文件，跳过 {split}")
            continue
        
        # 限制样本数量
        sample_files = image_files[:min(num_samples, len(image_files))]
        
        split_stats = {
            'total_images': len(image_files),
            'checked_images': len(sample_files),
            'label_values': set(),
            'image_shapes': [],
            'label_shapes': [],
            'green_ratios': []
        }
        
        print(f"\n检查 {split} 分割...")
        print(f"总图像数: {len(image_files)}")
        print(f"检查样本数: {len(sample_files)}")
        
        for i, img_file in enumerate(sample_files):
            try:
                # 构建路径
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file)  # 假设标签文件名与图像相同
                
                # 如果标签文件不存在，尝试其他扩展名
                if not os.path.exists(label_path):
                    # 尝试.png扩展名
                    base_name = os.path.splitext(img_file)[0]
                    possible_labels = [f for f in os.listdir(label_dir) 
                                    if f.startswith(base_name) and f.endswith(('.png', '.jpg', '.jpeg'))]
                    if possible_labels:
                        label_path = os.path.join(label_dir, possible_labels[0])
                    else:
                        print(f"警告: 找不到 {img_file} 的标签文件")
                        continue
                
                # 加载图像和标签
                image = Image.open(img_path).convert('RGB')
                label = Image.open(label_path)
                
                # 转换为numpy数组进行检查
                image_np = np.array(image)
                label_np = np.array(label)
                
                # 记录统计信息
                split_stats['image_shapes'].append(image_np.shape)
                split_stats['label_shapes'].append(label_np.shape)
                split_stats['label_values'].update(np.unique(label_np))
                

                green_ratio = np.mean(label_np == 1) 
                
                split_stats['green_ratios'].append(green_ratio)
                
                # 创建可视化
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # 显示原始图像
                axes[0, 0].imshow(image_np)
                axes[0, 0].set_title(f'Original Image\nShape: {image_np.shape}')
                axes[0, 0].axis('off')
                

                axes[0, 1].imshow(label_np, cmap='gray', vmin=0, vmax=1)
                axes[0, 1].set_title(f'Label (Grayscale)\nShape: {label_np.shape}')
                axes[0, 1].axis('off')
                
                # 显示标签的唯一值
                unique_vals = np.unique(label_np)

                axes[0, 2].hist(label_np.flatten(), bins=len(unique_vals))
                axes[0, 2].set_title(f'Label Values: {unique_vals}\nGreen Ratio: {green_ratio:.4f}')
                
                # 显示叠加图像（绿植区域用绿色覆盖）
                overlay = image_np.copy()

                green_mask = label_np == 1
                
                # 在绿植区域叠加绿色
                overlay[green_mask] = [0, 255, 0]  # 绿色覆盖
                
                axes[1, 0].imshow(overlay)
                axes[1, 0].set_title('Green Overlay')
                axes[1, 0].axis('off')
                
                # 显示绿植掩码
                axes[1, 1].imshow(green_mask, cmap='viridis')
                axes[1, 1].set_title(f'Green Mask\nGreen Pixels: {np.sum(green_mask)}')
                axes[1, 1].axis('off')
                
                # 显示统计信息
                axes[1, 2].axis('off')
                info_text = f"文件: {img_file}\n"
                info_text += f"图像尺寸: {image_np.shape}\n"
                info_text += f"标签尺寸: {label_np.shape}\n"
                info_text += f"标签值: {unique_vals}\n"
                info_text += f"绿植比例: {green_ratio:.4f}\n"
                info_text += f"绿植像素: {np.sum(green_mask)}\n"
                info_text += f"总像素: {label_np.size}"
                
                axes[1, 2].text(0.1, 0.9, info_text, transform=axes[1, 2].transAxes, 
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.suptitle(f'{split} - Sample {i+1}/{len(sample_files)}', fontsize=16)
                plt.tight_layout()
                
                # 保存图像
                output_filename = f"{split}_{i:03d}_{os.path.splitext(img_file)[0]}.png"
                output_path = os.path.join(split_output_dir, output_filename)
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i+1}/{len(sample_files)} 个样本")
                    
            except Exception as e:
                print(f"处理 {img_file} 时出错: {str(e)}")
                continue
        
        # 计算分割的总体统计
        if split_stats['green_ratios']:
            split_stats['avg_green_ratio'] = np.mean(split_stats['green_ratios'])
            split_stats['min_green_ratio'] = np.min(split_stats['green_ratios'])
            split_stats['max_green_ratio'] = np.max(split_stats['green_ratios'])
        
        stats[split] = split_stats
        
        # 为每个分割创建汇总图
        create_split_summary(split, split_stats, split_output_dir)
    
    # 创建总体报告
    create_overall_report(stats, output_dir)
    
    return stats

def create_split_summary(split, stats, output_dir):
    """为每个分割创建汇总统计图"""
    if not stats['green_ratios']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绿植比例分布
    axes[0, 0].hist(stats['green_ratios'], bins=20, alpha=0.7, color='green')
    axes[0, 0].axvline(stats['avg_green_ratio'], color='red', linestyle='--', 
                      label=f'平均: {stats["avg_green_ratio"]:.4f}')
    axes[0, 0].set_xlabel('绿植比例')
    axes[0, 0].set_ylabel('图像数量')
    axes[0, 0].set_title(f'{split}分割 - 绿植比例分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图像尺寸分布
    if stats['image_shapes']:
        heights = [shape[0] for shape in stats['image_shapes']]
        widths = [shape[1] for shape in stats['image_shapes']]
        
        axes[0, 1].scatter(widths, heights, alpha=0.6)
        axes[0, 1].set_xlabel('宽度')
        axes[0, 1].set_ylabel('高度')
        axes[0, 1].set_title(f'{split}分割 - 图像尺寸分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加平均尺寸线
        avg_height = np.mean(heights)
        avg_width = np.mean(widths)
        axes[0, 1].axhline(avg_height, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(avg_width, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].text(0.05, 0.95, f'平均尺寸: {int(avg_width)}×{int(avg_height)}', 
                       transform=axes[0, 1].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 标签值统计
    axes[1, 0].bar([str(x) for x in stats['label_values']], 
                  [1] * len(stats['label_values']), color='skyblue')
    axes[1, 0].set_xlabel('标签值')
    axes[1, 0].set_ylabel('出现')
    axes[1, 0].set_title(f'{split}分割 - 标签值分布')
    
    # 统计信息文本
    axes[1, 1].axis('off')
    summary_text = f"{split}分割统计摘要:\n\n"
    summary_text += f"总图像数: {stats['total_images']}\n"
    summary_text += f"检查样本数: {stats['checked_images']}\n"
    summary_text += f"标签值: {sorted(stats['label_values'])}\n"
    if stats['green_ratios']:
        summary_text += f"平均绿植比例: {stats['avg_green_ratio']:.4f}\n"
        summary_text += f"最小绿植比例: {stats['min_green_ratio']:.4f}\n"
        summary_text += f"最大绿植比例: {stats['max_green_ratio']:.4f}\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'{split}分割 - 数据集验证摘要', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{split}_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_overall_report(stats, output_dir):
    """创建总体报告"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 各分割绿植比例比较
    split_names = []
    avg_ratios = []
    
    for split, split_stats in stats.items():
        if split_stats.get('green_ratios'):
            split_names.append(split)
            avg_ratios.append(split_stats['avg_green_ratio'])
    
    if avg_ratios:
        bars = axes[0, 0].bar(split_names, avg_ratios, color=['blue', 'orange', 'green'])
        axes[0, 0].set_ylabel('平均绿植比例')
        axes[0, 0].set_title('各分割平均绿植比例比较')
        
        # 在柱状图上添加数值
        for bar, ratio in zip(bars, avg_ratios):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{ratio:.4f}', ha='center', va='bottom')
    
    # 各分割样本数量
    total_images = [stats[split]['total_images'] for split in stats if 'total_images' in stats[split]]
    checked_images = [stats[split]['checked_images'] for split in stats if 'checked_images' in stats[split]]
    
    x = range(len(split_names))
    width = 0.35
    
    axes[0, 1].bar(x, total_images, width, label='总图像数', alpha=0.8)
    axes[0, 1].bar([i + width for i in x], checked_images, width, label='检查样本数', alpha=0.8)
    axes[0, 1].set_xlabel('分割')
    axes[0, 1].set_ylabel('图像数量')
    axes[0, 1].set_title('各分割样本数量')
    axes[0, 1].set_xticks([i + width/2 for i in x])
    axes[0, 1].set_xticklabels(split_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 标签值汇总
    all_label_values = set()
    for split_stats in stats.values():
        if 'label_values' in split_stats:
            all_label_values.update(split_stats['label_values'])
    
    axes[1, 0].bar([str(x) for x in sorted(all_label_values)], 
                  [1] * len(all_label_values), color='lightcoral')
    axes[1, 0].set_xlabel('标签值')
    axes[1, 0].set_ylabel('出现')
    axes[1, 0].set_title('所有分割的标签值汇总')
    
    # 总体统计信息
    axes[1, 1].axis('off')
    
    total_summary = "数据集总体统计:\n\n"
    total_images_all = sum(total_images)
    total_checked_all = sum(checked_images)
    
    total_summary += f"总图像数: {total_images_all}\n"
    total_summary += f"总检查样本: {total_checked_all}\n"
    total_summary += f"检查比例: {total_checked_all/total_images_all*100:.1f}%\n\n"
    
    total_summary += "各分割详情:\n"
    for split in split_names:
        if split in stats and stats[split].get('green_ratios'):
            total_summary += f"{split}: {stats[split]['total_images']} 图像, "
            total_summary += f"平均绿植比例: {stats[split]['avg_green_ratio']:.4f}\n"
    
    total_summary += f"\n标签值范围: {sorted(all_label_values)}"
    
    axes[1, 1].text(0.05, 0.95, total_summary, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('CamVid二分类数据集验证报告', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_report.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存文本报告
    save_text_report(stats, output_dir)

def save_text_report(stats, output_dir):
    """保存文本格式的详细报告"""
    report_path = os.path.join(output_dir, 'dataset_validation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CamVid二分类数据集验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        for split, split_stats in stats.items():
            f.write(f"{split.upper()}分割:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总图像数: {split_stats.get('total_images', 'N/A')}\n")
            f.write(f"检查样本数: {split_stats.get('checked_images', 'N/A')}\n")
            f.write(f"标签值: {sorted(split_stats.get('label_values', []))}\n")
            
            if split_stats.get('green_ratios'):
                f.write(f"平均绿植比例: {split_stats['avg_green_ratio']:.4f}\n")
                f.write(f"最小绿植比例: {split_stats['min_green_ratio']:.4f}\n")
                f.write(f"最大绿植比例: {split_stats['max_green_ratio']:.4f}\n")
            
            if split_stats.get('image_shapes'):
                heights = [shape[0] for shape in split_stats['image_shapes']]
                widths = [shape[1] for shape in split_stats['image_shapes']]
                f.write(f"平均图像尺寸: {int(np.mean(widths))}×{int(np.mean(heights))}\n")
            
            f.write("\n")
        
        # 数据质量检查
        f.write("数据质量检查:\n")
        f.write("-" * 40 + "\n")
        
        all_label_values = set()
        for split_stats in stats.values():
            if 'label_values' in split_stats:
                all_label_values.update(split_stats['label_values'])
        
        expected_values = {0, 1}
        if all_label_values == expected_values:
            f.write("✓ 标签值正确: 只包含0和1\n")
        else:
            f.write("✗ 标签值异常: 包含非0/1的值\n")
            f.write(f"  期望值: {expected_values}\n")
            f.write(f"  实际值: {sorted(all_label_values)}\n")
        
        # 检查图像和标签尺寸匹配
        size_mismatch = False
        for split, split_stats in stats.items():
            if 'image_shapes' in split_stats and 'label_shapes' in split_stats:
                for i, (img_shape, label_shape) in enumerate(zip(split_stats['image_shapes'], split_stats['label_shapes'])):
                    if img_shape[:2] != label_shape[:2]:  # 比较高度和宽度
                        f.write(f"✗ 尺寸不匹配: {split}分割第{i}个样本\n")
                        f.write(f"  图像尺寸: {img_shape}\n")
                        f.write(f"  标签尺寸: {label_shape}\n")
                        size_mismatch = True
                        break
        
        if not size_mismatch:
            f.write("✓ 图像和标签尺寸匹配正常\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("验证完成\n")
        f.write("=" * 60 + "\n")
    
    print(f"详细报告已保存到: {report_path}")

def check_specific_problem(data_dir, sample_index=0, split='val'):
    """
    专门检查特定样本的问题
    """
    split_dir = os.path.join(data_dir, split)
    image_dir = os.path.join(split_dir, 'images')
    label_dir = os.path.join(split_dir, 'labels')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if sample_index >= len(image_files):
        print(f"样本索引 {sample_index} 超出范围，最大索引为 {len(image_files)-1}")
        return
    
    img_file = image_files[sample_index]
    img_path = os.path.join(image_dir, img_file)
    
    # 查找对应的标签文件
    base_name = os.path.splitext(img_file)[0]
    label_files = [f for f in os.listdir(label_dir) if f.startswith(base_name)]
    
    if not label_files:
        print(f"找不到 {img_file} 对应的标签文件")
        return
    
    label_path = os.path.join(label_dir, label_files[0])
    
    print(f"检查样本: {img_file}")
    print(f"标签文件: {label_files[0]}")
    
    # 加载并详细检查
    image = Image.open(img_path).convert('RGB')
    label = Image.open(label_path)
    
    image_np = np.array(image)
    label_np = np.array(label)
    
    print(f"图像尺寸: {image_np.shape}")
    print(f"标签尺寸: {label_np.shape}")
    print(f"标签模式: {label.mode}")
    print(f"标签唯一值: {np.unique(label_np)}")
    print(f"标签数据类型: {label_np.dtype}")
    
    # 显示详细统计
    if label_np.ndim == 3:
        print("标签是RGB格式")
        # 检查每个通道的统计
        for i, channel in enumerate(['R', 'G', 'B']):
            unique_vals = np.unique(label_np[:,:,i])
            print(f"  {channel}通道唯一值: {unique_vals}")
    else:
        print("标签是单通道格式")
        unique_vals = np.unique(label_np)
        counts = [(val, np.sum(label_np == val)) for val in unique_vals]
        print("像素值统计:")
        for val, count in counts:
            print(f"  值 {val}: {count} 像素 ({count/label_np.size*100:.2f}%)")

if __name__ == '__main__':
    # 使用示例
    data_dir = 'camvid_binary'  # 你的数据集路径
    output_dir = 'dataset_validation_results'
    
    print("开始验证CamVid二分类数据集...")
    stats = validate_binary_dataset(data_dir, output_dir, num_samples=20)
    
    print(f"\n验证完成！结果已保存到: {output_dir}")
    
    # 如果需要检查特定样本的问题，取消下面的注释
    # print("\n详细检查第一个样本:")
    # check_specific_problem(data_dir, sample_index=0, split='val')