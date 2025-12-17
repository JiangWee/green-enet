import os
import argparse
import numpy as np
from PIL import Image
import shutil

# 导入您的数据集模块
from data import CamVid, Cityscapes
from data.utils import pil_loader, remap

def create_binary_green_labels(root_dir, dataset_type, output_dir, split='all'):
    """
    创建二分类绿植标签（绿植=1，非绿植=0）
    
    Args:
        root_dir: 数据集根目录
        dataset_type: 数据集类型 ('camvid' 或 'cityscapes')
        output_dir: 输出目录
        split: 数据分割 ('train', 'val', 'test', 或 'all')
    """
    
    # 创建输出目录结构
    splits = []
    if split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [split]
    
    for current_split in splits:
        # 创建图像和标签输出目录
        img_output_dir = os.path.join(output_dir, current_split, 'images')
        label_output_dir = os.path.join(output_dir, current_split, 'labels')
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)
    
# 根据数据集类型创建数据集实例
    if dataset_type.lower() == 'camvid':
        DatasetClass = CamVid
        # 修正：CamVid中树木的索引是6，交通灯是7
        green_class_index = 5  # 'tree'类别
    elif dataset_type.lower() == 'cityscapes':
        DatasetClass = Cityscapes
        # Cityscapes中的植被类别：'vegetation'（索引为9）
        green_class_index = 9
    else:
        raise ValueError(f"不支持的dataset_type: {dataset_type}")
    
    processed_count = 0
    
    for current_split in splits:
        print(f"处理 {dataset_type} {current_split} 分割...")
        
        try:
            # 创建数据集实例
            dataset = DatasetClass(
                root_dir=root_dir,
                mode=current_split,
                transform=None,
                label_transform=None,
                loader=pil_loader
            )
            
            img_output_dir = os.path.join(output_dir, current_split, 'images')
            label_output_dir = os.path.join(output_dir, current_split, 'labels')
            
            # 遍历数据集
            for i in range(len(dataset)):
                try:
                    # 获取图像和标签路径
                    if current_split.lower() == 'train':
                        image_path = dataset.train_data[i]
                        label_path = dataset.train_labels[i]
                    elif current_split.lower() == 'val':
                        image_path = dataset.val_data[i]
                        label_path = dataset.val_labels[i]
                    elif current_split.lower() == 'test':
                        image_path = dataset.test_data[i]
                        label_path = dataset.test_labels[i]
                    else:
                        continue
                    
                    # 加载原始图像和标签
                    image, label = pil_loader(image_path, label_path)
                    
                    # 调试：打印标签信息
                    print(f"标签路径: {label_path}")
                    print(f"标签模式: {label.mode}")
                    
                    # 将标签转换为numpy数组
                    if isinstance(label, Image.Image):
                        label_array = np.array(label)
                    else:
                        label_array = label
                    
                    print(f"标签数组形状: {label_array.shape}")
                    print(f"标签唯一值: {np.unique(label_array)}")
                    
                    # 创建二分类标签：绿植=1，非绿植=0
                    binary_label = np.zeros_like(label_array)
                    
                    # 简化处理：直接使用索引比较
                    if dataset_type.lower() == 'camvid':
                        # ENet的CamVid标签应该是单通道索引格式
                        if len(label_array.shape) == 3:  # 如果是RGB格式
                            print("警告: CamVid标签是RGB格式，应该是索引格式")
                            # 尝试转换为灰度图，取第一个通道作为索引
                            if label_array.shape[2] == 3:
                                label_array = label_array[:,:,0]  # 取R通道
                        
                        # 直接使用索引比较
                        green_mask = (label_array == green_class_index)
                        
                    else:  # Cityscapes
                        # Cityscapes已经通过remap处理，直接使用索引
                        green_mask = (label_array == green_class_index)
                    
                    binary_label[green_mask] = 1
                    
                    print(f"绿植像素数量: {np.sum(green_mask)}")
                    
                    # 保存二分类标签图像（单通道PNG）
                    binary_label_img = Image.fromarray(binary_label.astype(np.uint8))
                    label_filename = os.path.basename(label_path)
                    binary_label_path = os.path.join(label_output_dir, label_filename)
                    binary_label_img.save(binary_label_path)
                    
                    # 复制原始图像到输出目录
                    image_filename = os.path.basename(image_path)
                    output_image_path = os.path.join(img_output_dir, image_filename)
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(output_image_path)
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:  # 每10张打印一次，便于调试
                        print(f"已处理 {processed_count} 张图像")
                        
                except Exception as e:
                    print(f"处理第 {i} 个样本时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"创建 {current_split} 数据集时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"处理完成! 共转换 {processed_count} 张图像的标签")

def create_dataset_info_file(output_dir, dataset_type):
    """创建数据集信息文件"""
    info_content = f"""
# 二分类绿植分割数据集信息

## 数据集类型
- 原始数据集: {dataset_type}
- 转换后: 二分类（绿植 vs 非绿植）

## 类别定义
- 0: 非绿植
- 1: 绿植

## 目录结构
{output_dir}/
├── train/
│   ├── images/     # 原始图像
│   └── labels/     # 二分类标签 (0-1)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

## 标签格式
- 单通道PNG图像
- 像素值: 0 (非绿植) 或 1 (绿植)
- 图像尺寸与原始图像相同
"""
    
    info_path = os.path.join(output_dir, 'README.md')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(info_content)
    
    print(f"数据集信息文件已保存到: {info_path}")

def main():
    parser = argparse.ArgumentParser(description='创建二分类绿植分割数据集')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='原始数据集根目录')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['camvid', 'cityscapes'],
                       help='数据集类型')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='二分类数据集输出目录')
    parser.add_argument('--split', type=str, default='all',
                       choices=['train', 'val', 'test', 'all'],
                       help='要处理的数据分割')
    
    args = parser.parse_args()
    
    # 创建二分类数据集
    create_binary_green_labels(
        root_dir=args.root_dir,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        split=args.split
    )
    
    # 创建数据集说明文件
    create_dataset_info_file(args.output_dir, args.dataset_type)

if __name__ == '__main__':
    main()