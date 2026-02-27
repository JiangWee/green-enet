import json
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2

# 基础路径 - 请根据您的实际情况修改
# base_path = r"D:\code\PyTorch-ENet-master\PyTorch-ENet-master\input\v220-2331-480360-segdata"
# 定义文件夹列表
# folders = ["train", "val", "test"]

base_path = r"D:\code\PyTorch-ENet-master\PyTorch-ENet-master\input\test-coco"
folders = ["."]


# 类别映射 - 根据您的COCO文件中的categories定义
# COCO文件中vegetable的id是1，背景为0
class_mapping = {
    1: 1,  # vegetable -> 1
    # 背景自动为0
}

def convert_coco_to_png(coco_json_path, output_dir, images_dir):
    """将COCO格式的JSON转换为PNG标签图"""
    
    # 读取COCO JSON文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图像ID到图像信息的映射
    image_info_map = {img['id']: img for img in coco_data['images']}
    
    # 按图像分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # 处理每个图像
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_info_map:
            print(f"警告: 图像ID {image_id} 在images列表中未找到")
            continue
            
        img_info = image_info_map[image_id]
        width = img_info['width']
        height = img_info['height']
        file_name = img_info['file_name']
        
        # 创建空白图像（全0表示背景）
        label_img = Image.new('L', (width, height), 0)  # 'L'表示8位灰度图
        draw = ImageDraw.Draw(label_img)
        
        # 处理该图像的所有标注
        for ann in annotations:
            category_id = ann['category_id']
            
            # 使用类别映射
            if category_id in class_mapping:
                class_id = class_mapping[category_id]
            else:
                print(f"警告: 跳过未定义类别ID {category_id}")
                continue
            
            # 处理分割多边形
            segmentation = ann['segmentation']
            
            # COCO分割格式可能是多个多边形（对于复杂形状）
            for polygon_points in segmentation:
                # 将[x1, y1, x2, y2, ...]格式转换为[(x1, y1), (x2, y2), ...]
                points = [(polygon_points[i], polygon_points[i+1]) 
                         for i in range(0, len(polygon_points), 2)]
                
                # 将浮点数坐标转换为整数
                int_points = [(int(x), int(y)) for x, y in points]
                
                # 绘制多边形
                if len(int_points) >= 3:  # 确保是多边形
                    draw.polygon(int_points, fill=class_id)
        
        # 生成输出文件名（保持与原图相同的文件名，但扩展名为.png）
        base_name = os.path.splitext(file_name)[0]
        png_filename = base_name + '.png'
        output_path = os.path.join(output_dir, png_filename)
        
        # 保存PNG文件
        label_img.save(output_path)
        
        print(f"  ✓ 已生成: {output_path}")
    
    return len(annotations_by_image)

def process_folder(folder_name):
    """处理单个文件夹"""
    folder_path = os.path.join(base_path, folder_name)
    json_path = os.path.join(folder_path, "annotations.json")
    
    if not os.path.exists(json_path):
        print(f"警告: {folder_name} 文件夹中没有找到 annotations.json")
        return 0
    
    print(f"正在处理 {folder_name} 文件夹...")
    
    # 图像文件所在的目录（假设图像文件与JSON在同一目录）
    images_dir = folder_path
    
    # 为每个文件夹创建独立的输出目录
    output_dir = os.path.join(base_path, folder_name, "labels")
    
    # 转换COCO JSON到PNG
    try:
        count = convert_coco_to_png(json_path, output_dir, images_dir)
        print(f"  ✓ 成功处理 {count} 张图像的标注")
        return count
    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        return 0

def main():
    print("=" * 50)
    print("开始转换COCO格式标注文件为PNG标签图")
    print("=" * 50)
    print(f"基础路径: {base_path}")
    print(f"类别映射: {class_mapping}")
    print("-" * 50)
    
    total_count = 0
    # 处理所有文件夹
    for folder in folders:
        count = process_folder(folder)
        total_count += count
        print()  # 空行分隔
    
    print("=" * 50)
    print(f"转换完成! 总共处理了 {total_count} 张图像的标注")
    print("=" * 50)
    
    # 显示最终的文件结构
    print("\n生成的文件结构:")
    for folder in folders:
        labels_dir = os.path.join(base_path, folder, "labels")
        if os.path.exists(labels_dir):
            png_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
            print(f"  {folder}/labels/: {len(png_files)} 个PNG文件")

if __name__ == "__main__":
    main()
