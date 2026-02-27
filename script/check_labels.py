import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 基础路径
base_path = r"D:\code\PyTorch-ENet-master\PyTorch-ENet-master\input\v220-2331-480360-segdata"

# 定义文件夹列表
folders = ["train", "val", "test"]

def create_validation_overlay(original_image_path, label_image_path, output_path):
    """创建验证叠加图像"""
    try:
        # 读取原图
        original_img = Image.open(original_image_path).convert('RGB')
        original_array = np.array(original_img)
        
        # 读取标签图
        label_img = Image.open(label_image_path)
        label_array = np.array(label_img)
        
        # 创建叠加图像（复制原图）
        overlay_array = original_array.copy()
        
        # 创建绿色掩码（RGBA格式，绿色带透明度）
        green_mask = np.zeros((label_array.shape[0], label_array.shape[1], 4), dtype=np.uint8)
        green_mask[:, :, 1] = 255  # 绿色通道
        green_mask[:, :, 3] = 128  # 透明度（128 = 50%透明度）
        
        # 将标签为1的区域应用绿色掩码
        mask = label_array == 1
        overlay_array[mask] = overlay_array[mask] * 0.5 + green_mask[mask, :3] * 0.5
        
        # 保存叠加图像
        overlay_img = Image.fromarray(overlay_array)
        overlay_img.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"错误处理图像 {original_image_path}: {e}")
        return False

def validate_folder(folder_name):
    """验证单个文件夹"""
    folder_path = os.path.join(base_path, folder_name)
    labels_dir = os.path.join(folder_path, "labels")
    check_dir = os.path.join(folder_path, "labels-check")
    
    if not os.path.exists(labels_dir):
        print(f"警告: {folder_name} 文件夹中没有找到 labels 目录")
        return 0
    
    # 创建验证输出目录
    os.makedirs(check_dir, exist_ok=True)
    
    print(f"正在验证 {folder_name} 文件夹...")
    
    success_count = 0
    total_count = 0
    
    # 获取所有PNG标签文件
    png_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
    
    for png_file in png_files:
        total_count += 1
        
        # 构建文件路径
        label_path = os.path.join(labels_dir, png_file)
        
        # 获取对应的原图文件名（将.png替换为.bmp）
        original_filename = png_file.replace('.png', '.bmp')
        original_path = os.path.join(folder_path, original_filename)
        
        # 检查原图是否存在
        if not os.path.exists(original_path):
            print(f"  警告: 原图文件 {original_filename} 不存在，跳过")
            continue
        
        # 构建输出路径
        output_path = os.path.join(check_dir, png_file)
        
        # 创建叠加图像
        if create_validation_overlay(original_path, label_path, output_path):
            success_count += 1
            print(f"  ✓ 已生成验证图: {output_path}")
        else:
            print(f"  ✗ 生成验证图失败: {output_path}")
    
    print(f"  ✓ 成功验证 {success_count}/{total_count} 张图像")
    return success_count

def main():
    print("=" * 50)
    print("开始验证PNG标签图")
    print("=" * 50)
    print(f"基础路径: {base_path}")
    print("-" * 50)
    
    total_success = 0
    total_images = 0
    
    # 处理所有文件夹
    for folder in folders:
        success_count = validate_folder(folder)
        total_success += success_count
        
        # 统计该文件夹的图片总数
        folder_path = os.path.join(base_path, folder)
        labels_dir = os.path.join(folder_path, "labels")
        if os.path.exists(labels_dir):
            png_count = len([f for f in os.listdir(labels_dir) if f.endswith('.png')])
            total_images += png_count
        
        print()  # 空行分隔
    
    print("=" * 50)
    print(f"验证完成! 成功处理 {total_success}/{total_images} 张图像")
    print("=" * 50)
    
    # 显示最终的文件结构
    print("\n生成的文件结构:")
    for folder in folders:
        check_dir = os.path.join(base_path, folder, "labels-check")
        if os.path.exists(check_dir):
            check_files = [f for f in os.listdir(check_dir) if f.endswith('.png')]
            print(f"  {folder}/labels-check/: {len(check_files)} 个验证图文件")

if __name__ == "__main__":
    main()

# v220-2331-480360-segdata/
# ├── train/
# │   ├── labels/                    # 原始标签图
# │   │   ├── output_s001_iso189_480360.png
# │   │   └── ...
# │   ├── labels-check/              # 新生成的验证图
# │   │   ├── output_s001_iso189_480360.png  # 绿色叠加的验证图
# │   │   └── ...
# │   ├── output_s001_iso189_480360.bmp      # 原图
# │   └── ...
# ├── val/
# │   └── labels-check/              # 验证集验证图
# └── test/
#     └── labels-check/             # 测试集验证图