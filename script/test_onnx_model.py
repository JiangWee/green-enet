import onnxruntime
import numpy as np
import torch
from pathlib import Path
import cv2
from PIL import Image
import torchvision.transforms as transforms

# 保存预处理后的图像用于对比
def save_preprocessed_image(tensor, filename):
    """保存预处理后的图像"""
    img_np = tensor.squeeze(0).numpy()
    img_np = img_np.transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img_np)
    img_pil.save(filename)
    print(f"Python预处理图像已保存: {filename}")
    
# 新增调试节点1：将input tensor逆向转换回原始尺寸
def tensor_to_image(tensor, target_size, original_size):
    """将tensor转换回图像并调整到原始尺寸"""
    # 移除batch维度并转换为numpy
    img_np = tensor.squeeze(0).numpy()
    # 从CHW转换为HWC
    img_np = img_np.transpose(1, 2, 0)
    # 反归一化到0-255
    img_np = (img_np * 255).astype(np.uint8)
    
    # 创建PIL图像
    img_pil = Image.fromarray(img_np)
    
    # 调整到原始尺寸
    img_restored = img_pil.resize(original_size, Image.BILINEAR)
    
    return img_restored

def preprocess_image_for_onnx(image_path, target_size=(360, 480)):
    """统一使用RGB顺序与训练保持一致"""
    # 使用OpenCV读取
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 关键修改：BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ← 添加这一行
    
    original_size = (image.shape[1], image.shape[0])
    print(f"Python原始图像尺寸: {original_size[0]}x{original_size[1]}")
    
    # 调整尺寸
    target_height, target_width = target_size
    if original_size != (target_width, target_height):
        image_resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        print(f"调整尺寸到: {target_width}x{target_height}")
    else:
        image_resized = image
        print("使用原始尺寸，跳过resize")
    
    # 转换为float32并归一化，与训练一致
    image_float = image_resized.astype(np.float32) / 255.0
    
    # 转换为CHW格式
    image_chw = image_float.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = np.expand_dims(image_chw, axis=0)  # [1, 3, H, W]
    
    return input_tensor, original_size
    
def save_preprocessed_image(tensor, filename):
    """保存预处理后的图像"""
    img_np = tensor.squeeze(0).numpy()
    img_np = img_np.transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img_np)
    img_pil.save(filename)
    print(f"Python预处理图像已保存: {filename}")

def test_onnx_model(onnx_path, input_tensor, model_type):
    """测试ONNX模型推理"""
    print(f"尝试加载模型: {onnx_path}")
        
    if not Path(onnx_path).exists():
        print(f"错误：模型文件不存在于 {onnx_path}")
        return None


    
    try:
        # 创建推理会话
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # 获取输入输出名称
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        print(f"输入名称: {input_name}, 形状: {input_shape}")
        
        output_names = [output.name for output in ort_session.get_outputs()]
        print(f"输出名称: {output_names}")
        
        # 运行推理
        ort_outputs = ort_session.run(
            output_names, 
            {input_name: input_tensor}
        )
        output_match = compare_output_tensors(ort_outputs, "./output/debug_cpp")
        # 添加详细的调试信息
        print(f"输入张量范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        print(f"输入张量均值: {input_tensor.mean():.3f}")
        # 详细的输出分析
        print(f"输出数量: {len(ort_outputs)}")
        for i, output in enumerate(ort_outputs):
            print(f"输出{i}: 形状{output.shape}, 范围[{output.min():.3f}, {output.max():.3f}]")
        results = {}
        
        # 根据模型类型处理输出
        if model_type == "multi_class":
            # 多标签库：输出分割图，统计植被比例（类别5）
            segmentation = ort_outputs[0]  # [1, num_classes, H, W]
            print(f"分割输出形状: {segmentation.shape}")
            
            # 取argmax得到预测类别
            pred_mask = np.argmax(segmentation[0], axis=0)  # [H, W]
            print(f"预测掩码形状: {pred_mask.shape}")
            print(f"预测掩码唯一值: {np.unique(pred_mask)}")
            
            # 统计植被比例（多标签库中植被是类别5）
            vegetation_pixels = np.sum(pred_mask == 5)
            total_pixels = pred_mask.size
            vegetation_ratio = vegetation_pixels / total_pixels
            
            results['vegetation_ratio'] = vegetation_ratio
            results['segmentation'] = pred_mask
            
        elif model_type == "full":
            # 检查输出结构，可能需要调整
            if len(ort_outputs) >= 2:
                # 根据形状判断哪个是分割图，哪个是比例
                for i, output in enumerate(ort_outputs):
                    if output.ndim == 4:  # [1, C, H, W] 可能是分割图
                        segmentation = output
                    elif output.ndim == 2 or output.size == 1:  # 可能是比例
                        direct_ratio = output
            
            print(f"分割输出形状: {segmentation.shape}")
            print(f"直接输出形状: {direct_ratio.shape}")
            print(f"直接输出值: {direct_ratio}")
            
            # 统计分割图的植被比例（新训练模型中植被是类别1）
            pred_mask = np.argmax(segmentation[0], axis=0)  # [H, W]
            vegetation_pixels = np.sum(pred_mask == 1)
            total_pixels = pred_mask.size
            segmentation_ratio = vegetation_pixels / total_pixels
            
            # 直接输出的绿植比例
            if direct_ratio.size == 1:
                direct_ratio_value = float(direct_ratio[0])
            else:
                direct_ratio_value = float(np.mean(direct_ratio))
            
            results['vegetation_ratio_segmentation'] = segmentation_ratio
            results['vegetation_ratio_direct'] = direct_ratio_value
            results['segmentation'] = pred_mask
                
        elif model_type == "encoder":
            # 新训练encoder库：直接输出绿植比例
            output_data = ort_outputs[0]
            print(f"输出形状: {output_data.shape}")
            print(f"输出值: {output_data}")
            
            if output_data.size == 1:
                vegetation_ratio = float(output_data[0])
            else:
                # 如果输出不是标量，取平均值
                vegetation_ratio = float(np.mean(output_data))
            
            results['vegetation_ratio'] = vegetation_ratio
            
        # 运行推理后
        print(f"推理完成，输出数量: {len(ort_outputs)}")
        return results
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_segmentation_visualization(segmentation, original_size, filename, model_type):
    """保存分割结果可视化"""
    try:
        # 将分割图上采样到原始尺寸
        if len(segmentation.shape) == 2:  # [H, W]
            seg_resized = cv2.resize(
                segmentation.astype(np.uint8), 
                (original_size[0], original_size[1]),  # 修复：使用(宽, 高)
                interpolation=cv2.INTER_NEAREST
            )
        else:
            print(f"不支持的分割图形状: {segmentation.shape}")
            return
        
        # 根据模型类型确定植被标签
        if model_type == "multi_class":
            vegetation_label = 5
        else:  # full模型
            vegetation_label = 1
        
        # 创建植被掩码（植被为白色，其他为黑色）
        vegetation_mask = np.zeros_like(seg_resized, dtype=np.uint8)
        vegetation_mask[seg_resized == vegetation_label] = 255
        
        # 保存为BMP
        cv2.imwrite(filename, vegetation_mask)
        print(f"植被掩码已保存: {filename}")
        
    except Exception as e:
        print(f"保存可视化失败: {e}")


def save_tensor_debug(tensor, filename_prefix, language):
    """保存张量调试信息"""
    # 保存二进制文件
    tensor.tofile(f"{filename_prefix}_{language}_tensor.bin")
    
    # 保存统计信息
    stats = {
        'min': tensor.min(),
        'max': tensor.max(),
        'mean': tensor.mean(),
        'std': tensor.std(),
        'shape': tensor.shape
    }
    
    print(f"{language}输入张量统计 - 最小值: {stats['min']:.6f}, "
          f"最大值: {stats['max']:.6f}, 均值: {stats['mean']:.6f}, "
          f"形状: {tensor.shape}")
    
    # 保存前10个值
    with open(f"{filename_prefix}_{language}_sample.txt", "w") as f:
        flat_tensor = tensor.flatten()
        for i in range(min(10, flat_tensor.size)):
            f.write(f"{flat_tensor[i]:.6f}\n")
    
    return stats

def compare_input_tensors(cpp_file, python_tensor, tolerance=1e-5):
    """比对C++和Python的输入张量"""
    try:
        # 读取C++保存的张量
        cpp_data = np.fromfile(cpp_file, dtype=np.float32)
        python_flat = python_tensor.flatten()
        
        print(f"C++张量大小: {cpp_data.size}, Python张量大小: {python_flat.size}")
        
        if cpp_data.size != python_flat.size:
            print("错误: 张量尺寸不匹配!")
            return False

        # 逐元素比对
        max_diff = 0
        diff_count = 0
        for i in range(cpp_data.size):
            diff = abs(cpp_data[i] - python_flat[i])
            if diff > tolerance:
                diff_count += 1
                max_diff = max(max_diff, diff)
                if diff_count <= 10:  # 只打印前10个差异
                    print(f"位置 {i}: C++={cpp_data[i]:.6f}, Python={python_flat[i]:.6f}, 差异={diff:.6f}")
        
        if diff_count > 0:
            print(f"发现 {diff_count} 个差异点，最大差异: {max_diff:.6f}")
            return False
        else:
            print("输入张量完全一致!")
            return True
            
    except Exception as e:
        print(f"比对输入张量时出错: {e}")
        return False
    
def compare_output_tensors(python_outputs, cpp_prefix, tolerance=1e-5):
    """比对C++和Python的输出张量"""
    results = {}
    
    for i, py_output in enumerate(python_outputs):
        cpp_filename = f"{cpp_prefix}_output_node_{i}.bin"
        
        try:
            # 读取C++输出
            cpp_data = np.fromfile(cpp_filename, dtype=np.float32)
            py_flat = py_output.flatten()
            
            print(f"\n比对输出节点 {i}:")
            print(f"C++形状: {cpp_data.shape}, Python形状: {py_output.shape}")
            print(f"C++范围: [{cpp_data.min():.6f}, {cpp_data.max():.6f}]")
            print(f"Python范围: [{py_output.min():.6f}, {py_output.max():.6f}]")
            
            if cpp_data.size != py_flat.size:
                print("错误: 输出张量尺寸不匹配!")
                results[i] = False
                continue
                
            # 逐元素比对
            max_diff = 0
            diff_count = 0
            for j in range(cpp_data.size):
                diff = abs(cpp_data[j] - py_flat[j])
                if diff > tolerance:
                    diff_count += 1
                    max_diff = max(max_diff, diff)
                    
            if diff_count > 0:
                print(f"发现 {diff_count} 个差异点，最大差异: {max_diff:.6f}")
                results[i] = False
            else:
                print("输出张量完全一致!")
                results[i] = True
                
        except Exception as e:
            print(f"比对输出节点 {i} 时出错: {e}")
            results[i] = False
            
    return results


def debug_tensor_comparison(cpp_tensor, python_tensor):
    """详细对比两个张量的差异"""
    cpp_flat = cpp_tensor.flatten()
    python_flat = python_tensor.flatten()
    
    print("=== 详细张量对比 ===")
    print(f"CPP张量形状: {cpp_tensor.shape}")
    print(f"Python张量形状: {python_tensor.shape}")
    print(f"CPP范围: [{cpp_flat.min():.6f}, {cpp_flat.max():.6f}]")
    print(f"Python范围: [{python_flat.min():.6f}, {python_flat.max():.6f}]")
    
    # 检查前10个像素的差异
    print("前10个像素对比:")
    for i in range(min(10, len(cpp_flat))):
        diff = abs(cpp_flat[i] - python_flat[i])
        print(f"像素{i}: C++={cpp_flat[i]:.6f}, Python={python_flat[i]:.6f}, 差异={diff:.6f}")


def main():
    """主函数 - 对比三个模型在Python和C++中的结果"""
    # 图像路径
    # image_path = "./new_labeled_data_2331/train/images/output_s001_iso189_480360.bmp"
    image_path = "./new_labeled_data_2331/train/images/output_s001_iso189_480360.bmp"
    if not Path(image_path).exists():
        print(f"错误：图像文件不存在: {image_path}")
        return
    
    # 模型路径
    model_configs = [
        {
            "path": "./output/models/enet_model_opset11.onnx",
            "type": "multi_class", 
            "name": "多标签库"
        },
        {
            "path": "./output/models/enet_green_ratio_full_static_opset11_new_data.onnx", 
            "type": "full",
            "name": "新训练full库"
        },
        {
            "path": "./output/models/enet_green_ratio_encoder_static_opset11_new_data.onnx",
            "type": "encoder", 
            "name": "新训练encoder库"
        }
    ]
    
    # 预处理图像
    print("预处理图像...")
    input_tensor, original_size = preprocess_image_for_onnx(image_path)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"原始图像尺寸: {original_size}")

    python_stats = save_tensor_debug(input_tensor, "./output/onnx/debug", "python")
    input_match = compare_input_tensors("./output/onnx/cpp_actual_input_tensor.bin", input_tensor)


    # 存储结果
    python_results = {}
    
    # 测试每个模型
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"测试 {config['name']} ({config['type']})")
        print(f"{'='*60}")
        
        if not Path(config['path']).exists():
            print(f"警告：模型文件不存在: {config['path']}")
            continue
            
        results = test_onnx_model(config['path'], input_tensor, config['type'])

        if results is not None:
            python_results[config['type']] = results
            
            # 保存分割结果可视化
            if 'segmentation' in results:
                viz_filename = f"./output/onnx/python_{config['type']}_vegetation_mask.bmp"
                save_segmentation_visualization(
                    results['segmentation'], 
                    original_size, 
                    viz_filename, 
                    config['type']
                )
    
    # 打印Python结果
    print(f"\n{'='*60}")
    print("Python ONNX推理结果")
    print(f"{'='*60}")
    
    multi_class_result = python_results.get('multi_class', {})
    full_result = python_results.get('full', {})
    encoder_result = python_results.get('encoder', {})
    
    # 多标签库结果
    if 'vegetation_ratio' in multi_class_result:
        multi_ratio = multi_class_result['vegetation_ratio'] * 100
        print(f"多标签库植被比例: {multi_ratio:.5f}%")
    
    # 新训练full库结果
    if 'vegetation_ratio_segmentation' in full_result:
        full_seg_ratio = full_result['vegetation_ratio_segmentation'] * 100
        full_direct_ratio = full_result.get('vegetation_ratio_direct', 0) * 100
        print(f"新训练full库植被比例（分割统计）: {full_seg_ratio:.5f}%")
        print(f"新训练full库植被比例（直接输出）: {full_direct_ratio:.5f}%")
    
    # 新训练encoder库结果
    if 'vegetation_ratio' in encoder_result:
        encoder_ratio = encoder_result['vegetation_ratio'] * 100
        print(f"新训练encoder库植被比例: {encoder_ratio:.5f}%")
    
    # 与C++结果对比
    print(f"\n{'='*60}")
    print("Python vs C++ 结果对比")
    print(f"{'='*60}")
    
    # C++结果（从您提供的数据）
    cpp_results = {
        'multi_class': 6.59336,      # 多标签库植被比例
        'full_segmentation': 4.23438,  # full库分割统计
        'full_direct': 4.55093,       # full库直接输出
        'encoder': 4.55093           # encoder库直接输出
    }
    
    # 对比分析
    if 'multi_class' in python_results and 'full' in python_results:
        python_multi = multi_class_result['vegetation_ratio'] * 100
        python_full_seg = full_result['vegetation_ratio_segmentation'] * 100
        python_full_direct = full_result.get('vegetation_ratio_direct', 0) * 100
        
        print("1. 植被覆盖对比（多标签库 vs 新训练full库）:")
        print(f"   - Python多标签库植被比例: {python_multi:.5f}%")
        print(f"   - Python新训练full库植被比例（分割统计）: {python_full_seg:.5f}%")
        print(f"   - Python新训练full库植被比例（直接输出）: {python_full_direct:.5f}%")
        print(f"   - C++多标签库植被比例: {cpp_results['multi_class']:.5f}%")
        print(f"   - C++新训练full库植被比例（分割统计）: {cpp_results['full_segmentation']:.5f}%")
        print(f"   - C++新训练full库植被比例（直接输出）: {cpp_results['full_direct']:.5f}%")
        
        # 计算差异
        diff_seg = abs(python_multi - python_full_seg)
        diff_direct = abs(python_multi - python_full_direct)
        internal_diff = abs(python_full_seg - python_full_direct)
        
        print(f"   - Python差异（分割统计）: {diff_seg:.5f}%")
        print(f"   - Python差异（直接输出）: {diff_direct:.5f}%")
        print(f"   - Python Full库内部一致性差异: {internal_diff:.5f}%")
        
        # 与C++的差异
        cpp_diff_seg = abs(cpp_results['multi_class'] - cpp_results['full_segmentation'])
        cpp_diff_direct = abs(cpp_results['multi_class'] - cpp_results['full_direct'])
        cpp_internal_diff = abs(cpp_results['full_segmentation'] - cpp_results['full_direct'])
        
        print(f"   - C++差异（分割统计）: {cpp_diff_seg:.5f}%")
        print(f"   - C++差异（直接输出）: {cpp_diff_direct:.5f}%")
        print(f"   - C++ Full库内部一致性差异: {cpp_internal_diff:.5f}%")
    
    if 'multi_class' in python_results and 'encoder' in python_results:
        python_multi = multi_class_result['vegetation_ratio'] * 100
        python_encoder = encoder_result['vegetation_ratio'] * 100
        
        print("2. 植被比例对比（多标签库 vs 新训练encoder库）:")
        print(f"   - Python多标签库植被比例: {python_multi:.5f}%")
        print(f"   - Python新训练encoder库植被比例: {python_encoder:.5f}%")
        print(f"   - Python差异: {abs(python_multi - python_encoder):.5f}%")
        print(f"   - C++多标签库植被比例: {cpp_results['multi_class']:.5f}%")
        print(f"   - C++新训练encoder库植被比例: {cpp_results['encoder']:.5f}%")
        print(f"   - C++差异: {abs(cpp_results['multi_class'] - cpp_results['encoder']):.5f}%")
    
    if 'full' in python_results and 'encoder' in python_results:
        python_full_direct = full_result.get('vegetation_ratio_direct', 0) * 100
        python_encoder = encoder_result['vegetation_ratio'] * 100
        
        print("3. 直接输出对比（新训练full库 vs 新训练encoder库）:")
        print(f"   - Python新训练full库直接输出比例: {python_full_direct:.5f}%")
        print(f"   - Python新训练encoder库直接输出比例: {python_encoder:.5f}%")
        print(f"   - Python差异: {abs(python_full_direct - python_encoder):.5f}%")
        print(f"   - C++新训练full库直接输出比例: {cpp_results['full_direct']:.5f}%")
        print(f"   - C++新训练encoder库直接输出比例: {cpp_results['encoder']:.5f}%")
        print(f"   - C++差异: {abs(cpp_results['full_direct'] - cpp_results['encoder']):.5f}%")
    
    # 保存详细对比报告
    with open("./output/onnx/python_cpp_comparison_report.txt", "w") as f:
        f.write("Python vs C++ ONNX模型对比报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Python结果:\n")
        for model_type, results in python_results.items():
            f.write(f"{model_type}: ")
            for key, value in results.items():
                if 'ratio' in key:
                    f.write(f"{key}: {value*100:.5f}% ")
            f.write("\n")
        
        f.write("\nC++结果:\n")
        f.write(f"多标签库: {cpp_results['multi_class']:.5f}%\n")
        f.write(f"Full库分割统计: {cpp_results['full_segmentation']:.5f}%\n")
        f.write(f"Full库直接输出: {cpp_results['full_direct']:.5f}%\n")
        f.write(f"Encoder库: {cpp_results['encoder']:.5f}%\n")
        
        f.write("\n差异分析:\n")
        if 'multi_class' in python_results and 'full' in python_results:
            python_multi = multi_class_result['vegetation_ratio'] * 100
            python_full_seg = full_result['vegetation_ratio_segmentation'] * 100
            diff = abs(python_multi - cpp_results['multi_class'])
            f.write(f"多标签库差异: {diff:.5f}%\n")
            diff = abs(python_full_seg - cpp_results['full_segmentation'])
            f.write(f"Full库分割统计差异: {diff:.5f}%\n")

if __name__ == "__main__":
    # 创建输出目录
    Path("./output/onnx").mkdir(exist_ok=True)
    main()
