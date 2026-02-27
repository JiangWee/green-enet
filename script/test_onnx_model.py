import onnxruntime
import numpy as np
import torch
from pathlib import Path
import cv2
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image_for_onnx(image_path, target_size=(360, 480), debug_save_path=None):
    """统一使用RGB顺序与训练保持一致"""
    # 使用OpenCV读取
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 关键修改：BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
    
    if debug_save_path:
        # 保存调试图像
        debug_image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(debug_save_path, debug_image_bgr)
        print(f"调试图像已保存: {debug_save_path}")

    # 转换为float32并归一化
    image_float = image_resized.astype(np.float32) / 255.0
    
    # 转换为CHW格式
    image_chw = image_float.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = np.expand_dims(image_chw, axis=0)  # [1, 3, H, W]
    
    return input_tensor, original_size

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
        
        # 添加详细的调试信息
        print(f"输入张量范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        print(f"输入张量均值: {input_tensor.mean():.3f}")
        print(f"输出数量: {len(ort_outputs)}")
        
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
            segmentation = ort_outputs[0]
            direct_ratio = ort_outputs[1] if len(ort_outputs) > 1 else None
            
            print(f"分割输出形状: {segmentation.shape}")
            if direct_ratio is not None:
                print(f"直接输出形状: {direct_ratio.shape}")
                print(f"直接输出值: {direct_ratio}")
            
            # 统计分割图的植被比例（新训练模型中植被是类别1）
            pred_mask = np.argmax(segmentation[0], axis=0)  # [H, W]
            vegetation_pixels = np.sum(pred_mask == 1)
            total_pixels = pred_mask.size
            segmentation_ratio = vegetation_pixels / total_pixels
            
            # 直接输出的绿植比例
            direct_ratio_value = 0
            if direct_ratio is not None:
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
            
        print(f"推理完成，输出数量: {len(ort_outputs)}")
        return results
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_green_overlay(original_image, segmentation, original_size, target_size, model_type, alpha=0.5):
    """创建绿色透明掩膜覆盖在原图上"""
    # 将分割图上采样到原始尺寸
    seg_resized = cv2.resize(
        segmentation.astype(np.uint8), 
        original_size,  # (宽, 高)
        interpolation=cv2.INTER_NEAREST
    )
    
    # 根据模型类型确定植被标签
    if model_type == "multi_class":
        vegetation_label = 5
    else:  # full模型
        vegetation_label = 1
    
    # 创建绿色掩膜（植被区域为绿色，其他区域透明）
    overlay = original_image.copy()
    
    # 创建植被区域的掩码
    vegetation_mask = seg_resized == vegetation_label
    
    # 将植被区域设置为绿色 (BGR格式)
    overlay[vegetation_mask] = [0, 255, 0]  # 绿色
    
    # 将原图与绿色掩膜混合
    result = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
    
    return result

def save_results_with_overlay(original_image_path, segmentation, original_size, target_size, model_type, output_path, alpha=0.5):
    """保存带有绿色透明掩膜的结果图像"""
    # 重新读取原始图像（确保是BGR格式用于显示）
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"无法读取原始图像: {original_image_path}")
        return
    
    # 创建绿色透明掩膜
    overlay_result = create_green_overlay(original_image, segmentation, original_size, target_size, model_type, alpha)
    
    # 保存结果
    cv2.imwrite(output_path, overlay_result)
    print(f"绿色掩膜覆盖图已保存: {output_path}")

def main():
    """主函数 - 配置参数和运行三个模型"""
    # ========== 配置参数 ==========
    # 图像路径
    image_path = "./input/test/indoor.png"
    
    # 输出目录
    output_dir = "./output/onnx"
    
    # 目标尺寸 (高度, 宽度)
    target_size = (360, 480)
    
    # 绿色掩膜透明度 (0-1之间，越大越不透明)
    overlay_alpha = 0.5
    
    # 是否保存调试图像
    save_debug_images = True
    
    # 模型配置
    model_configs = [
        {
            "path": "./output/models/enet_model_opset11.onnx",
            "type": "multi_class", 
            "name": "多标签库",
            "enabled": True
        },
        {
            "path": "./output/models/enet_green_ratio_full_static_opset11_officeGreenAndOriInput.onnx", 
            "type": "full",
            "name": "新训练full库",
            "enabled": True
        },
        {
            "path": "./output/models/enet_green_ratio_encoder_static_opset11_officeGreenAndOriInput.onnx",
            "type": "encoder", 
            "name": "新训练encoder库",
            "enabled": True
        }
    ]
    # ========== 配置结束 ==========
    
    # 检查图像文件是否存在
    if not Path(image_path).exists():
        print(f"错误：图像文件不存在: {image_path}")
        return
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 预处理图像
    print("预处理图像...")
    debug_save_path = f"{output_dir}/input_tensor.bmp" if save_debug_images else None
    input_tensor, original_size = preprocess_image_for_onnx(
        image_path, 
        target_size=target_size,
        debug_save_path=debug_save_path
    )
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"原始图像尺寸: {original_size}")

    # 存储结果
    python_results = {}
    
    # 测试每个模型
    for config in model_configs:
        if not config["enabled"]:
            continue
            
        print(f"\n{'='*60}")
        print(f"测试 {config['name']} ({config['type']})")
        print(f"{'='*60}")
        
        if not Path(config['path']).exists():
            print(f"警告：模型文件不存在: {config['path']}")
            continue
            
        results = test_onnx_model(config['path'], input_tensor, config['type'])

        if results is not None:
            python_results[config['type']] = results
            
            # 保存带有绿色掩膜的结果
            if 'segmentation' in results:
                overlay_filename = f"{output_dir}/{config['type']}_green_overlay.bmp"
                save_results_with_overlay(
                    image_path,
                    results['segmentation'], 
                    original_size, 
                    target_size,
                    config['type'],
                    overlay_filename,
                    alpha=overlay_alpha
                )
    
    # 打印结果
    print(f"\n{'='*60}")
    print("模型推理结果")
    print(f"{'='*60}")
    
    for config in model_configs:
        if not config["enabled"]:
            continue
            
        model_type = config["type"]
        results = python_results.get(model_type, {})
        
        if model_type == "multi_class" and 'vegetation_ratio' in results:
            ratio = results['vegetation_ratio'] * 100
            print(f"{config['name']}植被比例: {ratio:.5f}%")
        
        elif model_type == "full":
            if 'vegetation_ratio_segmentation' in results:
                seg_ratio = results['vegetation_ratio_segmentation'] * 100
                direct_ratio = results.get('vegetation_ratio_direct', 0) * 100
                print(f"{config['name']}植被比例（分割统计）: {seg_ratio:.5f}%")
                print(f"{config['name']}植被比例（直接输出）: {direct_ratio:.5f}%")
        
        elif model_type == "encoder" and 'vegetation_ratio' in results:
            ratio = results['vegetation_ratio'] * 100
            print(f"{config['name']}植被比例: {ratio:.5f}%")
    
    print(f"\n所有结果图像已保存到: {output_dir}")

if __name__ == "__main__":
    main()
