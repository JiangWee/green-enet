import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import os
import sys
import io


# 添加模型路径到系统路径，以便导入ENetGreenRatio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def safe_print(msg):
    """安全打印函数，替换emoji为文本"""
    # 替换emoji为文本描述
    emoji_replacements = {
        '🧪🧪': '[测试]',
        '✅': '[成功]',
        '⚠️': '[警告]',
        '❌❌': '[错误]',
        '📊📊': '[信息]',
        '🎉🎉': '[完成]',
        '📋📋': '[说明]'
    }
    
    for emoji, text in emoji_replacements.items():
        msg = msg.replace(emoji, text)
    
    # 修复：使用print而不是递归调用safe_print
    print(msg)  # 这行是关键修复！

from models.enet_green import ENetGreenRatio

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_green_ratio_to_onnx(model_path, output_path, input_size=(360, 480), 
                               encoder_only=False, opset_version=13):
    """
    将绿植比例估计模型转换为ONNX格式（静态模式）
    
    参数:
    - model_path: 训练好的模型路径
    - output_path: ONNX模型输出路径
    - input_size: 输入图像尺寸 (H, W)
    - encoder_only: 是否只使用编码器模式
    - opset_version: ONNX算子集版本
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    safe_print(f"使用设备: {device}")
    
    # 创建模型
    num_classes = 2  # 二分类：绿植和非绿植
    model = ENetGreenRatio(num_classes=num_classes, encoder_only=encoder_only)
    model.to(device)
    model.eval()
    
    safe_print(f"模型架构: {'编码器模式' if encoder_only else '完整模式'}")
    safe_print(f"模型参数量: {count_parameters(model):,}")
    safe_print(f"输入尺寸: {input_size} (静态模式)")
    
    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        safe_print(f"成功加载模型: {model_path}")
        
        if 'encoder_state_dict' in checkpoint:
            # 加载编码器权重
            encoder_state_dict = checkpoint['encoder_state_dict']
            model.load_state_dict(encoder_state_dict, strict=False)
            safe_print("加载编码器权重")
        elif 'state_dict' in checkpoint:
            # 加载完整模型权重
            model.load_state_dict(checkpoint['state_dict'])
            safe_print("加载完整模型权重")
        else:
            # 直接加载模型权重
            model.load_state_dict(checkpoint)
            safe_print("直接加载模型权重")
            
    except Exception as e:
        safe_print(f"加载模型失败: {e}")
        return False
    
    # 创建虚拟输入
    batch_size = 1
    channels = 3
    height, width = input_size
    
    # 检查模型权重
    safe_print("检查模型权重...")
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name and param.dim() > 1:
            safe_print(f"参数 {name}: 形状 {param.shape}, 均值 {param.mean().item():.6f}, 标准差 {param.std().item():.6f}")
            if torch.all(param == 0):
                safe_print(f"[警告] 参数 {name} 全为零!")
            break  # 只显示第一个参数作为示例
    
    # 创建测试输入
    dummy_input = torch.randn(batch_size, channels, height, width, device=device) * 0.5 + 0.5
    safe_print(f"测试输入范围: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    safe_print(f"测试输入形状: {dummy_input.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        if encoder_only:
            output = model(dummy_input)
            safe_print(f"输出类型: {type(output)}")
            safe_print(f"输出形状: {output.shape}")
            safe_print(f"输出值: {output.item():.6f}")
            
            # 检查输出是否合理
            if abs(output.item()) < 1e-6:
                safe_print("[警告] 编码器输出接近0，可能权重加载有问题")
            output_names = ['green_ratio']
        else:
            segmentation, green_ratio = model(dummy_input)
            safe_print(f"分割输出形状: {segmentation.shape}")
            safe_print(f"绿植比例值: {green_ratio.item():.6f}")
            
            # 检查分割输出
            seg_softmax = torch.softmax(segmentation, dim=1)
            safe_print(f"分割softmax范围: [{seg_softmax.min():.3f}, {seg_softmax.max():.3f}]")
            safe_print(f"类别0平均概率: {seg_softmax[0, 0].mean().item():.3f}")
            safe_print(f"类别1平均概率: {seg_softmax[0, 1].mean().item():.3f}")
            output_names = ['segmentation', 'green_ratio']
    
    # 导出ONNX模型（静态模式）
    try:
        # 静态模式：不设置dynamic_axes，使用固定尺寸
        safe_print("使用静态模式导出ONNX模型...")
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=output_names,
            # 不设置dynamic_axes参数，使用静态尺寸
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False  # 设置为False减少输出
        )
        
        safe_print(f"✅ ONNX模型导出成功: {output_path}")
        safe_print(f"✅ ONNX算子集版本: {opset_version}")
        safe_print(f"✅ 模式: 静态 (固定输入尺寸: {input_size})")
        
        # 验证导出的ONNX模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            safe_print("✅ ONNX模型验证通过")
            
            # 打印模型信息
            safe_print("[信息] ONNX模型信息:")
            
            # 处理输入信息
            input_info = []
            for i in onnx_model.graph.input:
                dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
                input_info.append(f"{i.name}: {dims}")
            safe_print(f"   输入: {input_info}")
            
            # 处理输出信息
            output_info = []
            for o in onnx_model.graph.output:
                dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
                output_info.append(f"{o.name}: {dims}")
            safe_print(f"   输出: {output_info}")
            
            # 验证输入输出尺寸是否固定
            for i in onnx_model.graph.input:
                dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
                if all(dim > 0 for dim in dims):  # 所有维度都有具体数值
                    safe_print("✅ 输入尺寸已固定")
                else:
                    safe_print("⚠️  输入尺寸可能包含动态维度")
            
        except ImportError:
            safe_print("⚠️  无法导入onnx库，跳过模型验证")
        except Exception as e:
            safe_print(f"⚠️  ONNX模型验证失败: {e}")
            
        return True
        
    except Exception as e:
        safe_print(f"❌❌ ONNX导出失败: {e}")
        return False

def convert_all_models(encoder_model_path, full_model_path, output_dir="./output", suffix="afterstage2"):
    """转换所有相关的模型（静态模式）"""
    
    # 检查模型文件是否存在
    models_to_convert = []
    
    if os.path.exists(encoder_model_path):
        models_to_convert.append(('encoder', encoder_model_path, True))
    else:
        safe_print(f"[警告] 编码器模型文件不存在: {encoder_model_path}")
    
    if os.path.exists(full_model_path):
        models_to_convert.append(('full', full_model_path, False))
    else:
        safe_print(f"[警告] 完整模型文件不存在: {full_model_path}")
    
    if not models_to_convert:
        safe_print("[错误] 没有找到可转换的模型文件")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试不同的ONNX算子集版本
    opset_versions = [11, 12, 13]
    
    for model_type, model_path, encoder_only in models_to_convert:
        safe_print(f"\n{'='*60}")
        safe_print(f"转换 {model_type} 模型 (静态模式)")
        safe_print(f"{'='*60}")
        
        success = False
        for opset in opset_versions:
            safe_print(f"\n尝试使用opset版本 {opset}...")
            
            # 使用参数化的输出目录，添加static标识
            output_name = os.path.join(output_dir, f"enet_green_ratio_{model_type}_static_opset{opset}_{suffix}.onnx")
            
            if convert_green_ratio_to_onnx(
                model_path=model_path,
                output_path=output_name,
                input_size=(360, 480),  # 固定尺寸
                encoder_only=encoder_only,
                opset_version=opset
            ):
                success = True
                break  # 成功则跳出循环
            else:
                safe_print(f"opset {opset} 转换失败，尝试下一个版本...")
        
        if not success:
            safe_print(f"[错误] {model_type} 模型所有opset版本转换均失败")

def test_model_inference(encoder_model_path, full_model_path):
    """测试模型推理功能"""
    safe_print("\n[测试] 测试模型推理...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    safe_print(f"使用设备: {device}")
    
    # 测试编码器模型
    if os.path.exists(encoder_model_path):
        safe_print("测试编码器模型推理...")
        try:
            model = ENetGreenRatio(num_classes=2, encoder_only=True)
            checkpoint = torch.load(encoder_model_path, map_location=device)
            
            safe_print(f"检查点键: {list(checkpoint.keys())}")
            
            if 'encoder_state_dict' in checkpoint:
                encoder_state_dict = checkpoint['encoder_state_dict']
                model.load_state_dict(encoder_state_dict, strict=False)
                safe_print("加载编码器权重")
            else:
                model.load_state_dict(checkpoint, strict=False)
                safe_print("直接加载模型权重")
                
            model.to(device)
            model.eval()
            
            # 创建测试输入
            dummy_input = torch.randn(1, 3, 360, 480, device=device) * 0.5 + 0.5
            safe_print(f"测试输入形状: {dummy_input.shape}")
            safe_print(f"测试输入范围: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
            
            with torch.no_grad():
                output = model(dummy_input)
                
                if isinstance(output, tuple):
                    green_ratio = output[-1]
                    safe_print(f"输出为元组，使用最后一个元素")
                else:
                    green_ratio = output
                
                safe_print(f"编码器模型输出绿植比例: {green_ratio.item():.6f}")
                
                if abs(green_ratio.item()) < 1e-6:
                    safe_print("[警告] 绿植比例接近0，可能存在问题")
                    
        except Exception as e:
            safe_print(f"[错误] 编码器模型推理失败: {e}")
    
    # 测试完整模型
    if os.path.exists(full_model_path):
        safe_print("测试完整模型推理...")
        try:
            model = ENetGreenRatio(num_classes=2, encoder_only=False)
            checkpoint = torch.load(full_model_path, map_location=device)
            
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(device)
            model.eval()
            
            dummy_input = torch.randn(1, 3, 360, 480, device=device) * 0.5 + 0.5
            safe_print(f"测试输入形状: {dummy_input.shape}")
            
            with torch.no_grad():
                output = model(dummy_input)
                
                if isinstance(output, tuple) and len(output) >= 2:
                    segmentation, green_ratio = output[0], output[1]
                    safe_print(f"完整模型输出分割形状: {segmentation.shape}")
                    safe_print(f"完整模型输出绿植比例: {green_ratio.item():.6f}")
                else:
                    safe_print(f"[警告] 完整模型输出结构不符合预期: {type(output)}")
                    
        except Exception as e:
            safe_print(f"[错误] 完整模型推理失败: {e}")


def create_random_model_if_needed():
    """如果模型文件不存在，创建随机权重模型"""
    encoder_path = './output/models/enet_green_ratio_random_encoder.pth'
    full_path = './output/models/enet_green_ratio_random_full.pth'
    
    if not os.path.exists(encoder_path):
        print("创建随机权重编码器模型...")
        model = ENetGreenRatio(num_classes=2, encoder_only=True)
        torch.save({
            'encoder_state_dict': model.state_dict(),
            'epoch': 0
        }, encoder_path)
        print(f"✅ 已创建: {encoder_path}")
    
    if not os.path.exists(full_path):
        print("创建随机权重完整模型...")
        model = ENetGreenRatio(num_classes=2, encoder_only=False)
        torch.save(model.state_dict(), full_path)
        print(f"✅ 已创建: {full_path}")


def main():
    # create_random_model_if_needed()

    """主函数 - 集中管理所有路径配置"""
    safe_print("绿植比例估计模型转换工具 (静态模式)")
    safe_print("=" * 50)
    
    # 在这里集中配置所有路径
    encoder_model_path = './output/models/enet_green_ratio_encoder_officeGreenAndOriInput.pth'
    full_model_path = './output/models/enet_green_ratio_full_officeGreenAndOriInput.pth'
    output_dir = './output/models'
    suffix = 'officeGreenAndOriInput'

    # 验证路径是否存在
    safe_print("[信息] 配置检查:")
    safe_print(f"  编码器模型: {encoder_model_path} - {'存在' if os.path.exists(encoder_model_path) else '不存在'}")
    safe_print(f"  完整模型: {full_model_path} - {'存在' if os.path.exists(full_model_path) else '不存在'}")
    safe_print(f"  输出目录: {output_dir}")
    safe_print(f"  模式: 静态 (固定尺寸 360x480)")
    
    # 首先测试模型推理
    test_model_inference(encoder_model_path, full_model_path)
    
    # 转换所有模型
    convert_all_models(encoder_model_path, full_model_path, output_dir, suffix=suffix)
    
    safe_print("\n[完成] 转换完成！")
    safe_print("\n[说明] 使用说明:")
    safe_print("1. 静态模式: 输入尺寸固定为480x360")
    safe_print("2. 编码器模型 (encoder_only=True): 只输出绿植比例")
    safe_print("3. 完整模型 (encoder_only=False): 输出分割结果和绿植比例")
    safe_print("4. 静态模型在某些推理引擎上可能有更好的性能优化")

if __name__ == "__main__":
    main()
