import os
import onnx

def get_onnx_model_info(model_path):
    # 获取文件大小
    file_size = os.path.getsize(model_path)
    file_size_mb = file_size / (1024 * 1024)
    
    # 加载模型获取结构信息
    model = onnx.load(model_path)
    
    # 计算参数数量
    total_params = 0
    for initializer in model.graph.initializer:
        param_count = 1
        for dim in initializer.dims:
            param_count *= dim
        total_params += param_count
    
    # 估算内存占用
    # 假设大多数参数是float32（4字节）
    memory_estimate_mb = (total_params * 4) / (1024 * 1024)
    
    print(f"=== ENet ONNX模型信息 ===")
    print(f"文件路径: {model_path}")
    print(f"文件大小: {file_size_mb:.2f} MB")
    print(f"参数量: {total_params:,}")
    print(f"估算内存占用: {memory_estimate_mb:.2f} MB")
    print(f"输入节点数: {len(model.graph.input)}")
    print(f"输出节点数: {len(model.graph.output)}")
    print(f"层数: {len(model.graph.node)}")
    
    # 显示输入输出形状
    for input in model.graph.input:
        print(f"输入: {input.name}")
        for dim in input.type.tensor_type.shape.dim:
            print(f"  维度: {dim.dim_value}")
    
    for output in model.graph.output:
        print(f"输出: {output.name}")
        for dim in output.type.tensor_type.shape.dim:
            print(f"  维度: {dim.dim_value}")

# 使用示例
get_onnx_model_info("enet_model_opset11.onnx")