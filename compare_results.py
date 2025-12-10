# compare_results.py
import numpy as np
import cv2

def compare_results():
    # 加载Python结果
    py_preprocessed = np.load("./v220-test-predict/preprocessed_s001_iso189_10_17.npy")
    py_model_output = np.load("./v220-test-predict/model_output_s001_iso189_10_17.npy")
    py_argmax = np.load("./v220-test-predict/argmax_output_s001_iso189_10_17.npy")
    
    # 加载C++结果（需要先运行C++代码生成）
    try:
        # C++预处理数据（二进制格式）
        cpp_preprocessed = np.fromfile("./v220-test-predict/cpp_preprocessed.bin", dtype=np.float32)
        cpp_preprocessed = cpp_preprocessed.reshape(py_preprocessed.shape)
        
        # C++模型输出
        cpp_model_output = np.fromfile("./v220-test-predict/cpp_model_output.bin", dtype=np.float32)
        cpp_model_output = cpp_model_output.reshape(py_model_output.shape)
        
        print("=== 预处理数据对比 ===")
        print(f"Python形状: {py_preprocessed.shape}, C++形状: {cpp_preprocessed.shape}")
        print(f"Python范围: {py_preprocessed.min():.4f} ~ {py_preprocessed.max():.4f}")
        print(f"C++范围: {cpp_preprocessed.min():.4f} ~ {cpp_preprocessed.max():.4f}")
        print(f"数据差异: {np.abs(py_preprocessed - cpp_preprocessed).max():.6f}")
        
        print("\n=== 模型输出对比 ===")
        print(f"Python输出范围: {py_model_output.min():.4f} ~ {py_model_output.max():.4f}")
        print(f"C++输出范围: {cpp_model_output.min():.4f} ~ {cpp_model_output.max():.4f}")
        
        # 比较argmax结果
        py_final = np.argmax(py_model_output, axis=1)[0]  # 与torch.argmax对应
        cpp_final = np.argmax(cpp_model_output, axis=1)[0]
        
        print(f"\n=== 最终预测对比 ===")
        print(f"Python预测唯一值: {np.unique(py_final)}")
        print(f"C++预测唯一值: {np.unique(cpp_final)}")
        print(f"预测一致率: {np.mean(py_final == cpp_final):.4f}")
        
    except FileNotFoundError as e:
        print(f"请先运行C++代码生成调试文件: {e}")


# 详细分析脚本
import numpy as np
import cv2

def detailed_analysis():
    # 加载数据
    py_preprocessed = np.load("./v220-test-predict/preprocessed_s001_iso189_10_17.npy")
    cpp_preprocessed = np.fromfile("./v220-test-predict/cpp_preprocessed.bin", dtype=np.float32)
    cpp_preprocessed = cpp_preprocessed.reshape(py_preprocessed.shape)
    
    # 计算每个通道的差异
    diff = np.abs(py_preprocessed - cpp_preprocessed)
    print("=== 详细差异分析 ===")
    
    for channel in range(3):
        channel_diff = diff[channel]
        max_diff_idx = np.unravel_index(np.argmax(channel_diff), channel_diff.shape)
        max_diff_val = channel_diff[max_diff_idx]
        
        print(f"通道 {channel}:")
        print(f"  最大差异: {max_diff_val:.6f} 位置: {max_diff_idx}")
        print(f"  Python值: {py_preprocessed[channel][max_diff_idx]:.6f}")
        print(f"  C++值: {cpp_preprocessed[channel][max_diff_idx]:.6f}")
    
    # 检查前几个像素点的值
    print("\n=== 前5个像素点对比 ===")
    for i in range(5):
        print(f"像素 {i}: Python={py_preprocessed[:, i//480, i%480]} vs C++={cpp_preprocessed[:, i//480, i%480]}")
    
    # 检查是否颜色通道顺序问题
    print("\n=== 颜色通道顺序检查 ===")
    # 尝试交换通道顺序
    cpp_swapped = cpp_preprocessed[[2, 1, 0], :, :]  # BGR -> RGB
    diff_swapped = np.abs(py_preprocessed - cpp_swapped)
    print(f"交换通道后最大差异: {np.max(diff_swapped):.6f}")




if __name__ == "__main__":
    compare_results()
    # detailed_analysis()