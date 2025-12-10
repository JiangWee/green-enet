# cross_validation_direct.py
import numpy as np
import cv2

def validate_with_cpp_mask():
    # 直接加载C++生成的mask PNG文件
    cpp_mask = cv2.imread("s001_sc2331_segmentation_mask_nv12_bt709.png", cv2.IMREAD_GRAYSCALE)
    
    if cpp_mask is None:
        print("错误: 无法加载C++生成的mask文件")
        return
    
    print(f"C++ mask形状: {cpp_mask.shape}")
    print(f"C++ mask唯一值: {np.unique(cpp_mask)}")
    print(f"C++ mask数据类型: {cpp_mask.dtype}")
    
    # 使用Python的可视化逻辑
    h, w = cpp_mask.shape
    python_visualized = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Python的颜色映射（与您的代码一致）
    python_visualized[cpp_mask == 0] = [255, 0, 0]     # 天空: 红色 (BGR中的蓝色)
    python_visualized[cpp_mask == 5] = [0, 255, 0]     # 植被: 绿色
    python_visualized[cpp_mask == 9] = [0, 0, 255]    # 行人: 红色 (BGR中的蓝色)
    
    # 背景
    bg = (cpp_mask != 0) & (cpp_mask != 5) & (cpp_mask != 9)
    python_visualized[bg] = [0, 0, 0]
    
    # 保存Python可视化结果
    cv2.imwrite("cpp_mask_python_visualized.jpg", python_visualized)
    
    # 加载C++的可视化结果进行对比
    cpp_visualized = cv2.imread("s001_sc2331_color_segmentation_nv12_bt709.jpg")
    
    if cpp_visualized is not None:
        print(f"C++可视化形状: {cpp_visualized.shape}")
        
        # 比较两个结果
        diff = cv2.absdiff(python_visualized, cpp_visualized)
        diff_pixels = np.sum(diff > 0)
        total_pixels = h * w * 3
        
        print(f"差异像素数: {diff_pixels}/{total_pixels}")
        print(f"差异比例: {diff_pixels/total_pixels*100:.2f}%")
        
        # 保存差异图像
        cv2.imwrite("difference_mask.jpg", diff)
        
        # 检查植被区域
        vegetation_mask = (cpp_mask == 5)
        if np.any(vegetation_mask):
            veg_py = python_visualized[vegetation_mask]
            veg_cpp = cpp_visualized[vegetation_mask]
            
            print(f"Python植被平均颜色(BGR): {np.mean(veg_py, axis=0)}")
            print(f"C++植被平均颜色(BGR): {np.mean(veg_cpp, axis=0)}")
            
            # 检查颜色差异
            color_diff = np.abs(veg_py.astype(float) - veg_cpp.astype(float))
            print(f"植被区域平均颜色差异: {np.mean(color_diff, axis=0)}")
    
    print("交叉验证完成!")

if __name__ == "__main__":
    validate_with_cpp_mask()