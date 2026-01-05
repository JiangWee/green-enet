#!/usr/bin/env python3
import torch
import subprocess
import os

def check_gpu_info():
    print("=" * 50)
    print("GPU Information Check")
    print("=" * 50)
    
    # 方法1: 使用torch
    print("\n1. PyTorch GPU Info:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("No GPU available for PyTorch")
    
    # 方法2: 使用nvidia-smi
    print("\n2. NVIDIA SMI Info:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # 只显示前几行关键信息
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                print(line)
        else:
            print("nvidia-smi command failed")
    except FileNotFoundError:
        print("nvidia-smi not found")
    
    # 方法3: 环境变量
    print("\n3. Environment Variables:")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER']
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        print(f"{var}: {value}")

if __name__ == "__main__":
    check_gpu_info()