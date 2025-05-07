#!/usr/bin/env python
# 测试环境设置和依赖安装

try:
    # 导入核心依赖
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from PIL import Image
    import tensorboard
    import sklearn
    from sklearn.metrics import accuracy_score
    
    # 打印各依赖包版本
    print("Python 环境和依赖测试")
    print("-" * 40)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"TorchVision 版本: {torchvision.__version__}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"Pandas 版本: {pd.__version__}")
    print(f"Matplotlib 版本: {plt.matplotlib.__version__}")
    print(f"TensorBoard 版本: {tensorboard.__version__}")
    print(f"Scikit-Learn 版本: {sklearn.__version__}")
    print("-" * 40)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用，版本: {torch.version.cuda}")
        print(f"GPU设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA不可用，将使用CPU")
    
    # 创建小型测试张量
    print("\n创建并操作测试张量...")
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    z = torch.matmul(x, y)
    print(f"矩阵乘法结果形状: {z.shape}")
    
    print("\n环境测试完成，所有依赖已成功安装！")
    success = True
    
except Exception as e:
    print(f"环境测试失败，错误: {str(e)}")
    success = False

# 返回测试结果
import sys
sys.exit(0 if success else 1) 