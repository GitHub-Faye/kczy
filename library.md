# 库IDs

## 模型与可视化相关库

1. `/huggingface/transformers` - 包含Transformer模型（包括Vision Transformer）的实现和使用示例
2. `/pytorch/pytorch` - PyTorch核心库，包含张量操作和基本的可视化功能

## 数据处理相关库

3. `/pytorch/tutorials` - PyTorch教程库，包含数据集加载、预处理和拆分的实用示例
4. `/pytorch/vision` - PyTorch视觉库，提供图像处理、数据集和预训练模型
5. `/numpy/numpy` - NumPy库，用于数值计算和数组操作
6. `/pandas-dev/pandas` - Pandas库，提供数据分析和操作功能

## 注意事项

这些库可用于参考Vision Transformer(ViT)模型的实现和可视化方法，同时提供数据加载和预处理的参考实现。我们的项目已经有了自己的ViT实现和基本可视化功能，主要任务是测试和改进现有功能，包括数据处理和可视化工具。

数据加载器模块现已支持训练(70%)、验证(20%)和测试(10%)三部分数据集的划分，并提供了数据集验证机制和图像归一化处理功能。
