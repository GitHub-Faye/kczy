# ONNX导出功能使用指南

本文档介绍如何使用KCZY项目中的ONNX导出功能，将训练好的Vision Transformer模型导出为ONNX格式，以便在不同的推理环境中使用。

## 功能概述

ONNX（Open Neural Network Exchange）是一种用于表示深度学习模型的开放格式，可以在不同的框架和工具之间转换模型。通过将模型导出为ONNX格式，可以实现以下目标：

1. **跨框架兼容性**：导出的模型可以在ONNX支持的框架中使用，如ONNX Runtime、TensorRT等
2. **推理性能优化**：ONNX支持模型简化和优化，提高推理速度
3. **部署灵活性**：导出的ONNX模型可以部署在多种硬件和环境中

## 前提条件

使用ONNX导出功能需要安装以下依赖：

```bash
pip install onnx onnxruntime
```

对于高级功能，还需要安装：

```bash
# 用于模型简化
pip install onnx-simplifier

# 用于性能优化
pip install onnxruntime-extensions
```

## 基本用法

### 导出模型

最基本的模型导出方式：

```python
from src.models import VisionTransformer, export_to_onnx

# 创建或加载您的模型
model = VisionTransformer.create_base(num_classes=10)
# 或者加载已训练的模型
# model, _ = load_model("models/my_model.pt")

# 导出为ONNX格式
export_to_onnx(
    model=model,
    file_path="models/my_model.onnx"
)
```

### 高级导出选项

导出时可以指定多种参数控制导出过程：

```python
export_to_onnx(
    model=model,
    file_path="models/my_model.onnx",
    input_shape=(1, 3, 224, 224),  # 指定输入形状
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # 定义动态轴
    export_params=True,  # 是否导出权重
    opset_version=11,  # ONNX操作集版本
    simplify=True,  # 是否简化模型
    verify=True,  # 是否验证导出结果
    optimize=True,  # 是否优化模型
    target_providers=['CPUExecutionProvider']  # 目标运行环境
)
```

## 模型验证

导出后可以使用以下代码验证ONNX模型与PyTorch模型输出是否一致：

```python
import numpy as np
import torch
from src.models import load_onnx_model, onnx_inference

# 准备输入数据
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
torch_input = torch.tensor(input_data)

# PyTorch模型推理
model.eval()
with torch.no_grad():
    torch_output = model(torch_input).numpy()

# ONNX模型推理
session = load_onnx_model("models/my_model.onnx")
onnx_output = onnx_inference(session, input_data)

# 比较输出
is_close = np.allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-5)
print(f"输出是否匹配: {is_close}")
max_diff = np.max(np.abs(torch_output - onnx_output))
print(f"最大差异: {max_diff}")
```

## 模型信息查看

可以查看ONNX模型的基本信息：

```python
from src.models import get_onnx_model_info

info = get_onnx_model_info("models/my_model.onnx")
print(f"模型版本: {info['ir_version']}")
print(f"节点数量: {info['node_count']}")
print(f"操作类型: {info['operation_types']}")
```

## 模型优化

对于追求更高推理性能的场景，可以优化ONNX模型：

```python
from src.models.model_utils import optimize_onnx_model

# 优化模型
optimized_path = optimize_onnx_model(
    "models/my_model.onnx", 
    target_providers=['CPUExecutionProvider']
)
print(f"优化后的模型: {optimized_path}")
```

## 简化模型

简化模型以减少不必要的计算图节点：

```python
from src.models.model_utils import simplify_onnx_model

# 简化模型
simplified_path = simplify_onnx_model("models/my_model.onnx")
print(f"简化后的模型: {simplified_path}")
```

## 多环境支持

ONNX模型可以在多种环境中运行，通过指定providers参数：

```python
# CPU推理
session_cpu = load_onnx_model(
    "models/my_model.onnx",
    providers=['CPUExecutionProvider']
)

# GPU推理（需要CUDA）
session_gpu = load_onnx_model(
    "models/my_model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

## 完整演示脚本

项目提供了一个完整的演示脚本，展示ONNX导出和推理的完整流程：

```bash
python scripts/demo_onnx_export.py
```

该脚本会创建一个Vision Transformer模型，导出为ONNX格式，进行模型验证，并比较PyTorch和ONNX的推理性能。

## 注意事项

1. ONNX导出需要模型具有明确的输入尺寸，如果使用动态尺寸，请正确设置dynamic_axes参数
2. 不同的opset_version支持不同的操作，如遇导出错误，请尝试调整opset版本
3. 某些复杂操作可能在ONNX中不被支持，需要进行模型调整
4. 验证ONNX模型时，可能存在微小的数值差异，这通常是正常的
5. 优化后的模型可能在某些特定情况下与原始模型有细微差异

## 参考资源

- [ONNX官方文档](https://onnx.ai/)
- [ONNX Runtime文档](https://onnxruntime.ai/)
- [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier) 