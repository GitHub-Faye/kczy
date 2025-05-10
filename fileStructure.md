# KCZY 项目文件结构

## 项目概述

本项目是一个视觉转换器(Vision Transformer, ViT)模型分类器训练系统，专为计算机视觉研究人员和深度学习工程师设计。系统用于训练ViT模型并对其性能进行可视化分析，提供直观的性能指标跟踪和模型结构可视化功能。

## 目录组织逻辑

项目采用模块化组织结构，遵循标准Python机器学习项目设计原则，主要分为以下几个部分：

```
kczy/
├── src/                # 源代码目录
│   ├── data/           # 数据处理相关模块
│   ├── models/         # 模型定义和训练逻辑
│   ├── utils/          # 通用工具函数
│   └── visualization/  # 可视化相关功能
├── data/               # 数据文件存储
│   ├── examples/       # 示例数据
│   └── images/         # 图像文件
├── models/             # 训练好的模型保存目录
├── notebooks/          # Jupyter笔记本，用于实验和演示
├── scripts/            # 实用脚本，如训练脚本和演示脚本
├── temp_metrics/       # 临时指标存储目录
│   └── plots/          # 指标可视化图表
├── tests/              # 测试代码
├── tasks/              # 任务管理文件
└── requirements.txt    # 项目依赖
```

## 文件及目录详细说明

### 源代码目录 (`src/`)

#### 数据处理模块 (`src/data/`)
- `__init__.py` - 模块初始化文件，暴露关键数据处理接口
- `config.py` - 数据处理相关配置参数定义
- `custom_dataset.py` - 自定义数据集类实现
- `data_loader.py` - 数据加载器实现
- `dataset.py` - 基本数据集类定义
- `preprocessing.py` - 数据预处理函数
- `augmentation.py` - 数据增强方法实现

#### 模型模块 (`src/models/`)
- `__init__.py` - 模型模块初始化文件，导出模型类和模型工具函数
- `vit.py` - Vision Transformer模型定义，包含模型保存/加载功能
- `train.py` - 模型训练循环实现，支持检查点保存和恢复
- `optimizer_manager.py` - 优化器管理类
- `model_utils.py` - 模型保存、加载和转换工具函数，支持多种格式和用例

#### 工具函数 (`src/utils/`)
- `__init__.py` - 工具模块初始化文件
- `config.py` - 全局配置参数定义
- `metrics_logger.py` - 性能指标记录和分析工具

#### 可视化模块 (`src/visualization/`)
- 用于实现模型性能和结构的可视化功能（待实现）

### 数据目录 (`data/`)
- `examples/` - 存放示例数据
- `images/` - 存放训练和测试图像
- `annotations.csv` - 数据标注文件

### 模型目录 (`models/`)
- 用于存储训练好的模型权重和配置
- `*.pt`/`*.pth` - PyTorch格式的模型权重和配置
- `*.onnx` - ONNX格式的导出模型
- `*_config.json` - 模型配置文件

### 笔记本目录 (`notebooks/`)
- 用于存放Jupyter笔记本，进行实验和演示

### 临时指标目录 (`temp_metrics/`)
- `plots/` - 存放指标可视化图表
- `simulation_train_metrics.csv` - 模拟训练指标数据
- `simulation_eval_metrics.csv` - 模拟评估指标数据

### 脚本目录 (`scripts/`)
- `train_with_config.py` - 使用配置文件进行模型训练
- `test_training_loop.py` - 测试训练循环
- `demo_custom_dataset.py` - 自定义数据集演示
- `demo_augmentation.py` - 数据增强演示
- `task-complexity-report.json` - 任务复杂度报告
- `example_prd.txt` - 示例产品需求文档

### 测试目录 (`tests/`)
- `__init__.py` - 测试模块初始化文件
- `test_vit.py` - ViT模型测试
- `test_data_sample.py` - 数据采样测试
- `test_custom_dataset.py` - 自定义数据集测试
- `test_augmentation.py` - 数据增强测试
- `test_preprocessing.py` - 数据预处理测试
- `test_data_loader.py` - 数据加载器测试
- `test_metrics_logger.py` - 性能指标记录工具测试
- `test_model_saving.py` - 模型保存和加载功能测试
- `sample_image.png` - 测试用图像
- `batch_images.png` - 测试用批量图像
- `outputs/` - 测试输出目录

### 任务目录 (`tasks/`)
- `tasks.json` - 任务定义文件
- `task_00X.txt` - 个别任务详细描述文件

### 其他文件
- `requirements.txt` - 项目依赖列表
- `.taskmasterconfig` - 任务管理器配置文件
- `prd.txt` - 产品需求文档
- `test_setup.py` - 测试环境设置
- `.gitignore` - Git忽略文件配置
- `.env.example` - 环境变量示例文件

## 关键文件之间的关系

### 数据流关系
1. 数据处理流程：
   - 原始数据 → `src/data/preprocessing.py` → `src/data/augmentation.py` → `src/data/custom_dataset.py` → `src/data/data_loader.py` → 模型训练
   - 配置由 `src/data/config.py` 控制

2. 模型训练流程：
   - 数据加载器 → `src/models/train.py` (使用 `src/models/vit.py` 定义的模型) → 训练结果 → 模型保存到 `models/` 目录
   - 优化过程由 `src/models/optimizer_manager.py` 管理
   - 训练和评估指标由 `src/utils/metrics_logger.py` 记录和分析
   - 配置由 `src/utils/config.py` 控制
   - 模型保存和加载由 `src/models/model_utils.py` 提供支持

3. 模型保存和加载流程:
   - 训练完成或检查点 → `src/models/model_utils.py` 保存功能 → 模型文件存储在 `models/` 目录
   - 模型文件 → `src/models/model_utils.py` 加载功能 → 恢复模型用于推理或继续训练
   - 支持多种格式：原生PyTorch模型（.pt/.pth）和ONNX格式（.onnx）

4. 性能指标流程：
   - 训练循环 → `src/utils/metrics_logger.py` → 指标数据保存到CSV/JSON文件
   - 指标数据 → `src/utils/metrics_logger.py` → 可视化结果保存到 `temp_metrics/plots/` 目录

5. 测试流程：
   - `tests/` 目录下的各测试文件分别测试对应模块的功能
   - 测试结果保存在 `tests/outputs/` 目录

### 配置依赖关系
- `src/utils/config.py` 提供全局配置，包括训练参数和指标记录选项
- `src/data/config.py` 提供数据处理特定配置
- 脚本通过导入这些配置模块访问参数

## 命名约定

1. **文件命名**
   - 使用小写字母和下划线命名法(snake_case)，如 `data_loader.py`
   - 测试文件以 `test_` 前缀命名，如 `test_vit.py`
   - 演示脚本以 `demo_` 前缀命名，如 `demo_augmentation.py`

2. **模块命名**
   - 模块名使用小写字母，如 `data`, `models`, `utils`
   - 每个模块目录包含 `__init__.py` 文件

3. **类命名**
   - 使用大驼峰命名法(PascalCase)，如 `CustomDataset`
   - 模型类以功能命名，如 `ViT`
   - 工具类以功能命名，如 `MetricsLogger`

4. **函数命名**
   - 使用小写字母和下划线命名法，如 `preprocess_image`
   - 私有函数以单下划线前缀，如 `_extract_features`

5. **常量命名**
   - 使用大写字母和下划线，如 `DEFAULT_BATCH_SIZE`

6. **变量命名**
   - 使用小写字母和下划线命名法，遵循描述性原则
   - 明确变量含义，如 `batch_size`, `learning_rate`

## 开发工作流

项目使用Task Master进行任务管理：
- `tasks/tasks.json` 定义了项目任务
- 个别任务在 `tasks/task_00X.txt` 中有详细描述
- 开发按照任务优先级和依赖关系进行
- 使用测试驱动开发，确保代码质量 

