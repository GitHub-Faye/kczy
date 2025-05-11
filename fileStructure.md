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
│   └── onnx/           # ONNX格式模型目录
├── notebooks/          # Jupyter笔记本，用于实验和演示
├── scripts/            # 实用脚本，如训练脚本和演示脚本
├── docs/               # 文档目录
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
- `optimizer_manager.py` - 优化器管理类，包含优化器状态保存和恢复功能
- `model_utils.py` - 模型和优化器状态保存、加载和转换工具函数，支持多种格式（PyTorch、ONNX）和完整训练状态恢复，包含ONNX模型验证、优化和简化功能

#### 工具函数 (`src/utils/`)
- `__init__.py` - 工具模块初始化文件
- `config.py` - 全局配置参数定义
- `metrics_logger.py` - 性能指标记录和分析工具

#### 可视化模块 (`src/visualization/`)
- `__init__.py` - 可视化模块初始化文件，导出指标绘图功能
- `metrics_plots.py` - 指标绘图模块，用于绘制训练和评估指标的可视化图表
  - `save_plot()` - 通用图表保存函数，支持自动添加时间戳和保存元数据
  - `plot_loss()` - 绘制训练和验证损失曲线，支持保存到文件并可添加时间戳和元数据
  - `plot_accuracy()` - 绘制训练和验证准确率曲线，支持百分比显示、Y轴范围控制和文件保存（含时间戳和元数据）
  - `plot_training_history()` - 绘制多种训练指标历史曲线，支持批量生成并保存图表

### 数据目录 (`data/`)
- `examples/` - 存放示例数据
- `images/` - 存放训练和测试图像
- `annotations.csv` - 数据标注文件

### 模型目录 (`models/`)
- 用于存储训练好的模型权重和配置
- `*.pt`/`*.pth` - PyTorch格式的模型权重和配置
- `*.onnx` - ONNX格式的导出模型
- `*_config.json` - 模型配置文件
- `onnx/` - ONNX格式模型专用目录，包含导出的模型及配置文件

### 文档目录 (`docs/`)
- `onnx_export.md` - ONNX导出功能使用指南，包含详细示例和注意事项

### 笔记本目录 (`notebooks/`)
- 用于存放Jupyter笔记本，进行实验和演示

### 临时指标目录 (`temp_metrics/`)
- `plots/` - 存放指标可视化图表
  - `loss_curve.png` - 损失曲线图
  - `accuracy_curve.png` - 准确率曲线图
  - `learning_rate_curve.png` - 学习率曲线图
- `simulation_train_metrics.csv` - 模拟训练指标数据
- `simulation_eval_metrics.csv` - 模拟评估指标数据

### 脚本目录 (`scripts/`)
- `train_with_config.py` - 使用配置文件进行模型训练
- `test_training_loop.py` - 测试训练循环
- `demo_custom_dataset.py` - 自定义数据集演示
- `demo_augmentation.py` - 数据增强演示
- `demo_onnx_export.py` - ONNX导出功能演示脚本，展示模型导出、验证和推理性能比较
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
- `test_model_saving.py` - 模型保存和加载功能测试，包含ONNX导出和推理验证
- `test_optimizer_saving.py` - 优化器状态保存和恢复功能测试
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
   - 优化器状态保存 → `optimizer_manager.state_dict()` → 通过`save_checkpoint()`保存到检查点文件
   - 优化器状态恢复 → 从检查点文件读取 → `optimizer_manager.load_state_dict()` → 恢复训练状态

4. ONNX模型导出和推理流程:
   - PyTorch模型 → `export_to_onnx()` → ONNX格式模型文件（.onnx）
   - ONNX模型文件 → `load_onnx_model()` → ONNX Runtime会话 → `onnx_inference()` → 推理结果
   - ONNX模型优化: 原始ONNX模型 → `simplify_onnx_model()`/`optimize_onnx_model()` → 优化后的ONNX模型
   - ONNX模型验证: PyTorch输出 vs ONNX输出 → `verify_onnx_model()` → 验证结果

5. 性能指标可视化流程：
   - 训练循环 → `src/utils/metrics_logger.py` → 指标数据保存到CSV/JSON文件
   - 指标数据 → `src/visualization/metrics_plots.py` → 可视化结果保存到 `temp_metrics/plots/` 目录
   - 指标可视化支持通用工具和专用功能：
     - 通用图表保存: `save_plot()` 提供统一的保存接口，支持添加时间戳和元数据
     - 损失曲线可视化: `plot_loss()` 生成训练和验证损失曲线，支持时间戳和元数据
     - 准确率曲线可视化: `plot_accuracy()` 生成训练和验证准确率曲线，支持百分比显示和元数据
     - 多指标可视化: `plot_training_history()` 生成多种指标历史曲线，支持批量保存

6. 测试流程：
   - `tests/` 目录下的各测试文件分别测试对应模块的功能
   - 测试结果保存在 `tests/outputs/` 目录
   - 指标可视化测试: `test_plot_save.py` 测试图表保存功能，包括时间戳和元数据支持

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