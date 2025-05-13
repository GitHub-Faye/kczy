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
├── scripts/            # 实用脚本，如演示脚本和测试脚本
├── docs/               # 文档目录
├── logs/               # 日志目录
│   └── tensorboard_test/ # TensorBoard日志示例目录 
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
- `config.py` - 数据处理相关配置参数定义，包含DatasetConfig类用于管理数据集参数
- `custom_dataset.py` - 自定义数据集类实现，支持图像分类任务
- `data_loader.py` - 数据加载器实现，提供创建训练/验证/测试数据加载器功能
  - `create_dataloaders()` - 创建数据加载器，支持70%/20%/10%的训练/验证/测试集划分
  - `create_dataloaders_from_config()` - 基于配置对象创建数据加载器
  - `get_transforms()` - 获取数据变换函数，支持归一化和数据增强
  - `verify_dataset_splits()` - 验证数据集拆分是否符合预期比例
- `dataset.py` - 基本数据集类定义，提供数据加载基础功能
- `preprocessing.py` - 数据预处理函数集合
  - `normalize_image()` - 图像归一化处理函数
  - `denormalize_image()` - 图像归一化逆操作函数
  - 其他数据预处理工具如标准化、缺失值填充、异常值处理等
- `augmentation.py` - 数据增强方法实现，提供多种图像增强选项

#### 模型模块 (`src/models/`)
- `__init__.py` - 模型模块初始化文件，导出模型类和模型工具函数
- `vit.py` - Vision Transformer模型定义，包含模型保存/加载功能
- `train.py` - 模型训练循环实现，支持检查点保存和恢复
- `optimizer_manager.py` - 优化器管理类，包含优化器状态保存和恢复功能
- `model_utils.py` - 模型和优化器状态保存、加载和转换工具函数，支持多种格式（PyTorch、ONNX）和完整训练状态恢复，包含ONNX模型验证、优化和简化功能

#### 工具函数 (`src/utils/`)
- `__init__.py` - 工具模块初始化文件
- `config.py` - 全局配置参数定义
- `metrics_logger.py` - 性能指标记录和分析工具，支持TensorBoard集成，可记录标量指标、直方图、图像和超参数
- `tensorboard_utils.py` - TensorBoard工具模块，提供TensorBoard启动、检查和停止功能
- `cli.py` - 命令行接口实现，包含参数解析和处理功能，支持从配置文件加载和组织化参数显示。提供丰富的配置选项，包括：(1)超参数配置：损失函数类型、各种优化器特定参数（如SGD的momentum、Adam的beta值等）和学习率调度器参数（如cosine的t-max、eta-min等）；(2)数据集规范：数据集类型和来源（支持自定义、ImageNet、CIFAR10等多种数据集）、数据拆分（训练/验证/测试比例或预定义目录）、数据增强（旋转、缩放、翻转、颜色调整、Mixup/CutMix等高级增强）、数据预处理（归一化、调整大小方法）和采样策略（加权采样、过采样、欠采样等）

#### 可视化模块 (`src/visualization/`)
- `__init__.py` - 可视化模块初始化文件，导出指标绘图、注意力可视化、模型结构可视化和静态可视化功能
- `metrics_plots.py` - 指标绘图模块，用于绘制训练和评估指标的可视化图表
  - `save_plot()` - 通用图表保存函数，支持自动添加时间戳和保存元数据
  - `plot_loss()` - 绘制训练和验证损失曲线，支持保存到文件并可添加时间戳和元数据
  - `plot_accuracy()` - 绘制训练和验证准确率曲线，支持百分比显示、Y轴范围控制和文件保存（含时间戳和元数据）
  - `plot_training_history()` - 绘制多种训练指标历史曲线，支持批量生成并保存图表
- `attention_viz.py` - 注意力权重可视化模块，用于创建Vision Transformer模型注意力权重的热力图和可视化
  - `plot_attention_weights()` - 生成注意力热力图，可视化模型中特定层和头的注意力分布
  - `visualize_attention_on_image()` - 将注意力权重叠加到原始图像上，直观展示模型关注的区域
  - `visualize_all_heads()` - 同时展示特定层的所有注意力头，方便对比不同头关注的区域差异
- `model_viz.py` - 模型结构可视化模块，用于展示Vision Transformer模型的整体结构和层连接
  - `plot_model_structure()` - 生成整体模型结构图，展示从输入到输出的数据流和主要组件间的连接
  - `plot_encoder_block()` - 详细展示Transformer编码器块的内部结构，包括多头注意力、MLP和残差连接
  - `visualize_layer_weights()` - 分析并可视化模型中各层权重的分布和连接强度，包括权重范数、层间相似度和连接示意图
- `static_viz.py` - 静态可视化综合模块，整合注意力权重和模型结构可视化为一套连贯的静态图表
  - `create_model_overview()` - 生成模型结构概览图，包含模型结构和主要参数信息
  - `create_attention_analysis()` - 生成注意力权重分析图，包含注意力热力图和图像叠加视图
  - `create_comprehensive_visualization()` - 生成一套完整的可视化图表，包括模型结构、注意力权重和层连接
  - `compare_models()` - 比较多个模型的结构和注意力特性，支持并排展示不同模型的参数和注意力特征
  - `generate_visualization_report()` - 生成包含各种可视化结果的HTML格式报告，便于整体查看和分享

### 数据目录 (`data/`)
- `examples/` - 存放示例数据
- `images/` - 存放训练和测试图像
- `annotations.csv` - 数据标注文件

### 模型目录 (`models/`)
- 用于存储训练好的模型权重和配置

### 日志目录 (`logs/`)
- 存放训练日志和TensorBoard日志文件
- `tensorboard_test/` - TensorBoard示例日志目录
  - 包含TensorBoard事件文件用于可视化训练指标
  - 支持实时监控训练过程中的各种指标（损失、准确率等）
  - 可视化模型参数和梯度分布的直方图
  - 记录训练样本图像和可视化结果

### 文档目录 (`docs/`)
- `onnx_export.md` - ONNX导出功能使用指南，包含详细示例和注意事项
- `tensorboard_usage_guide.md` - TensorBoard使用指南，详细说明启动方式、界面功能及使用技巧
- `tensorboard_validation_report.md` - TensorBoard验证报告，记录验证过程和测试结果
- `visualization_testing.md` - 模型可视化测试指南，详细说明可视化测试脚本的使用方法、参数选项和输出文件说明

### 笔记本目录 (`notebooks/`)
- 用于存放Jupyter笔记本，进行实验和演示

### 临时指标目录 (`temp_metrics/`)
- `plots/` - 存放指标可视化图表
  - `loss_curve.png` - 损失曲线图
  - `accuracy_curve.png` - 准确率曲线图
  - `learning_rate_curve.png` - 学习率曲线图
  - `attention_heatmap.png` - 注意力权重热力图
  - `attention_on_image.png` - 注意力权重与原始图像叠加图
  - `all_attention_heads.png` - 多头注意力可视化图
  - `vit_*_structure.png` - 模型整体结构图，展示模型各组件和数据流向
  - `vit_encoder_block_default.png` - 编码器块通用结构图，展示编码器块的内部组件和连接
  - `vit_*_encoder_block.png` - 特定模型编码器块结构图，带有实际模型参数
  - `vit_*_layer_weights.png` - 模型层权重分析图，包含权重分布、层间相似度和连接示意图
  - `vit_model_overview_*.png` - 模型结构和参数信息的概览图
  - `vit_attention_analysis_*.png` - 不同层和头的注意力分析图
  - `vit_models_comparison_*.png` - 不同模型结构和注意力特性的比较图
  - `vit_visualization_report_*.html` - 包含所有可视化结果的HTML格式报告
  - `<前缀>_attention_heatmap_<时间戳>.png` - 带时间戳的注意力热力图，支持定制化前缀
  - `<前缀>_model_structure_<时间戳>.png` - 带时间戳的模型结构图
  - `<前缀>_comprehensive_visualization_report_<时间戳>.html` - 综合可视化报告，包含完整的模型分析结果
  - `<前缀>_models_comparison_<时间戳>.png` - 不同模型的结构和注意力特性比较图
- `simulation_train_metrics.csv` - 模拟训练指标数据
- `simulation_eval_metrics.csv` - 模拟评估指标数据

### 脚本目录 (`scripts/`)
- `test_cli.py` - 命令行参数解析功能测试脚本
- `demo_custom_dataset.py` - 自定义数据集演示
- `demo_augmentation.py` - 数据增强演示
- `demo_data_loader.py` - 数据加载器演示脚本，展示数据集加载、三部分拆分和可视化功能
- `demo_onnx_export.py` - ONNX导出功能演示脚本，展示模型导出、验证和推理性能比较
- `start_tensorboard.py` - TensorBoard启动脚本，提供独立的TensorBoard启动功能
- `verify_tensorboard_web.py` - TensorBoard Web界面验证脚本，运行测试并生成报告
- `test_attention_viz.py` - 注意力权重可视化测试脚本，展示如何提取和可视化模型的注意力权重
- `test_model_viz.py` - 模型结构可视化测试脚本，演示如何生成和保存模型整体结构图、编码器块结构图和层权重可视化
- `test_static_viz.py` - 静态可视化综合测试脚本，展示如何生成模型概览、注意力分析、综合可视化和模型比较
- `test_visualization_with_model.py` - 综合可视化测试脚本，实际使用模型（随机初始化或预训练）进行可视化测试，支持多种模型类型、多种图像输入选项和完整的可视化功能测试
- `example_config.json` - 示例配置文件，用于CLI测试
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
- `test_cli.py` - 命令行参数解析功能的全面测试，包含基本功能、参数类型与配置文件加载测试
- `test_visualization_script.py` - 可视化测试脚本的单元测试，验证脚本的参数解析、模型创建、图像加载和各类可视化功能
- `test_tensorboard_cli.py` - TensorBoard相关CLI选项的测试，验证参数解析和配置文件加载
- `test_tensorboard_utils.py` - TensorBoard工具模块测试，验证TensorBoard服务器的启动、检查和停止功能
- `test_tensorboard_web_interface.py` - TensorBoard Web界面验证测试，测试界面可访问性和数据显示功能
- `cli_test_edge_cases.py` - 命令行参数解析的边界情况测试，专注于错误处理和特殊输入
- `cli_integration_test.py` - CLI与训练脚本的集成测试，验证参数解析与训练流程的衔接
- `run_all_cli_tests.py` - 运行所有CLI相关测试并生成综合报告的脚本
- `sample_image.png` - 测试用图像
- `batch_images.png` - 测试用批量图像
- `outputs/` - 测试输出目录
  - `cli_tests/` - CLI测试输出目录
    - `integration/` - CLI集成测试输出子目录，包含各类测试场景的结果
    - `temp_configs/` - 测试用临时配置文件

### 任务目录 (`tasks/`)
- `tasks.json` - 任务定义文件
- `task_00X.txt` - 个别任务详细描述文件

### 报告目录 (`reports/`)
- 用于存放测试报告和验证结果

### 其他文件
- `requirements.txt` - 项目依赖列表
- `.taskmasterconfig` - 任务管理器配置文件
- `prd.txt` - 产品需求文档
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

3. 命令行接口流程：
   - 用户提供命令行参数 → `src/utils/cli.py` 解析参数 → 转换为配置对象 → 用于训练和评估
   - 配置文件通过 `src/utils/cli.py` 的 `load_config()` 函数加载，与命令行参数结合
   - 命令行参数覆盖配置文件中的参数，提供灵活配置能力
   - CLI支持全面的训练配置，包括:
     - 模型超参数：各种优化器参数、学习率调度策略、损失函数类型
     - 数据集配置：数据集类型、拆分策略、增强选项、预处理方法、采样策略
     - 训练控制：早停、梯度裁剪、混合精度训练、检查点保存
     - TensorBoard配置：启用/禁用记录、日志目录、服务器端口、直方图和图像记录选项
   - CLI测试完整覆盖所有参数解析场景，包括:
     - 基础参数解析与默认值测试(`test_cli.py`)
     - 边界条件与错误处理测试(`cli_test_edge_cases.py`)
     - 配置文件加载与命令行参数混合测试
     - 各种格式(JSON/YAML)配置文件测试
     - 与训练流程的完整集成测试(`cli_integration_test.py`)
     - TensorBoard选项解析和验证测试(`test_tensorboard_cli.py`)

4. 模型保存和加载流程:
   - 训练完成或检查点 → `src/models/model_utils.py` 保存功能 → 模型文件存储在 `models/` 目录
   - 模型文件 → `src/models/model_utils.py` 加载功能 → 恢复模型用于推理或继续训练
   - 支持多种格式：原生PyTorch模型（.pt/.pth）和ONNX格式（.onnx）
   - 优化器状态保存 → `optimizer_manager.state_dict()` → 通过`save_checkpoint()`保存到检查点文件
   - 优化器状态恢复 → 从检查点文件读取 → `optimizer_manager.load_state_dict()` → 恢复训练状态

5. ONNX模型导出和推理流程:
   - PyTorch模型 → `export_to_onnx()` → ONNX格式模型文件（.onnx）
   - ONNX模型文件 → `load_onnx_model()` → ONNX Runtime会话 → `onnx_inference()` → 推理结果
   - ONNX模型优化: 原始ONNX模型 → `simplify_onnx_model()`/`optimize_onnx_model()` → 优化后的ONNX模型
   - ONNX模型验证: PyTorch输出 vs ONNX输出 → `verify_onnx_model()` → 验证结果

6. 性能指标可视化流程：
   - 训练循环 → `src/utils/metrics_logger.py` → 指标数据保存到CSV/JSON文件
   - 指标数据 → `src/visualization/metrics_plots.py` → 可视化结果保存到 `temp_metrics/plots/` 目录
   - 指标可视化支持通用工具和专用功能：
     - 通用图表保存: `save_plot()` 提供统一的保存接口，支持添加时间戳和元数据
     - 损失曲线可视化: `plot_loss()` 生成训练和验证损失曲线，支持时间戳和元数据
     - 准确率曲线可视化: `plot_accuracy()` 生成训练和验证准确率曲线，支持百分比显示和元数据
     - 多指标可视化: `plot_training_history()` 生成多种指标历史曲线，支持批量保存

7. 注意力权重可视化流程:
   - ViT模型 → `return_attention=True` 参数 → 返回注意力权重列表
   - 注意力权重 → `src/visualization/attention_viz.py` → 可视化结果保存到 `temp_metrics/plots/` 目录
   - 可视化测试流程: ViT模型(随机初始化/预训练) → `scripts/test_visualization_with_model.py` → 全面测试所有可视化功能
   - 注意力可视化支持三种主要功能:
     - 热力图可视化: `plot_attention_weights()` 生成层级或特定头的注意力热力图
     - 图像叠加可视化: `visualize_attention_on_image()` 将注意力权重叠加到原始图像上
     - 多头对比可视化: `visualize_all_heads()` 并排展示所有注意力头，便于比较不同头的关注区域

8. 模型结构可视化流程:
   - ViT模型实例 → `src/visualization/model_viz.py` → 可视化结果保存到 `temp_metrics/plots/` 目录
   - 支持多种可视化方式:
     - 整体结构可视化: `plot_model_structure()` 生成从输入到输出的完整模型结构图
     - 编码器块详情: `plot_encoder_block()` 展示单个Transformer编码器块的内部结构和数据流
     - 层权重分析: `visualize_layer_weights()` 分析模型各层权重并展示层间关系和连接强度
   - 提供双重实现:
     - Graphviz实现: 生成高质量的有向图，支持多种输出格式(PNG/SVG/PDF)
     - Matplotlib实现: 作为备选方案，无需额外依赖也能生成模型结构图
   - 支持不同的可视化配置:
     - 自定义方向: 支持垂直布局(上到下)或水平布局(左到右)
     - 详细程度控制: 支持简略或详细显示内部组件
     - 格式选择: 支持多种输出格式，适应不同用途

9. 静态可视化综合流程:
   - ViT模型实例 + 输入图像 → `src/visualization/static_viz.py` → 可视化结果保存到 `temp_metrics/plots/` 目录
   - 模型概览: `create_model_overview()` → 生成模型结构和参数信息的综合图表
   - 注意力分析: `create_attention_analysis()` → 生成注意力热力图和图像叠加视图
   - 综合可视化: `create_comprehensive_visualization()` → 生成包含所有可视化类型的完整套件
   - 模型比较: `compare_models()` → 比较多个模型的结构和注意力特性
   - HTML报告: `generate_visualization_report()` → 生成包含所有可视化结果的网页报告
   - 单组件测试: `scripts/test_static_viz.py` → 演示和验证各静态可视化组件功能
   - 综合测试: `scripts/test_visualization_with_model.py` → 使用实际模型进行所有可视化功能的全面测试，支持多种配置
   - 单元测试: `tests/test_visualization_script.py` → 验证测试脚本功能和准确性
   - 多种输出格式支持: PNG、JPG、SVG、PDF等图像格式，以及HTML格式报告
   - 支持定制化: 可自定义图像尺寸、分辨率、标题和输出路径

10. TensorBoard集成流程：
   - 训练开始 → 初始化TensorBoard SummaryWriter → 日志保存到`logs/`目录
   - 训练过程 → MetricsLogger记录指标到TensorBoard → 实时生成事件文件
   - TensorBoard启动方式：
     - 代码调用：`start_tensorboard()` → 在代码中启动TensorBoard
     - 独立脚本：`python scripts/start_tensorboard.py` → 单独启动TensorBoard服务器
   - 访问TensorBoard：http://localhost:6006/（默认）或自定义主机和端口
   - 指标记录的类型：
     - 标量指标：训练损失、验证损失、准确率等
     - 直方图：模型参数和梯度分布
     - 图像：训练样本和可视化结果
     - 超参数：实验配置和最终性能指标
   - TensorBoard的配置由TrainingConfig中的参数控制，可通过CLI配置：
     - enable_tensorboard(--enable-tensorboard)：是否启用TensorBoard
     - tensorboard_dir(--tensorboard-dir)：TensorBoard日志目录
     - tensorboard_port(--tensorboard-port)：TensorBoard服务器端口号
     - log_histograms(--log-histograms)：是否记录模型参数和梯度直方图
     - log_images(--log-images)：是否记录样本图像
     - start_tensorboard(--start-tensorboard)：是否启动TensorBoard服务器
     - tensorboard_host(--tensorboard-host)：TensorBoard服务器主机地址
     - tensorboard_background(--tensorboard-background)：是否在后台运行TensorBoard
   - 支持通过配置文件(JSON/YAML)或命令行参数设置TensorBoard选项
   - 命令行参数优先级高于配置文件设置
   - TensorBoard工具模块(tensorboard_utils.py)提供了启动、检查和停止TensorBoard的通用功能

11. 测试流程：
   - `tests/` 目录下的各测试文件分别测试对应模块的功能
   - 测试结果保存在 `tests/outputs/` 目录
   - CLI测试流程: 从基础单元测试(`test_cli.py`)到边界条件测试(`cli_test_edge_cases.py`)再到集成测试(`cli_integration_test.py`)，系统性验证CLI功能的正确性和健壮性
   - 综合CLI测试: `run_all_cli_tests.py`集成运行所有CLI测试并生成详细报告，支持中文输出和完整测试统计

### 配置依赖关系
- `src/utils/config.py` 提供全局配置，包括训练参数和指标记录选项
- `src/data/config.py` 提供数据处理特定配置
- `src/utils/cli.py` 提供命令行参数解析和配置文件加载功能
- 脚本通过导入这些配置模块或CLI模块访问参数

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