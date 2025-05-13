import os
import argparse
import logging
import subprocess
import json
import yaml
from typing import Dict, Optional, Any, Union
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import TrainingConfig
from src.utils.tensorboard_utils import (
    start_tensorboard, check_tensorboard_running, find_tensorboard_executable
)

logger = logging.getLogger(__name__)

def create_training_config(
    args: Optional[argparse.Namespace] = None,
    config_file: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    profile: str = "default"
) -> TrainingConfig:
    """
    创建训练配置，优先级：命令行参数 > 配置字典 > 配置文件 > 预定义配置文件 > 默认值
    
    参数:
        args (Optional[argparse.Namespace]): 命令行参数
        config_file (Optional[str]): 配置文件路径
        config_dict (Optional[Dict[str, Any]]): 配置字典
        profile (str): 预定义配置文件名称，可选值：default, fast_dev, high_performance
        
    返回:
        TrainingConfig: 训练配置
    """
    # 创建基础配置
    if profile == "default":
        config = TrainingConfig.create_default()
    elif profile == "fast_dev":
        config = TrainingConfig.create_fast_dev()
    elif profile == "high_performance":
        config = TrainingConfig.create_high_performance()
    else:
        raise ValueError(f"不支持的预定义配置文件: {profile}")
    
    # 从配置文件加载（如果提供）
    if config_file and os.path.exists(config_file):
        loaded_config = load_training_config(config_file)
        # 更新当前配置
        for key, value in asdict(loaded_config).items():
            if value is not None:
                setattr(config, key, value)
    
    # 从配置字典更新（如果提供）
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
    
    # 从命令行参数更新（如果提供）
    if args:
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
    
    return config

def validate_training_params(config: TrainingConfig) -> bool:
    """
    验证训练参数的有效性并提供反馈
    
    参数:
        config (TrainingConfig): 训练配置
        
    返回:
        bool: 配置是否有效
        
    抛出:
        ValueError: 如果配置无效
    """
    # 学习率验证
    if config.learning_rate <= 0:
        raise ValueError(f"学习率必须大于0，当前值: {config.learning_rate}")
    if config.learning_rate > 0.1:
        logger.warning(f"学习率较高: {config.learning_rate}，可能导致训练不稳定")
    
    # 批大小验证
    if config.batch_size <= 0:
        raise ValueError(f"批大小必须大于0，当前值: {config.batch_size}")
    
    # 检查是否使用CUDA
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("配置指定使用CUDA，但CUDA不可用，将回退到CPU")
        config.device = "cpu"
        config.use_mixed_precision = False  # 在CPU上禁用混合精度
    
    # 验证优化器类型
    supported_optimizers = ["sgd", "adam", "adamw", "rmsprop"]
    if config.optimizer_type.lower() not in supported_optimizers:
        raise ValueError(f"不支持的优化器类型: {config.optimizer_type}，支持的类型: {supported_optimizers}")
    
    # 验证学习率调度器类型
    if config.scheduler_type:
        supported_schedulers = ["step", "multistep", "exponential", "cosine", "plateau", "warmup_cosine"]
        if config.scheduler_type.lower() not in supported_schedulers:
            raise ValueError(f"不支持的调度器类型: {config.scheduler_type}，支持的类型: {supported_schedulers}")
    
    # 验证损失函数类型
    supported_loss_types = ["cross_entropy", "mse", "bce", "focal"]
    if config.loss_type.lower() not in supported_loss_types:
        raise ValueError(f"不支持的损失函数类型: {config.loss_type}，支持的类型: {supported_loss_types}")
    
    # 验证混合精度训练
    if config.use_mixed_precision and config.device == "cpu":
        logger.warning("混合精度训练在CPU上无效，已禁用")
        config.use_mixed_precision = False
    
    # 验证TensorBoard设置
    if config.enable_tensorboard and not config.tensorboard_dir:
        logger.warning("启用了TensorBoard但未指定日志目录，将使用默认目录")
        config.tensorboard_dir = "logs/tensorboard"
    
    # 验证正则化和参数调整参数
    if config.weight_decay < 0:
        raise ValueError(f"权重衰减必须非负，当前值: {config.weight_decay}")
    
    # 检查是否有dropout_rate属性，如果有则验证
    if hasattr(config, "dropout_rate"):
        if config.dropout_rate < 0 or config.dropout_rate >= 1:
            raise ValueError(f"丢弃率必须在[0, 1)范围内，当前值: {config.dropout_rate}")
    
    # 验证训练轮数
    if config.num_epochs <= 0:
        raise ValueError(f"训练轮数必须大于0，当前值: {config.num_epochs}")
    
    # 验证梯度裁剪参数
    if config.grad_clip_value is not None and config.grad_clip_value <= 0:
        raise ValueError(f"梯度裁剪值必须大于0，当前值: {config.grad_clip_value}")
    
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise ValueError(f"梯度范数裁剪值必须大于0，当前值: {config.grad_clip_norm}")
    
    return True

def print_training_config(config: TrainingConfig) -> None:
    """
    格式化打印训练配置摘要
    
    参数:
        config (TrainingConfig): 训练配置
    """
    print("\n" + "=" * 50)
    print("训练配置摘要")
    print("=" * 50)
    
    # 基本训练参数
    print(f"训练轮数: {config.num_epochs}")
    print(f"批大小: {config.batch_size}")
    print(f"设备: {config.device}")
    
    # 优化器和学习率
    print(f"优化器: {config.optimizer_type}")
    print(f"学习率: {config.learning_rate}")
    print(f"权重衰减: {config.weight_decay}")
    
    # 学习率调度器
    if config.scheduler_type:
        print(f"学习率调度器: {config.scheduler_type}")
        if hasattr(config, "scheduler_params") and config.scheduler_params:
            for key, value in config.scheduler_params.items():
                print(f"  - {key}: {value}")
    
    # 损失函数
    print(f"损失函数: {config.loss_type}")
    
    # 高级训练设置
    print(f"混合精度训练: {'启用' if config.use_mixed_precision else '禁用'}")
    if config.grad_clip_value:
        print(f"梯度裁剪值: {config.grad_clip_value}")
    if config.grad_clip_norm:
        print(f"梯度范数裁剪: {config.grad_clip_norm}")
    
    # 丢弃率（如果存在）
    if hasattr(config, "dropout_rate"):
        print(f"丢弃率: {config.dropout_rate}")
    
    # TensorBoard和日志设置
    print(f"TensorBoard: {'启用' if config.enable_tensorboard else '禁用'}")
    if config.enable_tensorboard:
        print(f"  - 日志目录: {config.tensorboard_dir}")
        print(f"  - 记录直方图: {'启用' if config.log_histograms else '禁用'}")
        print(f"  - 记录图像: {'启用' if config.log_images else '禁用'}")
    
    # 检查点设置
    if hasattr(config, "checkpoint_dir") and config.checkpoint_dir:
        print(f"检查点目录: {config.checkpoint_dir}")
    
    # 数据加载设置
    if hasattr(config, "num_workers") and config.num_workers is not None:
        print(f"数据加载器工作进程数: {config.num_workers}")
    
    print("=" * 50 + "\n")

def setup_tensorboard(config: TrainingConfig) -> Optional[SummaryWriter]:
    """
    配置TensorBoard，包括创建日志目录和启动服务器（如果需要）
    
    参数:
        config (TrainingConfig): 训练配置
        
    返回:
        Optional[SummaryWriter]: TensorBoard的SummaryWriter对象，如果启用了TensorBoard
    """
    if not config.enable_tensorboard:
        return None
    
    # 创建TensorBoard日志目录
    tensorboard_dir = config.tensorboard_dir
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 创建SummaryWriter
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # 如果需要，启动TensorBoard服务器
    if hasattr(config, "start_tensorboard") and config.start_tensorboard:
        port = config.tensorboard_port if hasattr(config, "tensorboard_port") else 6006
        
        # 检查TensorBoard是否已经在运行
        if not check_tensorboard_running(port):
            # 尝试启动TensorBoard
            try:
                tb_process = start_tensorboard(
                    log_dir=tensorboard_dir, 
                    port=port, 
                    host="0.0.0.0"  # 允许外部访问
                )
                
                if tb_process:
                    logger.info(f"TensorBoard服务器已启动，可在浏览器中访问 http://localhost:{port}")
                else:
                    logger.warning("尝试启动TensorBoard服务器失败")
            except Exception as e:
                logger.error(f"启动TensorBoard时出错: {str(e)}")
        else:
            logger.info(f"TensorBoard已在端口 {port} 运行")
    
    return writer

def save_training_config(config: TrainingConfig, file_path: str, format: str = None) -> str:
    """
    保存训练配置到文件
    
    参数:
        config (TrainingConfig): 训练配置
        file_path (str): 保存路径
        format (str): 保存格式，支持'json'和'yaml'，如果为None则从文件扩展名推断
        
    返回:
        str: 保存的文件路径
    """
    # 创建目录
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # 将配置转换为字典
    config_dict = asdict(config)
    
    # 确定格式
    if format is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.json':
            format = 'json'
        elif ext in ['.yaml', '.yml']:
            format = 'yaml'
        else:
            format = 'json'  # 默认为JSON
    
    # 保存配置
    if format.lower() == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    elif format.lower() == 'yaml':
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
    else:
        raise ValueError(f"不支持的格式: {format}，支持的格式: json, yaml")
    
    logger.info(f"训练配置已保存到: {file_path}")
    return file_path

def load_training_config(file_path: str) -> TrainingConfig:
    """
    从文件加载训练配置
    
    参数:
        file_path (str): 配置文件路径
        
    返回:
        TrainingConfig: 加载的训练配置
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    
    # 确定文件格式
    ext = os.path.splitext(file_path)[1].lower()
    
    # 加载配置
    config_dict = {}
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    elif ext in ['.yaml', '.yml']:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，支持的格式: .json, .yaml, .yml")
    
    # 创建配置对象
    config = TrainingConfig()
    
    # 更新配置
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    logger.info(f"已从 {file_path} 加载训练配置")
    return config

def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息，包括CPU和GPU（如果可用）
    
    返回:
        Dict[str, Any]: 设备信息
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_type": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # CPU信息
    try:
        import platform
        import psutil
        device_info["cpu"] = {
            "name": platform.processor(),
            "arch": platform.machine(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
        }
    except ImportError:
        device_info["cpu"] = {"name": "Unknown (psutil not available)"}
    
    # GPU信息
    if torch.cuda.is_available():
        device_info["cuda_version"] = torch.version.cuda
        device_info["device_name"] = torch.cuda.get_device_name(0)
        device_info["gpu_count"] = torch.cuda.device_count()
        
        # 尝试获取更多GPU信息
        try:
            device_info["gpu"] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB"
                }
                device_info["gpu"].append(gpu_info)
        except Exception as e:
            logger.warning(f"获取详细GPU信息时出错: {str(e)}")
    else:
        device_info["device_name"] = "CPU"
    
    return device_info

def get_optimal_config_for_device() -> TrainingConfig:
    """
    根据当前设备自动生成最优训练配置
    
    返回:
        TrainingConfig: 优化的训练配置
    """
    # 获取设备信息
    device_info = get_device_info()
    
    # 创建基础配置
    config = TrainingConfig.create_default()
    
    # 根据设备类型配置
    if device_info["cuda_available"]:
        # GPU配置
        config.device = "cuda"
        config.use_mixed_precision = True
        
        # 获取GPU显存大小并据此调整批大小
        if "gpu" in device_info and device_info["gpu"]:
            memory_gb = float(device_info["gpu"][0]["memory_total"].split(" ")[0])
            
            # 根据显存大小调整批大小
            if memory_gb >= 16:
                config.batch_size = 128
            elif memory_gb >= 8:
                config.batch_size = 64
            elif memory_gb >= 4:
                config.batch_size = 32
            else:
                config.batch_size = 16
                
            logger.info(f"根据GPU显存({memory_gb} GB)自动设置批大小为 {config.batch_size}")
    else:
        # CPU配置
        config.device = "cpu"
        config.use_mixed_precision = False
        config.batch_size = 32  # CPU上使用较小的批大小
        
        # 如果可用，根据CPU核心数配置工作进程数
        if "cpu" in device_info and isinstance(device_info["cpu"], dict) and "cores" in device_info["cpu"]:
            cores = device_info["cpu"]["cores"]
            if cores:
                config.num_workers = max(1, cores - 1)  # 留出一个核心给系统
                logger.info(f"根据CPU核心数({cores})自动设置数据加载器工作进程数为 {config.num_workers}")
    
    # 优化器和学习率配置
    if device_info["cuda_available"]:
        config.optimizer_type = "adamw"
        config.learning_rate = 0.001
        config.weight_decay = 0.01
        config.scheduler_type = "cosine"
        config.scheduler_params = {"T_max": 100, "eta_min": 1e-6}
    else:
        config.optimizer_type = "adam"
        config.learning_rate = 0.0005
        config.weight_decay = 0.001
    
    return config 