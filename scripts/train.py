#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ViT模型训练脚本，使用命令行参数配置训练过程
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入项目模块
from src.utils.cli import parse_args, print_args_info
from src.utils.config import ViTConfig, TrainingConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment(args):
    """
    设置训练环境，如创建必要的目录
    
    参数:
        args: 解析后的命令行参数
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 创建指标目录
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # 设置随机种子
    import torch
    import numpy as np
    import random
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"环境设置完成，随机种子: {seed}")

def create_vit_config(args):
    """
    从命令行参数创建ViT模型配置
    
    参数:
        args: 解析后的命令行参数
        
    返回:
        ViTConfig: ViT模型配置对象
    """
    # 根据model_type选择基础配置
    if args.model_type == 'tiny':
        config = ViTConfig.create_tiny(num_classes=args.num_classes)
    elif args.model_type == 'small':
        config = ViTConfig.create_small(num_classes=args.num_classes)
    elif args.model_type == 'base':
        config = ViTConfig.create_base(num_classes=args.num_classes)
    elif args.model_type == 'large':
        config = ViTConfig.create_large(num_classes=args.num_classes)
    elif args.model_type == 'huge':
        config = ViTConfig.create_huge(num_classes=args.num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 使用命令行参数覆盖默认配置
    config.img_size = args.img_size
    config.patch_size = args.patch_size
    config.pretrained = args.pretrained
    config.pretrained_path = args.pretrained_path
    
    # 如果指定了embed_dim, depth和num_heads，则覆盖预设配置
    if args.embed_dim is not None:
        config.embed_dim = args.embed_dim
    if args.depth is not None:
        config.depth = args.depth
    if args.num_heads is not None:
        config.num_heads = args.num_heads
    
    # 验证配置有效性
    config.validate()
    
    return config

def create_training_config(args):
    """
    从命令行参数创建训练配置
    
    参数:
        args: 解析后的命令行参数
        
    返回:
        TrainingConfig: 训练配置对象
    """
    # 创建基础配置
    config = TrainingConfig()
    
    # 使用命令行参数覆盖默认配置
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.optimizer_type = args.optimizer
    config.scheduler_type = args.scheduler if args.scheduler != 'none' else None
    config.val_split = args.val_split
    config.early_stopping = args.early_stopping
    config.early_stopping_patience = args.patience
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.log_freq = args.log_freq
    config.random_seed = args.seed
    config.device = args.device
    config.num_workers = args.num_workers
    config.pin_memory = args.pin_memory
    config.grad_clip_value = args.grad_clip_value
    config.grad_clip_norm = args.grad_clip_norm
    config.use_mixed_precision = args.use_mixed_precision
    config.metrics_dir = args.metrics_dir
    config.metrics_format = args.metrics_format
    config.metrics_experiment_name = args.experiment_name
    
    # 验证配置有效性
    config.validate()
    
    return config

def main():
    """训练脚本主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 打印参数信息
    print_args_info(args)
    
    # 设置训练环境
    setup_environment(args)
    
    # 创建ViT模型配置
    try:
        vit_config = create_vit_config(args)
        logger.info("成功创建ViT模型配置")
    except Exception as e:
        logger.error(f"创建ViT模型配置失败: {e}")
        return 1
    
    # 创建训练配置
    try:
        training_config = create_training_config(args)
        logger.info("成功创建训练配置")
    except Exception as e:
        logger.error(f"创建训练配置失败: {e}")
        return 1
    
    # 保存配置到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"experiment_{timestamp}"
    
    config_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(config_dir, exist_ok=True)
    
    vit_config_path = os.path.join(config_dir, "vit_config.json")
    training_config_path = os.path.join(config_dir, "training_config.json")
    
    vit_config.save(vit_config_path)
    training_config.save(training_config_path)
    
    logger.info(f"配置已保存到: {config_dir}")
    
    # TODO: 在这里实现实际的数据加载、模型训练等功能
    # 例如：
    # dataset = load_dataset(args.data_dir, args.anno_file)
    # model = build_model(vit_config)
    # trainer = Trainer(model, dataset, training_config)
    # trainer.train()
    
    logger.info("训练脚本执行完成 (模拟)")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 