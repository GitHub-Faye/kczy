#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模型的主脚本，提供命令行接口和完整的训练循环
"""
import os
import sys
import argparse
import logging
import time
import json
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import create_dataloaders_from_config
from src.data.config import DatasetConfig
from src.models.vit import VisionTransformer
from src.models.train import TrainingLoop
from src.models.optimizer_manager import OptimizerManager
from src.models.model_utils import save_checkpoint, save_model
from src.utils.config import TrainingConfig
from src.utils.metrics_logger import MetricsLogger
from src.models.train_config import (
    create_training_config, 
    validate_training_params, 
    print_training_config,
    setup_tensorboard
)
from src.utils.tensorboard_utils import start_tensorboard, check_tensorboard_running

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练 Vision Transformer 模型')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='data/images',
                        help='数据集目录路径')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集占比 (默认: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集占比 (默认: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集占比 (默认: 0.1)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像大小 (默认: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批量大小 (默认: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载的工作线程数 (默认: 4)')
    
    # 模型参数
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='嵌入维度 (默认: 768)')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='注意力头数量 (默认: 12)')
    parser.add_argument('--num-layers', type=int, default=12,
                        help='Transformer层数 (默认: 12)')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='图像块大小 (默认: 16)')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='丢弃率 (默认: 0.1)')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='分类类别数量')
    
    # 训练参数
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='权重衰减 (默认: 1e-5)')
    parser.add_argument('--optimizer-type', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw'],
                        help='优化器类型 (默认: adamw)')
    parser.add_argument('--scheduler-type', type=str, default='cosine',
                        choices=['step', 'multistep', 'exponential', 'cosine', 'plateau', 'warmup_cosine'],
                        help='学习率调度器类型 (默认: cosine)')
    parser.add_argument('--loss-type', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal', 'bce', 'mse'],
                        help='损失函数类型 (默认: cross_entropy)')
    parser.add_argument('--grad-clip-value', type=float, default=None,
                        help='梯度裁剪值 (默认: None)')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                        help='梯度范数裁剪值 (默认: 1.0)')
    parser.add_argument('--use-mixed-precision', action='store_true',
                        help='启用混合精度训练')
    parser.add_argument('--early-stopping', action='store_true',
                        help='启用早停止')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停止的耐心参数 (默认: 10)')
    parser.add_argument('--device', type=str, default='',
                        help='训练设备 (留空则自动选择)')
    
    # 输出和日志参数
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录 (默认: output)')
    parser.add_argument('--experiment-name', type=str, default='',
                        help='实验名称 (默认: 自动生成)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='检查点保存目录 (默认: checkpoints)')
    parser.add_argument('--checkpoint-freq', type=int, default=1,
                        help='检查点保存频率，每多少个epoch保存一次 (默认: 1)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='日志目录 (默认: logs)')
    parser.add_argument('--metrics-dir', type=str, default='metrics',
                        help='指标保存目录 (默认: metrics)')
    parser.add_argument('--save-format', type=str, default='json',
                        choices=['json', 'csv'],
                        help='指标保存格式 (默认: json)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='日志打印间隔，每多少批次打印一次 (默认: 10)')
    
    # TensorBoard参数
    parser.add_argument('--enable-tensorboard', action='store_true',
                        help='启用TensorBoard记录')
    parser.add_argument('--tensorboard-dir', type=str, default='logs/tensorboard',
                        help='TensorBoard日志目录 (默认: logs/tensorboard)')
    parser.add_argument('--log-histograms', action='store_true',
                        help='记录模型参数直方图到TensorBoard')
    parser.add_argument('--log-images', action='store_true',
                        help='记录图像到TensorBoard')
    parser.add_argument('--start-tensorboard', action='store_true',
                        help='自动启动TensorBoard服务器')
    parser.add_argument('--tensorboard-port', type=int, default=6006,
                        help='TensorBoard服务器端口 (默认: 6006)')
    
    # 配置文件
    parser.add_argument('--config-file', type=str, default=None,
                        help='配置文件路径 (JSON或YAML)')
    parser.add_argument('--profile', type=str, default='default',
                        choices=['default', 'fast_dev', 'high_performance'],
                        help='预定义配置文件 (默认: default)')
    
    # 模型恢复参数
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    return parser.parse_args()

def setup_experiment_dir(args):
    """设置实验目录结构"""
    # 生成实验名称（如果未提供）
    if not args.experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"experiment_{timestamp}"
    
    # 创建实验目录
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(experiment_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.join(experiment_dir, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建指标目录
    metrics_dir = os.path.join(experiment_dir, args.metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 创建TensorBoard目录
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 返回更新后的目录路径
    return {
        'experiment_dir': experiment_dir,
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
        'metrics_dir': metrics_dir,
        'tensorboard_dir': tensorboard_dir
    }

def create_model(args):
    """创建Vision Transformer模型"""
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,  # 假设RGB图像
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=4,
        dropout_rate=args.dropout_rate,
        embed_dropout_rate=args.dropout_rate,
        attention_dropout_rate=args.dropout_rate,
    )
    return model

def save_experiment_config(config, experiment_dir):
    """保存实验配置到文件"""
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    
    # 如果配置对象有to_dict方法
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    # 尝试使用dataclasses的asdict方法
    elif hasattr(config, '__dataclass_fields__'):
        from dataclasses import asdict
        config_dict = asdict(config)
    # 如果上述方法都不适用，但对象可字典化
    elif hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = vars(config)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"实验配置已保存到: {config_path}")

def main():
    """主函数，训练Vision Transformer模型"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置实验目录
    dirs = setup_experiment_dir(args)
    
    # 确定设备
    device = args.device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"使用设备: {device}")
    
    # 创建数据集配置
    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders_from_config(dataset_config)
    logger.info(f"数据加载器已创建，训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
    
    # 创建训练配置
    training_config = create_training_config(
        args=args,
        config_file=args.config_file,
        profile=args.profile
    )
    
    # 更新训练配置中的路径
    training_config.tensorboard_dir = dirs['tensorboard_dir']
    training_config.checkpoint_dir = dirs['checkpoint_dir']
    training_config.log_dir = dirs['log_dir']
    training_config.device = device
    
    # 验证训练参数
    validate_training_params(training_config)
    
    # 打印训练配置
    print_training_config(training_config)
    
    # 保存实验配置
    save_experiment_config(training_config, dirs['experiment_dir'])
    
    # 创建模型
    model = create_model(args)
    model.to(device)
    logger.info(f"模型已创建: {model.__class__.__name__} - {sum(p.numel() for p in model.parameters())} 参数")
    
    # 设置TensorBoard（如果启用）
    tensorboard_writer = None
    if args.enable_tensorboard:
        tensorboard_writer = setup_tensorboard(training_config)
        
        # 启动TensorBoard服务器（如果请求）
        if args.start_tensorboard and not check_tensorboard_running(args.tensorboard_port):
            start_tensorboard(
                log_dir=dirs['tensorboard_dir'],
                port=args.tensorboard_port,
                background=True
            )
            logger.info(f"TensorBoard服务器已启动，可在 http://localhost:{args.tensorboard_port} 访问")
    
    # 创建指标记录器
    metrics_logger = MetricsLogger(
        save_dir=dirs['metrics_dir'],
        experiment_name=args.experiment_name,
        save_format=args.save_format,
        enable_tensorboard=args.enable_tensorboard,
        tensorboard_log_dir=dirs['tensorboard_dir']
    )
    
    # 从TrainingConfig创建TrainingLoop
    training_loop = TrainingLoop.from_config(
        model=model,
        config=training_config,
        validate_config=True,
        setup_tb=False,  # 我们已经手动设置了TensorBoard
        print_summary=True
    )
    
    # 如果提供了恢复检查点，则加载
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"从检查点恢复训练: {args.resume}")
            # 使用TrainingLoop的方法加载检查点（包括模型、优化器状态等）
            model, optimizer_state, start_epoch, history = training_loop.load_checkpoint(args.resume)
            logger.info(f"恢复训练，从epoch {start_epoch} 开始")
        else:
            logger.warning(f"检查点文件不存在: {args.resume}，将从头开始训练")
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行训练
    try:
        logger.info("开始训练...")
        training_history = training_loop.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            checkpoint_dir=dirs['checkpoint_dir'],
            checkpoint_freq=args.checkpoint_freq,
            log_interval=args.log_interval,
            early_stopping=args.early_stopping,
            early_stopping_patience=args.patience,
            config=training_config  # 传递完整的训练配置以便保存
        )
        
        # 记录训练完成时间
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"训练完成！用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        
        # 保存最终模型
        final_model_path = os.path.join(dirs['checkpoint_dir'], f"{args.experiment_name}_final_model.pth")
        save_model(model, final_model_path, metadata={
            'experiment_name': args.experiment_name,
            'epochs': args.num_epochs,
            'final_train_loss': training_history['loss'][-1] if 'loss' in training_history else None,
            'final_val_loss': training_history['val_loss'][-1] if 'val_loss' in training_history else None,
            'final_val_acc': training_history['val_acc'][-1] if 'val_acc' in training_history else None,
            'training_time': training_time
        })
        logger.info(f"最终模型已保存到: {final_model_path}")
        
        # 对测试集进行评估
        if test_loader:
            logger.info("在测试集上评估模型...")
            test_results = training_loop.evaluate(
                test_loader=test_loader,
                num_classes=args.num_classes,
                visualize_confusion_matrix=True,
                output_path=os.path.join(dirs['experiment_dir'], f"{args.experiment_name}_confusion_matrix.png")
            )
            
            # 记录测试结果
            logger.info(f"测试集结果:")
            for metric, value in test_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
            
            # 保存测试结果
            test_results_path = os.path.join(dirs['experiment_dir'], f"{args.experiment_name}_test_results.json")
            with open(test_results_path, 'w', encoding='utf-8') as f:
                # 确保所有的numpy数组都被转换为列表
                test_results_serializable = {}
                for k, v in test_results.items():
                    if hasattr(v, 'tolist'):  # 如果是numpy数组
                        test_results_serializable[k] = v.tolist()
                    elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                        test_results_serializable[k] = v
                json.dump(test_results_serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"测试结果已保存到: {test_results_path}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        # 保存中断时的模型
        interrupted_model_path = os.path.join(dirs['checkpoint_dir'], f"{args.experiment_name}_interrupted.pth")
        save_model(model, interrupted_model_path)
        logger.info(f"中断时的模型已保存到: {interrupted_model_path}")
    except Exception as e:
        logger.exception(f"训练过程中出错: {e}")
    finally:
        # 关闭TensorBoard写入器
        if tensorboard_writer:
            tensorboard_writer.close()
        
        # 关闭指标记录器
        metrics_logger.close()

if __name__ == "__main__":
    main() 