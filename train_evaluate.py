#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_evaluate.py - ViT模型训练和评估脚本

import os
import argparse
import torch
from datetime import datetime

# 导入项目模块
from src.data.data_loader import create_dataloaders
from src.data.dataset import BaseDataset
from src.models.vit import VisionTransformer
from src.models.train import TrainingLoop, LossCalculator, BackpropManager
from src.utils.metrics_logger import MetricsLogger
from src.visualization.metrics_plots import plot_training_history
from src.models.optimizer_manager import OptimizerManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练和评估ViT模型')
    
    # 数据集参数
    parser.add_argument('--data-dir', type=str, default='data/images/', help='图像数据目录')
    parser.add_argument('--annotation-file', type=str, default='data/annotations.csv', help='标注文件路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--image-size', type=int, default=224, help='输入图像尺寸')
    
    # 模型参数
    parser.add_argument('--patch-size', type=int, default=16, help='图像块大小')
    parser.add_argument('--num-classes', type=int, default=10, help='分类类别数')
    parser.add_argument('--dim', type=int, default=768, help='Transformer维度')
    parser.add_argument('--depth', type=int, default=12, help='Transformer深度')
    parser.add_argument('--heads', type=int, default=12, help='多头注意力头数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'none'], help='学习率调度器')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比例')
    parser.add_argument('--loss-type', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'], help='损失函数类型')
    parser.add_argument('--grad-clip', type=float, default=None, help='梯度裁剪值')
    
    # 日志和输出参数
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--enable-wandb', action='store_true', help='启用Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='vit_training', help='W&B项目名称')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B实体(组织或个人账户)')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=[], help='W&B标签')
    parser.add_argument('--wandb-log-model', action='store_true', help='将模型保存到W&B')
    parser.add_argument('--save-model', action='store_true', help='保存模型')
    parser.add_argument('--model-dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔(批次)')
    
    # 数据增强参数
    parser.add_argument('--augmentation-preset', type=str, choices=['light', 'medium', 'heavy'], help='数据增强预设')
    
    # GPU相关参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    
    return parser.parse_args()


def main():
    """主函数，执行训练和评估流程"""
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"vit_training_{timestamp}"
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    logs_dir = os.path.join(experiment_dir, 'logs')
    metrics_dir = os.path.join(experiment_dir, 'metrics')
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    print(f"实验目录: {experiment_dir}")
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 确保数据目录路径正确
    data_dir = args.data_dir
    # 如果传入的是 'data'，自动修正为 'data/images/'
    if data_dir == 'data' or data_dir == 'data/':
        data_dir = 'data/images/'
    
    print(f"使用图像目录: {data_dir}")
    print(f"使用标注文件: {args.annotation_file}")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"图像目录 {data_dir} 不存在")
    if not os.path.exists(args.annotation_file):
        raise FileNotFoundError(f"标注文件 {args.annotation_file} 不存在")
    
    # 加载数据
    print("加载数据集...")
    img_size = (args.image_size, args.image_size)
    
    # 创建训练和验证数据加载器
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        anno_file=args.annotation_file,
        batch_size=args.batch_size,
        img_size=img_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        augmentation_preset=args.augmentation_preset
    )
    
    # 为测试集创建单独的数据加载器（使用验证集数据作为测试，因为我们没有单独的测试集创建函数）
    # 在实际项目中，应该创建单独的测试集
    test_loader = val_loader
    
    print(f"数据集加载完成: 训练批次={len(train_loader)}, 验证批次={len(val_loader)}")
    
    # 创建模型
    print("创建ViT模型...")
    model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        embed_dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        drop_rate=args.dropout
    )
    model.to(device)
    print(f"模型创建完成，参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建损失计算器
    loss_calculator = LossCalculator(loss_type=args.loss_type)
    
    # 创建优化器管理器
    optimizer_manager = OptimizerManager(
        optimizer_type=args.optimizer.lower(),
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler if args.scheduler != 'none' else None,
        scheduler_params={
            'T_max': args.epochs,
            'step_size': 30,
            'gamma': 0.1
        }
    )
    
    # 创建反向传播管理器
    backprop_manager = BackpropManager(grad_clip_value=args.grad_clip)
    
    # 准备Weights & Biases配置
    wandb_config = None
    if args.enable_wandb:
        print("准备Weights & Biases...")
        wandb_config = {
            # 模型参数
            'model_type': 'VisionTransformer',
            'image_size': args.image_size,
            'patch_size': args.patch_size,
            'num_classes': args.num_classes,
            'embed_dim': args.dim,
            'depth': args.depth,
            'num_heads': args.heads,
            'dropout': args.dropout,
            
            # 训练参数
            'optimizer': args.optimizer,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'scheduler': args.scheduler,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'loss_type': args.loss_type,
            
            # 数据参数
            'val_split': args.val_split,
            'augmentation': args.augmentation_preset
        }
    
    # 设置指标记录器
    metrics_logger = MetricsLogger(
        save_dir=logs_dir,
        experiment_name=experiment_name,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_config=wandb_config
    )
    
    # 如果启用了wandb，设置更多配置
    if args.enable_wandb and metrics_logger.wandb_initialized:
        if args.wandb_entity:
            wandb.run.entity = args.wandb_entity
        if args.wandb_tags:
            wandb.run.tags = args.wandb_tags
        
        # 记录代码版本信息
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        except:
            print("无法记录代码版本信息，可能git未安装或当前目录不是git仓库")
    

    
    # 创建训练循环
    training_loop = TrainingLoop(
        model=model,
        loss_calculator=loss_calculator,
        optimizer_manager=optimizer_manager,
        backprop_manager=backprop_manager,
        device=device,
        metrics_logger=metrics_logger
    )
    
    # 训练模型
    print(f"开始训练，共{args.epochs}轮...")
    train_metrics = training_loop.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
        checkpoint_dir=checkpoints_dir
    )
    
    # 保存训练指标
    metrics_logger.export_metrics(os.path.join(metrics_dir, 'training_metrics.json'), format='json')
    
    # 绘制训练历史
    if hasattr(training_loop, 'train_losses') and hasattr(training_loop, 'val_losses'):
        # 创建临时MetricsLogger来保存训练历史数据
        temp_metrics_logger = MetricsLogger(
            save_dir=metrics_dir,
            experiment_name=experiment_name,
            save_format='json'
        )
        
        # 添加每个epoch的数据
        for i, (loss, acc) in enumerate(zip(training_loop.train_losses, training_loop.train_accs)):
            epoch = i + 1
            temp_metrics_logger.log_train_metrics({'loss': loss, 'accuracy': acc}, epoch)
            
        # 添加验证数据
        for i, (val_loss, val_acc) in enumerate(zip(training_loop.val_losses, training_loop.val_accs)):
            epoch = i + 1
            temp_metrics_logger.log_eval_metrics({'val_loss': val_loss, 'val_accuracy': val_acc}, epoch)
        
        # 使用正确的参数调用绘图函数
        plot_training_history(
            metrics_logger=temp_metrics_logger,
            metrics=['loss', 'accuracy'],
            output_dir=metrics_dir,
            experiment_name=experiment_name,
        )
        
        # 将图表上传到W&B
        if args.enable_wandb and metrics_logger.wandb_initialized:
            for file in os.listdir(metrics_dir):
                if file.endswith('.png'):
                    image_path = os.path.join(metrics_dir, file)
                    wandb.log({f"final_plots/{file}": wandb.Image(image_path)})
    
    print(f"训练完成，指标已保存至 {metrics_dir}")
    
    # 评估模型
    print("在测试集上评估模型...")
    test_metrics = training_loop.evaluate(
        test_loader=test_loader,
        num_classes=args.num_classes,
        visualize_confusion_matrix=True,
        output_path=os.path.join(metrics_dir, 'confusion_matrix.png')
    )
    
    print(f"测试集评估结果:")
    print(f"  准确率: {test_metrics['test_accuracy']:.2f}%")
    print(f"  损失: {test_metrics['test_loss']:.4f}")
    
    # 检查详细指标是否存在
    if 'f1_score' in test_metrics:
        print(f"  F1分数: {test_metrics['f1_score']:.4f}")
    if 'precision' in test_metrics:
        print(f"  精确率: {test_metrics['precision']:.4f}%")
    if 'recall' in test_metrics:
        print(f"  召回率: {test_metrics['recall']:.4f}%")
    
    # 保存模型(可选)
    if args.save_model:
        model_path = os.path.join(args.model_dir, f"vit_model_{timestamp}.pth")
        os.makedirs(args.model_dir, exist_ok=True)
        model.save_model(model_path, save_optimizer=True, optimizer_state=optimizer_manager.state_dict())
        print(f"模型已保存至 {model_path}")
        
        # 保存模型到W&B
        if args.enable_wandb and metrics_logger.wandb_initialized:
            wandb.save(model_path)
    
    # 关闭wandb连接 - 移到评估后才关闭
    if args.enable_wandb and metrics_logger.wandb_initialized:
        metrics_logger.close()
    
    print("训练和评估流程完成!")


if __name__ == "__main__":
    main()