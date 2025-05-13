#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练演示脚本，使用小型数据集进行快速训练和验证，展示完整训练流程
"""
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vit import VisionTransformer
from src.models.train import TrainingLoop, LossCalculator, BackpropManager
from src.models.optimizer_manager import OptimizerManager
from src.utils.metrics_logger import MetricsLogger
from src.utils.tensorboard_utils import start_tensorboard, check_tensorboard_running
from src.models.model_utils import save_checkpoint, save_model
from src.utils.config import TrainingConfig
from src.models.train_config import setup_tensorboard

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_dummy_dataset(num_samples=500, img_size=64, num_classes=10):
    """
    创建模拟数据集用于演示
    
    参数:
        num_samples (int): 样本数量
        img_size (int): 图像大小
        num_classes (int): 类别数量
        
    返回:
        torch.utils.data.TensorDataset: 模拟数据集
    """
    # 创建随机图像数据 (num_samples, 3, img_size, img_size)
    images = torch.randn(num_samples, 3, img_size, img_size)
    
    # 创建随机标签 (num_samples,)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return TensorDataset(images, labels)

def setup_experiment_dir(experiment_name=None):
    """
    设置实验目录
    
    参数:
        experiment_name (str): 实验名称
        
    返回:
        dict: 包含各种目录路径的字典
    """
    # 如果未提供实验名称，生成一个
    if not experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"demo_training_{timestamp}"
    
    # 创建基础目录
    base_dir = 'output'
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # 创建各子目录
    dirs = {
        'experiment_dir': experiment_dir,
        'checkpoint_dir': os.path.join(experiment_dir, 'checkpoints'),
        'log_dir': os.path.join(experiment_dir, 'logs'),
        'metrics_dir': os.path.join(experiment_dir, 'metrics'),
        'tensorboard_dir': os.path.join(experiment_dir, 'tensorboard')
    }
    
    # 确保所有目录都存在
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    logger.info(f"实验目录已设置: {experiment_dir}")
    return dirs, experiment_name

def create_demo_model(img_size=64, patch_size=8, num_classes=10, device='cpu'):
    """
    创建用于演示的ViT模型
    
    参数:
        img_size (int): 输入图像大小
        patch_size (int): 图像块大小
        num_classes (int): 分类类别数量
        device (str): 设备
        
    返回:
        src.models.vit.VisionTransformer: 模型
    """
    # 创建小型ViT模型
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,   # 较小的嵌入维度
        depth=4,         # 较少的Transformer层
        num_heads=4,     # 较少的注意力头
        mlp_ratio=2,     # 较小的MLP扩展比率
        drop_rate=0.1
    )
    
    model = model.to(device)
    logger.info(f"演示模型已创建: VisionTransformer - {sum(p.numel() for p in model.parameters())} 参数")
    return model

def main():
    """主函数，演示ViT模型训练"""
    # 参数设置
    img_size = 64
    patch_size = 8
    num_classes = 10
    batch_size = 32
    num_epochs = 5
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置实验目录
    dirs, experiment_name = setup_experiment_dir()
    
    # 创建模拟数据集
    logger.info("创建模拟数据集...")
    dataset = create_dummy_dataset(
        num_samples=800,  # 较小的数据集以加快演示
        img_size=img_size,
        num_classes=num_classes
    )
    
    # 划分数据集为训练、验证和测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"数据加载器已创建 - 训练: {len(train_loader)} 批次, 验证: {len(val_loader)} 批次, 测试: {len(test_loader)} 批次")
    
    # 创建模型
    model = create_demo_model(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        device=device
    )
    
    # 创建训练配置
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_type='adamw',
        scheduler_type='cosine',
        loss_type='cross_entropy',
        grad_clip_norm=1.0,
        device=str(device),  # 将设备对象转换为字符串
        enable_tensorboard=True,
        tensorboard_dir=dirs['tensorboard_dir'],
        log_histograms=True,
        log_images=True,
        checkpoint_dir=dirs['checkpoint_dir'],
        checkpoint_freq=1
    )
    
    # 打印训练配置摘要
    print("\n" + "=" * 50)
    print("演示训练配置摘要")
    print("=" * 50)
    print(f"模型类型: VisionTransformer")
    print(f"图像大小: {img_size}x{img_size}")
    print(f"图像块大小: {patch_size}")
    print(f"类别数量: {num_classes}")
    print(f"训练轮数: {num_epochs}")
    print(f"批大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"权重衰减: {weight_decay}")
    print(f"优化器: AdamW")
    print(f"学习率调度器: Cosine")
    print(f"损失函数: CrossEntropy")
    print(f"设备: {device}")
    print("=" * 50 + "\n")
    
    # 设置TensorBoard
    tensorboard_writer = setup_tensorboard(training_config)
    
    # 启动TensorBoard服务器
    tensorboard_port = 6006
    if not check_tensorboard_running(tensorboard_port):
        start_tensorboard(
            log_dir=dirs['tensorboard_dir'],
            port=tensorboard_port,
            background=True
        )
        logger.info(f"TensorBoard服务器已启动，可在 http://localhost:{tensorboard_port} 访问")
    
    # 创建指标记录器
    metrics_logger = MetricsLogger(
        save_dir=dirs['metrics_dir'],
        experiment_name=experiment_name,
        save_format='json',
        enable_tensorboard=True,
        tensorboard_log_dir=dirs['tensorboard_dir']
    )
    
    # 创建优化器管理器
    optimizer_manager = OptimizerManager(
        optimizer_type=training_config.optimizer_type,
        model=model,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        scheduler_type=training_config.scheduler_type,
        scheduler_params={'T_max': num_epochs}
    )
    
    # 创建损失计算器
    loss_calculator = LossCalculator(
        loss_type=training_config.loss_type
    )
    
    # 创建反向传播管理器
    backprop_manager = BackpropManager(
        grad_clip_value=training_config.grad_clip_value,
        grad_clip_norm=training_config.grad_clip_norm,
        grad_scaler=torch.cuda.amp.GradScaler() if training_config.use_mixed_precision else None
    )
    
    # 创建训练循环
    training_loop = TrainingLoop(
        model=model,
        loss_calculator=loss_calculator,
        optimizer_manager=optimizer_manager,
        backprop_manager=backprop_manager,
        device=device,
        metrics_logger=metrics_logger,
        log_histograms=training_config.log_histograms,
        log_images=training_config.log_images
    )
    
    # 开始训练
    logger.info("开始训练...")
    try:
        training_history = training_loop.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            checkpoint_dir=dirs['checkpoint_dir'],
            checkpoint_freq=1,
            log_interval=2,  # 更频繁地打印日志以便查看进度
            early_stopping=False,
            config=training_config
        )
        
        # 保存最终模型
        final_model_path = os.path.join(dirs['checkpoint_dir'], f"{experiment_name}_final_model.pth")
        save_model(model, final_model_path, metadata={
            'experiment_name': experiment_name,
            'epochs': num_epochs,
            'img_size': img_size,
            'patch_size': patch_size,
            'num_classes': num_classes
        })
        logger.info(f"最终模型已保存到: {final_model_path}")
        
        # 在测试集上评估模型
        logger.info("在测试集上评估模型...")
        test_results = training_loop.evaluate(
            test_loader=test_loader,
            num_classes=num_classes,
            visualize_confusion_matrix=True,
            output_path=os.path.join(dirs['experiment_dir'], f"{experiment_name}_confusion_matrix.png")
        )
        
        # 打印测试结果
        print("\n" + "=" * 50)
        print("测试集评估结果")
        print("=" * 50)
        for metric, value in test_results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        print("=" * 50 + "\n")
        
        logger.info("演示训练完成！")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.exception(f"训练过程中出错: {e}")
    finally:
        # 关闭TensorBoard写入器
        if tensorboard_writer:
            tensorboard_writer.close()
        
        # 关闭指标记录器
        metrics_logger.close()
        
        logger.info(f"演示脚本执行完毕，输出目录: {dirs['experiment_dir']}")
        logger.info(f"您可以使用以下命令查看训练指标曲线：")
        logger.info(f"tensorboard --logdir={dirs['tensorboard_dir']}")

if __name__ == "__main__":
    main() 