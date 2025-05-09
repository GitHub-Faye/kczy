#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import TrainingConfig, ViTConfig
from src.models.vit import VisionTransformer
from src.models.train import TrainingLoop, LossCalculator, BackpropManager
from src.models.optimizer_manager import OptimizerManager

def generate_synthetic_data(num_samples=1000, img_size=32, num_classes=10, batch_size=32):
    """
    生成合成数据集用于测试
    
    参数:
        num_samples (int): 样本数量
        img_size (int): 图像大小
        num_classes (int): 类别数量
        batch_size (int): 批次大小
        
    返回:
        tuple: (train_loader, val_loader) 训练和验证数据加载器
    """
    print(f"正在生成合成数据: {num_samples}个样本，{img_size}x{img_size}图像，{num_classes}个类别")
    
    # 生成随机图像和标签
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # 创建数据集
    dataset = TensorDataset(images, labels)
    
    # 划分训练集和验证集
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"数据集创建完成: 训练集{train_size}样本，验证集{val_size}样本")
    
    return train_loader, val_loader

def test_training_loop_components():
    """测试训练循环的各个组件"""
    print("\n=== 测试训练循环组件 ===\n")
    
    # 创建一个小型模型
    model_config = ViTConfig.create_tiny(num_classes=10)
    model_config.img_size = 32
    model_config.patch_size = 8
    model = VisionTransformer.from_config(model_config)
    print("ViT模型创建成功")
    
    # 测试损失计算器
    loss_calculator = LossCalculator(loss_type='cross_entropy')
    print("损失计算器创建成功")
    
    # 测试优化器管理器
    optimizer_manager = OptimizerManager(
        optimizer_type='adam',
        model=model,
        lr=0.001,
        weight_decay=1e-4,
        scheduler_type='cosine',
        scheduler_params={'T_max': 10}
    )
    print("优化器管理器创建成功")
    print(f"当前学习率: {optimizer_manager.get_lr()}")
    
    # 测试反向传播管理器
    backprop_manager = BackpropManager(
        grad_clip_value=1.0,
        grad_clip_norm=1.0
    )
    print("反向传播管理器创建成功")
    
    # 测试TrainingLoop创建
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_loop = TrainingLoop(
        model=model,
        loss_calculator=loss_calculator,
        optimizer_manager=optimizer_manager,
        backprop_manager=backprop_manager,
        device=device
    )
    print(f"训练循环创建成功，使用设备: {device}")
    
    return training_loop, model, model_config

def test_from_config():
    """测试从配置创建训练循环"""
    print("\n=== 测试从配置创建训练循环 ===\n")
    
    # 创建训练配置
    training_config = TrainingConfig.create_fast_dev()
    training_config.batch_size = 32
    training_config.num_epochs = 3
    print("训练配置创建成功")
    
    # 创建模型配置
    model_config = ViTConfig.create_tiny(num_classes=10)
    model_config.img_size = 32
    model_config.patch_size = 8
    print("模型配置创建成功")
    
    # 创建模型
    model = VisionTransformer.from_config(model_config)
    print("模型创建成功")
    
    # 从配置创建训练循环
    training_loop = TrainingLoop.from_config(
        model=model,
        config=training_config
    )
    print("从配置创建训练循环成功")
    
    return training_loop, model, model_config, training_config

def run_training_test(training_loop, train_loader, val_loader, num_epochs=3):
    """运行训练测试"""
    print("\n=== 运行训练测试 ===\n")
    
    # 创建检查点目录
    checkpoint_dir = os.path.join('tests', 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 运行训练
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"test_model_{timestamp}.pt")
    
    history = training_loop.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        log_interval=1,
        early_stopping=True,
        early_stopping_patience=5
    )
    
    print(f"训练完成，模型保存到: {checkpoint_path}")
    
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='训练损失')
    if val_loader:
        plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='训练准确率')
    if val_loader:
        plt.plot(history['val_accuracy'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f"training_curves_{timestamp}.png"))
    print(f"训练曲线已保存到: {os.path.join(checkpoint_dir, f'training_curves_{timestamp}.png')}")
    
    return history, checkpoint_path

def test_evaluation(training_loop, val_loader, checkpoint_path=None, num_classes=10):
    """测试评估功能"""
    print("\n=== 测试评估功能 ===\n")
    
    # 创建输出目录
    output_dir = os.path.join('tests', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_metrics = training_loop.evaluate(
        test_loader=val_loader,
        num_classes=num_classes,
        visualize_confusion_matrix=True,
        output_path=os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    )
    
    print("评估完成，指标:")
    print(f"测试损失: {eval_metrics['test_loss']:.4f}")
    print(f"测试准确率: {eval_metrics['test_accuracy']:.2f}%")
    print(f"精确率: {eval_metrics['precision']:.2f}%")
    print(f"召回率: {eval_metrics['recall']:.2f}%")
    print(f"F1分数: {eval_metrics['f1_score']:.2f}%")
    
    # 测试评估模型功能
    if checkpoint_path:
        print("\n=== 测试从检查点加载模型并评估 ===\n")
        
        # 创建新模型
        model_config = ViTConfig.create_tiny(num_classes=num_classes)
        model_config.img_size = 32
        model_config.patch_size = 8
        model = VisionTransformer.from_config(model_config)
        
        # 使用evaluate_model
        eval_metrics = TrainingLoop.evaluate_model(
            test_loader=val_loader,
            model=model,
            checkpoint_path=checkpoint_path,
            loss_type='cross_entropy',
            num_classes=num_classes,
            visualize_confusion_matrix=True,
            output_path=os.path.join(output_dir, f"loaded_confusion_matrix_{timestamp}.png")
        )
        
        print("从检查点加载模型并评估完成")
    
    return eval_metrics

def main():
    """主函数"""
    print("=== 开始训练循环测试 ===")
    
    # 1. 测试训练循环组件
    # training_loop, model, model_config = test_training_loop_components()
    
    # 2. 测试从配置创建训练循环（推荐方式）
    training_loop, model, model_config, training_config = test_from_config()
    
    # 3. 生成合成数据
    train_loader, val_loader = generate_synthetic_data(
        num_samples=1000,
        img_size=model_config.img_size,
        num_classes=model_config.num_classes,
        batch_size=training_config.batch_size if 'training_config' in locals() else 32
    )
    
    # 4. 运行训练测试
    history, checkpoint_path = run_training_test(
        training_loop, 
        train_loader, 
        val_loader,
        num_epochs=training_config.num_epochs if 'training_config' in locals() else 3
    )
    
    # 5. 测试评估功能
    eval_metrics = test_evaluation(
        training_loop, 
        val_loader, 
        checkpoint_path,
        num_classes=model_config.num_classes
    )
    
    print("\n=== 测试完成 ===")
    print("所有训练循环组件测试通过")
    print("综合测试结果: 成功")

if __name__ == "__main__":
    main() 