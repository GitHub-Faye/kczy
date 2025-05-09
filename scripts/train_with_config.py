#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import TrainingConfig, ViTConfig
from src.models.vit import VisionTransformer
from src.models.train import TrainingLoop
from src.data.datasets import ImageClassificationDataset

def main():
    """
    使用TrainingConfig进行模型训练的主函数
    """
    parser = argparse.ArgumentParser(description='使用配置训练ViT模型')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--config_type', type=str, default='default',
                        choices=['default', 'fast_dev', 'high_performance'],
                        help='训练配置类型')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['tiny', 'small', 'base', 'large', 'huge'],
                        help='ViT模型大小')
    parser.add_argument('--num_classes', type=int, default=10, help='分类类别数量')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='模型保存目录')
    parser.add_argument('--config_path', type=str, help='自定义训练配置文件路径')
    args = parser.parse_args()
    
    # 确保模型保存目录存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 创建或加载训练配置
    if args.config_path:
        try:
            training_config = TrainingConfig.load(args.config_path)
            print(f"从{args.config_path}加载训练配置")
        except Exception as e:
            print(f"加载配置失败: {e}")
            print("使用默认配置")
            training_config = TrainingConfig.create_default()
    else:
        # 根据选择的类型创建配置
        if args.config_type == 'fast_dev':
            training_config = TrainingConfig.create_fast_dev()
        elif args.config_type == 'high_performance':
            training_config = TrainingConfig.create_high_performance()
        else:
            training_config = TrainingConfig.create_default()
        
        print(f"使用{args.config_type}训练配置")
    
    # 创建ViT模型配置
    if args.model_size == 'tiny':
        model_config = ViTConfig.create_tiny(num_classes=args.num_classes)
    elif args.model_size == 'small':
        model_config = ViTConfig.create_small(num_classes=args.num_classes)
    elif args.model_size == 'base':
        model_config = ViTConfig.create_base(num_classes=args.num_classes)
    elif args.model_size == 'large':
        model_config = ViTConfig.create_large(num_classes=args.num_classes)
    elif args.model_size == 'huge':
        model_config = ViTConfig.create_huge(num_classes=args.num_classes)
    
    print(f"使用ViT-{args.model_size.capitalize()}模型配置")
    
    # 创建模型
    model = VisionTransformer(model_config)
    print(f"创建ViT模型: {args.model_size.capitalize()}, 类别数: {args.num_classes}")
    
    # 加载数据集
    try:
        # 这里假设数据集已经预处理为ImageClassificationDataset可以直接加载的格式
        dataset = ImageClassificationDataset(
            data_dir=args.data_dir,
            img_size=model_config.img_size
        )
        
        # 划分训练集和验证集
        train_size = int((1 - training_config.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory
        )
        
        print(f"数据集加载完成，训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("如果数据集格式有误，请确保数据已经按照正确的方式预处理")
        return
    
    # 从配置创建训练循环
    training_loop = TrainingLoop.from_config(
        model=model,
        config=training_config
    )
    
    # 设置检查点路径
    checkpoint_path = os.path.join(args.save_dir, f"vit_{args.model_size}_final.pt")
    checkpoint_dir = os.path.join(args.save_dir, f"vit_{args.model_size}_checkpoints")
    
    # 记录配置
    model_config_path = os.path.join(args.save_dir, f"vit_{args.model_size}_model_config.json")
    training_config_path = os.path.join(args.save_dir, f"vit_{args.model_size}_training_config.json")
    model_config.save(model_config_path)
    training_config.save(training_config_path)
    
    print(f"模型配置已保存到: {model_config_path}")
    print(f"训练配置已保存到: {training_config_path}")
    
    # 开始训练
    print("\n=== 开始训练 ===\n")
    history = training_loop.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config.num_epochs,
        checkpoint_path=checkpoint_path,
        log_interval=training_config.log_freq,
        early_stopping=training_config.early_stopping,
        early_stopping_patience=training_config.early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=training_config.checkpoint_freq
    )
    
    print(f"\n=== 训练完成，最终模型已保存到: {checkpoint_path} ===")
    
    # 训练后评估
    print("\n=== 最终模型评估 ===\n")
    eval_metrics = training_loop.evaluate(
        test_loader=val_loader,
        num_classes=args.num_classes,
        visualize_confusion_matrix=True,
        output_path=os.path.join(args.save_dir, f"vit_{args.model_size}_confusion_matrix.png")
    )
    
    print("\n=== 全部完成 ===")

if __name__ == "__main__":
    main() 