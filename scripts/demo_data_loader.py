#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载器演示脚本
展示如何使用更新的数据加载器功能，支持训练、验证和测试三部分数据集
"""

import os
import sys
import argparse
import torch
from typing import Dict, Any
import matplotlib.pyplot as plt

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    create_dataloaders,
    create_dataloaders_from_config,
    verify_dataset_splits,
    DatasetConfig
)

def show_sample_images(data_loader, title, num_samples=5):
    """显示数据集中的样本图像"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle(title)
    
    # 获取一个批次的数据
    data_iter = iter(data_loader)
    images, targets = next(data_iter)
    
    # 显示图像
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()  # 从CHW转换为HWC
        
        # 反标准化图像以便更好地显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        img = img * std.numpy() + mean.numpy()
        img = img.clip(0, 1)
        
        axes[i].imshow(img)
        target_info = f"bbox: {targets[i]['boxes'].tolist()[0]}" if 'boxes' in targets[i] else ""
        axes[i].set_title(f"Label: {targets[i]['labels'].item()}\n{target_info}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def main(args):
    """主函数"""
    
    if args.use_config:
        # 从配置文件加载数据集
        try:
            config = DatasetConfig.load(args.config_path)
            print(f"使用配置加载数据集: {config.name}")
        except FileNotFoundError:
            # 如果配置文件不存在，创建一个默认配置
            print(f"配置文件 {args.config_path} 不存在，创建默认配置")
            config = DatasetConfig(
                name="示例数据集",
                data_dir=args.data_dir,
                anno_file=args.anno_file,
                val_split=args.val_split,
                test_split=args.test_split,
                batch_size=args.batch_size
            )
            # 保存配置
            os.makedirs(os.path.dirname(args.config_path), exist_ok=True)
            config.save(args.config_path)
            print(f"已保存默认配置到: {args.config_path}")
            
        # 使用配置创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders_from_config(config)
    else:
        # 直接从参数创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            anno_file=args.anno_file,
            batch_size=args.batch_size,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed
        )
    
    # 打印数据集信息
    print(f"数据集已加载:")
    print(f" - 训练集: {len(train_loader.dataset)} 样本")
    print(f" - 验证集: {len(val_loader.dataset)} 样本")
    print(f" - 测试集: {len(test_loader.dataset)} 样本")
    
    # 验证数据集拆分
    verification = verify_dataset_splits(
        train_loader, 
        val_loader, 
        test_loader,
        expected_train_ratio=1-args.val_split-args.test_split,
        expected_val_ratio=args.val_split,
        expected_test_ratio=args.test_split
    )
    
    print(f"\n拆分验证结果:")
    print(f" - 训练集比例: {verification['train_ratio']:.4f} (预期: {1-args.val_split-args.test_split:.4f})")
    print(f" - 验证集比例: {verification['val_ratio']:.4f} (预期: {args.val_split:.4f})")
    print(f" - 测试集比例: {verification['test_ratio']:.4f} (预期: {args.test_split:.4f})")
    print(f" - 拆分有效性: {'通过' if verification['is_valid'] else '未通过'}")
    
    # 显示样本图像
    if args.show_samples:
        train_fig = show_sample_images(train_loader, "训练集样本")
        val_fig = show_sample_images(val_loader, "验证集样本")
        test_fig = show_sample_images(test_loader, "测试集样本")
        
        # 保存图像
        if args.save_samples:
            os.makedirs(args.output_dir, exist_ok=True)
            train_fig.savefig(os.path.join(args.output_dir, "train_samples.png"))
            val_fig.savefig(os.path.join(args.output_dir, "val_samples.png"))
            test_fig.savefig(os.path.join(args.output_dir, "test_samples.png"))
            print(f"样本图像已保存到: {args.output_dir}")
        
        plt.show()
    
    print("演示完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据加载器演示")
    
    # 数据相关参数
    parser.add_argument('--data-dir', type=str, default='data/images',
                        help='图像数据目录路径')
    parser.add_argument('--anno-file', type=str, default='data/annotations.csv',
                        help='标注文件路径')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
                        
    # 配置相关参数
    parser.add_argument('--use-config', action='store_true',
                        help='是否使用配置文件')
    parser.add_argument('--config-path', type=str, default='data/examples/dataset_demo_config.yaml',
                        help='配置文件路径')
                        
    # 显示相关参数
    parser.add_argument('--show-samples', action='store_true',
                        help='是否显示样本图像')
    parser.add_argument('--save-samples', action='store_true',
                        help='是否保存样本图像')
    parser.add_argument('--output-dir', type=str, default='temp_metrics/plots',
                        help='输出目录')
    
    args = parser.parse_args()
    main(args) 