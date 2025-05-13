"""
自定义数据集示例
展示如何使用配置文件创建自定义数据集并使用配置加载数据
"""

import os
import argparse
import torch
from typing import Tuple, Dict, Any

from .config import DatasetConfig, AugmentationConfig
from .data_loader import create_dataloaders_from_config
    
def load_dataset_from_config(config_path: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    从配置文件加载数据集
    
    参数:
        config_path (str): 配置文件路径
        
    返回:
        Tuple[DataLoader, DataLoader]: 训练和验证数据加载器
    """
    # 加载配置
    config = DatasetConfig.load(config_path)
    print(f"已加载数据集配置: {config.name}")
    
    # 从配置创建数据加载器
    return create_dataloaders_from_config(config)

def create_custom_dataset_config(
    name: str,
    data_dir: str,
    anno_file: str,
    output_path: str,
    preset: str = None,
    advanced_aug: bool = False
) -> DatasetConfig:
    """
    创建并保存自定义数据集配置
    
    参数:
        name (str): 数据集名称
        data_dir (str): 数据目录路径
        anno_file (str): 标注文件路径
        output_path (str): 输出配置文件路径
        preset (str): 使用的预设 ('light', 'medium', 'heavy')
        advanced_aug (bool): 是否使用高级增强
    
    返回:
        DatasetConfig: 创建的配置对象
    """
    if advanced_aug:
        # 创建高级增强配置
        aug_config = AugmentationConfig(
            rotate={
                'degrees': 15,
                'expand': True,
                'fill': 0,
                'p': 0.7
            },
            flip={
                'horizontal': True,
                'vertical': False,
                'p': 0.5
            },
            color_jitter={
                'brightness': [0.8, 1.2],
                'contrast': [0.8, 1.2],
                'saturation': [0.8, 1.2],
                'hue': [-0.1, 0.1],
                'p': 0.7
            },
            blur={
                'blur_type': 'gaussian',
                'radius': [0.5, 1.5],
                'p': 0.3
            }
        )
        
        # 创建带高级增强的配置
        config = DatasetConfig(
            name=name,
            data_dir=data_dir,
            anno_file=anno_file,
            img_size=(384, 384),
            batch_size=16,
            augmentation_config=aug_config
        )
    else:
        # 使用预设的配置
        config = DatasetConfig(
            name=name,
            data_dir=data_dir,
            anno_file=anno_file,
            img_size=(224, 224),
            batch_size=32,
            augmentation_preset=preset or 'medium'
        )
    
    # 保存配置到文件
    config.save(output_path)
    print(f"配置已保存到: {output_path}")
    
    return config

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='自定义数据集配置示例')
    
    parser.add_argument('--mode', type=str, choices=['create', 'load'], default='create',
                        help='运行模式: 创建配置 (create) 或 加载配置 (load)')
    
    parser.add_argument('--name', type=str, default='MyCustomDataset',
                        help='数据集名称')
    parser.add_argument('--data-dir', type=str, default='data/images',
                        help='数据目录路径')
    parser.add_argument('--anno-file', type=str, default='data/annotations.csv',
                        help='标注文件路径')
    
    parser.add_argument('--config-path', type=str, default='data/examples/my_dataset_config.yaml',
                        help='配置文件路径 (创建或加载)')
    
    parser.add_argument('--preset', type=str, choices=['light', 'medium', 'heavy'], default='medium',
                        help='数据增强预设')
    parser.add_argument('--advanced-aug', action='store_true',
                        help='使用高级增强配置 (会忽略预设)')
    
    args = parser.parse_args()
    
    if args.mode == 'create':
        create_custom_dataset_config(
            name=args.name,
            data_dir=args.data_dir,
            anno_file=args.anno_file,
            output_path=args.config_path,
            preset=args.preset,
            advanced_aug=args.advanced_aug
        )
    else:  # load
        train_loader, val_loader = load_dataset_from_config(args.config_path)
        print(f"已加载训练集 ({len(train_loader.dataset)} 样本) 和验证集 ({len(val_loader.dataset)} 样本)")
        
        # 打印一个批次的形状
        for images, targets in train_loader:
            print(f"批次形状: {images.shape}")
            print(f"目标数量: {len(targets)}")
            break

if __name__ == '__main__':
    main() 