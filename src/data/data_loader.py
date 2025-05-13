import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from typing import Tuple, Dict, Optional, List, Union, Any

from .dataset import BaseDataset
from .augmentation import (
    create_transform_from_preset,
    create_augmentation_pipeline,
    AugmentationPresets
)
from .config import DatasetConfig

def get_transforms(
    train: bool = True,
    img_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    augmentation_preset: Optional[str] = None,
    augmentation_config: Optional[Dict] = None
) -> transforms.Compose:
    """
    获取数据转换操作
    
    参数:
        train (bool): 是否为训练模式
        img_size (Tuple[int, int]): 图像大小
        mean (List[float]): 归一化均值
        std (List[float]): 归一化标准差
        augmentation_preset (Optional[str]): 预设增强方案，可选['light', 'medium', 'heavy']
        augmentation_config (Optional[Dict]): 自定义增强配置
        
    返回:
        transforms.Compose: 图像转换组合
    """
    if train:
        if augmentation_preset:
            # 使用预设增强配置
            return create_transform_from_preset(
                preset_name=augmentation_preset,
                img_size=img_size,
                mean=mean,
                std=std
            )
        elif augmentation_config:
            # 使用自定义增强配置
            base_transforms = [transforms.Resize(img_size)]
            final_transforms = [transforms.Normalize(mean=mean, std=std)]
            
            return create_augmentation_pipeline(
                rotate=augmentation_config.get('rotate'),
                flip=augmentation_config.get('flip'),
                color_jitter=augmentation_config.get('color_jitter'),
                blur=augmentation_config.get('blur'),
                noise=augmentation_config.get('noise'),
                elastic=augmentation_config.get('elastic'),
                erasing=augmentation_config.get('erasing'),
                additional_transforms=base_transforms,
                final_transforms=final_transforms
            )
        else:
            # 默认的基础增强
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    else:
        # 验证/测试时的基本转换
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform

# 将TransformWrapper类提升为模块级类
class TransformWrapper(Dataset):
    """
    数据集包装器，用于应用图像转换
    
    参数:
        dataset: 原始数据集
        transform: 要应用的转换
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

def create_dataloaders(
    data_dir: str,
    anno_file: str,
    batch_size: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    augmentation_preset: Optional[str] = None,
    augmentation_config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    参数:
        data_dir (str): 数据目录路径
        anno_file (str): 标注文件路径
        batch_size (int): 批量大小
        img_size (Tuple[int, int]): 图像大小
        val_split (float): 验证集比例
        num_workers (int): 数据加载线程数
        seed (int): 随机种子
        mean (List[float]): 归一化均值
        std (List[float]): 归一化标准差
        augmentation_preset (Optional[str]): 预设增强方案，可选['light', 'medium', 'heavy']
        augmentation_config (Optional[Dict]): 自定义增强配置
        
    返回:
        Tuple[DataLoader, DataLoader]: 训练数据加载器和验证数据加载器
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    
    # 获取训练和验证转换
    train_transform = get_transforms(
        train=True, 
        img_size=img_size, 
        mean=mean, 
        std=std,
        augmentation_preset=augmentation_preset,
        augmentation_config=augmentation_config
    )
    val_transform = get_transforms(train=False, img_size=img_size, mean=mean, std=std)
    
    # 创建完整数据集
    full_dataset = BaseDataset(data_dir, anno_file, transform=None)
    
    # 计算训练集和验证集大小
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # 随机拆分数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 应用相应的转换
    train_dataset = TransformWrapper(train_dataset, train_transform)
    val_dataset = TransformWrapper(val_dataset, val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,  # 使用自定义整理函数
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,  # 使用自定义整理函数
    )
    
    return train_loader, val_loader

def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    自定义批次整理函数，处理不同大小的图像和目标
    
    参数:
        batch (List[Tuple[torch.Tensor, Dict]]): 批次数据
        
    返回:
        Tuple[torch.Tensor, List[Dict]]: 整理后的批次
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # 将图像堆叠为批次
    images = torch.stack(images, dim=0)
    
    return images, targets

def create_dataloaders_from_config(config: DatasetConfig) -> Tuple[DataLoader, DataLoader]:
    """
    根据配置对象创建训练和验证数据加载器
    
    参数:
        config (DatasetConfig): 数据集配置对象
        
    返回:
        Tuple[DataLoader, DataLoader]: 训练数据加载器和验证数据加载器
    """
    return create_dataloaders(
        data_dir=config.data_dir,
        anno_file=config.anno_file,
        batch_size=config.batch_size,
        img_size=config.img_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        seed=config.seed,
        mean=config.mean,
        std=config.std,
        augmentation_preset=config.augmentation_preset,
        augmentation_config=config.augmentation_config.to_dict() if not config.augmentation_config.is_empty() else None
    ) 