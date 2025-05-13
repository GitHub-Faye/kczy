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
    test_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    augmentation_preset: Optional[str] = None,
    augmentation_config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    参数:
        data_dir (str): 数据目录路径
        anno_file (str): 标注文件路径
        batch_size (int): 批量大小
        img_size (Tuple[int, int]): 图像大小
        val_split (float): 验证集比例
        test_split (float): 测试集比例
        num_workers (int): 数据加载线程数
        seed (int): 随机种子
        mean (List[float]): 归一化均值
        std (List[float]): 归一化标准差
        augmentation_preset (Optional[str]): 预设增强方案，可选['light', 'medium', 'heavy']
        augmentation_config (Optional[Dict]): 自定义增强配置
        
    返回:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证和测试数据加载器
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    
    # 获取训练和验证/测试转换
    train_transform = get_transforms(
        train=True, 
        img_size=img_size, 
        mean=mean, 
        std=std,
        augmentation_preset=augmentation_preset,
        augmentation_config=augmentation_config
    )
    eval_transform = get_transforms(train=False, img_size=img_size, mean=mean, std=std)
    
    # 创建完整数据集
    full_dataset = BaseDataset(data_dir, anno_file, transform=None)
    
    # 计算训练集、验证集和测试集大小
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size - test_size
    
    # 确保拆分比例合法
    if train_size <= 0:
        raise ValueError(f"训练集大小为{train_size}，无效！请调整val_split和test_split参数。")
    
    # 随机拆分数据集为三部分 (70%/20%/10% 默认)
    # 注意：我们需要按照先分离测试集，再分离验证集的顺序来保证测试集的纯度
    remaining_dataset, test_dataset = random_split(
        full_dataset, [dataset_size - test_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataset, val_dataset = random_split(
        remaining_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 应用相应的转换
    train_dataset = TransformWrapper(train_dataset, train_transform)
    val_dataset = TransformWrapper(val_dataset, eval_transform)
    test_dataset = TransformWrapper(test_dataset, eval_transform)
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,  # 使用自定义整理函数
    )
    
    return train_loader, val_loader, test_loader

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

def create_dataloaders_from_config(config: DatasetConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    根据配置对象创建训练、验证和测试数据加载器
    
    参数:
        config (DatasetConfig): 数据集配置对象
        
    返回:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证和测试数据加载器
    """
    return create_dataloaders(
        data_dir=config.data_dir,
        anno_file=config.anno_file,
        batch_size=config.batch_size,
        img_size=config.img_size,
        val_split=config.val_split,
        test_split=config.test_split,
        num_workers=config.num_workers,
        seed=config.seed,
        mean=config.mean,
        std=config.std,
        augmentation_preset=config.augmentation_preset,
        augmentation_config=config.augmentation_config.to_dict() if not config.augmentation_config.is_empty() else None
    )

def verify_dataset_splits(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    expected_train_ratio: float = 0.7,
    expected_val_ratio: float = 0.2,
    expected_test_ratio: float = 0.1,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    验证数据集拆分是否符合预期比例
    
    参数:
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        test_loader (DataLoader): 测试数据加载器
        expected_train_ratio (float): 预期训练集比例
        expected_val_ratio (float): 预期验证集比例
        expected_test_ratio (float): 预期测试集比例
        tolerance (float): 容差范围
        
    返回:
        Dict[str, Any]: 包含验证结果的字典
    """
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    total_size = train_size + val_size + test_size
    
    train_ratio = train_size / total_size
    val_ratio = val_size / total_size
    test_ratio = test_size / total_size
    
    is_valid = (
        abs(train_ratio - expected_train_ratio) <= tolerance and
        abs(val_ratio - expected_val_ratio) <= tolerance and
        abs(test_ratio - expected_test_ratio) <= tolerance
    )
    
    # 查看三个数据集的几个样本，确保数据完整性
    # 注意：这只是一个简单的示例检查，实际应用中可能需要更复杂的验证
    sample_check = {}
    
    # 检查训练集
    train_iter = iter(train_loader)
    train_sample_images, train_sample_targets = next(train_iter)
    sample_check['train'] = {
        'image_shape': tuple(train_sample_images.shape),
        'batch_size': train_sample_images.shape[0],
        'target_sample': train_sample_targets[0] if train_sample_targets else None
    }
    
    # 检查验证集
    val_iter = iter(val_loader)
    val_sample_images, val_sample_targets = next(val_iter)
    sample_check['val'] = {
        'image_shape': tuple(val_sample_images.shape),
        'batch_size': val_sample_images.shape[0],
        'target_sample': val_sample_targets[0] if val_sample_targets else None
    }
    
    # 检查测试集
    test_iter = iter(test_loader)
    test_sample_images, test_sample_targets = next(test_iter)
    sample_check['test'] = {
        'image_shape': tuple(test_sample_images.shape),
        'batch_size': test_sample_images.shape[0],
        'target_sample': test_sample_targets[0] if test_sample_targets else None
    }
    
    result = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'total_size': total_size,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'is_valid': is_valid,
        'sample_check': sample_check
    }
    
    return result 