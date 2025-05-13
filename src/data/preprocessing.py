import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from PIL import Image
import torchvision.transforms as transforms

# 数据标准化函数
def standardize_data(data: torch.Tensor, 
                     mean: Optional[torch.Tensor] = None, 
                     std: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    对数据进行标准化处理 (z-score标准化)
    
    参数:
        data (torch.Tensor): 输入数据
        mean (torch.Tensor, 可选): 均值，如果为None则计算data的均值
        std (torch.Tensor, 可选): 标准差，如果为None则计算data的标准差
        
    返回:
        torch.Tensor: 标准化后的数据
    """
    if mean is None:
        mean = torch.mean(data, dim=0)
    if std is None:
        std = torch.std(data, dim=0)
    
    # 避免除以零
    std = torch.clamp(std, min=1e-8)
    
    return (data - mean) / std

def normalize_data(data: torch.Tensor, 
                   min_val: Optional[torch.Tensor] = None, 
                   max_val: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    对数据进行归一化处理 (Min-Max归一化)
    
    参数:
        data (torch.Tensor): 输入数据
        min_val (torch.Tensor, 可选): 最小值，如果为None则计算data的最小值
        max_val (torch.Tensor, 可选): 最大值，如果为None则计算data的最大值
        
    返回:
        torch.Tensor: 归一化后的数据，范围[0,1]
    """
    if min_val is None:
        min_val = torch.min(data, dim=0)[0]
    if max_val is None:
        max_val = torch.max(data, dim=0)[0]
    
    # 避免除以零
    divisor = max_val - min_val
    divisor = torch.clamp(divisor, min=1e-8)
    
    return (data - min_val) / divisor

# 缺失值处理函数
def fill_missing_values(data: pd.DataFrame, 
                        strategy: str = 'mean', 
                        fill_values: Optional[Dict] = None) -> pd.DataFrame:
    """
    填充DataFrame中的缺失值
    
    参数:
        data (pd.DataFrame): 输入数据
        strategy (str): 填充策略，可选['mean', 'median', 'mode', 'constant']
        fill_values (Dict, 可选): 当strategy='constant'时，指定各列填充值
        
    返回:
        pd.DataFrame: 填充缺失值后的数据
    """
    df = data.copy()
    
    if strategy == 'constant' and fill_values is not None:
        for col, value in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
    elif strategy == 'mean':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(df[col].median())
    elif strategy == 'mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        raise ValueError(f"不支持的填充策略: {strategy}")
    
    return df

def fill_missing_tensor(data: torch.Tensor, 
                        mask: torch.Tensor, 
                        value: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
    """
    填充张量中的缺失值
    
    参数:
        data (torch.Tensor): 输入数据
        mask (torch.Tensor): 布尔掩码，True表示缺失
        value (float 或 torch.Tensor): 填充值
        
    返回:
        torch.Tensor: 填充缺失值后的数据
    """
    result = data.clone()
    result[mask] = value
    return result

# 数据缩放函数
class MinMaxScaler:
    """最小-最大缩放器，将数据缩放到指定范围"""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """
        初始化缩放器
        
        参数:
            feature_range (Tuple[float, float]): 目标范围，默认为(0,1)
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.min_target, self.max_target = feature_range
    
    def fit(self, data: torch.Tensor) -> 'MinMaxScaler':
        """
        计算缩放参数
        
        参数:
            data (torch.Tensor): 输入数据
            
        返回:
            self: 当前对象
        """
        self.min_ = torch.min(data, dim=0)[0]
        self.max_ = torch.max(data, dim=0)[0]
        
        # 避免除以零
        data_range = self.max_ - self.min_
        data_range = torch.clamp(data_range, min=1e-8)
        
        self.scale_ = (self.max_target - self.min_target) / data_range
        
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用缩放
        
        参数:
            data (torch.Tensor): 输入数据
            
        返回:
            torch.Tensor: 缩放后的数据
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("尚未拟合缩放器，请先调用fit方法")
            
        scaled_data = self.min_target + self.scale_ * (data - self.min_)
        return scaled_data
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        拟合并应用缩放
        
        参数:
            data (torch.Tensor): 输入数据
            
        返回:
            torch.Tensor: 缩放后的数据
        """
        return self.fit(data).transform(data)

class StandardScaler:
    """标准化缩放器，将数据转换为均值为0，标准差为1"""
    
    def __init__(self):
        """初始化缩放器"""
        self.mean_ = None
        self.std_ = None
    
    def fit(self, data: torch.Tensor) -> 'StandardScaler':
        """
        计算缩放参数
        
        参数:
            data (torch.Tensor): 输入数据
            
        返回:
            self: 当前对象
        """
        self.mean_ = torch.mean(data, dim=0)
        self.std_ = torch.std(data, dim=0)
        
        # 避免除以零
        self.std_ = torch.clamp(self.std_, min=1e-8)
        
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用缩放
        
        参数:
            data (torch.Tensor): 输入数据
            
        返回:
            torch.Tensor: 缩放后的数据
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("尚未拟合缩放器，请先调用fit方法")
            
        return (data - self.mean_) / self.std_
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        拟合并应用缩放
        
        参数:
            data (torch.Tensor): 输入数据
            
        返回:
            torch.Tensor: 缩放后的数据
        """
        return self.fit(data).transform(data)

# 异常值检测与处理
def detect_outliers_iqr(data: torch.Tensor, 
                         factor: float = 1.5) -> torch.Tensor:
    """
    使用IQR方法检测异常值
    
    参数:
        data (torch.Tensor): 输入数据
        factor (float): IQR系数，默认为1.5
        
    返回:
        torch.Tensor: 布尔掩码，True表示异常值
    """
    q1 = torch.quantile(data, 0.25, dim=0)
    q3 = torch.quantile(data, 0.75, dim=0)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

def remove_outliers(data: torch.Tensor, 
                    outlier_mask: torch.Tensor) -> torch.Tensor:
    """
    移除异常值
    
    参数:
        data (torch.Tensor): 输入数据
        outlier_mask (torch.Tensor): 异常值掩码，True表示异常值
        
    返回:
        torch.Tensor: 移除异常值后的数据
    """
    # 计算每行是否包含异常值
    row_has_outlier = torch.any(outlier_mask, dim=1)
    
    # 保留不包含异常值的行
    return data[~row_has_outlier]

def clip_outliers(data: torch.Tensor, 
                  lower_bound: Optional[Union[float, torch.Tensor]] = None, 
                  upper_bound: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
    """
    裁剪异常值
    
    参数:
        data (torch.Tensor): 输入数据
        lower_bound (float 或 torch.Tensor, 可选): 下限
        upper_bound (float 或 torch.Tensor, 可选): 上限
        
    返回:
        torch.Tensor: 裁剪异常值后的数据
    """
    result = data.clone()
    
    if lower_bound is not None:
        result = torch.clamp(result, min=lower_bound)
    if upper_bound is not None:
        result = torch.clamp(result, max=upper_bound)
        
    return result

# 预处理管道
class PreprocessingPipeline:
    """数据预处理管道，将多个预处理步骤组合在一起"""
    
    def __init__(self, steps: List[Tuple[str, Callable]]):
        """
        初始化预处理管道
        
        参数:
            steps (List[Tuple[str, Callable]]): 预处理步骤列表，
                                              每个步骤是(名称, 处理函数)对
        """
        self.steps = steps
        
    def __call__(self, data: Any) -> Any:
        """
        应用预处理管道
        
        参数:
            data (Any): 输入数据
            
        返回:
            Any: 处理后的数据
        """
        result = data
        for _, transform in self.steps:
            result = transform(result)
        return result
    
    def add_step(self, name: str, transform: Callable) -> None:
        """
        添加预处理步骤
        
        参数:
            name (str): 步骤名称
            transform (Callable): 处理函数
        """
        self.steps.append((name, transform))

# 图像特定预处理函数
def create_image_preprocessing_pipeline(
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    resize: Optional[Tuple[int, int]] = None,
    additional_transforms: Optional[List[Callable]] = None
) -> transforms.Compose:
    """
    创建图像预处理管道
    
    参数:
        normalize (bool): 是否进行标准化
        mean (List[float]): 归一化均值
        std (List[float]): 归一化标准差
        resize (Tuple[int, int], 可选): 调整大小的目标尺寸
        additional_transforms (List[Callable], 可选): 额外的转换
        
    返回:
        transforms.Compose: 图像转换组合
    """
    transform_list = []
    
    # 调整大小
    if resize is not None:
        transform_list.append(transforms.Resize(resize))
    
    # 添加额外的转换
    if additional_transforms is not None:
        transform_list.extend(additional_transforms)
    
    # 转换为Tensor
    transform_list.append(transforms.ToTensor())
    
    # 标准化
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)

def normalize_image(
    image: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    归一化图像
    
    参数:
        image (torch.Tensor): 输入图像，形状为(C, H, W)
        mean (List[float]): 各通道均值
        std (List[float]): 各通道标准差
        
    返回:
        torch.Tensor: 归一化后的图像
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"图像必须是torch.Tensor类型，而不是{type(image)}")
    
    if image.dim() != 3:
        raise ValueError(f"图像必须是3维张量(C, H, W)，而不是{image.dim()}维")
    
    # 确保mean和std是tensor并且维度正确
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    
    # 应用归一化：(x - mean) / std
    normalized_image = (image - mean_tensor) / std_tensor
    
    return normalized_image

def denormalize_image(
    image: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    将归一化后的图像转换回原始范围，用于可视化
    
    参数:
        image (torch.Tensor): 归一化后的图像，形状为(C, H, W)
        mean (List[float]): 各通道均值
        std (List[float]): 各通道标准差
        
    返回:
        torch.Tensor: 反归一化后的图像
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"图像必须是torch.Tensor类型，而不是{type(image)}")
    
    if image.dim() != 3:
        raise ValueError(f"图像必须是3维张量(C, H, W)，而不是{image.dim()}维")
    
    # 确保mean和std是tensor并且维度正确
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    
    # 应用反归一化：x * std + mean
    denormalized_image = image * std_tensor + mean_tensor
    
    # 裁剪到[0, 1]范围
    denormalized_image = torch.clamp(denormalized_image, 0, 1)
    
    return denormalized_image 