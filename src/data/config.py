import os
import json
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """
    数据增强配置类
    
    参数:
        rotate (Optional[Dict]): 旋转增强配置
        flip (Optional[Dict]): 翻转增强配置
        color_jitter (Optional[Dict]): 颜色抖动增强配置
        blur (Optional[Dict]): 模糊增强配置
        noise (Optional[Dict]): 噪声增强配置
        elastic (Optional[Dict]): 弹性变换配置
        erasing (Optional[Dict]): 随机擦除配置
    """
    rotate: Optional[Dict] = None
    flip: Optional[Dict] = None
    color_jitter: Optional[Dict] = None
    blur: Optional[Dict] = None
    noise: Optional[Dict] = None
    elastic: Optional[Dict] = None
    erasing: Optional[Dict] = None
    
    def is_empty(self) -> bool:
        """检查配置是否为空"""
        return all(v is None for v in self.__dict__.values())

@dataclass
class DatasetConfig:
    """
    数据集配置类
    
    参数:
        name (str): 数据集名称
        data_dir (str): 数据目录路径
        anno_file (str): 标注文件路径
        img_size (Tuple[int, int]): 图像尺寸
        batch_size (int): 批量大小
        val_split (float): 验证集划分比例
        test_split (float): 测试集划分比例
        num_workers (int): 数据加载线程数
        augmentation_preset (Optional[str]): 预设增强方案
        augmentation_config (AugmentationConfig): 自定义增强配置
        mean (List[float]): 归一化均值
        std (List[float]): 归一化标准差
        seed (int): 随机种子
        dev_mode (bool): 开发模式，跳过文件存在性检查
    """
    name: str
    data_dir: str
    anno_file: str
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 16
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 4
    augmentation_preset: Optional[str] = None
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    seed: int = 42
    dev_mode: bool = False
    
    def __post_init__(self):
        """初始化后进行验证"""
        self.validate()
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        返回:
            bool: 配置是否有效
        
        异常:
            ValueError: 当配置无效时抛出
        """
        # 仅在非开发模式下验证文件存在性
        if not self.dev_mode:
            # 验证数据目录存在
            if not os.path.exists(self.data_dir):
                raise ValueError(f"数据目录不存在: {self.data_dir}")
            
            # 验证标注文件存在
            if not os.path.exists(self.anno_file):
                raise ValueError(f"标注文件不存在: {self.anno_file}")
        
        # 验证图像尺寸
        if not isinstance(self.img_size, tuple) or len(self.img_size) != 2:
            raise ValueError(f"图像尺寸必须为二元组: {self.img_size}")
        
        # 验证批量大小
        if self.batch_size <= 0:
            raise ValueError(f"批量大小必须为正数: {self.batch_size}")
        
        # 验证拆分比例
        if not 0 < self.val_split < 1:
            raise ValueError(f"验证集划分比例必须在0到1之间: {self.val_split}")
            
        if not 0 < self.test_split < 1:
            raise ValueError(f"测试集划分比例必须在0到1之间: {self.test_split}")
            
        # 确保拆分比例总和不超过1
        train_split = 1 - self.val_split - self.test_split
        if train_split <= 0:
            raise ValueError(f"拆分比例总和超过1，无法分配训练集: val_split={self.val_split}, test_split={self.test_split}")
        
        # 验证数据加载线程数
        if self.num_workers < 0:
            raise ValueError(f"数据加载线程数必须为非负数: {self.num_workers}")
        
        # 验证增强预设有效性
        if self.augmentation_preset and self.augmentation_preset not in ["light", "medium", "heavy"]:
            raise ValueError(f"无效的增强预设: {self.augmentation_preset}, 可选值为 'light', 'medium', 'heavy'")
        
        # 验证增强预设和自定义增强配置不能同时存在
        if self.augmentation_preset and not self.augmentation_config.is_empty():
            logger.warning("同时指定了增强预设和自定义增强配置，将使用增强预设")
        
        return True
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        返回:
            Dict: 包含配置信息的字典
        """
        # 使用dataclasses的asdict函数将数据类转为字典
        config_dict = asdict(self)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DatasetConfig':
        """
        从字典创建配置
        
        参数:
            config_dict (Dict): 配置字典
            
        返回:
            DatasetConfig: 配置对象
        """
        # 提取并创建增强配置
        aug_config_dict = config_dict.pop('augmentation_config', {})
        # 使用字典展开操作创建增强配置对象
        aug_config = AugmentationConfig(**aug_config_dict)
        
        # 处理img_size类型转换（如果是列表则转为元组）
        if 'img_size' in config_dict and isinstance(config_dict['img_size'], list):
            config_dict['img_size'] = tuple(config_dict['img_size'])
        
        # 使用剩余参数创建数据集配置对象，并传入增强配置
        return cls(**config_dict, augmentation_config=aug_config)
    
    def save(self, file_path: str) -> None:
        """
        保存配置到文件
        
        参数:
            file_path (str): 文件路径
        """
        config_dict = self.to_dict()
        
        # 根据文件扩展名选择格式
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif ext.lower() in ['.yml', '.yaml']:
            # 配置YAML Dumper，避免使用Python特定标签
            class SafeDumper(yaml.SafeDumper):
                pass
            
            # 自定义元组表示方法，转换为列表
            def tuple_representer(dumper, data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', list(data))
            
            # 注册自定义表示方法
            SafeDumper.add_representer(tuple, tuple_representer)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, Dumper=SafeDumper)
        else:
            raise ValueError(f"不支持的文件格式: {ext}, 请使用 .json, .yml 或 .yaml")
        
        logger.info(f"配置已保存到: {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'DatasetConfig':
        """
        从文件加载配置
        
        参数:
            file_path (str): 文件路径
            
        返回:
            DatasetConfig: 配置对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        # 根据文件扩展名选择格式
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif ext.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的文件格式: {ext}, 请使用 .json, .yml 或 .yaml")
        
        logger.info(f"配置已从 {file_path} 加载")
        return cls.from_dict(config_dict)
    
    @classmethod
    def create_default(cls, name: str, data_dir: str, anno_file: str) -> 'DatasetConfig':
        """
        创建默认配置
        
        参数:
            name (str): 数据集名称
            data_dir (str): 数据目录
            anno_file (str): 标注文件
            
        返回:
            DatasetConfig: 默认配置对象
        """
        return cls(
            name=name,
            data_dir=data_dir,
            anno_file=anno_file
        )

    @classmethod
    def create_for_testing(cls, name: str, data_dir: str, anno_file: str, **kwargs) -> 'DatasetConfig':
        """
        创建用于测试的配置，开启开发模式
        
        参数:
            name (str): 数据集名称
            data_dir (str): 数据目录
            anno_file (str): 标注文件
            **kwargs: 其他配置参数
            
        返回:
            DatasetConfig: 配置对象
        """
        return cls(
            name=name,
            data_dir=data_dir,
            anno_file=anno_file,
            dev_mode=True,
            **kwargs
        ) 