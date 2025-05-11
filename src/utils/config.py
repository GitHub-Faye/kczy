import os
import json
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import torch

logger = logging.getLogger(__name__)

@dataclass
class ViTConfig:
    """
    Vision Transformer 模型配置类
    
    参数:
        img_size (int): 输入图像大小（假设是方形）
        patch_size (int): 补丁大小（假设是方形）
        in_channels (int): 输入图像的通道数
        num_classes (int): 分类类别数量
        embed_dim (int): 嵌入向量的维度
        depth (int): Transformer编码器块的数量
        num_heads (int): 注意力头的数量
        mlp_ratio (float): MLP中隐藏层维度与输入维度的比例
        qkv_bias (bool): 是否为QKV投影添加可学习的偏置项
        representation_size (Optional[int]): 表示层的维度，如果为None则不使用额外的表示层
        drop_rate (float): 丢弃率
        attn_drop_rate (float): 注意力丢弃率
        drop_path_rate (float): 随机深度丢弃率
        pretrained (bool): 是否使用预训练权重
        pretrained_path (Optional[str]): 预训练权重路径，当pretrained=True时有效
    """
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    representation_size: Optional[int] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    
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
        # 验证图像大小
        if self.img_size <= 0:
            raise ValueError(f"图像大小必须为正数: {self.img_size}")
        
        # 验证补丁大小
        if self.patch_size <= 0:
            raise ValueError(f"补丁大小必须为正数: {self.patch_size}")
            
        # 验证图像大小是否为补丁大小的整数倍
        if self.img_size % self.patch_size != 0:
            raise ValueError(f"图像大小 ({self.img_size}) 必须是补丁大小 ({self.patch_size}) 的整数倍")
        
        # 验证通道数
        if self.in_channels <= 0:
            raise ValueError(f"输入通道数必须为正数: {self.in_channels}")
        
        # 验证类别数
        if self.num_classes <= 0:
            raise ValueError(f"类别数量必须为正数: {self.num_classes}")
        
        # 验证嵌入维度
        if self.embed_dim <= 0:
            raise ValueError(f"嵌入维度必须为正数: {self.embed_dim}")
        
        # 验证Transformer深度
        if self.depth <= 0:
            raise ValueError(f"Transformer深度必须为正数: {self.depth}")
        
        # 验证注意力头数量
        if self.num_heads <= 0:
            raise ValueError(f"注意力头数量必须为正数: {self.num_heads}")
            
        # 验证注意力头数量能否被嵌入维度整除
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"嵌入维度 ({self.embed_dim}) 必须是注意力头数量 ({self.num_heads}) 的整数倍")
        
        # 验证MLP比例
        if self.mlp_ratio <= 0:
            raise ValueError(f"MLP比例必须为正数: {self.mlp_ratio}")
            
        # 验证表示层维度
        if self.representation_size is not None and self.representation_size <= 0:
            raise ValueError(f"表示层维度必须为正数: {self.representation_size}")
        
        # 验证丢弃率
        if not 0 <= self.drop_rate < 1:
            raise ValueError(f"丢弃率必须在0到1之间: {self.drop_rate}")
        
        # 验证注意力丢弃率
        if not 0 <= self.attn_drop_rate < 1:
            raise ValueError(f"注意力丢弃率必须在0到1之间: {self.attn_drop_rate}")
        
        # 验证随机深度丢弃率
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError(f"随机深度丢弃率必须在0到1之间: {self.drop_path_rate}")
        
        # 当pretrained为True时，验证预训练权重路径
        if self.pretrained and self.pretrained_path is not None and not os.path.exists(self.pretrained_path):
            # 这里只是给出警告，而不是错误，因为在某些情况下可能会在运行时下载权重
            logger.warning(f"预训练权重路径不存在: {self.pretrained_path}")
        
        return True
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        返回:
            Dict: 包含配置信息的字典
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ViTConfig':
        """
        从字典创建配置
        
        参数:
            config_dict (Dict): 配置字典
            
        返回:
            ViTConfig: 配置对象
        """
        return cls(**config_dict)
    
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
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的文件格式: {ext}, 请使用 .json, .yml 或 .yaml")
        
        logger.info(f"配置已保存到: {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'ViTConfig':
        """
        从文件加载配置
        
        参数:
            file_path (str): 文件路径
            
        返回:
            ViTConfig: 配置对象
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
    def create_tiny(cls, num_classes: int = 1000) -> 'ViTConfig':
        """
        创建ViT-Tiny配置
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            ViTConfig: ViT-Tiny配置对象
        """
        return cls(
            embed_dim=192,
            depth=12,
            num_heads=3,
            num_classes=num_classes
        )
    
    @classmethod
    def create_small(cls, num_classes: int = 1000) -> 'ViTConfig':
        """
        创建ViT-Small配置
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            ViTConfig: ViT-Small配置对象
        """
        return cls(
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=num_classes
        )
    
    @classmethod
    def create_base(cls, num_classes: int = 1000) -> 'ViTConfig':
        """
        创建ViT-Base配置
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            ViTConfig: ViT-Base配置对象
        """
        return cls(
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=num_classes
        )
    
    @classmethod
    def create_large(cls, num_classes: int = 1000) -> 'ViTConfig':
        """
        创建ViT-Large配置
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            ViTConfig: ViT-Large配置对象
        """
        return cls(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            num_classes=num_classes
        )
    
    @classmethod
    def create_huge(cls, num_classes: int = 1000) -> 'ViTConfig':
        """
        创建ViT-Huge配置
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            ViTConfig: ViT-Huge配置对象
        """
        return cls(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            num_classes=num_classes
        )

@dataclass
class TrainingConfig:
    """
    训练配置类，用于管理训练过程中的各种超参数
    
    参数:
        batch_size (int): 批次大小
        num_epochs (int): 训练总轮数
        learning_rate (float): 学习率
        weight_decay (float): 权重衰减系数
        optimizer_type (str): 优化器类型，支持 'sgd', 'adam', 'adamw', 'rmsprop'
        loss_type (str): 损失函数类型，支持 'cross_entropy', 'mse', 'bce', 'focal'
        scheduler_type (Optional[str]): 学习率调度器类型，支持 'step', 'multistep', 'exponential', 'cosine', None
        scheduler_params (Dict): 学习率调度器参数
        grad_clip_value (Optional[float]): 梯度裁剪阈值
        grad_clip_norm (Optional[float]): 梯度范数裁剪阈值
        use_mixed_precision (bool): 是否使用混合精度训练
        val_split (float): 验证集比例（仅当不提供单独的验证集时使用）
        early_stopping (bool): 是否使用早停策略
        early_stopping_patience (int): 早停策略的耐心值
        checkpoint_dir (str): 检查点保存目录
        checkpoint_freq (int): 检查点保存频率（每多少个epoch保存一次）
        log_dir (str): 日志保存目录
        log_freq (int): 日志记录频率（每多少个batch记录一次）
        random_seed (int): 随机种子
        device (str): 训练设备，'cuda', 'cpu' 或 'auto'（自动选择）
        num_workers (int): 数据加载器的工作进程数
        pin_memory (bool): 数据加载器是否使用锁页内存
        metrics_dir (str): 指标保存目录
        metrics_format (str): 指标保存格式，支持 'csv', 'json'
        metrics_save_freq (int): 指标保存频率（每多少个epoch保存一次）
        metrics_experiment_name (Optional[str]): 指标实验名称，用于生成文件名
        plot_metrics (bool): 是否在训练结束后自动绘制指标曲线
        plot_metrics_dir (Optional[str]): 指标曲线保存目录
        enable_tensorboard (bool): 是否启用TensorBoard记录
        tensorboard_dir (str): TensorBoard日志目录
        tensorboard_port (int): TensorBoard服务器端口号
        log_histograms (bool): 是否记录模型参数和梯度的直方图到TensorBoard
        log_images (bool): 是否记录样本图像到TensorBoard
    """
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_type: str = 'adam'
    loss_type: str = 'cross_entropy'
    scheduler_type: Optional[str] = 'cosine'
    scheduler_params: Dict = field(default_factory=lambda: {'T_max': 100})
    grad_clip_value: Optional[float] = None
    grad_clip_norm: Optional[float] = None
    use_mixed_precision: bool = False
    val_split: float = 0.2
    early_stopping: bool = True
    early_stopping_patience: int = 10
    checkpoint_dir: str = 'checkpoints'
    checkpoint_freq: int = 1
    log_dir: str = 'logs'
    log_freq: int = 10
    random_seed: int = 42
    device: str = 'auto'
    num_workers: int = 4
    pin_memory: bool = True
    optimizer_params: Dict = field(default_factory=dict)
    # 新增指标记录相关参数
    metrics_dir: str = 'metrics'
    metrics_format: str = 'csv'
    metrics_save_freq: int = 1
    metrics_experiment_name: Optional[str] = None
    plot_metrics: bool = True
    plot_metrics_dir: Optional[str] = None
    # 添加TensorBoard相关参数
    enable_tensorboard: bool = False
    tensorboard_dir: str = 'logs'
    tensorboard_port: int = 6006
    log_histograms: bool = False
    log_images: bool = False
    start_tensorboard: bool = False
    tensorboard_host: str = 'localhost'
    tensorboard_background: bool = False
    
    def __post_init__(self):
        """初始化后进行验证"""
        self.validate()
        
        # 如果设备设置为auto，自动选择合适的设备
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        返回:
            bool: 配置是否有效
        
        异常:
            ValueError: 当配置无效时抛出
        """
        # 验证批次大小
        if self.batch_size <= 0:
            raise ValueError(f"批次大小必须为正数: {self.batch_size}")
        
        # 验证训练轮数
        if self.num_epochs <= 0:
            raise ValueError(f"训练轮数必须为正数: {self.num_epochs}")
        
        # 验证学习率
        if self.learning_rate <= 0:
            raise ValueError(f"学习率必须为正数: {self.learning_rate}")
        
        # 验证权重衰减
        if self.weight_decay < 0:
            raise ValueError(f"权重衰减必须为非负数: {self.weight_decay}")
        
        # 验证优化器类型
        valid_optimizers = ['sgd', 'adam', 'adamw', 'rmsprop']
        if self.optimizer_type.lower() not in valid_optimizers:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}. 支持的类型: {valid_optimizers}")
        
        # 验证损失函数类型
        valid_losses = ['cross_entropy', 'mse', 'bce', 'focal']
        if self.loss_type.lower() not in valid_losses:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}. 支持的类型: {valid_losses}")
        
        # 验证学习率调度器类型
        valid_schedulers = ['step', 'multistep', 'exponential', 'cosine', 'plateau', None]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(f"不支持的学习率调度器类型: {self.scheduler_type}. 支持的类型: {valid_schedulers}")
        
        # 验证梯度裁剪相关参数
        if self.grad_clip_value is not None and self.grad_clip_value <= 0:
            raise ValueError(f"梯度裁剪阈值必须为正数: {self.grad_clip_value}")
            
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"梯度范数裁剪阈值必须为正数: {self.grad_clip_norm}")
        
        # 验证验证集比例
        if not 0 <= self.val_split < 1:
            raise ValueError(f"验证集比例必须在0和1之间: {self.val_split}")
        
        # 验证早停策略相关参数
        if self.early_stopping_patience <= 0:
            raise ValueError(f"早停策略的耐心值必须为正数: {self.early_stopping_patience}")
        
        # 验证检查点频率
        if self.checkpoint_freq <= 0:
            raise ValueError(f"检查点保存频率必须为正数: {self.checkpoint_freq}")
            
        # 验证日志频率
        if self.log_freq <= 0:
            raise ValueError(f"日志记录频率必须为正数: {self.log_freq}")
        
        # 验证设备
        valid_devices = ['cuda', 'cpu', 'auto']
        if self.device not in valid_devices:
            raise ValueError(f"不支持的设备类型: {self.device}. 支持的类型: {valid_devices}")
        
        # 验证工作进程数
        if self.num_workers < 0:
            raise ValueError(f"工作进程数必须为非负数: {self.num_workers}")
        
        # 验证指标保存格式
        valid_metrics_formats = ['csv', 'json']
        if self.metrics_format.lower() not in valid_metrics_formats:
            raise ValueError(f"不支持的指标保存格式: {self.metrics_format}. 支持的格式: {valid_metrics_formats}")
        
        # 验证指标保存频率
        if self.metrics_save_freq <= 0:
            raise ValueError(f"指标保存频率必须为正数: {self.metrics_save_freq}")
        
        # 验证TensorBoard端口
        if self.tensorboard_port <= 0 or self.tensorboard_port > 65535:
            raise ValueError(f"TensorBoard端口必须在1-65535范围内: {self.tensorboard_port}")
            
        return True
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        返回:
            Dict: 包含配置信息的字典
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """
        从字典创建配置
        
        参数:
            config_dict (Dict): 配置字典
            
        返回:
            TrainingConfig: 配置对象
        """
        return cls(**config_dict)
    
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
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的文件格式: {ext}, 请使用 .json, .yml 或 .yaml")
        
        logger.info(f"训练配置已保存到: {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'TrainingConfig':
        """
        从文件加载配置
        
        参数:
            file_path (str): 文件路径
            
        返回:
            TrainingConfig: 配置对象
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
        
        logger.info(f"训练配置已从 {file_path} 加载")
        return cls.from_dict(config_dict)
    
    @classmethod
    def create_default(cls) -> 'TrainingConfig':
        """
        创建默认训练配置
        
        返回:
            TrainingConfig: 默认配置对象
        """
        return cls()
    
    @classmethod
    def create_fast_dev(cls) -> 'TrainingConfig':
        """
        创建用于快速开发/测试的训练配置
        
        返回:
            TrainingConfig: 快速开发配置对象
        """
        return cls(
            batch_size=8,
            num_epochs=5,
            learning_rate=1e-3,
            scheduler_type=None,
            early_stopping=False,
            checkpoint_freq=5,
            num_workers=0
        )
    
    @classmethod
    def create_high_performance(cls) -> 'TrainingConfig':
        """
        创建用于高性能训练的配置
        
        返回:
            TrainingConfig: 高性能训练配置对象
        """
        return cls(
            batch_size=64,
            num_epochs=200,
            learning_rate=1e-4,
            weight_decay=1e-5,
            optimizer_type='adamw',
            scheduler_type='cosine',
            scheduler_params={'T_max': 200},
            use_mixed_precision=True,
            early_stopping=True,
            early_stopping_patience=15,
            checkpoint_freq=1,
            grad_clip_norm=1.0
        ) 