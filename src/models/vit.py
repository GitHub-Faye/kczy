import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple
from src.utils.config import ViTConfig
import os

class PatchEmbedding(nn.Module):
    """
    将输入图像转换为一系列平铺的补丁嵌入向量
    
    参数:
        img_size (int): 输入图像大小（假设是正方形）
        patch_size (int): 补丁大小（假设是正方形）
        in_channels (int): 输入图像的通道数
        embed_dim (int): 嵌入向量的维度
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # 计算图像的补丁数量
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层实现补丁嵌入
        # 卷积核大小和步长设置为patch_size，可以将图像分割成不重叠的补丁
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        参数:
            x: 形状为 [batch_size, in_channels, img_size, img_size] 的张量
            
        返回:
            形状为 [batch_size, num_patches, embed_dim] 的补丁嵌入张量
        """
        # 检查输入图像大小
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入图像大小 ({H}*{W}) 与预期大小 ({self.img_size}*{self.img_size}) 不匹配"
            
        # 使用卷积投影得到补丁嵌入
        # 形状: [B, embed_dim, H//patch_size, W//patch_size]
        x = self.proj(x)
        
        # 将补丁展平
        # 形状: [B, embed_dim, num_patches]
        x = x.flatten(2)
        
        # 转置以获得最终形状 [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    
    参数:
        dim (int): 输入/输出的特征维度
        num_heads (int): 注意力头的数量
        qkv_bias (bool): 是否为QKV投影添加可学习的偏置项
        attn_drop (float): 注意力权重的丢弃率
        proj_drop (float): 输出投影的丢弃率
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = head_dim ** -0.5  # 缩放因子
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """
        参数:
            x: 形状为 [batch_size, seq_len, dim] 的张量
            
        返回:
            形状为 [batch_size, seq_len, dim] 的自注意力输出
        """
        B, N, C = x.shape  # batch_size, 序列长度, 嵌入维度
        
        # 计算query, key, value
        # 形状: [B, N, 3*C] -> 3 x [B, N, C]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 形状: [B, num_heads, N, head_dim]
        
        # 计算注意力分数
        # 转置key以进行矩阵乘法: [B, num_heads, N, head_dim] x [B, num_heads, head_dim, N]
        # 结果形状: [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 应用softmax得到注意力权重
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 将注意力权重与value相乘，并重塑
        # 形状: [B, num_heads, N, N] x [B, num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 最终投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    多层感知机（MLP）/前馈网络
    
    参数:
        in_features (int): 输入特征的维度
        hidden_features (int): 隐藏层特征的维度
        out_features (int): 输出特征的维度
        drop (float): 丢弃率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # ViT使用GELU作为激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        """
        参数:
            x: 形状为 [batch_size, seq_len, in_features] 的张量
            
        返回:
            形状为 [batch_size, seq_len, out_features] 的张量
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer编码器块，包含自注意力和前馈网络
    
    参数:
        dim (int): 输入/输出的特征维度
        num_heads (int): 注意力头的数量
        mlp_ratio (float): MLP中隐藏层维度与输入维度的比例
        qkv_bias (bool): 是否为QKV投影添加可学习的偏置项
        drop (float): 丢弃率
        attn_drop (float): 注意力权重的丢弃率
        drop_path (float): 随机深度
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        
        # 第一个层规范化
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        
        # 多头自注意力
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # 第二个层规范化
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # 前馈网络
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            drop=drop
        )
        
        # 丢弃路径（随机深度）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        """
        参数:
            x: 形状为 [batch_size, seq_len, dim] 的张量
            
        返回:
            形状为 [batch_size, seq_len, dim] 的张量
        """
        # 残差连接1: 输入 + 自注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # 残差连接2: 前一层输出 + 前馈网络
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """
    随机深度: 在训练时随机丢弃整个模块的输出
    
    参数:
        drop_prob (float): 丢弃概率
    """
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # 处理维度为[B, ...]的张量，丢弃整批中的一些样本
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, 1, ...]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化为0或1
        output = x.div(keep_prob) * random_tensor  # 缩放保留的激活值
        return output
        

class TransformerEncoder(nn.Module):
    """
    Transformer编码器，由多个编码器块堆叠而成
    
    参数:
        dim (int): 输入/输出的特征维度
        depth (int): 编码器块的数量
        num_heads (int): 注意力头的数量
        mlp_ratio (float): MLP中隐藏层维度与输入维度的比例
        qkv_bias (bool): 是否为QKV投影添加可学习的偏置项
        drop (float): 丢弃率
        attn_drop (float): 注意力权重的丢弃率
        drop_path (float): 随机深度
    """
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        
        # 创建多个编码器块
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop, 
                attn_drop=attn_drop,
                # 随机深度: 逐渐增加丢弃率
                drop_path=drop_path * i / (depth - 1) if depth > 1 else 0,
            )
            for i in range(depth)
        ])
    
    def forward(self, x):
        """
        参数:
            x: 形状为 [batch_size, seq_len, dim] 的张量
            
        返回:
            形状为 [batch_size, seq_len, dim] 的张量
        """
        # 依次通过所有编码器块
        for block in self.blocks:
            x = block(x)
        return x


class ClassificationHead(nn.Module):
    """
    分类头，用于将Transformer的输出转换为类别预测
    
    参数:
        in_features (int): 输入特征的维度
        num_classes (int): 类别数量
        hidden_features (int): 隐藏层的特征维度，如果不为None，则使用两层MLP
        drop (float): 丢弃率
    """
    def __init__(self, in_features, num_classes, hidden_features=None, drop=0.):
        super().__init__()
        
        # 层规范化
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        
        # 如果提供了隐藏层维度，则使用双层MLP
        if hidden_features is not None:
            self.head = MLP(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=num_classes,
                drop=drop
            )
        else:
            # 否则使用单层线性投影
            self.head = nn.Linear(in_features, num_classes)
            if drop > 0:
                self.drop = nn.Dropout(drop)
            else:
                self.drop = nn.Identity()
                
    def forward(self, x):
        """
        参数:
            x: 形状为 [batch_size, in_features] 的张量，通常是提取的[CLS]标记
            
        返回:
            形状为 [batch_size, num_classes] 的类别预测
        """
        # 应用层规范化
        x = self.norm(x)
        
        # 应用分类头
        if isinstance(self.head, MLP):
            # 如果是MLP，直接前向传播
            x = self.head(x)
        else:
            # 如果是单层线性投影，先应用丢弃，再应用线性层
            x = self.drop(x)
            x = self.head(x)
            
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 模型
    
    参数:
        img_size (int): 输入图像大小
        patch_size (int): 补丁大小
        in_channels (int): 输入图像的通道数
        num_classes (int): 类别数量
        embed_dim (int): 嵌入向量的维度
        depth (int): Transformer编码器块的数量
        num_heads (int): 注意力头的数量
        mlp_ratio (float): MLP中隐藏层维度与输入维度的比例
        qkv_bias (bool): 是否为QKV投影添加可学习的偏置项
        representation_size (int): 表示层的维度，如果为None则不使用额外的表示层
        drop_rate (float): 丢弃率
        attn_drop_rate (float): 注意力丢弃率
        drop_path_rate (float): 随机深度丢弃率
        config (ViTConfig): 模型配置对象，如果提供，将覆盖其他参数
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 config: Optional[ViTConfig] = None):
        super().__init__()
        
        # 如果提供了配置对象，使用配置对象的参数
        if config is not None:
            img_size = config.img_size
            patch_size = config.patch_size
            in_channels = config.in_channels
            num_classes = config.num_classes
            embed_dim = config.embed_dim
            depth = config.depth
            num_heads = config.num_heads
            mlp_ratio = config.mlp_ratio
            qkv_bias = config.qkv_bias
            representation_size = config.representation_size
            drop_rate = config.drop_rate
            attn_drop_rate = config.attn_drop_rate
            drop_path_rate = config.drop_path_rate
        
        # 补丁嵌入
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        
        # 补丁数量
        num_patches = self.patch_embed.num_patches
        
        # 类别标记 (CLS token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 丢弃
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer编码器
        self.transformer = TransformerEncoder(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate
        )
        
        # 分类头
        if representation_size is None:
            self.head = ClassificationHead(
                in_features=embed_dim,
                num_classes=num_classes,
                drop=drop_rate
            )
        else:
            # 如果提供了表示层维度，则使用带有隐藏层的分类头
            self.head = ClassificationHead(
                in_features=embed_dim,
                num_classes=num_classes,
                hidden_features=representation_size,
                drop=drop_rate
            )
            
        # 初始化位置嵌入和类别标记
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 应用权重初始化
        self.apply(self._init_weights)
        
        # 存储配置供后续使用
        self.config = config or ViTConfig(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            representation_size=representation_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        """提取特征，不包括分类头"""
        # 补丁嵌入 [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        
        # 添加类别标记
        # [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # [B, num_patches, embed_dim] -> [B, num_patches+1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 应用Transformer编码器
        x = self.transformer(x)
        
        # 返回类别标记
        return x[:, 0]
    
    def forward(self, x):
        """完整的前向传播"""
        # 提取特征
        x = self.forward_features(x)
        
        # 应用分类头
        x = self.head(x)
        
        return x
    
    @classmethod
    def from_config(cls, config: ViTConfig) -> 'VisionTransformer':
        """
        从配置创建模型
        
        参数:
            config (ViTConfig): 模型配置
            
        返回:
            VisionTransformer: 模型实例
        """
        return cls(config=config)
    
    def get_config(self) -> ViTConfig:
        """
        获取模型的配置
        
        返回:
            ViTConfig: 模型配置
        """
        return self.config
    
    def save_config(self, file_path: str) -> None:
        """
        保存模型配置到文件
        
        参数:
            file_path (str): 文件路径
        """
        self.config.save(file_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将模型配置转换为字典
        
        返回:
            Dict[str, Any]: 模型配置字典
        """
        return self.config.to_dict()
    
    def save_model(self, file_path: str, save_optimizer: bool = False, optimizer_state: Optional[Dict] = None) -> None:
        """
        保存完整模型（包括结构和权重）
        
        参数:
            file_path (str): 保存路径（.pt或.pth文件）
            save_optimizer (bool): 是否保存优化器状态
            optimizer_state (Optional[Dict]): 优化器状态字典（当save_optimizer为True时需要）
            
        返回:
            None
        """
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 构建保存字典
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'model_type': 'VisionTransformer',
            'version': '1.0.0'  # 版本信息用于未来的兼容性检查
        }
        
        # 如果需要，添加优化器状态
        if save_optimizer and optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        # 保存模型
        torch.save(checkpoint, file_path)
        print(f"模型已保存到: {file_path}")
        
        # 额外保存配置文件（便于后续检查）
        config_path = os.path.splitext(file_path)[0] + '_config.json'
        self.save_config(config_path)
    
    @classmethod
    def load_model(cls, file_path: str, device: Optional[torch.device] = None) -> Tuple['VisionTransformer', Optional[Dict]]:
        """
        从文件加载模型
        
        参数:
            file_path (str): 模型文件路径
            device (Optional[torch.device]): 加载模型的设备
            
        返回:
            Tuple[VisionTransformer, Optional[Dict]]: 加载的模型和可选的优化器状态
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
        
        # 设置加载设备
        map_location = device if device is not None else torch.device('cpu')
        
        # 加载检查点
        checkpoint = torch.load(file_path, map_location=map_location)
        
        # 兼容性检查
        if 'model_type' not in checkpoint or checkpoint['model_type'] != 'VisionTransformer':
            print("警告: 加载的模型可能不是VisionTransformer或格式不兼容")
        
        # 从检查点获取配置
        if 'config' in checkpoint:
            config = ViTConfig.from_dict(checkpoint['config'])
        else:
            # 尝试寻找配置文件
            config_path = os.path.splitext(file_path)[0] + '_config.json'
            if os.path.exists(config_path):
                config = ViTConfig.load(config_path)
            else:
                raise ValueError(f"无法找到模型配置，请提供有效的检查点或配置文件: {file_path}")
        
        # 创建模型
        model = cls.from_config(config)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 将模型移动到指定设备
        model.to(device if device is not None else torch.device('cpu'))
        
        # 返回模型和可能的优化器状态
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        
        return model, optimizer_state
    
    def save_weights(self, file_path: str) -> None:
        """
        仅保存模型权重
        
        参数:
            file_path (str): 保存路径
            
        返回:
            None
        """
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 直接保存模型状态字典
        torch.save(self.state_dict(), file_path)
        print(f"模型权重已保存到: {file_path}")
    
    def load_weights(self, file_path: str, strict: bool = True) -> None:
        """
        加载模型权重
        
        参数:
            file_path (str): 权重文件路径
            strict (bool): 是否执行严格加载（所有权重必须匹配）
            
        返回:
            None
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"权重文件不存在: {file_path}")
        
        # 加载权重
        weights = torch.load(file_path, map_location=next(self.parameters()).device)
        
        # 权重可能是状态字典或完整的检查点
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            # 从完整检查点中提取模型状态
            weights = weights['model_state_dict']
        
        # 加载权重到模型
        self.load_state_dict(weights, strict=strict)
        print(f"模型权重已从 {file_path} 加载")
    
    def export_to_onnx(self, file_path: str, input_shape: Optional[Tuple] = None, 
                      dynamic_axes: Optional[Dict] = None, export_params: bool = True,
                      opset_version: int = 11) -> None:
        """
        将模型导出为ONNX格式
        
        参数:
            file_path (str): 保存路径（.onnx文件）
            input_shape (Optional[Tuple]): 输入形状，默认为(1, in_channels, img_size, img_size)
            dynamic_axes (Optional[Dict]): 动态轴信息
            export_params (bool): 是否导出参数
            opset_version (int): ONNX操作集版本
            
        返回:
            None
        """
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 设置模型为评估模式
        self.eval()
        
        # 默认输入形状
        if input_shape is None:
            input_shape = (1, self.config.in_channels, self.config.img_size, self.config.img_size)
            
        # 默认动态轴 - 使批次大小可变
        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            
        # 创建示例输入
        dummy_input = torch.randn(input_shape, device=next(self.parameters()).device)
        
        # 导出为ONNX
        torch.onnx.export(
            self,                        # 要导出的模型
            dummy_input,                 # 示例输入
            file_path,                   # 输出文件
            export_params=export_params, # 存储训练过的参数权重
            opset_version=opset_version, # ONNX版本
            input_names=['input'],       # 输入名称
            output_names=['output'],     # 输出名称
            dynamic_axes=dynamic_axes,   # 动态轴
            verbose=False
        )
            
        print(f"模型已导出为ONNX格式: {file_path}")
        
        # 尝试保存配置文件
        try:
            config_path = os.path.splitext(file_path)[0] + '_config.json'
            self.save_config(config_path)
        except Exception as e:
            print(f"保存配置文件时出错: {str(e)}")
    
    @classmethod
    def create_tiny(cls, num_classes: int = 1000) -> 'VisionTransformer':
        """
        创建ViT-Tiny模型
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            VisionTransformer: ViT-Tiny模型实例
        """
        config = ViTConfig.create_tiny(num_classes=num_classes)
        return cls.from_config(config)
    
    @classmethod
    def create_small(cls, num_classes: int = 1000) -> 'VisionTransformer':
        """
        创建ViT-Small模型
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            VisionTransformer: ViT-Small模型实例
        """
        config = ViTConfig.create_small(num_classes=num_classes)
        return cls.from_config(config)
    
    @classmethod
    def create_base(cls, num_classes: int = 1000) -> 'VisionTransformer':
        """
        创建ViT-Base模型
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            VisionTransformer: ViT-Base模型实例
        """
        config = ViTConfig.create_base(num_classes=num_classes)
        return cls.from_config(config)
    
    @classmethod
    def create_large(cls, num_classes: int = 1000) -> 'VisionTransformer':
        """
        创建ViT-Large模型
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            VisionTransformer: ViT-Large模型实例
        """
        config = ViTConfig.create_large(num_classes=num_classes)
        return cls.from_config(config)
    
    @classmethod
    def create_huge(cls, num_classes: int = 1000) -> 'VisionTransformer':
        """
        创建ViT-Huge模型
        
        参数:
            num_classes (int): 类别数量
            
        返回:
            VisionTransformer: ViT-Huge模型实例
        """
        config = ViTConfig.create_huge(num_classes=num_classes)
        return cls.from_config(config)


if __name__ == "__main__":
    # 测试补丁嵌入模块
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    batch_size = 4
    
    # 创建随机输入
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # 实例化补丁嵌入层
    patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    
    # 前向传播
    embeddings = patch_embed(x)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"嵌入形状: {embeddings.shape}")
    print(f"预期形状: [batch_size={batch_size}, num_patches={(img_size//patch_size)**2}, embed_dim={embed_dim}]")
    
    # 测试Transformer编码器
    seq_len = (img_size // patch_size) ** 2  # 序列长度 = 补丁数量
    depth = 12  # Transformer块的数量
    num_heads = 12  # 注意力头的数量
    
    # 实例化Transformer编码器
    encoder = TransformerEncoder(
        dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )
    
    # 前向传播
    encoded = encoder(embeddings)
    
    # 打印输出形状
    print(f"\nTransformer编码器输出形状: {encoded.shape}")
    print(f"预期形状: [batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}]")
    
    # 测试完整的ViT模型
    num_classes = 1000  # 例如ImageNet
    
    # 实例化ViT模型 - 通过直接参数
    vit_model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )
    
    # 前向传播
    logits = vit_model(x)
    
    # 打印输出形状
    print(f"\nViT模型输出形状: {logits.shape}")
    print(f"预期形状: [batch_size={batch_size}, num_classes={num_classes}]")
    
    # 测试从配置创建ViT模型
    config = ViTConfig(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )
    
    # 从配置创建模型
    vit_from_config = VisionTransformer.from_config(config)
    
    # 前向传播
    logits_from_config = vit_from_config(x)
    
    # 打印输出形状
    print(f"\nViT模型(从配置创建)输出形状: {logits_from_config.shape}")
    print(f"预期形状: [batch_size={batch_size}, num_classes={num_classes}]")
    
    # 测试预设模型创建
    vit_small = VisionTransformer.create_small(num_classes=10)
    print(f"\nViT-Small配置: {vit_small.get_config().to_dict()}")
    
    vit_base = VisionTransformer.create_base(num_classes=10)
    print(f"ViT-Base配置: {vit_base.get_config().to_dict()}")
    
    # 保存配置到文件(可选)
    # vit_model.save_config("vit_config.json")
    # config_dict = vit_model.to_dict()
    # print(f"\n模型配置字典: {config_dict}") 