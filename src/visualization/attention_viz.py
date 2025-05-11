"""
注意力权重可视化模块。
提供用于创建Vision Transformer注意力权重热力图的工具。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import List, Tuple, Dict, Optional, Union, Any
import os
import math

def plot_attention_weights(attention_weights: List[torch.Tensor], 
                          layer_idx: Optional[int] = None,
                          head_idx: Optional[int] = None,
                          cmap: str = 'viridis',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 10),
                          dpi: int = 100,
                          title: Optional[str] = None,
                          show_colorbar: bool = True) -> plt.Figure:
    """
    可视化注意力权重热力图。
    
    参数:
        attention_weights: 注意力权重列表，每个元素形状为 [batch_size, num_heads, seq_len, seq_len]
        layer_idx: 要可视化的特定层索引，如果为None，则可视化所有层
        head_idx: 要可视化的特定注意力头索引，如果为None，则可视化所有头的平均值
        cmap: 热力图颜色图
        save_path: 图像保存路径，如果为None，则不保存
        figsize: 图形大小
        dpi: 图像分辨率
        title: 图表标题
        show_colorbar: 是否显示颜色条
        
    返回:
        matplotlib Figure对象
    """
    if layer_idx is not None and (layer_idx < 0 or layer_idx >= len(attention_weights)):
        raise ValueError(f"layer_idx {layer_idx} 超出范围 [0, {len(attention_weights)-1}]")
    
    # 确定要可视化的层
    if layer_idx is not None:
        layers_to_viz = [attention_weights[layer_idx]]
        layer_names = [f"层 {layer_idx}"]
    else:
        layers_to_viz = attention_weights
        layer_names = [f"层 {i}" for i in range(len(attention_weights))]
    
    num_layers = len(layers_to_viz)
    
    # 创建图表
    if num_layers == 1:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        axes = [ax]
    else:
        # 计算行数和列数以接近正方形布局
        n_cols = min(3, int(math.ceil(math.sqrt(num_layers))))
        n_rows = int(math.ceil(num_layers / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
        if n_rows * n_cols > 1:
            axes = axes.flatten()
    
    # 全局标题
    if title:
        fig.suptitle(title, fontsize=16)
    
    # 绘制每一层的注意力热力图
    for i, (attn, ax, layer_name) in enumerate(zip(layers_to_viz, axes, layer_names)):
        # 移动到CPU和Numpy
        attn = attn.detach().cpu().numpy()
        
        # 形状: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = attn.shape
        
        if head_idx is not None:
            if head_idx < 0 or head_idx >= num_heads:
                raise ValueError(f"head_idx {head_idx} 超出范围 [0, {num_heads-1}]")
            # 获取特定头的注意力权重
            attn_map = attn[0, head_idx]  # 选取第一个批次样本
            subtitle = f"{layer_name} - 头 {head_idx}"
        else:
            # 计算所有头的平均值
            attn_map = attn[0].mean(axis=0)  # 选取第一个批次样本并平均所有头
            subtitle = f"{layer_name} - 所有头的平均值"
        
        # 绘制热力图
        im = ax.imshow(attn_map, cmap=cmap)
        ax.set_title(subtitle)
        
        # 如果是第一个CLS token，为坐标轴添加特殊标记
        if seq_len > 10:  # 只有当序列足够长时才添加标签
            ax.set_xticks([0] + list(range(1, seq_len, max(1, seq_len // 10))))
            ax.set_yticks([0] + list(range(1, seq_len, max(1, seq_len // 10))))
            
            # 为CLS令牌添加特殊标签
            labels = ['CLS'] + [str(i) for i in range(1, seq_len)]
            labels = [labels[i] for i in ax.get_xticks()]
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        
        # 去除不必要的坐标轴
        if num_layers > 1:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # 隐藏额外的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # 添加颜色条
    if show_colorbar:
        fig.colorbar(im, ax=axes, shrink=0.8, label='注意力权重')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def visualize_attention_on_image(image: torch.Tensor,
                                attention_weights: List[torch.Tensor],
                                patch_size: int,
                                image_size: Optional[Tuple[int, int]] = None,
                                layer_idx: Optional[int] = None,
                                head_idx: Optional[int] = None,
                                cls_token_attention: bool = True,
                                alpha: float = 0.6,
                                cmap: str = 'inferno',
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (16, 8),
                                dpi: int = 100,
                                title: Optional[str] = None) -> plt.Figure:
    """
    将注意力权重叠加到原始图像上以可视化模型关注的区域。
    
    参数:
        image: 输入图像张量，形状为 [3, H, W] 或 [1, 3, H, W]
        attention_weights: 注意力权重列表，每个元素形状为 [batch_size, num_heads, seq_len, seq_len]
        patch_size: 补丁大小（用于将注意力映射回图像空间）
        image_size: 图像大小，如果为None，则从image推断
        layer_idx: 要可视化的特定层索引，如果为None，则使用最后一层
        head_idx: 要可视化的特定注意力头索引，如果为None，则可视化所有头的平均值
        cls_token_attention: 是否可视化CLS令牌对其他补丁的注意力（而不是补丁之间的注意力）
        alpha: 注意力热力图的透明度
        cmap: 热力图颜色图
        save_path: 图像保存路径，如果为None，则不保存
        figsize: 图形大小
        dpi: 图像分辨率
        title: 图表标题
        
    返回:
        matplotlib Figure对象
    """
    # 处理输入图像
    if image.dim() == 4:
        image = image[0]  # 获取第一个批次样本
    
    # 将图像转换为Numpy数组，并调整为0-1范围
    img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    
    # 如果图像为灰度，转换为RGB
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    
    # 标准化到0-1范围
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # 确定图像大小
    if image_size is None:
        image_size = (img_np.shape[0], img_np.shape[1])
    
    # 确定要可视化的层
    if layer_idx is None:
        layer_idx = len(attention_weights) - 1  # 使用最后一层
    elif layer_idx < 0 or layer_idx >= len(attention_weights):
        raise ValueError(f"layer_idx {layer_idx} 超出范围 [0, {len(attention_weights)-1}]")
    
    # 获取指定层的注意力权重
    attn = attention_weights[layer_idx]
    
    # 移动到CPU和Numpy
    attn = attn.detach().cpu().numpy()
    
    # 形状: [batch_size, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attn.shape
    
    # 根据head_idx选择注意力权重
    if head_idx is not None:
        if head_idx < 0 or head_idx >= num_heads:
            raise ValueError(f"head_idx {head_idx} 超出范围 [0, {num_heads-1}]")
        # 获取特定头的注意力权重
        attn_map = attn[0, head_idx]  # 选取第一个批次样本
        head_text = f"头 {head_idx}"
    else:
        # 计算所有头的平均值
        attn_map = attn[0].mean(axis=0)  # 选取第一个批次样本并平均所有头
        head_text = "所有头的平均值"
    
    # 确定要可视化的注意力类型
    if cls_token_attention:
        # 获取CLS令牌对其他补丁的注意力（第一行，不包括对CLS自身的注意力）
        patch_attn = attn_map[0, 1:]
        attn_title = f"CLS令牌对其他补丁的注意力"
    else:
        # 计算补丁之间的平均注意力（不包括CLS令牌）
        patch_attn = attn_map[1:, 1:].mean(axis=0)
        attn_title = f"补丁之间的平均注意力"
    
    # 计算补丁网格尺寸
    grid_size = int(np.sqrt(len(patch_attn)))
    
    # 将注意力权重重塑为网格
    attention_grid = patch_attn.reshape(grid_size, grid_size)
    
    # 将注意力权重上采样到原始图像大小
    h, w = image_size
    attention_upsampled = np.kron(attention_grid, np.ones((patch_size, patch_size)))
    
    # 如果上采样后的大小与原图像不匹配，进行裁剪或填充
    if attention_upsampled.shape[0] > h:
        attention_upsampled = attention_upsampled[:h, :]
    if attention_upsampled.shape[1] > w:
        attention_upsampled = attention_upsampled[:, :w]
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    
    # 显示原始图像
    ax1.imshow(img_np)
    ax1.set_title("原始图像")
    ax1.axis('off')
    
    # 显示注意力热力图
    norm = Normalize(vmin=attention_upsampled.min(), vmax=attention_upsampled.max())
    ax2.imshow(attention_upsampled, cmap=cmap, norm=norm)
    ax2.set_title(f"注意力热力图\n{attn_title}\n{head_text}")
    ax2.axis('off')
    
    # 显示叠加后的图像
    colored_attention = cm.get_cmap(cmap)(norm(attention_upsampled))
    colored_attention = colored_attention[:, :, :3]  # 移除alpha通道
    
    # 叠加注意力热力图到原始图像上
    overlay = (1 - alpha) * img_np + alpha * colored_attention
    
    ax3.imshow(overlay)
    ax3.set_title(f"叠加图像\n{attn_title}\n{head_text}")
    ax3.axis('off')
    
    # 添加全局标题
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f"层 {layer_idx} 的注意力可视化", fontsize=16)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def visualize_all_heads(image: torch.Tensor,
                       attention_weights: List[torch.Tensor],
                       patch_size: int,
                       layer_idx: Optional[int] = None,
                       cls_token_attention: bool = True,
                       alpha: float = 0.6,
                       cmap: str = 'inferno',
                       save_path: Optional[str] = None,
                       figsize: Optional[Tuple[int, int]] = None,
                       dpi: int = 100,
                       max_heads_per_row: int = 4) -> plt.Figure:
    """
    可视化特定层的所有注意力头。
    
    参数:
        image: 输入图像张量，形状为 [3, H, W] 或 [1, 3, H, W]
        attention_weights: 注意力权重列表，每个元素形状为 [batch_size, num_heads, seq_len, seq_len]
        patch_size: 补丁大小（用于将注意力映射回图像空间）
        layer_idx: 要可视化的特定层索引，如果为None，则使用最后一层
        cls_token_attention: 是否可视化CLS令牌对其他补丁的注意力（而不是补丁之间的注意力）
        alpha: 注意力热力图的透明度
        cmap: 热力图颜色图
        save_path: 图像保存路径，如果为None，则不保存
        figsize: 图形大小，如果为None，则自动计算
        dpi: 图像分辨率
        max_heads_per_row: 每行最大显示的头数量
        
    返回:
        matplotlib Figure对象
    """
    # 处理输入图像
    if image.dim() == 4:
        image = image[0]  # 获取第一个批次样本
    
    # 将图像转换为Numpy数组，并调整为0-1范围
    img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    
    # 如果图像为灰度，转换为RGB
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    
    # 标准化到0-1范围
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # 确定要可视化的层
    if layer_idx is None:
        layer_idx = len(attention_weights) - 1  # 使用最后一层
    elif layer_idx < 0 or layer_idx >= len(attention_weights):
        raise ValueError(f"layer_idx {layer_idx} 超出范围 [0, {len(attention_weights)-1}]")
    
    # 获取指定层的注意力权重
    attn = attention_weights[layer_idx]
    
    # 移动到CPU和Numpy
    attn = attn.detach().cpu().numpy()
    
    # 形状: [batch_size, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attn.shape
    
    # 计算行数和列数
    n_cols = min(max_heads_per_row, num_heads)
    n_rows = int(math.ceil(num_heads / n_cols))
    
    # 自动计算figsize（如果未提供）
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)
    
    # 创建图表
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 计算补丁网格尺寸
    grid_size = int(np.sqrt(seq_len - 1))  # 减1是因为有CLS令牌
    
    # 确定要可视化的注意力类型的描述
    attn_type = "CLS令牌对其他补丁的注意力" if cls_token_attention else "补丁之间的平均注意力"
    
    # 为每个头创建可视化
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        
        # 获取特定头的注意力权重
        attn_map = attn[0, head_idx]  # 选取第一个批次样本
        
        # 确定要可视化的注意力类型
        if cls_token_attention:
            # 获取CLS令牌对其他补丁的注意力（第一行，不包括对CLS自身的注意力）
            patch_attn = attn_map[0, 1:]
        else:
            # 计算补丁之间的平均注意力（不包括CLS令牌）
            patch_attn = attn_map[1:, 1:].mean(axis=0)
        
        # 将注意力权重重塑为网格
        attention_grid = patch_attn.reshape(grid_size, grid_size)
        
        # 将注意力权重上采样到原始图像大小
        h, w = img_np.shape[:2]
        attention_upsampled = np.kron(attention_grid, np.ones((patch_size, patch_size)))
        
        # 如果上采样后的大小与原图像不匹配，进行裁剪或填充
        if attention_upsampled.shape[0] > h:
            attention_upsampled = attention_upsampled[:h, :]
        if attention_upsampled.shape[1] > w:
            attention_upsampled = attention_upsampled[:, :w]
        
        # 标准化注意力权重
        norm = Normalize(vmin=attention_upsampled.min(), vmax=attention_upsampled.max())
        
        # 获取着色的注意力热力图
        colored_attention = cm.get_cmap(cmap)(norm(attention_upsampled))
        colored_attention = colored_attention[:, :, :3]  # 移除alpha通道
        
        # 叠加注意力热力图到原始图像上
        overlay = (1 - alpha) * img_np + alpha * colored_attention
        
        # 显示叠加后的图像
        ax.imshow(overlay)
        ax.set_title(f"头 {head_idx}")
        ax.axis('off')
    
    # 隐藏额外的子图
    for j in range(num_heads, len(axes)):
        axes[j].axis('off')
    
    # 添加全局标题
    fig.suptitle(f"层 {layer_idx} 的注意力可视化 - {attn_type}", fontsize=16)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig 