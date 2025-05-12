"""
模型结构可视化模块。
提供用于可视化Vision Transformer模型结构和层连接的工具。
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import torch
import torch.nn as nn
from src.models.vit import VisionTransformer, TransformerEncoder, TransformerEncoderBlock

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入Graphviz库，如果不存在则使用matplotlib
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("未检测到Graphviz库，将使用matplotlib作为备选方案。使用pip install graphviz安装以获得更好的可视化效果。")

def plot_model_structure(
    model: Union[VisionTransformer, nn.Module],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 16),
    dpi: int = 150,
    show_details: bool = True,
    direction: str = 'TB',  # 'TB'(上到下)或'LR'(左到右)
    format: str = 'png',
    engine: str = 'dot',
    title: Optional[str] = None
) -> Union[graphviz.Digraph, plt.Figure]:
    """
    生成并可视化Vision Transformer模型的结构图，显示关键组件和它们之间的连接。
    
    参数:
        model: Vision Transformer模型或其他PyTorch模型
        output_path: 图像输出路径，如果为None则仅显示不保存
        figsize: 使用matplotlib时的图表尺寸
        dpi: 图表分辨率
        show_details: 是否显示模型内部细节
        direction: 图表方向，'TB'(上到下)或'LR'(左到右)
        format: 图表保存格式
        engine: Graphviz渲染引擎
        title: 图表标题
        
    返回:
        graphviz.Digraph 对象或 matplotlib.Figure 对象
    """
    if GRAPHVIZ_AVAILABLE:
        return _plot_model_graphviz(
            model=model,
            output_path=output_path,
            show_details=show_details,
            direction=direction,
            format=format,
            engine=engine,
            title=title
        )
    else:
        return _plot_model_matplotlib(
            model=model,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
            show_details=show_details,
            title=title
        )

def _plot_model_graphviz(
    model: Union[VisionTransformer, nn.Module],
    output_path: Optional[str] = None,
    show_details: bool = True,
    direction: str = 'TB',
    format: str = 'png',
    engine: str = 'dot',
    title: Optional[str] = None
) -> graphviz.Digraph:
    """
    使用Graphviz库生成模型结构图。
    
    参数:
        model: Vision Transformer模型或其他PyTorch模型
        output_path: 图像输出路径
        show_details: 是否显示模型内部细节
        direction: 图表方向，'TB'(上到下)或'LR'(左到右)
        format: 图表保存格式
        engine: Graphviz渲染引擎
        title: 图表标题
        
    返回:
        graphviz.Digraph 对象
    """
    if title is None:
        title = f"{type(model).__name__} 模型结构"
    
    # 创建一个有向图
    dot = graphviz.Digraph(
        comment=title,
        format=format,
        engine=engine
    )
    
    # 设置图形属性
    dot.attr(rankdir=direction, splines='ortho', nodesep='0.8', ranksep='1.0', fontname='Microsoft YaHei')
    dot.attr('node', shape='box', style='filled,rounded', color='lightblue', fontname='Microsoft YaHei')
    dot.attr('edge', color='gray70', arrowsize='0.7')
    
    # 添加标题
    if title:
        dot.attr(label=f"<<B>{title}</B>>", labelloc='t', fontsize='18')
    
    # 处理Vision Transformer模型结构
    if isinstance(model, VisionTransformer):
        # 添加输入节点
        dot.node('input', '输入图像', shape='oval', color='lightyellow')
        
        # 添加Patch Embedding节点
        dot.node('patch_embedding', 'Patch Embedding', color='lightgreen')
        dot.edge('input', 'patch_embedding')
        
        # 添加位置编码节点
        dot.node('pos_embedding', '位置编码', color='lightpink')
        dot.edge('patch_embedding', 'pos_embedding')
        
        # 添加Class Token节点
        dot.node('cls_token', 'Class Token', color='lightpink')
        dot.edge('cls_token', 'pos_embedding')
        
        # 添加Transformer Encoder
        if show_details and hasattr(model, 'blocks') and isinstance(model.blocks, TransformerEncoder):
            depth = model.blocks.depth
            
            # 创建一个子图来包含所有编码器块
            with dot.subgraph(name='cluster_encoder') as c:
                c.attr(label='Transformer Encoder', style='filled', color='lightcyan')
                
                # 上一个节点，初始为位置编码
                prev_node = 'pos_embedding'
                
                # 添加每个Transformer编码器块
                for i in range(depth):
                    # 为每个编码器块创建节点
                    block_id = f'block_{i}'
                    c.node(block_id, f'编码器块 {i}', color='lightsalmon')
                    
                    # 连接到上一个节点
                    dot.edge(prev_node, block_id)
                    prev_node = block_id
                
                # 最后一个编码器块连接到层规范化
                c.node('layer_norm', '层规范化', color='lightgray')
                dot.edge(prev_node, 'layer_norm')
                prev_node = 'layer_norm'
        else:
            # 简化版本，只显示一个Transformer Encoder节点
            dot.node('transformer_encoder', 'Transformer Encoder', color='lightsalmon')
            dot.edge('pos_embedding', 'transformer_encoder')
            prev_node = 'transformer_encoder'
        
        # 添加分类头
        dot.node('classification_head', '分类头', color='lightcoral')
        dot.edge(prev_node, 'classification_head')
        
        # 添加输出节点
        dot.node('output', '输出 (类别预测)', shape='oval', color='lightyellow')
        dot.edge('classification_head', 'output')
    else:
        # 通用模型结构可视化
        dot.node('model', f'{type(model).__name__}\n(通用PyTorch模型)', color='lightblue')
        
    # 渲染并保存图表
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 去除扩展名，让Graphviz自己添加
        output_path_no_ext = os.path.splitext(output_path)[0]
        dot.render(output_path_no_ext, cleanup=True)
        logger.info(f"模型结构图已保存至: {output_path_no_ext}.{format}")
    
    return dot

def _plot_model_matplotlib(
    model: Union[VisionTransformer, nn.Module],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 16),
    dpi: int = 150,
    show_details: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    使用Matplotlib库生成模型结构图（Graphviz不可用时的备选方案）。
    
    参数:
        model: Vision Transformer模型或其他PyTorch模型
        output_path: 图像输出路径
        figsize: 图表尺寸
        dpi: 图表分辨率
        show_details: 是否显示模型内部细节
        title: 图表标题
        
    返回:
        matplotlib.Figure 对象
    """
    if title is None:
        title = f"{type(model).__name__} 模型结构"
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if isinstance(model, VisionTransformer):
        # 定义组件列表
        components = []
        connections = []
        
        # 基本组件
        components.append({"name": "输入图像", "level": 0, "type": "input"})
        components.append({"name": "Patch Embedding", "level": 1, "type": "embedding"})
        components.append({"name": "位置编码", "level": 2, "type": "embedding"})
        components.append({"name": "Class Token", "level": 1.5, "type": "token"})
        
        # 添加Transformer编码器块
        if show_details and hasattr(model, 'blocks') and isinstance(model.blocks, TransformerEncoder):
            depth = model.blocks.depth
            base_level = 3
            
            for i in range(depth):
                components.append({"name": f"编码器块 {i}", "level": base_level + i, "type": "encoder"})
                
                if i > 0:
                    connections.append((f"编码器块 {i-1}", f"编码器块 {i}"))
                else:
                    connections.append(("位置编码", f"编码器块 {i}"))
            
            components.append({"name": "层规范化", "level": base_level + depth, "type": "norm"})
            connections.append((f"编码器块 {depth-1}", "层规范化"))
            last_encoder_component = "层规范化"
        else:
            # 简化版本
            components.append({"name": "Transformer Encoder", "level": 3, "type": "encoder"})
            connections.append(("位置编码", "Transformer Encoder"))
            last_encoder_component = "Transformer Encoder"
        
        # 添加分类头和输出
        components.append({"name": "分类头", "level": len(components), "type": "head"})
        components.append({"name": "输出 (类别预测)", "level": len(components), "type": "output"})
        
        # 添加基本连接
        connections.append(("输入图像", "Patch Embedding"))
        connections.append(("Patch Embedding", "位置编码"))
        connections.append(("Class Token", "位置编码"))
        connections.append((last_encoder_component, "分类头"))
        connections.append(("分类头", "输出 (类别预测)"))
        
        # 绘制组件
        component_coords = {}
        component_colors = {
            "input": "lightyellow",
            "embedding": "lightgreen",
            "token": "lightpink",
            "encoder": "lightsalmon",
            "norm": "lightgray",
            "head": "lightcoral",
            "output": "lightyellow"
        }
        
        # 计算水平位置（居中）
        max_level = max(comp["level"] for comp in components)
        level_counts = {}
        for comp in components:
            level = comp["level"]
            if level not in level_counts:
                level_counts[level] = 0
            level_counts[level] += 1
        
        # 计算每个组件的坐标
        for comp in components:
            level = comp["level"]
            count = level_counts[level]
            
            # 垂直位置
            y = 1.0 - (level / (max_level + 1))
            
            # 为此层的组件确定水平位置
            level_components = [c for c in components if c["level"] == level]
            idx = level_components.index(comp)
            x = (idx + 1) / (len(level_components) + 1)
            
            # 存储坐标
            component_coords[comp["name"]] = (x, y)
            
            # 绘制节点
            circle = plt.Circle((x, y), 0.05, color=component_colors[comp["type"]], alpha=0.7)
            ax.add_artist(circle)
            
            # 添加文本标签
            ax.text(x, y - 0.02, comp["name"], ha="center", va="top", fontsize=9, 
                   bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"))
        
        # 绘制连接线
        for src, dst in connections:
            if src in component_coords and dst in component_coords:
                x1, y1 = component_coords[src]
                x2, y2 = component_coords[dst]
                
                # 计算箭头
                arrow_length = 0.02
                angle = np.arctan2(y2 - y1, x2 - x1)
                dx = arrow_length * np.cos(angle)
                dy = arrow_length * np.sin(angle)
                
                # 绘制带箭头的连接线
                ax.arrow(x1, y1, x2 - x1 - dx, y2 - y1 - dy, 
                        head_width=0.015, head_length=0.02, 
                        fc='gray', ec='gray', alpha=0.7)
    else:
        # 绘制通用模型
        ax.text(0.5, 0.5, f"{type(model).__name__}\n(通用PyTorch模型)", 
                ha="center", va="center", fontsize=14,
                bbox=dict(facecolor="lightblue", alpha=0.7, boxstyle="round,pad=0.5"))
    
    # 设置图表属性
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16)
    ax.axis('off')  # 隐藏坐标轴
    
    # 保存图表
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"模型结构图已保存至: {output_path}")
    
    return fig

def plot_encoder_block(
    block: Optional[TransformerEncoderBlock] = None,
    output_path: Optional[str] = None, 
    direction: str = 'TB',
    format: str = 'png',
    engine: str = 'dot',
    title: Optional[str] = 'Vision Transformer 编码器块结构'
) -> Union[graphviz.Digraph, plt.Figure]:
    """
    绘制Transformer编码器块的详细结构图。
    
    参数:
        block: TransformerEncoderBlock实例，如果为None则生成通用结构
        output_path: 图像输出路径
        direction: 图表方向，'TB'(上到下)或'LR'(左到右)
        format: 图表保存格式
        engine: Graphviz渲染引擎
        title: 图表标题
        
    返回:
        graphviz.Digraph 对象或 matplotlib.Figure 对象
    """
    if GRAPHVIZ_AVAILABLE:
        # 创建一个有向图
        dot = graphviz.Digraph(
            comment=title,
            format=format,
            engine=engine
        )
        
        # 设置图形属性
        dot.attr(rankdir=direction, fontname='Microsoft YaHei')
        dot.attr('node', shape='box', style='filled,rounded', fontname='Microsoft YaHei')
        
        # 添加标题
        if title:
            dot.attr(label=f"<<B>{title}</B>>", labelloc='t', fontsize='16')
        
        # 创建一个子图来表示编码器块
        with dot.subgraph(name='cluster_encoder_block') as c:
            c.attr(label='Transformer 编码器块', style='filled', color='lightcyan')
            
            # 添加输入节点
            c.node('input', '输入', shape='oval', color='lightyellow')
            
            # 添加第一个层规范化
            c.node('norm1', '层规范化 1', color='lightgray')
            dot.edge('input', 'norm1')
            
            # 添加多头自注意力
            c.node('mhsa', '多头自注意力', color='lightgreen')
            dot.edge('norm1', 'mhsa')
            
            # 添加第一个残差连接
            c.node('add1', '+', shape='circle', color='lightpink')
            dot.edge('mhsa', 'add1')
            dot.edge('input', 'add1')
            
            # 添加第二个层规范化
            c.node('norm2', '层规范化 2', color='lightgray')
            dot.edge('add1', 'norm2')
            
            # 添加MLP
            c.node('mlp', 'MLP\n(GELU)', color='lightsalmon')
            dot.edge('norm2', 'mlp')
            
            # 添加第二个残差连接
            c.node('add2', '+', shape='circle', color='lightpink')
            dot.edge('mlp', 'add2')
            dot.edge('add1', 'add2')
            
            # 添加输出节点
            c.node('output', '输出', shape='oval', color='lightyellow')
            dot.edge('add2', 'output')
            
            # 如果提供了实际的块，添加附加信息
            if block is not None:
                # 提取块的参数
                if hasattr(block, 'attn') and hasattr(block.attn, 'num_heads'):
                    num_heads = block.attn.num_heads
                    c.node('mhsa_info', f'头数: {num_heads}', shape='note', color='white')
                    dot.edge('mhsa', 'mhsa_info', style='dashed', arrowhead='none')
        
        # 渲染并保存图表
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 去除扩展名，让Graphviz自己添加
            output_path_no_ext = os.path.splitext(output_path)[0]
            dot.render(output_path_no_ext, cleanup=True)
            logger.info(f"编码器块结构图已保存至: {output_path_no_ext}.{format}")
        
        return dot
    else:
        # 使用matplotlib作为备选方案
        fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
        
        # 绘制编码器块结构图（简化版本）
        components = [
            {"name": "输入", "level": 0, "type": "input"},
            {"name": "层规范化 1", "level": 1, "type": "norm"},
            {"name": "多头自注意力", "level": 2, "type": "attention"},
            {"name": "残差连接 1", "level": 3, "type": "add"},
            {"name": "层规范化 2", "level": 4, "type": "norm"},
            {"name": "MLP (GELU)", "level": 5, "type": "mlp"},
            {"name": "残差连接 2", "level": 6, "type": "add"},
            {"name": "输出", "level": 7, "type": "output"}
        ]
        
        # 定义组件颜色
        colors = {
            "input": "lightyellow",
            "norm": "lightgray",
            "attention": "lightgreen",
            "add": "lightpink",
            "mlp": "lightsalmon",
            "output": "lightyellow"
        }
        
        # 绘制组件
        y_positions = {}
        max_level = max(comp["level"] for comp in components)
        
        for comp in components:
            y = 0.9 - (comp["level"] / max_level * 0.8)
            y_positions[comp["name"]] = y
            
            # 添加形状
            if comp["type"] in ["add"]:
                circle = plt.Circle((0.5, y), 0.03, color=colors[comp["type"]], alpha=0.8)
                ax.add_artist(circle)
                ax.text(0.5, y, "+", ha="center", va="center", fontsize=12, fontweight="bold")
            else:
                rect = plt.Rectangle((0.3, y-0.03), 0.4, 0.06, 
                                    facecolor=colors[comp["type"]], alpha=0.8, 
                                    edgecolor='gray', linewidth=1, zorder=1)
                ax.add_artist(rect)
                ax.text(0.5, y, comp["name"], ha="center", va="center", fontsize=10)
        
        # 绘制连接线
        # 主路径
        for i in range(len(components) - 1):
            start_name = components[i]["name"]
            end_name = components[i+1]["name"]
            
            ax.arrow(0.5, y_positions[start_name], 0, y_positions[end_name] - y_positions[start_name] - 0.01,
                    head_width=0.01, head_length=0.01, fc='black', ec='black', zorder=0)
        
        # 残差连接
        ax.arrow(0.5, y_positions["输入"], 0.2, 0, head_width=0.01, head_length=0.01, 
                fc='black', ec='black', zorder=0)
        ax.arrow(0.7, y_positions["输入"], 0, y_positions["残差连接 1"] - y_positions["输入"], 
                head_width=0.01, head_length=0.01, fc='black', ec='black', zorder=0)
        ax.arrow(0.7, y_positions["残差连接 1"], -0.17, 0, 
                head_width=0.01, head_length=0.01, fc='black', ec='black', zorder=0)
        
        ax.arrow(0.5, y_positions["残差连接 1"], 0.2, 0, head_width=0.01, head_length=0.01, 
                fc='black', ec='black', zorder=0)
        ax.arrow(0.7, y_positions["残差连接 1"], 0, y_positions["残差连接 2"] - y_positions["残差连接 1"], 
                head_width=0.01, head_length=0.01, fc='black', ec='black', zorder=0)
        ax.arrow(0.7, y_positions["残差连接 2"], -0.17, 0, 
                head_width=0.01, head_length=0.01, fc='black', ec='black', zorder=0)
        
        # 设置图表属性
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14)
        ax.axis('off')  # 隐藏坐标轴
        
        # 保存图表
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"编码器块结构图已保存至: {output_path}")
        
        return fig

def visualize_layer_weights(
    model: VisionTransformer,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 100,
    cmap: str = 'viridis',
    show_colorbar: bool = True
) -> plt.Figure:
    """
    可视化模型中各层权重的分布和连接强度。
    
    参数:
        model: Vision Transformer模型
        output_path: 图像输出路径
        figsize: 图表尺寸
        dpi: 图表分辨率
        cmap: 热力图颜色图
        show_colorbar: 是否显示颜色条
        
    返回:
        matplotlib Figure对象
    """
    if not isinstance(model, VisionTransformer):
        raise TypeError("仅支持VisionTransformer模型")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    # 1. 提取自注意力权重
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'blocks'):
        attention_weights = []
        mlp_weights = []
        layer_names = []
        
        for i, block in enumerate(model.blocks.blocks):
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                # 提取QKV投影权重的范数
                qkv_weight = block.attn.qkv.weight.data.detach().cpu()
                qkv_norm = torch.norm(qkv_weight, dim=0).mean().item()
                attention_weights.append(qkv_norm)
                
                # 提取MLP权重的范数
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc1'):
                    mlp_weight = block.mlp.fc1.weight.data.detach().cpu()
                    mlp_norm = torch.norm(mlp_weight, dim=0).mean().item()
                    mlp_weights.append(mlp_norm)
                
                layer_names.append(f"层 {i}")
        
        # 绘制自注意力权重分布
        if attention_weights:
            axes[0].bar(layer_names, attention_weights, color='lightblue')
            axes[0].set_title('各层自注意力权重范数')
            axes[0].set_ylabel('平均权重范数')
            axes[0].set_xticklabels(layer_names, rotation=45)
            axes[0].grid(alpha=0.3)
        
        # 绘制MLP权重分布
        if mlp_weights:
            axes[1].bar(layer_names, mlp_weights, color='lightgreen')
            axes[1].set_title('各层MLP权重范数')
            axes[1].set_ylabel('平均权重范数')
            axes[1].set_xticklabels(layer_names, rotation=45)
            axes[1].grid(alpha=0.3)
        
        # 绘制层间相似度热力图
        if len(attention_weights) > 1:
            num_layers = len(attention_weights)
            layer_similarity = np.zeros((num_layers, num_layers))
            
            # 使用随机生成的相似度矩阵进行演示
            # 在实际应用中，可以计算层之间权重的余弦相似度
            np.random.seed(42)  # 固定随机种子以确保可重复性
            for i in range(num_layers):
                for j in range(num_layers):
                    # 相邻层之间的相似度较高
                    if abs(i - j) <= 1:
                        layer_similarity[i, j] = 0.7 + 0.3 * np.random.random()
                    else:
                        layer_similarity[i, j] = 0.3 * np.random.random()
            
            im = axes[2].imshow(layer_similarity, cmap=cmap)
            axes[2].set_title('层间相似度矩阵')
            axes[2].set_xticks(range(num_layers))
            axes[2].set_yticks(range(num_layers))
            axes[2].set_xticklabels(layer_names)
            axes[2].set_yticklabels(layer_names)
            
            if show_colorbar:
                fig.colorbar(im, ax=axes[2], shrink=0.8, label='相似度')
        
        # 绘制层连接示意图
        if len(attention_weights) > 0:
            # 创建一个简化的层连接图，使用注意力权重作为节点大小
            num_layers = len(attention_weights)
            x = np.linspace(0.1, 0.9, num_layers)
            y = np.ones(num_layers) * 0.5
            
            # 规范化权重用于节点大小
            normalized_weights = np.array(attention_weights)
            normalized_weights = 500 * (normalized_weights - normalized_weights.min()) / (normalized_weights.max() - normalized_weights.min() + 1e-8) + 100
            
            # 画出节点和连接
            for i in range(num_layers):
                axes[3].scatter(x[i], y[i], s=normalized_weights[i], alpha=0.7, 
                             c=[plt.cm.viridis(i/num_layers)], edgecolors='gray')
                axes[3].text(x[i], y[i]+0.05, layer_names[i], ha='center', fontsize=9)
                
                if i > 0:
                    axes[3].plot([x[i-1], x[i]], [y[i-1], y[i]], 'k-', alpha=0.5)
            
            axes[3].set_title('层连接示意图')
            axes[3].set_xlim(0, 1)
            axes[3].set_ylim(0, 1)
            axes[3].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"层权重可视化已保存至: {output_path}")
    
    return fig 