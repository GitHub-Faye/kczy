"""
静态可视化模块。
整合注意力权重和模型结构可视化为一套连贯的静态图表。
"""

import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib.colors import Normalize
from typing import List, Tuple, Dict, Optional, Union, Any, Set
from pathlib import Path
import base64
import io
from datetime import datetime

from src.models.vit import VisionTransformer
from src.visualization.attention_viz import (
    plot_attention_weights,
    visualize_attention_on_image,
    visualize_all_heads
)
from src.visualization.model_viz import (
    plot_model_structure,
    plot_encoder_block,
    visualize_layer_weights
)

# 设置日志
logger = logging.getLogger(__name__)

def create_model_overview(
    model: VisionTransformer,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150,
    show_details: bool = True,
    format: str = 'png',
    title: Optional[str] = None
) -> plt.Figure:
    """
    生成模型结构概览图，包含模型结构和主要参数信息。
    
    参数:
        model: Vision Transformer模型
        output_path: 输出文件路径，如果为None则不保存文件
        figsize: 图表大小
        dpi: 图像分辨率
        show_details: 是否显示详细信息
        format: 输出文件格式
        title: 图表标题
        
    返回:
        matplotlib Figure对象
    """
    if title is None:
        title = f"{type(model).__name__} 模型结构概览"
    
    # 创建一个包含两个子图的布局
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2])
    
    # 左侧放模型结构图（只显示主要组件）
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.axis('off')
    
    # 临时保存模型结构图
    tmp_model_img_path = "temp_model_structure.png" if output_path else None
    model_fig = plot_model_structure(
        model=model, 
        output_path=tmp_model_img_path,
        show_details=False,
        format=format,
        title=None
    )
    
    # 如果使用了Graphviz，需要加载保存的图像
    if tmp_model_img_path and os.path.exists(tmp_model_img_path):
        img = plt.imread(tmp_model_img_path)
        ax_left.imshow(img)
        os.remove(tmp_model_img_path)
    else:
        # 如果使用了matplotlib创建结构图，将图像添加到当前图表
        ax_left.text(0.5, 0.5, "模型结构图 (需要Graphviz库)", 
                    ha='center', va='center', fontsize=12)
    
    ax_left.set_title("模型结构", fontsize=14)
    
    # 右侧放模型参数信息表格
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.axis('off')
    
    # 获取模型参数信息
    model_info = []
    model_info.append(["模型类型", type(model).__name__])
    
    if hasattr(model, 'img_size'):
        model_info.append(["图像大小", f"{model.img_size} x {model.img_size}"])
    
    if hasattr(model, 'patch_size'):
        model_info.append(["补丁大小", f"{model.patch_size} x {model.patch_size}"])
    
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'num_patches'):
        model_info.append(["补丁数量", model.patch_embed.num_patches])
    
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'depth'):
        model_info.append(["Transformer层数", model.blocks.depth])
    
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'blocks') and len(model.blocks.blocks) > 0:
        if hasattr(model.blocks.blocks[0], 'attn') and hasattr(model.blocks.blocks[0].attn, 'num_heads'):
            model_info.append(["注意力头数", model.blocks.blocks[0].attn.num_heads])
    
    if hasattr(model, 'embed_dim'):
        model_info.append(["嵌入维度", model.embed_dim])
    
    if hasattr(model, 'num_classes'):
        model_info.append(["类别数量", model.num_classes])

    total_params = sum(p.numel() for p in model.parameters())
    model_info.append(["总参数量", f"{total_params:,}"])
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info.append(["可训练参数", f"{trainable_params:,}"])
    
    # 创建表格
    table = ax_right.table(
        cellText=model_info,
        colLabels=["参数", "值"],
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.6]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    
    # 为表格添加标题
    ax_right.set_title("模型参数信息", fontsize=14)
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 保存图表
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
        logger.info(f"模型概览图已保存至: {output_path}")
    
    return fig

def create_attention_analysis(
    model: VisionTransformer,
    input_image: torch.Tensor,
    output_path: Optional[str] = None,
    layer_indices: Optional[List[int]] = None,
    head_index: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150,
    format: str = 'png',
    title: Optional[str] = None
) -> plt.Figure:
    """
    生成注意力权重分析图，包含注意力热力图和图像叠加视图。
    
    参数:
        model: Vision Transformer模型
        input_image: 输入图像张量，形状为 [1, 3, H, W]
        output_path: 输出文件路径，如果为None则不保存文件
        layer_indices: 要可视化的层索引列表，如果为None则使用首尾和中间层
        head_index: 要可视化的注意力头索引，如果为None则使用所有头的平均值
        figsize: 图表大小
        dpi: 图像分辨率
        format: 输出文件格式
        title: 图表标题
        
    返回:
        matplotlib Figure对象
    """
    if title is None:
        title = "注意力权重分析"
    
    # 确保模型处于评估模式
    model.eval()
    
    # 获取注意力权重
    with torch.no_grad():
        outputs, attention_weights = model(input_image, return_attention=True)
    
    # 获取层数
    num_layers = len(attention_weights)
    
    # 如果没有指定层索引，则选择首尾和中间层
    if layer_indices is None:
        if num_layers <= 3:
            layer_indices = list(range(num_layers))
        else:
            layer_indices = [0, num_layers // 2, num_layers - 1]
    
    # 确保所有层索引有效
    layer_indices = [i for i in layer_indices if 0 <= i < num_layers]
    if not layer_indices:
        raise ValueError("没有有效的层索引")
    
    # 获取模型参数
    img_size = model.img_size if hasattr(model, 'img_size') else 224
    patch_size = model.patch_size if hasattr(model, 'patch_size') else 16
    
    # 创建一个大的图表
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    
    # 为每一层创建一个子图行（注意力热力图 + 图像叠加视图）
    num_selected_layers = len(layer_indices)
    gs = gridspec.GridSpec(num_selected_layers, 2, figure=fig)
    
    for i, layer_idx in enumerate(layer_indices):
        # 注意力热力图
        ax_heatmap = fig.add_subplot(gs[i, 0])
        plot_attention_weights(
            attention_weights, 
            layer_idx=layer_idx, 
            head_idx=head_index,
            cmap='viridis',
            title=None
        )
        layer_name = f"层 {layer_idx}"
        head_name = f"头 {head_index}" if head_index is not None else "所有头的平均值"
        ax_heatmap.set_title(f"{layer_name} - {head_name} - 注意力热力图", fontsize=12)
        
        # 图像叠加视图
        ax_overlay = fig.add_subplot(gs[i, 1])
        visualize_attention_on_image(
            input_image,
            attention_weights,
            patch_size=patch_size,
            layer_idx=layer_idx,
            head_idx=head_index,
            cls_token_attention=True,
            title=None
        )
        ax_overlay.set_title(f"{layer_name} - {head_name} - 注意力叠加图", fontsize=12)
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 保存图表
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
        logger.info(f"注意力分析图已保存至: {output_path}")
    
    return fig

def create_comprehensive_visualization(
    model: VisionTransformer,
    input_image: torch.Tensor,
    output_dir: str,
    prefix: str = 'vit',
    format: str = 'png',
    dpi: int = 150,
    create_html: bool = True
) -> Dict[str, str]:
    """
    生成一套完整的可视化图表，包括模型结构、注意力权重和层连接。
    
    参数:
        model: Vision Transformer模型
        input_image: 输入图像张量，形状为 [1, 3, H, W]
        output_dir: 输出目录路径
        prefix: 文件名前缀
        format: 输出文件格式
        dpi: 图像分辨率
        create_html: 是否创建HTML报告
        
    返回:
        包含各个生成文件路径的字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = {}
    
    # 1. 生成模型概览图
    overview_path = os.path.join(output_dir, f"{prefix}_model_overview_{timestamp}.{format}")
    create_model_overview(
        model=model,
        output_path=overview_path,
        format=format,
        dpi=dpi
    )
    output_files['model_overview'] = overview_path
    
    # 2. 生成模型结构图
    structure_path = os.path.join(output_dir, f"{prefix}_structure_{timestamp}.{format}")
    plot_model_structure(
        model=model,
        output_path=structure_path,
        format=format,
        title=f"Vision Transformer 模型结构"
    )
    output_files['model_structure'] = structure_path
    
    # 3. 生成编码器块结构图
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'blocks') and len(model.blocks.blocks) > 0:
        encoder_block = model.blocks.blocks[0]
        block_path = os.path.join(output_dir, f"{prefix}_encoder_block_{timestamp}.{format}")
        plot_encoder_block(
            block=encoder_block,
            output_path=block_path,
            format=format,
            title=f"Vision Transformer 编码器块结构"
        )
        output_files['encoder_block'] = block_path
    
    # 4. 生成层权重可视化
    weights_path = os.path.join(output_dir, f"{prefix}_layer_weights_{timestamp}.{format}")
    visualize_layer_weights(
        model=model,
        output_path=weights_path,
        dpi=dpi
    )
    output_files['layer_weights'] = weights_path
    
    # 5. 生成注意力分析图
    attn_path = os.path.join(output_dir, f"{prefix}_attention_analysis_{timestamp}.{format}")
    create_attention_analysis(
        model=model,
        input_image=input_image,
        output_path=attn_path,
        format=format,
        dpi=dpi
    )
    output_files['attention_analysis'] = attn_path
    
    # 6. 生成所有注意力头的可视化（使用最后一层）
    with torch.no_grad():
        _, attention_weights = model(input_image, return_attention=True)
    
    patch_size = model.patch_size if hasattr(model, 'patch_size') else 16
    layer_idx = len(attention_weights) - 1  # 最后一层
    
    all_heads_path = os.path.join(output_dir, f"{prefix}_all_attention_heads_{timestamp}.{format}")
    visualize_all_heads(
        input_image,
        attention_weights,
        patch_size=patch_size,
        layer_idx=layer_idx,
        save_path=all_heads_path,
        dpi=dpi
    )
    output_files['all_heads'] = all_heads_path
    
    # 7. 如果需要，创建HTML报告
    if create_html:
        html_path = os.path.join(output_dir, f"{prefix}_visualization_report_{timestamp}.html")
        generate_visualization_report(
            model=model,
            input_image=input_image,
            output_files=output_files,
            html_path=html_path,
            report_title=f"Vision Transformer 模型可视化报告 - {timestamp}"
        )
        output_files['html_report'] = html_path
    
    return output_files

def compare_models(
    models: List[VisionTransformer],
    model_names: List[str],
    input_image: torch.Tensor,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150,
    format: str = 'png',
    title: Optional[str] = None
) -> plt.Figure:
    """
    比较多个模型的结构和注意力特性。
    
    参数:
        models: Vision Transformer模型列表
        model_names: 模型名称列表
        input_image: 输入图像张量，形状为 [1, 3, H, W]
        output_path: 输出文件路径，如果为None则不保存文件
        figsize: 图表大小
        dpi: 图像分辨率
        format: 输出文件格式
        title: 图表标题
        
    返回:
        matplotlib Figure对象
    """
    if len(models) != len(model_names):
        raise ValueError("models和model_names的长度必须一致")
    
    if title is None:
        title = "模型比较"
    
    num_models = len(models)
    
    # 创建一个大的图表
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    
    # 为每个模型创建一个行，包含模型信息和最后一层的注意力可视化
    gs = gridspec.GridSpec(num_models, 2, figure=fig, width_ratios=[1, 2])
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        # 左侧放模型参数信息
        ax_info = fig.add_subplot(gs[i, 0])
        ax_info.axis('off')
        
        # 获取模型参数信息
        model_info = []
        model_info.append(["模型名称", name])
        
        if hasattr(model, 'img_size'):
            model_info.append(["图像大小", f"{model.img_size} x {model.img_size}"])
        
        if hasattr(model, 'patch_size'):
            model_info.append(["补丁大小", f"{model.patch_size} x {model.patch_size}"])
        
        if hasattr(model, 'blocks') and hasattr(model.blocks, 'depth'):
            model_info.append(["Transformer层数", model.blocks.depth])
        
        if hasattr(model, 'blocks') and hasattr(model.blocks, 'blocks') and len(model.blocks.blocks) > 0:
            if hasattr(model.blocks.blocks[0], 'attn') and hasattr(model.blocks.blocks[0].attn, 'num_heads'):
                model_info.append(["注意力头数", model.blocks.blocks[0].attn.num_heads])
        
        if hasattr(model, 'embed_dim'):
            model_info.append(["嵌入维度", model.embed_dim])
        
        total_params = sum(p.numel() for p in model.parameters())
        model_info.append(["总参数量", f"{total_params:,}"])
        
        # 创建表格
        table = ax_info.table(
            cellText=model_info,
            colLabels=["参数", "值"],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.6]
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 为表格添加标题
        ax_info.set_title(f"模型 {i+1}: {name}", fontsize=12)
        
        # 右侧放最后一层的注意力可视化
        ax_attn = fig.add_subplot(gs[i, 1])
        
        # 获取注意力权重
        model.eval()
        with torch.no_grad():
            _, attention_weights = model(input_image, return_attention=True)
        
        # 获取最后一层
        layer_idx = len(attention_weights) - 1
        
        # 获取补丁大小
        patch_size = model.patch_size if hasattr(model, 'patch_size') else 16
        
        # 将注意力权重叠加到图像上
        visualize_attention_on_image(
            input_image,
            attention_weights,
            patch_size=patch_size,
            layer_idx=layer_idx,
            head_idx=None,  # 使用所有头的平均值
            cls_token_attention=True,
            title=None
        )
        
        ax_attn.set_title(f"{name} - 最后一层注意力权重", fontsize=12)
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 保存图表
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
        logger.info(f"模型比较图已保存至: {output_path}")
    
    return fig

def encode_image_base64(img_path: str) -> str:
    """
    将图像编码为base64字符串，用于在HTML中嵌入图像。
    
    参数:
        img_path: 图像文件路径
        
    返回:
        base64编码的图像字符串
    """
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    # 获取文件扩展名，确定MIME类型
    extension = os.path.splitext(img_path)[1].lower()
    if extension == '.png':
        mime_type = 'image/png'
    elif extension in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif extension == '.gif':
        mime_type = 'image/gif'
    elif extension == '.svg':
        mime_type = 'image/svg+xml'
    else:
        mime_type = 'image/png'  # 默认
    
    return f"data:{mime_type};base64,{encoded_string}"

def generate_visualization_report(
    model: VisionTransformer,
    input_image: torch.Tensor,
    output_files: Dict[str, str],
    html_path: str,
    report_title: str = "模型可视化报告"
) -> None:
    """
    生成包含各种可视化的HTML格式报告。
    
    参数:
        model: Vision Transformer模型
        input_image: 输入图像张量
        output_files: 包含各个可视化文件路径的字典
        html_path: HTML报告输出路径
        report_title: 报告标题
    """
    # 获取模型信息
    model_type = type(model).__name__
    img_size = model.img_size if hasattr(model, 'img_size') else "未知"
    patch_size = model.patch_size if hasattr(model, 'patch_size') else "未知"
    
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'depth'):
        num_layers = model.blocks.depth
    else:
        num_layers = "未知"
    
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'blocks') and len(model.blocks.blocks) > 0:
        if hasattr(model.blocks.blocks[0], 'attn') and hasattr(model.blocks.blocks[0].attn, 'num_heads'):
            num_heads = model.blocks.blocks[0].attn.num_heads
        else:
            num_heads = "未知"
    else:
        num_heads = "未知"
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # 准备HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {{
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2, h3 {{
                color: #0066cc;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
            }}
            h2 {{
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            .model-info {{
                background-color: #e9f7fe;
                border-left: 4px solid #0066cc;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .model-info table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .model-info th, .model-info td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .model-info th {{
                width: 40%;
                font-weight: bold;
            }}
            .visualization {{
                margin: 30px 0;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
                border: 1px solid #ddd;
            }}
            .caption {{
                text-align: center;
                margin-top: 10px;
                font-style: italic;
                color: #666;
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #666;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{report_title}</h1>
        
        <div class="model-info">
            <h2>模型信息</h2>
            <table>
                <tr><th>模型类型</th><td>{model_type}</td></tr>
                <tr><th>图像大小</th><td>{img_size} x {img_size}</td></tr>
                <tr><th>补丁大小</th><td>{patch_size} x {patch_size}</td></tr>
                <tr><th>Transformer层数</th><td>{num_layers}</td></tr>
                <tr><th>注意力头数</th><td>{num_heads}</td></tr>
                <tr><th>总参数量</th><td>{total_params:,}</td></tr>
                <tr><th>生成时间</th><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
        </div>
    """
    
    # 添加各个可视化部分
    if 'model_overview' in output_files:
        img_path = output_files['model_overview']
        img_data = encode_image_base64(img_path)
        html_content += f"""
        <div class="visualization">
            <h2>模型概览</h2>
            <img src="{img_data}" alt="模型概览" />
            <p class="caption">图1：模型结构和参数信息概览</p>
        </div>
        """
    
    if 'model_structure' in output_files:
        img_path = output_files['model_structure']
        img_data = encode_image_base64(img_path)
        html_content += f"""
        <div class="visualization">
            <h2>模型结构图</h2>
            <img src="{img_data}" alt="模型结构图" />
            <p class="caption">图2：Vision Transformer模型的整体结构，展示了从输入到输出的数据流和主要组件</p>
        </div>
        """
    
    if 'encoder_block' in output_files:
        img_path = output_files['encoder_block']
        img_data = encode_image_base64(img_path)
        html_content += f"""
        <div class="visualization">
            <h2>编码器块结构</h2>
            <img src="{img_data}" alt="编码器块结构" />
            <p class="caption">图3：Transformer编码器块的内部结构，包括多头注意力、MLP和残差连接</p>
        </div>
        """
    
    if 'layer_weights' in output_files:
        img_path = output_files['layer_weights']
        img_data = encode_image_base64(img_path)
        html_content += f"""
        <div class="visualization">
            <h2>层权重分析</h2>
            <img src="{img_data}" alt="层权重分析" />
            <p class="caption">图4：模型各层权重的分布和连接强度分析</p>
        </div>
        """
    
    if 'attention_analysis' in output_files:
        img_path = output_files['attention_analysis']
        img_data = encode_image_base64(img_path)
        html_content += f"""
        <div class="visualization">
            <h2>注意力权重分析</h2>
            <img src="{img_data}" alt="注意力权重分析" />
            <p class="caption">图5：不同层的注意力热力图和叠加到输入图像上的注意力权重</p>
        </div>
        """
    
    if 'all_heads' in output_files:
        img_path = output_files['all_heads']
        img_data = encode_image_base64(img_path)
        html_content += f"""
        <div class="visualization">
            <h2>所有注意力头</h2>
            <img src="{img_data}" alt="所有注意力头" />
            <p class="caption">图6：最后一层的所有注意力头，展示了不同头关注的区域差异</p>
        </div>
        """
    
    # 添加页脚
    html_content += f"""
        <footer>
            <p>此报告由Vision Transformer可视化工具自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML报告已生成：{html_path}") 