#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Vision Transformer模型结构可视化功能。
"""

import os
import sys
import argparse

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.vit import VisionTransformer
from src.visualization.model_viz import (
    plot_model_structure,
    plot_encoder_block,
    visualize_layer_weights
)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="ViT模型结构可视化测试")
    
    parser.add_argument('--output-dir', type=str, default='temp_metrics/plots',
                        help='输出目录路径')
    parser.add_argument('--show-details', action='store_true', default=True,
                        help='显示模型内部细节')
    parser.add_argument('--model-type', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large', 'huge'],
                        help='ViT模型类型')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'svg', 'pdf'],
                        help='输出图表格式')
    parser.add_argument('--direction', type=str, default='TB',
                        choices=['TB', 'LR'],
                        help='图表方向 (TB: 上到下, LR: 左到右)')
    parser.add_argument('--default-block', action='store_true',
                        help='生成默认编码器块结构图（无需模型实例）')
    
    return parser.parse_args()

def main():
    """主函数。"""
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果指定了默认编码器块，不需要加载模型
    if args.default_block:
        print("正在生成默认编码器块结构图...")
        block_structure_path = os.path.join(args.output_dir, f'vit_encoder_block_default.{args.format}')
        plot_encoder_block(
            block=None,  # 使用None生成通用结构
            output_path=block_structure_path,
            direction=args.direction,
            format=args.format,
            title='Vision Transformer 编码器块通用结构'
        )
        print(f"默认编码器块结构图已保存至: {block_structure_path}")
        print("可视化完成！")
        return
    
    print(f"正在创建 ViT-{args.model_type} 模型...")
    
    # 创建ViT模型
    if args.model_type == 'tiny':
        model = VisionTransformer.create_tiny()
    elif args.model_type == 'small':
        model = VisionTransformer.create_small()
    elif args.model_type == 'base':
        model = VisionTransformer.create_base()
    elif args.model_type == 'large':
        model = VisionTransformer.create_large()
    elif args.model_type == 'huge':
        model = VisionTransformer.create_huge()
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 生成模型结构图
    print("正在生成整体模型结构图...")
    model_structure_path = os.path.join(args.output_dir, f'vit_{args.model_type}_structure.{args.format}')
    plot_model_structure(
        model=model,
        output_path=model_structure_path,
        show_details=args.show_details,
        direction=args.direction,
        format=args.format,
        title=f'Vision Transformer ({args.model_type}) 模型结构'
    )
    print(f"整体模型结构图已保存至: {model_structure_path}")
    
    # 生成编码器块结构图
    print("正在生成编码器块结构图...")
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'blocks') and len(model.blocks.blocks) > 0:
        encoder_block = model.blocks.blocks[0]
        block_structure_path = os.path.join(args.output_dir, f'vit_{args.model_type}_encoder_block.{args.format}')
        plot_encoder_block(
            block=encoder_block,
            output_path=block_structure_path,
            direction=args.direction,
            format=args.format,
            title=f'Vision Transformer ({args.model_type}) 编码器块结构'
        )
        print(f"编码器块结构图已保存至: {block_structure_path}")
    else:
        print("无法获取编码器块，将生成默认编码器块结构图...")
        block_structure_path = os.path.join(args.output_dir, f'vit_encoder_block_default.{args.format}')
        plot_encoder_block(
            block=None,  # 使用None生成通用结构
            output_path=block_structure_path,
            direction=args.direction,
            format=args.format,
            title='Vision Transformer 编码器块通用结构'
        )
        print(f"默认编码器块结构图已保存至: {block_structure_path}")
    
    # 生成层权重可视化
    print("正在生成层权重可视化...")
    weights_viz_path = os.path.join(args.output_dir, f'vit_{args.model_type}_layer_weights.{args.format}')
    visualize_layer_weights(
        model=model,
        output_path=weights_viz_path,
        dpi=150
    )
    print(f"层权重可视化已保存至: {weights_viz_path}")
    
    print("所有可视化已完成！")

if __name__ == '__main__':
    main() 