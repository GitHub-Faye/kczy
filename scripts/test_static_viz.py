#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试静态可视化功能。

本脚本用于测试整合的静态可视化功能，包括模型概览、注意力分析、综合可视化和模型比较。
"""

import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vit import VisionTransformer
from src.visualization import (
    create_model_overview,
    create_attention_analysis,
    create_comprehensive_visualization,
    compare_models
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试静态可视化功能")
    parser.add_argument('--image', type=str, default=None, 
                        help='输入图像路径。如果不提供，将使用随机生成的图像。')
    parser.add_argument('--model', type=str, default=None, 
                        help='预训练模型路径。如果不提供，将使用随机初始化的模型。')
    parser.add_argument('--output_dir', type=str, default='temp_metrics/plots', 
                        help='输出目录')
    parser.add_argument('--mode', type=str, default='comprehensive', 
                        choices=['overview', 'attention', 'comprehensive', 'compare', 'all'], 
                        help='要测试的可视化模式')
    parser.add_argument('--format', type=str, default='png', 
                        choices=['png', 'jpg', 'svg', 'pdf'], 
                        help='输出图像格式')
    parser.add_argument('--dpi', type=int, default=150, 
                        help='输出图像的DPI')
    parser.add_argument('--no_html', action='store_true', 
                        help='不生成HTML报告')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logger.info(f"开始静态可视化测试，模式：{args.mode}")
    logger.info(f"输出目录：{args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载或创建模型
    if args.model is not None and os.path.exists(args.model):
        logger.info(f"加载预训练模型: {args.model}")
        model, _ = VisionTransformer.load_model(args.model, device=device)
    else:
        logger.info("使用随机初始化的模型")
        # 创建一个小型ViT模型
        model = VisionTransformer.create_tiny(num_classes=10)
        model = model.to(device)
    
    # 加载或创建输入图像
    if args.image is not None and os.path.exists(args.image):
        logger.info(f"加载图像: {args.image}")
        img = Image.open(args.image).convert('RGB')
        img_size = model.img_size if hasattr(model, 'img_size') else 224
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
    else:
        logger.info("使用随机生成的图像")
        # 创建随机图像
        img_size = model.img_size if hasattr(model, 'img_size') else 224
        img_tensor = torch.randn(1, 3, img_size, img_size).to(device)
    
    # 根据选择的模式运行可视化测试
    if args.mode in ['overview', 'all']:
        logger.info("=== 测试模型概览 ===")
        overview_path = os.path.join(args.output_dir, f'vit_model_overview.{args.format}')
        create_model_overview(
            model=model,
            output_path=overview_path,
            format=args.format,
            dpi=args.dpi
        )
        logger.info(f"模型概览图已保存至: {overview_path}")
    
    if args.mode in ['attention', 'all']:
        logger.info("=== 测试注意力分析 ===")
        attention_path = os.path.join(args.output_dir, f'vit_attention_analysis.{args.format}')
        create_attention_analysis(
            model=model,
            input_image=img_tensor,
            output_path=attention_path,
            format=args.format,
            dpi=args.dpi
        )
        logger.info(f"注意力分析图已保存至: {attention_path}")
    
    if args.mode in ['comprehensive', 'all']:
        logger.info("=== 测试综合可视化 ===")
        output_files = create_comprehensive_visualization(
            model=model,
            input_image=img_tensor,
            output_dir=args.output_dir,
            prefix='vit',
            format=args.format,
            dpi=args.dpi,
            create_html=not args.no_html
        )
        logger.info("综合可视化文件:")
        for name, path in output_files.items():
            logger.info(f"- {name}: {path}")
    
    if args.mode in ['compare', 'all']:
        logger.info("=== 测试模型比较 ===")
        # 创建另一个不同配置的模型用于比较
        logger.info("创建第二个模型(VIT-Small)进行比较")
        model2 = VisionTransformer.create_small(num_classes=10)
        model2 = model2.to(device)
        
        compare_path = os.path.join(args.output_dir, f'vit_models_comparison.{args.format}')
        compare_models(
            models=[model, model2],
            model_names=["VIT-Tiny", "VIT-Small"],
            input_image=img_tensor,
            output_path=compare_path,
            format=args.format,
            dpi=args.dpi
        )
        logger.info(f"模型比较图已保存至: {compare_path}")
    
    logger.info("所有测试完成!")

if __name__ == '__main__':
    main() 