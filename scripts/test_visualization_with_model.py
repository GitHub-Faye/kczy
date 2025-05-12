#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用训练模型测试可视化功能。

本脚本测试视觉转换器(ViT)模型的可视化功能，包括注意力权重可视化、模型结构可视化和综合静态可视化。
它可以使用随机初始化的模型或预训练模型，并支持多种输入图像和输出格式。
"""

import os
import sys
import argparse
import torch
import logging
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目模块
from src.models.vit import VisionTransformer
from src.visualization import (
    # 注意力可视化
    plot_attention_weights,
    visualize_attention_on_image,
    visualize_all_heads,
    # 模型结构可视化
    plot_model_structure,
    plot_encoder_block,
    visualize_layer_weights,
    # 静态综合可视化
    create_model_overview,
    create_attention_analysis,
    create_comprehensive_visualization,
    compare_models
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用训练模型测试可视化功能")
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default=None, 
                        help='预训练模型路径。如果不提供，将使用随机初始化的模型。')
    parser.add_argument('--model_type', type=str, default='tiny', 
                        choices=['tiny', 'small', 'base', 'large', 'huge'], 
                        help='如果不使用预训练模型，指定要创建的模型类型')
    parser.add_argument('--num_classes', type=int, default=10, 
                        help='模型的类别数量')
    
    # 图像相关参数
    parser.add_argument('--image', type=str, default=None, 
                        help='输入图像路径。如果不提供，将使用随机生成的图像或从data/images中随机选择。')
    parser.add_argument('--random_image', action='store_true',
                        help='使用随机生成的图像，而不是加载真实图像')
    parser.add_argument('--sample_dir', type=str, default='data/images', 
                        help='如果未指定图像，从该目录随机选择图像')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='temp_metrics/plots', 
                        help='输出目录')
    parser.add_argument('--format', type=str, default='png', 
                        choices=['png', 'jpg', 'svg', 'pdf'], 
                        help='输出图像格式')
    parser.add_argument('--dpi', type=int, default=150, 
                        help='输出图像的DPI')
    parser.add_argument('--prefix', type=str, default='vit_test', 
                        help='输出文件名前缀')
    parser.add_argument('--add_timestamp', action='store_true',
                        help='在输出文件名中添加时间戳')
    
    # 可视化模式参数
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['attention', 'structure', 'static', 'all'], 
                        help='要测试的可视化模式')
    parser.add_argument('--no_html', action='store_true', 
                        help='不生成HTML报告')
    
    # 比较模式参数
    parser.add_argument('--compare', action='store_true',
                        help='启用模型比较模式，将创建两个不同大小的模型进行比较')
    
    # 调试参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于重现结果')
    parser.add_argument('--verbose', action='store_true',
                        help='启用详细输出')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"已设置随机种子: {seed}")

def get_device():
    """获取计算设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    return device

def load_or_create_model(args, device):
    """加载预训练模型或创建随机初始化的模型"""
    # 如果提供了模型路径且文件存在，加载模型
    if args.model is not None and os.path.exists(args.model):
        logger.info(f"加载预训练模型: {args.model}")
        model, _ = VisionTransformer.load_model(args.model, device=device)
        return model
    
    # 否则，根据指定的模型类型创建随机初始化的模型
    logger.info(f"创建随机初始化的模型，类型: {args.model_type}")
    
    if args.model_type == 'tiny':
        model = VisionTransformer.create_tiny(num_classes=args.num_classes)
    elif args.model_type == 'small':
        model = VisionTransformer.create_small(num_classes=args.num_classes)
    elif args.model_type == 'base':
        model = VisionTransformer.create_base(num_classes=args.num_classes)
    elif args.model_type == 'large':
        model = VisionTransformer.create_large(num_classes=args.num_classes)
    elif args.model_type == 'huge':
        model = VisionTransformer.create_huge(num_classes=args.num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    return model.to(device)

def get_input_image(args, model, device):
    """获取输入图像，可以是指定的图像、随机选择的图像或随机生成的图像"""
    img_size = model.img_size if hasattr(model, 'img_size') else 224
    
    # 1. 如果用户指定了使用随机生成的图像
    if args.random_image:
        logger.info("使用随机生成的图像")
        img_tensor = torch.randn(1, 3, img_size, img_size).to(device)
        return img_tensor, None
    
    # 2. 如果用户指定了具体的图像路径且文件存在
    if args.image is not None and os.path.exists(args.image):
        logger.info(f"加载指定图像: {args.image}")
        img = Image.open(args.image).convert('RGB')
        
    # 3. 否则，从样本目录随机选择一张图像
    else:
        if os.path.exists(args.sample_dir) and os.path.isdir(args.sample_dir):
            image_files = [f for f in os.listdir(args.sample_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                random_image = random.choice(image_files)
                image_path = os.path.join(args.sample_dir, random_image)
                logger.info(f"随机选择图像: {image_path}")
                img = Image.open(image_path).convert('RGB')
            else:
                logger.warning(f"在 {args.sample_dir} 中未找到有效图像，使用随机生成的图像")
                img_tensor = torch.randn(1, 3, img_size, img_size).to(device)
                return img_tensor, None
        else:
            logger.warning(f"样本目录 {args.sample_dir} 不存在，使用随机生成的图像")
            img_tensor = torch.randn(1, 3, img_size, img_size).to(device)
            return img_tensor, None
    
    # 处理加载的图像
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img

def get_output_path(args, filename):
    """构建输出文件路径，可选添加时间戳"""
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建基本文件名
    base_name = f"{args.prefix}_{filename}"
    
    # 如果指定了添加时间戳，在文件名中添加时间戳
    if args.add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{base_name}_{timestamp}"
    
    # 添加文件扩展名
    output_path = os.path.join(args.output_dir, f"{base_name}.{args.format}")
    
    return output_path

def test_attention_visualization(args, model, img_tensor, original_img):
    """测试注意力权重可视化功能"""
    logger.info("=== 测试注意力权重可视化 ===")
    results = {}
    
    # 确保模型处于评估模式
    model.eval()
    
    # 获取补丁大小
    patch_size = model.patch_size if hasattr(model, 'patch_size') else 16
    
    # 1. 使用模型进行前向传播，获取注意力权重
    with torch.no_grad():
        outputs, attention_weights = model(img_tensor, return_attention=True)
    
    # 2. 测试注意力热力图
    attention_heatmap_path = get_output_path(args, "attention_heatmap")
    logger.info(f"生成注意力热力图: {attention_heatmap_path}")
    
    # 选择一个中间层和注意力头
    if hasattr(model, 'depth'):
        layer_index = model.depth // 2  # 选择中间层
    else:
        layer_index = 0  # 默认选择第一层
        
    if hasattr(model, 'num_heads'):
        head_index = model.num_heads // 2  # 选择中间的注意力头
    else:
        head_index = 0  # 默认选择第一个注意力头
    
    plot_attention_weights(
        attention_weights=attention_weights,
        layer_idx=layer_index,
        head_idx=head_index,
        save_path=attention_heatmap_path,  # 注意：这里使用save_path而不是output_path
        dpi=args.dpi
    )
    results['attention_heatmap'] = attention_heatmap_path
    
    # 3. 测试注意力在图像上的可视化（仅当有原始图像时）
    if original_img is not None:
        attention_on_image_path = get_output_path(args, "attention_on_image")
        logger.info(f"生成注意力在图像上的可视化: {attention_on_image_path}")
        
        visualize_attention_on_image(
            image=img_tensor,  # 使用张量形式的图像
            attention_weights=attention_weights,
            patch_size=patch_size,  # 添加patch_size参数
            layer_idx=layer_index,
            head_idx=head_index,
            save_path=attention_on_image_path,  # 使用save_path参数
            dpi=args.dpi
        )
        results['attention_on_image'] = attention_on_image_path
    
    # 4. 测试所有注意力头的可视化
    all_heads_path = get_output_path(args, "all_attention_heads")
    logger.info(f"生成所有注意力头的可视化: {all_heads_path}")
    
    visualize_all_heads(
        image=img_tensor,  # 添加image参数
        attention_weights=attention_weights,
        patch_size=patch_size,  # 添加patch_size参数
        layer_idx=layer_index,
        save_path=all_heads_path,  # 使用save_path参数
        dpi=args.dpi
    )
    results['all_attention_heads'] = all_heads_path
    
    return results

def test_model_structure_visualization(args, model):
    """测试模型结构可视化功能"""
    logger.info("=== 测试模型结构可视化 ===")
    results = {}
    
    # 1. 测试整体模型结构可视化
    model_structure_path = get_output_path(args, "model_structure")
    logger.info(f"生成模型结构图: {model_structure_path}")
    
    plot_model_structure(
        model=model,
        output_path=model_structure_path,
        format=args.format,
        show_details=True,
        direction='TB',  # 从上到下的布局
        dpi=args.dpi
    )
    results['model_structure'] = model_structure_path
    
    # 2. 测试编码器块结构可视化
    encoder_block_path = get_output_path(args, "encoder_block")
    logger.info(f"生成编码器块结构图: {encoder_block_path}")
    
    # 如果模型有blocks属性并且blocks有blocks属性（嵌套结构）
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'blocks') and len(model.blocks.blocks) > 0:
        encoder_block = model.blocks.blocks[0]  # 获取第一个编码器块
        plot_encoder_block(
            block=encoder_block,
            output_path=encoder_block_path,
            format=args.format,
            direction='TB'
        )
    else:
        # 如果无法访问特定的编码器块，则用None作为参数，函数会创建一个通用的编码器块图
        plot_encoder_block(
            block=None,
            output_path=encoder_block_path,
            format=args.format,
            direction='TB'
        )
    results['encoder_block'] = encoder_block_path
    
    # 3. 测试层权重可视化
    layer_weights_path = get_output_path(args, "layer_weights")
    logger.info(f"生成层权重可视化: {layer_weights_path}")
    
    visualize_layer_weights(
        model=model,
        output_path=layer_weights_path,
        dpi=args.dpi
    )
    results['layer_weights'] = layer_weights_path
    
    return results

def test_static_visualization(args, model, img_tensor):
    """测试静态综合可视化功能"""
    logger.info("=== 测试静态综合可视化 ===")
    results = {}
    
    # 1. 测试模型概览
    overview_path = get_output_path(args, "model_overview")
    logger.info(f"生成模型概览: {overview_path}")
    
    create_model_overview(
        model=model,
        output_path=overview_path,
        format=args.format,
        dpi=args.dpi
    )
    results['model_overview'] = overview_path
    
    # 2. 测试注意力分析
    attention_analysis_path = get_output_path(args, "attention_analysis")
    logger.info(f"生成注意力分析: {attention_analysis_path}")
    
    create_attention_analysis(
        model=model,
        input_image=img_tensor,
        output_path=attention_analysis_path,
        format=args.format,
        dpi=args.dpi
    )
    results['attention_analysis'] = attention_analysis_path
    
    # 3. 测试综合可视化（生成多个文件）
    logger.info("生成综合可视化...")
    
    comprehensive_files = create_comprehensive_visualization(
        model=model,
        input_image=img_tensor,
        output_dir=args.output_dir,
        prefix=f"{args.prefix}_comprehensive",
        format=args.format,
        dpi=args.dpi,
        create_html=not args.no_html
    )
    
    if args.verbose:
        logger.info("综合可视化文件:")
        for name, path in comprehensive_files.items():
            logger.info(f"- {name}: {path}")
    
    results.update(comprehensive_files)
    
    return results

def test_model_comparison(args, model1, model2, img_tensor):
    """测试模型比较可视化功能"""
    logger.info("=== 测试模型比较可视化 ===")
    
    # 创建比较输出路径
    comparison_path = get_output_path(args, "models_comparison")
    logger.info(f"生成模型比较可视化: {comparison_path}")
    
    # 执行模型比较
    compare_models(
        models=[model1, model2],
        model_names=[f"ViT-{args.model_type.capitalize()}", "ViT-Small"],
        input_image=img_tensor,
        output_path=comparison_path,
        format=args.format,
        dpi=args.dpi
    )
    
    return {'models_comparison': comparison_path}

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取计算设备
    device = get_device()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载或创建主模型
    model = load_or_create_model(args, device)
    
    # 如果是比较模式，创建第二个模型
    model2 = None
    if args.compare:
        logger.info("创建第二个模型(ViT-Small)用于比较")
        model2 = VisionTransformer.create_small(num_classes=args.num_classes).to(device)
    
    # 获取输入图像
    img_tensor, original_img = get_input_image(args, model, device)
    
    # 根据指定的模式执行可视化测试
    results = {}
    
    if args.mode in ['attention', 'all']:
        attention_results = test_attention_visualization(args, model, img_tensor, original_img)
        results.update(attention_results)
    
    if args.mode in ['structure', 'all']:
        structure_results = test_model_structure_visualization(args, model)
        results.update(structure_results)
    
    if args.mode in ['static', 'all']:
        static_results = test_static_visualization(args, model, img_tensor)
        results.update(static_results)
    
    # 如果指定了比较模式，执行模型比较
    if args.compare and model2 is not None:
        comparison_results = test_model_comparison(args, model, model2, img_tensor)
        results.update(comparison_results)
    
    # 输出测试结果摘要
    logger.info("=== 测试完成，生成的文件 ===")
    for name, path in results.items():
        logger.info(f"{name}: {path}")
    
    logger.info(f"所有测试文件已保存到目录: {args.output_dir}")

if __name__ == '__main__':
    main() 