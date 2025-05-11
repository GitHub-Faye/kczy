#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试注意力权重可视化功能。
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vit import VisionTransformer
from src.visualization import (
    plot_attention_weights, 
    visualize_attention_on_image, 
    visualize_all_heads
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试注意力权重可视化")
    parser.add_argument('--image', type=str, default=None, help='输入图像路径。如果不提供，将使用随机生成的图像。')
    parser.add_argument('--model', type=str, default=None, help='预训练模型路径。如果不提供，将使用随机初始化的模型。')
    parser.add_argument('--layer', type=int, default=None, help='要可视化的层索引。默认为最后一层。')
    parser.add_argument('--head', type=int, default=None, help='要可视化的注意力头索引。默认为所有头的平均值。')
    parser.add_argument('--output_dir', type=str, default='temp_metrics/plots', help='输出目录')
    parser.add_argument('--cls_attention', action='store_true', help='可视化CLS令牌的注意力。默认为True。')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载或创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if args.model is not None and os.path.exists(args.model):
        print(f"加载预训练模型: {args.model}")
        model, _ = VisionTransformer.load_model(args.model, device=device)
    else:
        print("使用随机初始化的模型")
        # 创建一个小型ViT模型
        model = VisionTransformer.create_tiny(num_classes=10)
        model = model.to(device)
    
    # 提取模型参数
    img_size = model.patch_embed.img_size
    patch_size = model.patch_embed.patch_size
    num_layers = len(model.transformer.blocks)
    num_heads = model.transformer.blocks[0].attn.num_heads
    
    print(f"模型信息:")
    print(f"  - 图像大小: {img_size}x{img_size}")
    print(f"  - 补丁大小: {patch_size}x{patch_size}")
    print(f"  - 层数: {num_layers}")
    print(f"  - 每层的注意力头数: {num_heads}")
    
    # 加载或创建输入图像
    if args.image is not None and os.path.exists(args.image):
        print(f"加载图像: {args.image}")
        img = Image.open(args.image).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
    else:
        print("使用随机生成的图像")
        # 创建随机图像
        img_tensor = torch.randn(1, 3, img_size, img_size).to(device)
        # 为了可视化，也创建一个PIL图像版本
        img_np = (img_tensor[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
        img = Image.fromarray((img_np * 255).astype(np.uint8))
    
    # 提取注意力权重
    print("提取注意力权重...")
    model.eval()
    with torch.no_grad():
        outputs, attention_weights = model(img_tensor, return_attention=True)
    
    # 打印注意力权重形状
    print("注意力权重形状:")
    for i, attn in enumerate(attention_weights):
        print(f"  - 层 {i}: {attn.shape}")
    
    # 选择层索引
    if args.layer is not None and 0 <= args.layer < num_layers:
        layer_idx = args.layer
    else:
        layer_idx = num_layers - 1
    print(f"将可视化层 {layer_idx} 的注意力权重")
    
    # 选择头索引
    if args.head is not None and 0 <= args.head < num_heads:
        head_idx = args.head
        print(f"将可视化头 {head_idx} 的注意力权重")
    else:
        head_idx = None
        print("将可视化所有头的平均注意力权重")
    
    # 1. 绘制注意力热力图
    print("绘制注意力热力图...")
    save_path_1 = os.path.join(args.output_dir, 'attention_heatmap.png')
    fig1 = plot_attention_weights(
        attention_weights, 
        layer_idx=layer_idx, 
        head_idx=head_idx,
        save_path=save_path_1,
        title=f"层 {layer_idx} 的注意力热力图"
    )
    
    # 2. 在原始图像上可视化注意力
    print("在原始图像上可视化注意力...")
    save_path_2 = os.path.join(args.output_dir, 'attention_on_image.png')
    fig2 = visualize_attention_on_image(
        img_tensor,
        attention_weights,
        patch_size=patch_size,
        layer_idx=layer_idx,
        head_idx=head_idx,
        cls_token_attention=args.cls_attention,
        save_path=save_path_2,
        title=f"层 {layer_idx} 的注意力可视化"
    )
    
    # 3. 可视化所有头的注意力
    print("可视化所有注意力头...")
    save_path_3 = os.path.join(args.output_dir, 'all_attention_heads.png')
    fig3 = visualize_all_heads(
        img_tensor,
        attention_weights,
        patch_size=patch_size,
        layer_idx=layer_idx,
        cls_token_attention=args.cls_attention,
        save_path=save_path_3
    )
    
    # 4. 如果是交互式环境，显示图表
    if 'DISPLAY' in os.environ or 'JUPYTER_KERNEL' in os.environ:
        plt.show()
    
    print("完成! 可视化结果保存在:")
    print(f"  1. 注意力热力图: {save_path_1}")
    print(f"  2. 图像叠加注意力: {save_path_2}")
    print(f"  3. 所有注意力头: {save_path_3}")

if __name__ == '__main__':
    main() 