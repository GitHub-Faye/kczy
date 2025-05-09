"""
使用测试数据验证ViT模型的实现

此脚本生成随机图像数据并使用不同大小的ViT模型进行前向传播测试，
以验证模型结构是否正确实现，没有明显的错误。
"""
import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# 确保可以导入src模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.vit import VisionTransformer
from src.utils.config import ViTConfig

def generate_dummy_image(batch_size=1, img_size=224, channels=3):
    """
    生成随机图像数据
    
    参数:
        batch_size (int): 批次大小
        img_size (int): 图像大小(假设是方形)
        channels (int): 通道数
        
    返回:
        torch.Tensor: 形状为[batch_size, channels, img_size, img_size]的随机图像数据
    """
    return torch.randn(batch_size, channels, img_size, img_size)

def test_vit_forward(model, img_size=224, batch_size=4):
    """
    测试ViT模型的前向传播
    
    参数:
        model (VisionTransformer): 要测试的ViT模型
        img_size (int): 输入图像大小
        batch_size (int): 批次大小
        
    返回:
        bool: 测试是否成功
    """
    print(f"测试模型: {model.__class__.__name__}")
    
    # 生成随机图像数据
    dummy_input = generate_dummy_image(batch_size, img_size, model.config.in_channels)
    
    try:
        # 进入评估模式
        model.eval()
        
        # 前向传播
        with torch.no_grad():
            output = model(dummy_input)
        
        # 检查输出形状
        expected_shape = (batch_size, model.config.num_classes)
        actual_shape = tuple(output.shape)
        
        if expected_shape != actual_shape:
            print(f"输出形状错误! 期望: {expected_shape}, 实际: {actual_shape}")
            return False
            
        print(f"输出形状: {actual_shape}, 测试通过!")
        return True
        
    except Exception as e:
        print(f"测试失败! 错误: {e}")
        return False

def test_vit_variants():
    """
    测试所有预定义的ViT变体
    
    返回:
        int: 失败的测试数量
    """
    # 测试不同大小的模型
    print("=" * 50)
    print("测试不同的ViT变体")
    print("=" * 50)
    
    variants = [
        ("ViT-Tiny", VisionTransformer.create_tiny),
        ("ViT-Small", VisionTransformer.create_small),
        ("ViT-Base", VisionTransformer.create_base),
        ("ViT-Large", VisionTransformer.create_large),
    ]
    
    failures = 0
    for name, factory_method in variants:
        print(f"\n测试 {name}...")
        try:
            # 使用小批量进行测试
            batch_size = 2
            num_classes = 10
            model = factory_method(num_classes=num_classes)
            
            success = test_vit_forward(model, batch_size=batch_size)
            if not success:
                failures += 1
                print(f"{name} 测试失败!")
            else:
                print(f"{name} 测试通过!")
        except Exception as e:
            failures += 1
            print(f"{name} 创建或测试失败! 错误: {e}")
    
    return failures

def test_custom_config():
    """
    测试自定义配置的ViT模型
    
    返回:
        bool: 测试是否成功
    """
    print("\n" + "=" * 50)
    print("测试自定义配置的ViT模型")
    print("=" * 50)
    
    try:
        # 创建自定义配置
        config = ViTConfig(
            img_size=32,  # 更小的图像
            patch_size=4,  # 更小的补丁
            in_channels=3,
            num_classes=100,
            embed_dim=256,
            depth=6,  # 减少深度
            num_heads=8,
            mlp_ratio=2.0,
            qkv_bias=True
        )
        
        # 创建模型
        model = VisionTransformer(config=config)
        
        # 按照自定义图像大小测试
        success = test_vit_forward(model, img_size=config.img_size, batch_size=2)
        
        if not success:
            print("自定义配置测试失败!")
            return False
        else:
            print("自定义配置测试通过!")
            return True
            
    except Exception as e:
        print(f"自定义配置测试失败! 错误: {e}")
        return False

def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("开始测试ViT模型实现...")
    
    # 测试不同的模型变体
    variant_failures = test_vit_variants()
    
    # 测试自定义配置
    custom_success = test_custom_config()
    
    # 打印总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    if variant_failures == 0 and custom_success:
        print("所有测试通过! ViT模型实现正确。")
    else:
        print(f"测试失败! {variant_failures} 个变体测试失败, 自定义配置测试{'通过' if custom_success else '失败'}。")

if __name__ == "__main__":
    main() 