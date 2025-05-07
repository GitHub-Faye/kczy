import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    create_transform_from_preset,
    AugmentationPresets,
    create_augmentation_pipeline,
    RandomRotate,
    RandomFlip,
    RandomNoise,
    RandomBlur,
    RandomErasing,
    AugmentationPipeline
)

def display_augmented_images(original_img, transformed_images, titles, filename=None):
    """
    显示和保存原始图像及其增强版本
    
    参数:
        original_img: 原始图像
        transformed_images: 变换后的图像列表
        titles: 标题列表
        filename: 保存文件名
    """
    n = len(transformed_images) + 1
    fig, axes = plt.subplots(1, n, figsize=(n*4, 5))
    
    # 显示原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示变换后的图像
    for i, (img, title) in enumerate(zip(transformed_images, titles)):
        if isinstance(img, torch.Tensor):
            # 处理Tensor图像
            if img.dim() == 3 and img.size(0) == 3:
                # 转换通道顺序 CHW -> HWC
                img_np = img.permute(1, 2, 0).numpy()
                # 处理归一化图像，确保值在显示范围内
                if img_np.min() < 0 or img_np.max() > 1:
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = img.numpy()
            axes[i+1].imshow(img_np)
        else:
            axes[i+1].imshow(img)
        axes[i+1].set_title(title)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"图像已保存至: {filename}")
    
    plt.show()

def create_sample_image(size=(224, 224)):
    """
    创建一个示例图像用于演示
    """
    # 创建一个简单的图像
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # 添加背景色
    img_array[:, :, 0] = 240  # 红色背景
    
    # 添加一个蓝色的方块
    offset = size[0] // 10
    box_size = size[0] - 2 * offset
    img_array[offset:offset+box_size, offset:offset+box_size, 0] = 50
    img_array[offset:offset+box_size, offset:offset+box_size, 2] = 220
    
    # 添加一些绿色对角线
    for i in range(size[0]):
        if i < size[0]:
            thickness = 5
            for t in range(thickness):
                if i+t < size[0] and i+t < size[1]:
                    img_array[i+t, i+t, 0] = 50
                    img_array[i+t, i+t, 1] = 220
                    img_array[i+t, i+t, 2] = 50
                
                if i+t < size[0] and size[1]-i-t-1 >= 0:
                    img_array[i+t, size[1]-i-t-1, 0] = 50
                    img_array[i+t, size[1]-i-t-1, 1] = 220
                    img_array[i+t, size[1]-i-t-1, 2] = 50
    
    return Image.fromarray(img_array)

def demo_basic_transforms(img):
    """演示基本的增强变换"""
    print("\n### 基本图像增强演示 ###")
    
    # 应用旋转变换
    rotate_30 = RandomRotate(degrees=30, p=1.0)
    rotated_img = rotate_30(img)
    
    # 应用翻转变换
    flip_h = RandomFlip(horizontal=True, vertical=False, p=1.0)
    flipped_img = flip_h(img)
    
    # 应用噪声变换
    noise = RandomNoise(noise_type="gaussian", amount=0.05, p=1.0)
    noisy_img = noise(img)
    
    # 显示结果
    display_augmented_images(
        img, 
        [rotated_img, flipped_img, noisy_img],
        ['旋转30度', '水平翻转', '高斯噪声'],
        'outputs/demo_basic_transforms.png'
    )

def demo_augmentation_pipeline(img):
    """演示自定义增强管道"""
    print("\n### 增强管道演示 ###")
    
    # 创建一个自定义增强管道
    custom_pipeline = create_augmentation_pipeline(
        rotate={'degrees': 20, 'p': 1.0},
        flip={'horizontal': True, 'p': 1.0},
        color_jitter={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1, 'p': 1.0},
        blur={'blur_type': 'gaussian', 'radius': 1.5, 'p': 1.0},
        noise={'noise_type': 'gaussian', 'amount': 0.03, 'p': 1.0},
        additional_transforms=[T.Resize((224, 224))],
        final_transforms=[T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    
    # 应用管道
    transformed_img = custom_pipeline(img)
    
    # 仅用于显示的反归一化转换
    denormalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    display_img = denormalize(transformed_img.clone())
    
    # 显示结果
    display_augmented_images(
        img, 
        [display_img],
        ['自定义增强管道'],
        'outputs/demo_custom_pipeline.png'
    )

def demo_augmentation_presets(img):
    """演示预设增强配置"""
    print("\n### 预设增强配置演示 ###")
    
    # 使用轻度预设
    light_transform = create_transform_from_preset(
        preset_name='light',
        img_size=(224, 224)
    )
    light_img = light_transform(img)
    
    # 使用中度预设
    medium_transform = create_transform_from_preset(
        preset_name='medium',
        img_size=(224, 224)
    )
    medium_img = medium_transform(img)
    
    # 使用重度预设
    heavy_transform = create_transform_from_preset(
        preset_name='heavy',
        img_size=(224, 224)
    )
    heavy_img = heavy_transform(img)
    
    # 反归一化转换
    denormalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 显示结果
    display_augmented_images(
        img, 
        [denormalize(light_img.clone()), 
         denormalize(medium_img.clone()), 
         denormalize(heavy_img.clone())],
        ['轻度增强', '中度增强', '重度增强'],
        'outputs/demo_presets.png'
    )

def demo_augmentation_with_dataloader():
    """演示如何在数据加载器中使用增强"""
    print("\n### 数据加载器中使用增强 ###")
    print("以下是在创建数据加载器时整合增强的示例代码:")
    print("""
    # 方法1：使用预设
    train_loader, val_loader = create_dataloaders(
        data_dir='data/images',
        anno_file='data/annotations.csv',
        augmentation_preset='medium'  # 使用预设
    )
    
    # 方法2：使用自定义配置
    custom_aug_config = {
        'rotate': {'degrees': 20, 'p': 0.7},
        'flip': {'horizontal': True, 'vertical': False, 'p': 0.5},
        'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1},
        'blur': {'blur_type': 'gaussian', 'radius': (0.5, 1.5), 'p': 0.3},
        'noise': {'noise_type': 'gaussian', 'amount': 0.03, 'p': 0.2}
    }
    
    train_loader, val_loader = create_dataloaders(
        data_dir='data/images',
        anno_file='data/annotations.csv',
        augmentation_config=custom_aug_config  # 使用自定义配置
    )
    """)

def main():
    """主函数"""
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 创建示例图像
    sample_img = create_sample_image()
    
    # 保存原始图像
    sample_img.save('outputs/original_sample.png')
    print(f"原始示例图像已保存至: outputs/original_sample.png")
    
    # 演示基本变换
    demo_basic_transforms(sample_img)
    
    # 演示增强管道
    demo_augmentation_pipeline(sample_img)
    
    # 演示预设配置
    demo_augmentation_presets(sample_img)
    
    # 演示在数据加载器中使用
    demo_augmentation_with_dataloader()
    
    print("\n所有演示完成！生成的图像保存在 'outputs' 目录中。")

if __name__ == "__main__":
    main() 