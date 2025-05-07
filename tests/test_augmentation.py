import unittest
import torch
import numpy as np
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import (
    RandomRotate,
    RandomFlip,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
    RandomHue,
    RandomNoise,
    RandomBlur,
    RandomSharpness,
    RandomErasing,
    create_augmentation_pipeline,
    create_transform_from_preset,
    AugmentationPresets,
    AugmentationPipeline
)

# 创建一个自定义的PIL到Tensor转换模块
class PILToTensor(torch.nn.Module):
    """将PIL图像转换为规范化的Tensor"""
    
    def forward(self, x):
        """
        将PIL图像转换为tensor
        
        参数:
            x: PIL图像
            
        返回:
            torch.Tensor: 形状为[C,H,W]的tensor，值范围[0,1]
        """
        return torch.tensor(np.array(x)).permute(2, 0, 1).float() / 255.0
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class TestAugmentation(unittest.TestCase):
    """测试数据增强功能"""
    
    def setUp(self):
        """创建测试用的图像"""
        # 创建一个简单的测试图像 (50x50)
        self.test_img_size = (50, 50)
        img_array = np.zeros((50, 50, 3), dtype=np.uint8)
        
        # 添加一些基本形状使图像更有意义
        # 红色背景
        img_array[:, :, 0] = 200
        
        # 蓝色方块
        img_array[10:40, 10:40, 0] = 0
        img_array[10:40, 10:40, 2] = 200
        
        # 绿色对角线
        for i in range(50):
            if i < 50:
                img_array[i, i, 0] = 0
                img_array[i, i, 1] = 200
                img_array[i, i, 2] = 0
        
        self.test_img = Image.fromarray(img_array)
        
        # 创建输出目录以保存可视化图像
        self.output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_image_comparison(self, original, transformed, filename):
        """保存原始图像和变换后图像的对比图"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        if isinstance(transformed, torch.Tensor):
            # 转换Tensor为numpy数组用于显示
            if transformed.dim() == 3 and transformed.size(0) == 3:
                # CHW -> HWC
                img = transformed.permute(1, 2, 0).numpy()
                # 反归一化（如果需要）
                if img.min() < 0 or img.max() > 1:
                    img = (img - img.min()) / (img.max() - img.min())
            else:
                img = transformed.numpy()
            plt.imshow(img)
        else:
            plt.imshow(transformed)
        plt.title('增强后图像')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def test_basic_transforms(self):
        """测试基本的单个增强变换"""
        # 测试随机旋转
        rotate_transform = RandomRotate(degrees=30, p=1.0)
        rotated_img = rotate_transform(self.test_img)
        self.assertIsInstance(rotated_img, Image.Image)
        self.save_image_comparison(self.test_img, rotated_img, 'test_rotation.png')
        
        # 测试随机翻转
        flip_transform = RandomFlip(horizontal=True, vertical=True, p=1.0)
        flipped_img = flip_transform(self.test_img)
        self.assertIsInstance(flipped_img, Image.Image)
        self.save_image_comparison(self.test_img, flipped_img, 'test_flip.png')
        
        # 测试亮度调整
        brightness_transform = RandomBrightness(brightness_factor=(0.5, 1.5), p=1.0)
        bright_img = brightness_transform(self.test_img)
        self.assertIsInstance(bright_img, Image.Image)
        self.save_image_comparison(self.test_img, bright_img, 'test_brightness.png')
        
    def test_advanced_transforms(self):
        """测试高级增强变换"""
        # 测试高斯噪声
        noise_transform = RandomNoise(noise_type="gaussian", amount=0.1, p=1.0)
        noisy_img = noise_transform(self.test_img)
        self.assertIsInstance(noisy_img, Image.Image)
        self.save_image_comparison(self.test_img, noisy_img, 'test_noise.png')
        
        # 测试高斯模糊
        blur_transform = RandomBlur(blur_type="gaussian", radius=2.0, p=1.0)
        blurred_img = blur_transform(self.test_img)
        self.assertIsInstance(blurred_img, Image.Image)
        self.save_image_comparison(self.test_img, blurred_img, 'test_blur.png')
        
        # 测试随机擦除（需要转换为tensor）
        # 使用nn.Module子类代替lambda函数
        to_tensor = torch.nn.Sequential(PILToTensor())
        img_tensor = to_tensor(self.test_img)
        erasing_transform = RandomErasing(scale=(0.1, 0.3), p=1.0)
        erased_img = erasing_transform(img_tensor)
        self.assertIsInstance(erased_img, torch.Tensor)
        self.save_image_comparison(self.test_img, erased_img, 'test_erasing.png')
        
    def test_pipeline(self):
        """测试增强管道"""
        # 创建自定义管道
        pipeline = create_augmentation_pipeline(
            rotate={'degrees': 30, 'p': 1.0},
            flip={'horizontal': True, 'p': 1.0},
            color_jitter={'brightness': 0.2, 'contrast': 0, 'saturation': 0, 'hue': 0, 'p': 1.0},
            blur={'blur_type': 'gaussian', 'radius': 1.0, 'p': 1.0},
            noise={'noise_type': 'gaussian', 'amount': 0.05, 'p': 1.0},
            erasing={'p': 1.0, 'scale': (0.05, 0.1)}
        )
        
        # 应用管道变换
        transformed_img = pipeline(self.test_img)
        self.assertIsInstance(transformed_img, torch.Tensor)
        self.save_image_comparison(self.test_img, transformed_img, 'test_pipeline.png')
        
    def test_presets(self):
        """测试预设增强配置"""
        # 测试轻度增强
        light_transform = create_transform_from_preset(
            preset_name='light',
            img_size=self.test_img_size
        )
        light_img = light_transform(self.test_img)
        self.assertIsInstance(light_img, torch.Tensor)
        self.save_image_comparison(self.test_img, light_img, 'test_preset_light.png')
        
        # 测试重度增强
        heavy_transform = create_transform_from_preset(
            preset_name='heavy',
            img_size=self.test_img_size
        )
        heavy_img = heavy_transform(self.test_img)
        self.assertIsInstance(heavy_img, torch.Tensor)
        self.save_image_comparison(self.test_img, heavy_img, 'test_preset_heavy.png')
        
    def test_augmentation_pipeline_class(self):
        """测试AugmentationPipeline类"""
        pipeline = AugmentationPipeline()
        pipeline.add_transform(RandomRotate(degrees=30, p=1.0))
        pipeline.add_transform(RandomFlip(horizontal=True, p=1.0))
        pipeline.add_transform(RandomBrightness(brightness_factor=1.5, p=1.0))
        
        # 应用链式变换
        transformed_img = pipeline(self.test_img)
        self.assertIsInstance(transformed_img, Image.Image)
        self.save_image_comparison(self.test_img, transformed_img, 'test_pipeline_class.png')

if __name__ == '__main__':
    unittest.main() 