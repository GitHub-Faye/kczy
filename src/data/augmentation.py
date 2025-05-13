import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import random
try:
    from scipy.ndimage import gaussian_filter, map_coordinates
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 基础增强方法
class RandomRotate:
    """
    随机旋转图像
    
    参数:
        degrees (Union[float, Tuple[float, float]]): 旋转角度范围
        expand (bool): 是否扩展图像以避免裁剪
        center (Optional[Tuple[int, int]]): 旋转中心点，格式为(x, y)
        fill (Union[int, Tuple[int, int, int]]): 填充值
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        expand: bool = False,
        center: Optional[Tuple[int, int]] = None,
        fill: Union[int, Tuple[int, int, int]] = 0,
        p: float = 1.0
    ):
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.expand = expand
        
        # 确保center参数为None或(x,y)格式的序列
        if center is not None and (not isinstance(center, (list, tuple)) or len(center) != 2):
            raise ValueError("center参数必须是包含两个整数的元组或列表，格式为(x, y)")
        self.center = center
        
        self.fill = fill
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            # 正确处理center参数
            if self.center is None:
                return F.rotate(img, angle, expand=self.expand, fill=self.fill)
            else:
                return F.rotate(img, angle, expand=self.expand, center=self.center, fill=self.fill)
        return img

class RandomFlip:
    """
    随机翻转图像
    
    参数:
        horizontal (bool): 是否进行水平翻转
        vertical (bool): 是否进行垂直翻转
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = False,
        p: float = 0.5
    ):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if self.horizontal and random.random() < self.p:
            img = F.hflip(img)
        if self.vertical and random.random() < self.p:
            img = F.vflip(img)
        return img

class RandomBrightness:
    """
    随机调整图像亮度
    
    参数:
        brightness_factor (Union[float, Tuple[float, float]]): 亮度调整因子范围
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        brightness_factor: Union[float, Tuple[float, float]] = (0.8, 1.2),
        p: float = 0.5
    ):
        self.brightness_factor = brightness_factor
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(self.brightness_factor, (tuple, list)):
                factor = random.uniform(self.brightness_factor[0], self.brightness_factor[1])
            else:
                factor = self.brightness_factor
            return F.adjust_brightness(img, factor)
        return img

class RandomContrast:
    """
    随机调整图像对比度
    
    参数:
        contrast_factor (Union[float, Tuple[float, float]]): 对比度调整因子范围
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        contrast_factor: Union[float, Tuple[float, float]] = (0.8, 1.2),
        p: float = 0.5
    ):
        self.contrast_factor = contrast_factor
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(self.contrast_factor, (tuple, list)):
                factor = random.uniform(self.contrast_factor[0], self.contrast_factor[1])
            else:
                factor = self.contrast_factor
            return F.adjust_contrast(img, factor)
        return img

class RandomSaturation:
    """
    随机调整图像饱和度
    
    参数:
        saturation_factor (Union[float, Tuple[float, float]]): 饱和度调整因子范围
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        saturation_factor: Union[float, Tuple[float, float]] = (0.8, 1.2),
        p: float = 0.5
    ):
        self.saturation_factor = saturation_factor
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(self.saturation_factor, (tuple, list)):
                factor = random.uniform(self.saturation_factor[0], self.saturation_factor[1])
            else:
                factor = self.saturation_factor
            return F.adjust_saturation(img, factor)
        return img

class RandomHue:
    """
    随机调整图像色调
    
    参数:
        hue_factor (Union[float, Tuple[float, float]]): 色调调整因子范围
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        hue_factor: Union[float, Tuple[float, float]] = (-0.1, 0.1),
        p: float = 0.5
    ):
        self.hue_factor = hue_factor
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(self.hue_factor, (tuple, list)):
                factor = random.uniform(self.hue_factor[0], self.hue_factor[1])
            else:
                factor = self.hue_factor
            return F.adjust_hue(img, factor)
        return img

# 高级增强方法
class RandomNoise:
    """
    向图像添加随机噪声
    
    参数:
        noise_type (str): 噪声类型，可选["gaussian", "salt_pepper"]
        amount (float): 噪声强度
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        noise_type: str = "gaussian",
        amount: float = 0.05,
        p: float = 0.5
    ):
        self.noise_type = noise_type
        self.amount = amount
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_array = np.array(img)
            
            if self.noise_type == "gaussian":
                # 高斯噪声
                mean = 0
                std = self.amount * 255
                noise = np.random.normal(mean, std, img_array.shape)
                img_array = img_array + noise
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
            elif self.noise_type == "salt_pepper":
                # 椒盐噪声
                row, col, ch = img_array.shape
                s_vs_p = 0.5  # 盐比椒比例
                out = np.copy(img_array)
                
                # 添加盐噪声（白点）
                num_salt = np.ceil(self.amount * img_array.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
                out[coords[0], coords[1], :] = 255
                
                # 添加椒噪声（黑点）
                num_pepper = np.ceil(self.amount * img_array.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
                out[coords[0], coords[1], :] = 0
                
                img_array = out
                
            return Image.fromarray(img_array.astype(np.uint8))
        return img

class RandomBlur:
    """
    随机模糊图像
    
    参数:
        blur_type (str): 模糊类型，可选["gaussian", "box", "median"]
        radius (Union[float, Tuple[float, float]]): 模糊半径
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        blur_type: str = "gaussian",
        radius: Union[float, Tuple[float, float]] = (0.5, 2.0),
        p: float = 0.5
    ):
        self.blur_type = blur_type
        self.radius = radius
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(self.radius, (tuple, list)):
                radius = random.uniform(self.radius[0], self.radius[1])
            else:
                radius = self.radius
                
            if self.blur_type == "gaussian":
                return img.filter(ImageFilter.GaussianBlur(radius=radius))
            elif self.blur_type == "box":
                return img.filter(ImageFilter.BoxBlur(radius=radius))
            elif self.blur_type == "median":
                # PIL的中值滤波只接受整数半径
                int_radius = max(1, int(radius))
                return img.filter(ImageFilter.MedianFilter(size=int_radius))
        return img

class RandomSharpness:
    """
    随机调整图像锐度
    
    参数:
        sharpness_factor (Union[float, Tuple[float, float]]): 锐度调整因子范围
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        sharpness_factor: Union[float, Tuple[float, float]] = (0.8, 2.0),
        p: float = 0.5
    ):
        self.sharpness_factor = sharpness_factor
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(self.sharpness_factor, (tuple, list)):
                factor = random.uniform(self.sharpness_factor[0], self.sharpness_factor[1])
            else:
                factor = self.sharpness_factor
            return F.adjust_sharpness(img, factor)
        return img

class RandomErasing:
    """
    随机擦除部分图像区域
    
    参数:
        scale (Tuple[float, float]): 擦除区域面积占比范围
        ratio (Tuple[float, float]): 擦除区域长宽比范围
        value (Union[int, float, Tuple, str]): 填充值，可以是具体值或'random'
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[int, float, Tuple, str] = 0,
        p: float = 0.5
    ):
        self.eraser = T.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # 注意：此变换需要在图像已转换为Tensor后应用
        if not isinstance(img, torch.Tensor):
            raise TypeError("RandomErasing需要输入为torch.Tensor类型")
        return self.eraser(img)

class ElasticTransform:
    """
    弹性变换
    
    参数:
        alpha (float): 变形强度
        sigma (float): 高斯核标准差
        p (float): 应用此变换的概率
    """
    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 5.0,
        p: float = 0.5
    ):
        if not SCIPY_AVAILABLE:
            raise ImportError("弹性变换需要scipy库。请安装：pip install scipy")
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # 转换为numpy数组
            img_array = np.array(img)
            shape = img_array.shape
            
            # 生成位移场
            dx = np.random.rand(shape[0], shape[1]) * 2 - 1
            dy = np.random.rand(shape[0], shape[1]) * 2 - 1
            
            # 使用高斯滤波平滑位移场
            dx = gaussian_filter(dx, sigma=self.sigma) * self.alpha
            dy = gaussian_filter(dy, sigma=self.sigma) * self.alpha
            
            # 创建网格
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            
            # 应用变形
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            
            # 对每个通道应用变形
            distorted_image = np.zeros_like(img_array)
            for c in range(shape[2]):
                distorted_image[:, :, c] = np.reshape(
                    map_coordinates(img_array[:, :, c], indices, order=1, mode='reflect'), 
                    shape[:2]
                )
            
            return Image.fromarray(distorted_image)
        return img
        
    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, sigma={self.sigma}, p={self.p})'

# 实用工具：根据配置生成增强管道
def create_augmentation_pipeline(
    rotate: Optional[Dict] = None,
    flip: Optional[Dict] = None,
    color_jitter: Optional[Dict] = None,
    blur: Optional[Dict] = None,
    noise: Optional[Dict] = None,
    elastic: Optional[Dict] = None,
    erasing: Optional[Dict] = None,
    additional_transforms: Optional[List[Callable]] = None,
    final_transforms: Optional[List[Callable]] = None
) -> T.Compose:
    """
    创建自定义数据增强管道
    
    参数:
        rotate (Dict, 可选): 旋转参数, 例如 {'degrees': 30, 'p': 0.5}
        flip (Dict, 可选): 翻转参数, 例如 {'horizontal': True, 'vertical': False, 'p': 0.5}
        color_jitter (Dict, 可选): 颜色调整参数, 例如
            {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1, 'p': 0.5}
        blur (Dict, 可选): 模糊参数, 例如 {'blur_type': 'gaussian', 'radius': (0.5, 2.0), 'p': 0.3}
        noise (Dict, 可选): 噪声参数, 例如 {'noise_type': 'gaussian', 'amount': 0.05, 'p': 0.3}
        elastic (Dict, 可选): 弹性变换参数, 例如 {'alpha': 50.0, 'sigma': 5.0, 'p': 0.3}
        erasing (Dict, 可选): 随机擦除参数, 例如 
            {'scale': (0.02, 0.33), 'ratio': (0.3, 3.3), 'value': 'random', 'p': 0.3}
        additional_transforms (List[Callable], 可选): 额外的自定义转换
        final_transforms (List[Callable], 可选): 最终转换，将在所有增强后应用
        
    返回:
        T.Compose: 转换管道
    """
    transforms_list = []
    
    # 添加旋转
    if rotate:
        transforms_list.append(
            RandomRotate(
                degrees=rotate.get('degrees', 10),
                expand=rotate.get('expand', False),
                center=rotate.get('center', None),
                fill=rotate.get('fill', 0),
                p=rotate.get('p', 0.5)
            )
        )
    
    # 添加翻转
    if flip:
        transforms_list.append(
            RandomFlip(
                horizontal=flip.get('horizontal', True),
                vertical=flip.get('vertical', False),
                p=flip.get('p', 0.5)
            )
        )
    
    # 添加颜色调整
    if color_jitter:
        # 可以整合为ColorJitter，也可以使用单独的增强器
        if all(param in color_jitter for param in ['brightness', 'contrast', 'saturation', 'hue']):
            transforms_list.append(
                T.ColorJitter(
                    brightness=color_jitter.get('brightness', 0),
                    contrast=color_jitter.get('contrast', 0),
                    saturation=color_jitter.get('saturation', 0),
                    hue=color_jitter.get('hue', 0)
                )
            )
        else:
            if 'brightness' in color_jitter:
                transforms_list.append(
                    RandomBrightness(
                        brightness_factor=color_jitter.get('brightness'),
                        p=color_jitter.get('p', 0.5)
                    )
                )
            if 'contrast' in color_jitter:
                transforms_list.append(
                    RandomContrast(
                        contrast_factor=color_jitter.get('contrast'),
                        p=color_jitter.get('p', 0.5)
                    )
                )
            if 'saturation' in color_jitter:
                transforms_list.append(
                    RandomSaturation(
                        saturation_factor=color_jitter.get('saturation'),
                        p=color_jitter.get('p', 0.5)
                    )
                )
            if 'hue' in color_jitter:
                transforms_list.append(
                    RandomHue(
                        hue_factor=color_jitter.get('hue'),
                        p=color_jitter.get('p', 0.5)
                    )
                )
    
    # 添加模糊
    if blur:
        transforms_list.append(
            RandomBlur(
                blur_type=blur.get('blur_type', 'gaussian'),
                radius=blur.get('radius', (0.5, 2.0)),
                p=blur.get('p', 0.3)
            )
        )
    
    # 添加锐化（如果需要）
    if blur and blur.get('sharpen', False):
        transforms_list.append(
            RandomSharpness(
                sharpness_factor=blur.get('sharpen_factor', (1.0, 3.0)),
                p=blur.get('sharpen_p', 0.3)
            )
        )
    
    # 添加噪声
    if noise:
        transforms_list.append(
            RandomNoise(
                noise_type=noise.get('noise_type', 'gaussian'),
                amount=noise.get('amount', 0.05),
                p=noise.get('p', 0.3)
            )
        )
    
    # 添加弹性变换
    if elastic and SCIPY_AVAILABLE:
        try:
            transforms_list.append(
                ElasticTransform(
                    alpha=elastic.get('alpha', 50.0),
                    sigma=elastic.get('sigma', 5.0),
                    p=elastic.get('p', 0.3)
                )
            )
        except Exception as e:
            print(f"添加弹性变换失败: {e}")
    
    # 添加额外的自定义转换
    if additional_transforms:
        transforms_list.extend(additional_transforms)
    
    # 添加基本转换（转换为Tensor）
    transforms_list.append(T.ToTensor())
    
    # 添加随机擦除（需要在ToTensor后）
    if erasing:
        transforms_list.append(
            T.RandomErasing(
                p=erasing.get('p', 0.3),
                scale=erasing.get('scale', (0.02, 0.33)),
                ratio=erasing.get('ratio', (0.3, 3.3)),
                value=erasing.get('value', 0)
            )
        )
    
    # 添加最终转换（例如归一化）
    if final_transforms:
        transforms_list.extend(final_transforms)
    
    return T.Compose(transforms_list)

# 预定义的增强配置
class AugmentationPresets:
    """预定义的数据增强配置"""
    
    @staticmethod
    def light() -> Dict:
        """
        轻度增强配置
        
        返回:
            Dict: 轻度增强配置
        """
        return {
            'rotate': {'degrees': 10, 'p': 0.3},
            'flip': {'horizontal': True, 'p': 0.3},
            'color_jitter': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.05, 'p': 0.3},
        }
    
    @staticmethod
    def medium() -> Dict:
        """
        中度增强配置
        
        返回:
            Dict: 中度增强配置
        """
        return {
            'rotate': {'degrees': 15, 'p': 0.5},
            'flip': {'horizontal': True, 'vertical': False, 'p': 0.5},
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1, 'p': 0.5},
            'blur': {'blur_type': 'gaussian', 'radius': (0.5, 1.5), 'p': 0.2},
            'erasing': {'p': 0.2},
        }
    
    @staticmethod
    def heavy() -> Dict:
        """
        重度增强配置
        
        返回:
            Dict: 重度增强配置
        """
        return {
            'rotate': {'degrees': 30, 'p': 0.6},
            'flip': {'horizontal': True, 'vertical': True, 'p': 0.5},
            'color_jitter': {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.15, 'p': 0.6},
            'blur': {'blur_type': 'gaussian', 'radius': (0.5, 2.0), 'sharpen': True, 'p': 0.3},
            'noise': {'noise_type': 'gaussian', 'amount': 0.05, 'p': 0.3},
            'elastic': {'alpha': 50.0, 'sigma': 5.0, 'p': 0.2},
            'erasing': {'p': 0.3, 'value': 'random'},
        }
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict:
        """
        获取预设配置
        
        参数:
            preset_name (str): 预设名称，可选['light', 'medium', 'heavy']
            
        返回:
            Dict: 预设配置
        """
        presets = {
            'light': AugmentationPresets.light(),
            'medium': AugmentationPresets.medium(),
            'heavy': AugmentationPresets.heavy()
        }
        
        if preset_name not in presets:
            raise ValueError(f"不支持的预设: {preset_name}，可选: {list(presets.keys())}")
        
        return presets[preset_name]

def create_transform_from_preset(
    preset_name: str = 'medium',
    img_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> T.Compose:
    """
    使用预设配置创建转换
    
    参数:
        preset_name (str): 预设名称，可选['light', 'medium', 'heavy']
        img_size (Tuple[int, int]): 图像大小
        mean (List[float]): 归一化均值
        std (List[float]): 归一化标准差
        
    返回:
        T.Compose: 转换管道
    """
    # 获取预设配置
    config = AugmentationPresets.get_preset(preset_name)
    
    # 基本转换
    base_transforms = [T.Resize(img_size)]
    
    # 最终转换
    final_transforms = [T.Normalize(mean=mean, std=std)]
    
    # 创建管道
    return create_augmentation_pipeline(
        rotate=config.get('rotate'),
        flip=config.get('flip'),
        color_jitter=config.get('color_jitter'),
        blur=config.get('blur'),
        noise=config.get('noise'),
        elastic=config.get('elastic'),
        erasing=config.get('erasing'),
        additional_transforms=base_transforms,
        final_transforms=final_transforms
    )

# 增强操作类，用于组织和管理数据增强操作
class AugmentationPipeline:
    """
    数据增强管道，用于组合多个增强操作
    
    参数:
        transforms_list (List[Callable]): 转换列表
    """
    def __init__(self, transforms_list: List[Callable] = None):
        self.transforms = [] if transforms_list is None else transforms_list
        
    def add_transform(self, transform: Callable) -> 'AugmentationPipeline':
        """
        添加转换
        
        参数:
            transform (Callable): 要添加的转换
            
        返回:
            AugmentationPipeline: 自身，用于链式调用
        """
        self.transforms.append(transform)
        return self
        
    def build(self) -> T.Compose:
        """
        构建转换管道
        
        返回:
            T.Compose: 转换管道
        """
        return T.Compose(self.transforms)
        
    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """
        应用所有转换
        
        参数:
            img (Union[Image.Image, torch.Tensor]): 输入图像
            
        返回:
            Union[Image.Image, torch.Tensor]: 转换后的图像
        """
        for transform in self.transforms:
            img = transform(img)
        return img 