import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import (
    standardize_data,
    normalize_data,
    fill_missing_values,
    fill_missing_tensor,
    MinMaxScaler,
    StandardScaler,
    detect_outliers_iqr,
    remove_outliers,
    clip_outliers,
    PreprocessingPipeline,
    create_image_preprocessing_pipeline
)

class TestPreprocessing(unittest.TestCase):
    """预处理功能的测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.tensor_data = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        
        # 带异常值的数据
        self.outlier_data = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [100.0, 110.0, 120.0]  # 异常值
        ])
        
        # 带缺失值的DataFrame
        self.df_with_missing = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': [5.0, np.nan, 7.0, 8.0],
            'C': ['a', 'b', np.nan, 'd']
        })
        
        # 创建一个简单的测试图像
        self.test_image = Image.new('RGB', (100, 100), color='red')
    
    def test_standardize_data(self):
        """测试数据标准化功能"""
        # 使用函数进行标准化
        standardized = standardize_data(self.tensor_data)
        
        # 手动计算均值和标准差
        mean = torch.mean(self.tensor_data, dim=0)
        std = torch.std(self.tensor_data, dim=0)
        expected = (self.tensor_data - mean) / std
        
        # 验证结果
        self.assertTrue(torch.allclose(standardized, expected))
        
        # 验证均值接近0，标准差接近1
        self.assertTrue(torch.allclose(torch.mean(standardized, dim=0), 
                                       torch.zeros(3), 
                                       atol=1e-6))
        self.assertTrue(torch.allclose(torch.std(standardized, dim=0), 
                                      torch.ones(3), 
                                      atol=1e-6))
    
    def test_normalize_data(self):
        """测试数据归一化功能"""
        # 使用函数进行归一化
        normalized = normalize_data(self.tensor_data)
        
        # 手动计算最小值和最大值
        min_val = torch.min(self.tensor_data, dim=0)[0]
        max_val = torch.max(self.tensor_data, dim=0)[0]
        expected = (self.tensor_data - min_val) / (max_val - min_val)
        
        # 验证结果
        self.assertTrue(torch.allclose(normalized, expected))
        
        # 验证最小值接近0，最大值接近1
        self.assertTrue(torch.allclose(torch.min(normalized, dim=0)[0], 
                                      torch.zeros(3), 
                                      atol=1e-6))
        self.assertTrue(torch.allclose(torch.max(normalized, dim=0)[0], 
                                      torch.ones(3), 
                                      atol=1e-6))
    
    def test_fill_missing_values(self):
        """测试缺失值填充功能"""
        # 使用均值填充
        df_filled_mean = fill_missing_values(self.df_with_missing, strategy='mean')
        self.assertFalse(df_filled_mean['A'].isna().any())
        self.assertFalse(df_filled_mean['B'].isna().any())
        self.assertTrue(df_filled_mean['C'].isna().any())  # 字符串列不应被均值填充
        
        # 使用中位数填充
        df_filled_median = fill_missing_values(self.df_with_missing, strategy='median')
        self.assertFalse(df_filled_median['A'].isna().any())
        self.assertFalse(df_filled_median['B'].isna().any())
        
        # 使用众数填充
        df_filled_mode = fill_missing_values(self.df_with_missing, strategy='mode')
        self.assertFalse(df_filled_mode['A'].isna().any())
        self.assertFalse(df_filled_mode['B'].isna().any())
        self.assertFalse(df_filled_mode['C'].isna().any())  # 字符串列应被模式填充
        
        # 使用常数填充
        fill_values = {'A': 99.0, 'B': 88.0, 'C': 'unknown'}
        df_filled_const = fill_missing_values(
            self.df_with_missing, 
            strategy='constant', 
            fill_values=fill_values
        )
        self.assertEqual(df_filled_const.loc[2, 'A'], 99.0)
        self.assertEqual(df_filled_const.loc[1, 'B'], 88.0)
        self.assertEqual(df_filled_const.loc[2, 'C'], 'unknown')
    
    def test_fill_missing_tensor(self):
        """测试张量缺失值填充功能"""
        # 创建带缺失值的张量
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = torch.tensor([[False, False], [True, False], [False, True]])
        
        # 使用0填充
        filled = fill_missing_tensor(data, mask)
        expected = torch.tensor([[1.0, 2.0], [0.0, 4.0], [5.0, 0.0]])
        self.assertTrue(torch.equal(filled, expected))
        
        # 使用自定义值填充
        filled_custom = fill_missing_tensor(data, mask, value=9.9)
        expected_custom = torch.tensor([[1.0, 2.0], [9.9, 4.0], [5.0, 9.9]])
        self.assertTrue(torch.equal(filled_custom, expected_custom))
    
    def test_min_max_scaler(self):
        """测试MinMaxScaler类"""
        # 默认范围 [0, 1]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.tensor_data)
        
        # 验证范围在 [0, 1]
        self.assertTrue(torch.all(scaled_data >= 0))
        self.assertTrue(torch.all(scaled_data <= 1))
        
        # 自定义范围 [-1, 1]
        custom_scaler = MinMaxScaler(feature_range=(-1, 1))
        custom_scaled = custom_scaler.fit_transform(self.tensor_data)
        
        # 验证范围在 [-1, 1]
        self.assertTrue(torch.all(custom_scaled >= -1))
        self.assertTrue(torch.all(custom_scaled <= 1))
    
    def test_standard_scaler(self):
        """测试StandardScaler类"""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.tensor_data)
        
        # 验证均值接近0，标准差接近1
        self.assertTrue(torch.allclose(torch.mean(scaled_data, dim=0), 
                                      torch.zeros(3), 
                                      atol=1e-6))
        self.assertTrue(torch.allclose(torch.std(scaled_data, dim=0), 
                                      torch.ones(3), 
                                      atol=1e-6))
    
    def test_detect_outliers_iqr(self):
        """测试IQR异常值检测"""
        outliers = detect_outliers_iqr(self.outlier_data)
        
        # 验证最后一行被检测为异常值
        self.assertTrue(torch.all(outliers[3]))
        self.assertFalse(torch.any(outliers[0:3]))
    
    def test_remove_outliers(self):
        """测试异常值移除"""
        outlier_mask = detect_outliers_iqr(self.outlier_data)
        cleaned_data = remove_outliers(self.outlier_data, outlier_mask)
        
        # 验证数据形状
        self.assertEqual(cleaned_data.shape[0], 3)  # 应该只有3行了
        
        # 验证所有异常值都被移除
        expected = self.tensor_data[0:3]  # 前3行
        self.assertTrue(torch.equal(cleaned_data, expected))
    
    def test_clip_outliers(self):
        """测试异常值裁剪"""
        clipped = clip_outliers(self.outlier_data, lower_bound=0.0, upper_bound=10.0)
        
        # 验证所有值都在范围内
        self.assertTrue(torch.all(clipped >= 0.0))
        self.assertTrue(torch.all(clipped <= 10.0))
        
        # 验证最后一行被裁剪到上限
        self.assertTrue(torch.all(clipped[3] == 10.0))
        
        # 验证其他行没变化
        self.assertTrue(torch.equal(clipped[0:3], self.outlier_data[0:3]))
    
    def test_preprocessing_pipeline(self):
        """测试预处理管道"""
        # 创建预处理步骤
        steps = [
            ('standardize', lambda x: standardize_data(x)),
            ('clip', lambda x: clip_outliers(x, -2.0, 2.0))
        ]
        
        # 创建并应用管道
        pipeline = PreprocessingPipeline(steps)
        processed = pipeline(self.outlier_data)
        
        # 验证结果
        self.assertTrue(torch.all(processed >= -2.0))
        self.assertTrue(torch.all(processed <= 2.0))
        
        # 测试添加步骤
        pipeline.add_step('normalize', lambda x: normalize_data(x))
        processed_with_norm = pipeline(self.outlier_data)
        
        # 验证结果在 [0, 1] 范围内
        self.assertTrue(torch.all(processed_with_norm >= 0.0))
        self.assertTrue(torch.all(processed_with_norm <= 1.0))
    
    def test_create_image_preprocessing_pipeline(self):
        """测试图像预处理管道创建"""
        # 创建基本管道
        transform = create_image_preprocessing_pipeline()
        processed = transform(self.test_image)
        
        # 验证结果是torch张量且有正确的通道顺序 [C, H, W]
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape[0], 3)  # 3个通道
        
        # 验证添加额外变换
        resize = (50, 50)
        additional = [transforms.RandomHorizontalFlip(p=1.0)]  # 总是翻转
        custom_transform = create_image_preprocessing_pipeline(
            resize=resize,
            additional_transforms=additional
        )
        
        custom_processed = custom_transform(self.test_image)
        
        # 验证尺寸正确
        self.assertEqual(custom_processed.shape[1:], torch.Size(resize))

if __name__ == '__main__':
    unittest.main() 