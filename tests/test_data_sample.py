import os
import unittest
import torch
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from src.data import BaseDataset, create_dataloaders
from src.data.config import DatasetConfig

class TestSampleData(unittest.TestCase):
    """测试数据加载器处理实际数据集的能力"""
    
    def setUp(self):
        """设置测试环境"""
        self.data_dir = os.path.join('data', 'images')
        self.anno_file = os.path.join('data', 'annotations.csv')
        
        # 确保数据文件存在
        self.assertTrue(os.path.exists(self.data_dir), f"图像目录 {self.data_dir} 不存在")
        self.assertTrue(os.path.exists(self.anno_file), f"标注文件 {self.anno_file} 不存在")
    
    def test_annotations_csv_format(self):
        """测试标注文件的格式是否符合预期"""
        # 加载标注文件
        df = pd.read_csv(self.anno_file)
        
        # 检查列名
        expected_columns = ['file_name', 'width', 'height', 'x1', 'y1', 'x2', 'y2', 'category']
        self.assertListEqual(list(df.columns), expected_columns, "标注文件列名不符合预期")
        
        # 检查数据类型
        self.assertTrue(df['file_name'].dtype == 'object', "file_name列应为字符串类型")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['width']), "width列应为数值类型")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['height']), "height列应为数值类型")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['x1']), "x1列应为数值类型")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['y1']), "y1列应为数值类型")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['x2']), "x2列应为数值类型")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['y2']), "y2列应为数值类型")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['category']), "category列应为数值类型")
        
        # 检查数据有效性
        self.assertGreater(len(df), 0, "标注文件不应为空")
        print(f"标注文件包含 {len(df)} 个样本")
        
        # 随机抽取一些样本进行检查
        sample = df.sample(min(10, len(df)))
        for _, row in sample.iterrows():
            # 检查边界框坐标的有效性
            self.assertLess(row['x1'], row['x2'], f"x1应小于x2：{row}")
            self.assertLess(row['y1'], row['y2'], f"y1应小于y2：{row}")
            self.assertGreaterEqual(row['x1'], 0, f"x1应大于等于0：{row}")
            self.assertGreaterEqual(row['y1'], 0, f"y1应大于等于0：{row}")
            self.assertLessEqual(row['x2'], row['width'], f"x2应小于等于width：{row}")
            self.assertLessEqual(row['y2'], row['height'], f"y2应小于等于height：{row}")
    
    def test_image_files_exist(self):
        """测试标注文件中的图像是否都存在"""
        # 加载标注文件
        df = pd.read_csv(self.anno_file)
        
        # 获取图像目录下的所有文件
        all_images = set(os.listdir(self.data_dir))
        print(f"图像目录包含 {len(all_images)} 个文件")
        
        # 检查前10个标注中的图像是否存在
        missing_images = []
        for i, row in df.head(10).iterrows():
            img_name = row['file_name']
            if img_name not in all_images:
                missing_images.append(img_name)
        
        if missing_images:
            print(f"警告：以下 {len(missing_images)} 个图像文件缺失：{missing_images}")
        else:
            print("前10个标注的图像文件都存在")
    
    def test_base_dataset_loading(self):
        """测试BaseDataset能否加载数据集"""
        try:
            # 创建数据集对象
            dataset = BaseDataset(self.data_dir, self.anno_file)
            
            # 打印数据集大小
            print(f"数据集大小: {len(dataset)}")
            
            # 获取第一个样本
            image, target = dataset[0]
            
            # 检查图像和目标的类型
            self.assertIsInstance(image, Image.Image, "加载的图像应为PIL.Image.Image类型")
            self.assertIsInstance(target, dict, "目标应为字典类型")
            
            # 检查目标字典的键
            expected_keys = ['boxes', 'labels', 'image_id', 'area', 'iscrowd']
            for key in expected_keys:
                self.assertIn(key, target, f"目标中应包含{key}键")
            
            print("成功加载第一个样本")
            
            # 尝试加载更多样本
            for i in range(1, min(5, len(dataset))):
                image, target = dataset[i]
                self.assertIsInstance(image, Image.Image, f"第{i}个样本的图像加载失败")
            
            print("成功加载多个样本")
            
        except Exception as e:
            self.fail(f"加载数据集失败: {str(e)}")
    
    def test_dataloaders_creation(self):
        """测试能否创建数据加载器"""
        try:
            # 创建数据加载器
            train_loader, val_loader = create_dataloaders(
                data_dir=self.data_dir,
                anno_file=self.anno_file,
                batch_size=4,
                val_split=0.2
            )
            
            # 打印数据加载器大小
            print(f"训练集大小: {len(train_loader.dataset)}")
            print(f"验证集大小: {len(val_loader.dataset)}")
            
            # 获取第一个批次
            images, targets = next(iter(train_loader))
            
            # 检查批次的形状
            self.assertEqual(images.dim(), 4, "图像批次维度应为4（批次大小、通道数、高度、宽度）")
            self.assertEqual(images.size(0), min(4, len(train_loader.dataset)), "批次大小不符合预期")
            self.assertEqual(images.size(1), 3, "图像通道数应为3（RGB）")
            self.assertEqual(len(targets), min(4, len(train_loader.dataset)), "目标数量应等于批次大小")
            
            print("成功创建并验证数据加载器")
            
        except Exception as e:
            self.fail(f"创建数据加载器失败: {str(e)}")
    
    def test_custom_dataset_config(self):
        """测试使用自定义配置加载数据集"""
        try:
            # 创建数据集配置
            config = DatasetConfig(
                name="SampleDataset",
                data_dir=self.data_dir,
                anno_file=self.anno_file,
                batch_size=4,
                img_size=(224, 224),
                val_split=0.2,
                augmentation_preset="light"
            )
            
            # 检查配置的有效性
            self.assertEqual(config.name, "SampleDataset", "配置名称不正确")
            self.assertEqual(config.data_dir, self.data_dir, "数据目录不正确")
            self.assertEqual(config.anno_file, self.anno_file, "标注文件不正确")
            
            # 使用配置创建数据加载器
            from src.data.data_loader import create_dataloaders_from_config
            train_loader, val_loader = create_dataloaders_from_config(config)
            
            # 检查数据加载器
            self.assertGreater(len(train_loader), 0, "训练集数据加载器为空")
            self.assertGreater(len(val_loader), 0, "验证集数据加载器为空")
            
            print("成功使用自定义配置创建数据加载器")
            
        except Exception as e:
            self.fail(f"使用自定义配置加载数据集失败: {str(e)}")

if __name__ == '__main__':
    unittest.main() 