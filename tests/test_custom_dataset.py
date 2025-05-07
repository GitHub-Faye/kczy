import os
import sys
import unittest
import tempfile
import json
import yaml
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.config import DatasetConfig, AugmentationConfig

class TestCustomDatasetConfig(unittest.TestCase):
    """测试自定义数据集配置功能"""
    
    def setUp(self):
        """准备测试环境"""
        # 创建临时目录来保存测试配置
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # 创建测试用数据目录和标注文件
        self.test_data_dir = os.path.join(self.temp_dir.name, 'data')
        self.test_anno_file = os.path.join(self.temp_dir.name, 'annotations.csv')
        
        # 创建测试目录和空文件
        os.makedirs(self.test_data_dir, exist_ok=True)
        with open(self.test_anno_file, 'w') as f:
            f.write('file,width,height,x1,y1,x2,y2,class\n')
            f.write('test.jpg,100,100,10,10,90,90,0\n')
    
    def tearDown(self):
        """清理测试环境"""
        self.temp_dir.cleanup()
    
    def test_create_config(self):
        """测试创建基本配置"""
        config = DatasetConfig.create_for_testing(
            name="TestDataset",
            data_dir=self.test_data_dir,
            anno_file=self.test_anno_file
        )
        
        # 检查默认值
        self.assertEqual(config.name, "TestDataset")
        self.assertEqual(config.data_dir, self.test_data_dir)
        self.assertEqual(config.anno_file, self.test_anno_file)
        self.assertEqual(config.img_size, (224, 224))
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.val_split, 0.2)
        self.assertEqual(config.num_workers, 4)
        self.assertIsNone(config.augmentation_preset)
        self.assertTrue(isinstance(config.augmentation_config, AugmentationConfig))
        self.assertTrue(config.augmentation_config.is_empty())
        self.assertTrue(config.dev_mode)
    
    def test_create_config_with_preset(self):
        """测试创建带预设的配置"""
        config = DatasetConfig.create_for_testing(
            name="TestDataset",
            data_dir=self.test_data_dir,
            anno_file=self.test_anno_file,
            augmentation_preset="medium"
        )
        
        self.assertEqual(config.augmentation_preset, "medium")
    
    def test_create_config_with_augmentation(self):
        """测试创建带自定义增强的配置"""
        aug_config = AugmentationConfig(
            rotate={'degrees': 15, 'p': 0.5},
            flip={'horizontal': True, 'p': 0.5}
        )
        
        config = DatasetConfig.create_for_testing(
            name="TestDataset",
            data_dir=self.test_data_dir,
            anno_file=self.test_anno_file,
            augmentation_config=aug_config
        )
        
        self.assertEqual(config.augmentation_config.rotate, {'degrees': 15, 'p': 0.5})
        self.assertEqual(config.augmentation_config.flip, {'horizontal': True, 'p': 0.5})
    
    def test_save_load_json(self):
        """测试保存和加载JSON配置"""
        config = DatasetConfig.create_for_testing(
            name="TestDataset",
            data_dir=self.test_data_dir,
            anno_file=self.test_anno_file,
            batch_size=32,
            img_size=(256, 256)
        )
        
        # 保存为JSON
        json_path = os.path.join(self.temp_dir.name, 'config.json')
        config.save(json_path)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(json_path))
        
        # 手动解析JSON确认内容
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        self.assertEqual(json_data['name'], "TestDataset")
        self.assertEqual(json_data['batch_size'], 32)
        self.assertEqual(json_data['img_size'], [256, 256])  # JSON会将元组转换为列表
        
        # 使用开发模式加载JSON，避免文件存在性检查
        loaded_config = DatasetConfig.load(json_path)
        # 设置开发模式，因为load可能不会保留这个字段
        loaded_config.dev_mode = True
        
        # 验证加载的配置
        self.assertEqual(loaded_config.name, "TestDataset")
        self.assertEqual(loaded_config.batch_size, 32)
        # 验证转换回元组
        self.assertEqual(loaded_config.img_size, (256, 256))
    
    def test_save_load_yaml(self):
        """测试保存和加载YAML配置"""
        aug_config = AugmentationConfig(
            rotate={'degrees': 15, 'p': 0.5},
            blur={'blur_type': 'gaussian', 'radius': [0.5, 1.5], 'p': 0.3}
        )
        
        config = DatasetConfig.create_for_testing(
            name="TestDataset",
            data_dir=self.test_data_dir,
            anno_file=self.test_anno_file,
            augmentation_config=aug_config
        )
        
        # 保存为YAML
        yaml_path = os.path.join(self.temp_dir.name, 'config.yaml')
        config.save(yaml_path)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(yaml_path))
        
        # 手动解析YAML确认内容
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        self.assertEqual(yaml_data['name'], "TestDataset")
        self.assertEqual(yaml_data['augmentation_config']['rotate']['degrees'], 15)
        
        # 使用开发模式加载YAML，避免文件存在性检查
        loaded_config = DatasetConfig.load(yaml_path)
        # 设置开发模式，因为load可能不会保留这个字段
        loaded_config.dev_mode = True
        
        # 验证加载的配置
        self.assertEqual(loaded_config.name, "TestDataset")
        self.assertEqual(loaded_config.augmentation_config.rotate, {'degrees': 15, 'p': 0.5})
        self.assertEqual(loaded_config.augmentation_config.blur['blur_type'], 'gaussian')
    
    def test_to_from_dict(self):
        """测试配置和字典的相互转换"""
        config = DatasetConfig.create_for_testing(
            name="TestDataset",
            data_dir=self.test_data_dir,
            anno_file=self.test_anno_file,
            batch_size=64,
            img_size=(512, 512)
        )
        
        # 转换为字典
        config_dict = config.to_dict()
        
        # 验证字典中的值
        self.assertEqual(config_dict['name'], "TestDataset")
        self.assertEqual(config_dict['batch_size'], 64)
        self.assertEqual(config_dict['img_size'], (512, 512))
        self.assertTrue(config_dict['dev_mode'])
        
        # 从字典创建配置
        new_config = DatasetConfig.from_dict(config_dict)
        
        # 验证新创建的配置
        self.assertEqual(new_config.name, "TestDataset")
        self.assertEqual(new_config.batch_size, 64)
        self.assertEqual(new_config.img_size, (512, 512))
        self.assertTrue(new_config.dev_mode)
    
    def test_validation(self):
        """测试配置验证功能"""
        # 创建一个无效的配置（具有负批量大小）
        with self.assertRaises(ValueError):
            DatasetConfig.create_for_testing(
                name="TestDataset",
                data_dir=self.test_data_dir,
                anno_file=self.test_anno_file,
                batch_size=-1
            )
        
        # 创建一个使用无效的增强预设的配置
        with self.assertRaises(ValueError):
            DatasetConfig.create_for_testing(
                name="TestDataset",
                data_dir=self.test_data_dir,
                anno_file=self.test_anno_file,
                augmentation_preset="invalid_preset"
            )
        
        # 测试开发模式可以跳过文件存在检查
        nonexistent_dir = "/nonexistent/dir"
        nonexistent_file = "/nonexistent/file.csv"
        
        # 在非开发模式下应该引发异常
        with self.assertRaises(ValueError):
            DatasetConfig(
                name="FailingConfig", 
                data_dir=nonexistent_dir, 
                anno_file=nonexistent_file,
                dev_mode=False
            )
        
        # 在开发模式下不应该引发异常
        config = DatasetConfig(
            name="WorkingConfig", 
            data_dir=nonexistent_dir, 
            anno_file=nonexistent_file,
            dev_mode=True
        )
        self.assertEqual(config.data_dir, nonexistent_dir)
        self.assertEqual(config.anno_file, nonexistent_file)

if __name__ == '__main__':
    unittest.main() 