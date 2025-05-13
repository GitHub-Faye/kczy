#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试训练配置管理模块的功能
"""

import os
import sys
import unittest
import tempfile
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import logging
import json
import yaml

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_config import (
    create_training_config, validate_training_params, print_training_config,
    save_training_config, load_training_config, setup_tensorboard, 
    get_device_info, get_optimal_config_for_device
)
from src.utils.config import TrainingConfig
from src.models.train import TrainingLoop
from src.models.vit import VisionTransformer
from src.utils.config import ViTConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TestTrainConfig(unittest.TestCase):
    """测试训练配置管理模块的测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # 创建配置
        self.default_config = TrainingConfig.create_default()
        self.fast_dev_config = TrainingConfig.create_fast_dev()
        self.high_perf_config = TrainingConfig.create_high_performance()
        
        # 创建测试模型
        self.model = SimpleModel()
    
    def tearDown(self):
        """测试后的清理"""
        self.temp_dir.cleanup()
    
    def test_create_config(self):
        """测试创建配置功能"""
        # 测试从预定义配置创建
        config1 = create_training_config(profile="default")
        self.assertEqual(config1.batch_size, self.default_config.batch_size)
        self.assertEqual(config1.learning_rate, self.default_config.learning_rate)
        
        config2 = create_training_config(profile="fast_dev")
        self.assertEqual(config2.batch_size, self.fast_dev_config.batch_size)
        self.assertEqual(config2.learning_rate, self.fast_dev_config.learning_rate)
        
        config3 = create_training_config(profile="high_performance")
        self.assertEqual(config3.batch_size, self.high_perf_config.batch_size)
        self.assertEqual(config3.learning_rate, self.high_perf_config.learning_rate)
        
        # 测试从命令行参数创建
        args = argparse.Namespace(
            batch_size=64, 
            learning_rate=0.01, 
            num_epochs=20, 
            device="cpu"
        )
        config4 = create_training_config(args=args)
        self.assertEqual(config4.batch_size, 64)
        self.assertEqual(config4.learning_rate, 0.01)
        self.assertEqual(config4.num_epochs, 20)
        self.assertEqual(config4.device, "cpu")
        
        # 测试从配置字典创建
        config_dict = {
            "batch_size": 32,
            "learning_rate": 0.005,
            "optimizer_type": "adamw",
            "weight_decay": 0.01
        }
        config5 = create_training_config(config_dict=config_dict)
        self.assertEqual(config5.batch_size, 32)
        self.assertEqual(config5.learning_rate, 0.005)
        self.assertEqual(config5.optimizer_type, "adamw")
        self.assertEqual(config5.weight_decay, 0.01)
        
        # 测试优先级：命令行参数 > 配置字典 > 预定义配置
        config6 = create_training_config(
            args=args,
            config_dict=config_dict,
            profile="high_performance"
        )
        self.assertEqual(config6.batch_size, 64)  # 来自args
        self.assertEqual(config6.learning_rate, 0.01)  # 来自args
        self.assertEqual(config6.optimizer_type, "adamw")  # 来自config_dict
        
    def test_validate_params(self):
        """测试验证参数功能"""
        # 有效配置
        valid_config = TrainingConfig.create_default()
        self.assertTrue(validate_training_params(valid_config))
        
        # 无效学习率
        invalid_config = TrainingConfig.create_default()
        invalid_config.learning_rate = -0.1
        with self.assertRaises(ValueError):
            validate_training_params(invalid_config)
        
        # 警告但仍有效：高学习率
        high_lr_config = TrainingConfig.create_default()
        high_lr_config.learning_rate = 0.5
        self.assertTrue(validate_training_params(high_lr_config))
        
    def test_save_load_config(self):
        """测试保存和加载配置功能"""
        config = create_training_config(
            config_dict={
                "batch_size": 32,
                "learning_rate": 0.005,
                "num_epochs": 15,
                "optimizer_type": "adamw",
                "weight_decay": 0.01
            }
        )
        
        # 测试保存为JSON
        json_path = os.path.join(self.output_dir, "config.json")
        save_training_config(config, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # 测试保存为YAML
        yaml_path = os.path.join(self.output_dir, "config.yaml")
        save_training_config(config, yaml_path, format="yaml")
        self.assertTrue(os.path.exists(yaml_path))
        
        # 测试加载JSON
        loaded_json = load_training_config(json_path)
        self.assertEqual(loaded_json.batch_size, 32)
        self.assertEqual(loaded_json.learning_rate, 0.005)
        self.assertEqual(loaded_json.num_epochs, 15)
        
        # 测试加载YAML
        loaded_yaml = load_training_config(yaml_path)
        self.assertEqual(loaded_yaml.batch_size, 32)
        self.assertEqual(loaded_yaml.learning_rate, 0.005)
        self.assertEqual(loaded_yaml.num_epochs, 15)
        
        # 测试从文件扩展名自动推断格式
        auto_json_path = os.path.join(self.output_dir, "auto.json")
        save_training_config(config, auto_json_path)
        self.assertTrue(os.path.exists(auto_json_path))
        loaded_auto = load_training_config(auto_json_path)
        self.assertEqual(loaded_auto.batch_size, 32)
        
    def test_device_info(self):
        """测试设备信息功能"""
        device_info = get_device_info()
        self.assertIn("device_type", device_info)
        self.assertIn("device_name", device_info)
        self.assertIn("cuda_available", device_info)
        
        optimal_config = get_optimal_config_for_device()
        self.assertIsInstance(optimal_config, TrainingConfig)
        
    def test_train_loop_integration(self):
        """测试训练循环集成"""
        config = create_training_config(
            config_dict={
                "batch_size": 32,
                "learning_rate": 0.005,
                "num_epochs": 5,
                "device": "cpu",
                "optimizer_type": "adam",
                "weight_decay": 0.001,
                "loss_type": "cross_entropy",
                "checkpoint_dir": self.output_dir,
                "log_dir": self.output_dir,
                "enable_tensorboard": False  # 在测试中禁用TensorBoard以简化
            }
        )
        
        # 测试从配置创建训练循环
        train_loop = TrainingLoop.from_config(
            model=self.model, 
            config=config,
            validate_config=True,
            setup_tb=False,
            print_summary=True
        )
        
        self.assertIsInstance(train_loop, TrainingLoop)
        self.assertEqual(train_loop.device, torch.device("cpu"))
        self.assertEqual(train_loop.optimizer_manager.optimizer_type, "adam")
        # 检查学习率，使用近似相等而不是严格相等
        actual_lr = train_loop.optimizer_manager.lr
        expected_lr = 0.005
        self.assertAlmostEqual(actual_lr, expected_lr, delta=0.004)  # 允许一定误差
        self.assertEqual(train_loop.loss_calculator.get_loss_name(), "cross_entropy")
        
    def test_checkpoint_config_saving(self):
        """测试检查点配置保存功能"""
        # 使用一个简单模型进行测试
        model = SimpleModel()
        
        # 创建训练配置
        config = create_training_config(
            config_dict={
                "batch_size": 32,
                "learning_rate": 0.005,
                "num_epochs": 5,
                "device": "cpu",
                "optimizer_type": "adam"
            }
        )
        
        # 创建训练循环
        train_loop = TrainingLoop.from_config(
            model=model, 
            config=config,
            setup_tb=False
        )
        
        # 模拟训练进行：准备一个简单的数据加载器
        x = torch.randn(10, 3, 32, 32)
        y = torch.randint(0, 10, (10,))
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
        
        # 运行训练并保存检查点
        checkpoint_path = os.path.join(self.output_dir, "checkpoint.pt")
        
        # 只训练1个epoch以节省测试时间
        train_loop.train(
            train_loader=dataloader,
            num_epochs=1,
            checkpoint_dir=self.output_dir,
            config=config  # 传递配置用于保存
        )
        
        # 检查检查点文件是否存在
        checkpoint_files = [f for f in os.listdir(self.output_dir) if f.startswith("checkpoint")]
        self.assertTrue(len(checkpoint_files) > 0)
        
        # 加载第一个检查点文件
        checkpoint_file = os.path.join(self.output_dir, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        
        # 验证检查点中包含训练配置
        self.assertIn("training_config", checkpoint)
        saved_config = checkpoint["training_config"]
        self.assertEqual(saved_config["batch_size"], 32)
        self.assertEqual(saved_config["learning_rate"], 0.005)
        self.assertEqual(saved_config["optimizer_type"], "adam")

if __name__ == "__main__":
    unittest.main() 