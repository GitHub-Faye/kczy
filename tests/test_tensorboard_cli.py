#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试TensorBoard相关CLI选项
"""

import os
import sys
import unittest
import tempfile
from argparse import Namespace

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.cli import create_parser, parse_args
from src.utils.config import TrainingConfig


class TestTensorBoardCLI(unittest.TestCase):
    """测试TensorBoard相关CLI选项"""

    def test_tensorboard_arguments_exist(self):
        """测试TensorBoard参数是否存在"""
        parser = create_parser()
        args = parser.parse_args([])
        
        # 验证默认值
        self.assertFalse(args.enable_tensorboard)
        self.assertEqual(args.tensorboard_dir, "logs")
        self.assertEqual(args.tensorboard_port, 6006)
        self.assertFalse(args.log_histograms)
        self.assertFalse(args.log_images)
    
    def test_enable_tensorboard(self):
        """测试启用TensorBoard选项"""
        parser = create_parser()
        args = parser.parse_args(["--enable-tensorboard"])
        
        self.assertTrue(args.enable_tensorboard)
    
    def test_tensorboard_custom_dir(self):
        """测试自定义TensorBoard日志目录"""
        custom_dir = "./custom_logs"
        parser = create_parser()
        args = parser.parse_args(["--tensorboard-dir", custom_dir])
        
        self.assertEqual(args.tensorboard_dir, custom_dir)
    
    def test_tensorboard_custom_port(self):
        """测试自定义TensorBoard端口"""
        custom_port = 8080
        parser = create_parser()
        args = parser.parse_args(["--tensorboard-port", str(custom_port)])
        
        self.assertEqual(args.tensorboard_port, custom_port)
    
    def test_log_histograms(self):
        """测试记录直方图选项"""
        parser = create_parser()
        args = parser.parse_args(["--log-histograms"])
        
        self.assertTrue(args.log_histograms)
    
    def test_log_images(self):
        """测试记录图像选项"""
        parser = create_parser()
        args = parser.parse_args(["--log-images"])
        
        self.assertTrue(args.log_images)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 有效端口
        valid_config = TrainingConfig(tensorboard_port=8080)
        self.assertEqual(valid_config.tensorboard_port, 8080)
        
        # 无效端口
        with self.assertRaises(ValueError):
            TrainingConfig(tensorboard_port=0)
        
        with self.assertRaises(ValueError):
            TrainingConfig(tensorboard_port=70000)
    
    def test_config_from_args(self):
        """测试从参数创建配置"""
        parser = create_parser()
        args = parser.parse_args([
            "--enable-tensorboard",
            "--tensorboard-dir", "./custom_logs",
            "--tensorboard-port", "9090",
            "--log-histograms",
            "--log-images"
        ])
        
        config = TrainingConfig(
            enable_tensorboard=args.enable_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
            tensorboard_port=args.tensorboard_port,
            log_histograms=args.log_histograms,
            log_images=args.log_images
        )
        
        self.assertTrue(config.enable_tensorboard)
        self.assertEqual(config.tensorboard_dir, "./custom_logs")
        self.assertEqual(config.tensorboard_port, 9090)
        self.assertTrue(config.log_histograms)
        self.assertTrue(config.log_images)
    
    def test_config_file_override(self):
        """测试配置文件被命令行参数覆盖"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write('''{
                "enable_tensorboard": true,
                "tensorboard_dir": "./temp_logs",
                "tensorboard_port": 7000,
                "log_histograms": true,
                "log_images": true
            }''')
            config_path = tmp.name
        
        try:
            # 命令行参数应覆盖配置文件
            args = parse_args([
                "--config", config_path,
                "--tensorboard-port", "8000"
            ])
            
            self.assertTrue(args.enable_tensorboard)  # 从配置文件获取
            self.assertEqual(args.tensorboard_dir, "./temp_logs")  # 从配置文件获取
            self.assertEqual(args.tensorboard_port, 8000)  # 命令行参数覆盖
            self.assertTrue(args.log_histograms)  # 从配置文件获取
            self.assertTrue(args.log_images)  # 从配置文件获取
            
        finally:
            # 清理临时文件
            if os.path.exists(config_path):
                os.unlink(config_path)


if __name__ == '__main__':
    unittest.main() 