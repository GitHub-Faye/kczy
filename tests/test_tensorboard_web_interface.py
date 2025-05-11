#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorBoard Web界面验证测试脚本
用于测试TensorBoard的Web界面是否能正确显示训练指标
"""

import os
import sys
import unittest
import time
import json
import logging
import requests
import subprocess
import tempfile
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.tensorboard_utils import (
    start_tensorboard,
    stop_tensorboard,
    check_tensorboard_running,
    find_tensorboard_executable
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestTensorBoardWebInterface(unittest.TestCase):
    """TensorBoard Web界面验证测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，创建临时日志目录并启动TensorBoard"""
        # 创建临时日志目录
        cls.test_log_dir = tempfile.mkdtemp(prefix="tb_web_test_")
        logger.info(f"创建临时测试日志目录: {cls.test_log_dir}")
        
        # 使用特定端口，避免与默认的6006冲突
        cls.test_port = 8765
        cls.test_host = "localhost"
        
        # 检查端口是否可用
        try:
            # 启动TensorBoard
            logger.info(f"在端口 {cls.test_port} 启动TensorBoard服务器...")
            cls.tb_process = start_tensorboard(
                log_dir=cls.test_log_dir,
                port=cls.test_port,
                host=cls.test_host,
                background=True
            )
            
            # 等待TensorBoard完全启动
            max_retries = 10
            retry_interval = 1
            for i in range(max_retries):
                if check_tensorboard_running(cls.test_port, cls.test_host):
                    logger.info("TensorBoard服务器启动成功！")
                    break
                if i < max_retries - 1:
                    logger.info(f"等待TensorBoard启动... {i+1}/{max_retries}")
                    time.sleep(retry_interval)
            else:
                raise TimeoutError("TensorBoard启动超时")
            
            # 生成测试日志数据
            cls._generate_test_data()
        
        except Exception as e:
            logger.error(f"测试环境设置失败: {e}")
            # 清理
            cls._cleanup()
            raise
    
    @classmethod
    def tearDownClass(cls):
        """测试类结束时清理资源"""
        cls._cleanup()
    
    @classmethod
    def _cleanup(cls):
        """清理资源的辅助方法"""
        # 停止TensorBoard进程
        if hasattr(cls, 'tb_process') and cls.tb_process:
            logger.info("停止TensorBoard服务器...")
            stop_tensorboard(cls.tb_process)
        
        # 删除临时目录
        import shutil
        if hasattr(cls, 'test_log_dir') and os.path.exists(cls.test_log_dir):
            logger.info(f"删除临时测试日志目录: {cls.test_log_dir}")
            shutil.rmtree(cls.test_log_dir)
    
    @classmethod
    def _generate_test_data(cls):
        """生成TensorBoard测试数据"""
        logger.info("生成TensorBoard测试数据...")
        
        # 创建一个SummaryWriter
        writer = SummaryWriter(cls.test_log_dir)
        
        # 1. 记录标量指标
        for i in range(100):
            writer.add_scalar('training/loss', np.exp(-i/30), i)
            writer.add_scalar('training/accuracy', 1 - np.exp(-i/30), i)
            writer.add_scalar('validation/loss', np.exp(-i/40) + 0.05, i)
            writer.add_scalar('validation/accuracy', 1 - np.exp(-i/40) - 0.02, i)
        
        # 2. 记录直方图数据
        for i in range(5):
            writer.add_histogram('model/weights', np.random.normal(0, 1, 1000) + i*0.1, i)
        
        # 3. 记录图像数据 - 创建一个简单的3通道随机图像
        for i in range(3):
            img = np.random.rand(3, 64, 64)  # 3通道，64x64像素
            writer.add_image('example_images', img, i)
        
        # 4. 记录超参数
        writer.add_hparams(
            {
                'learning_rate': 0.01,
                'batch_size': 32,
                'optimizer': 'Adam',
                'model_type': 'ViT-B/16'
            },
            {
                'accuracy': 0.85,
                'loss': 0.35
            }
        )
        
        # 5. 记录模型架构图（可选）
        try:
            # 创建一个简单的PyTorch模型
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 10),
                torch.nn.Softmax(dim=1)
            )
            
            # 记录模型图
            sample_input = torch.randn(1, 10)
            writer.add_graph(model, sample_input)
        except:
            logger.warning("记录模型图失败，但这不会影响测试")
        
        # 确保数据写入磁盘
        writer.flush()
        writer.close()
        
        logger.info("测试数据生成完成")
    
    def test_tensorboard_server_running(self):
        """测试TensorBoard服务器是否正在运行"""
        self.assertTrue(check_tensorboard_running(self.test_port, self.test_host),
                        "TensorBoard服务器应该正在运行")
    
    def test_tensorboard_web_accessible(self):
        """测试TensorBoard Web界面是否可访问"""
        url = f"http://{self.test_host}:{self.test_port}"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, 
                         f"TensorBoard Web界面应该可以访问，URL: {url}")
        
        # 确认响应内容包含TensorBoard特有的内容
        self.assertIn(b"TensorBoard", response.content, 
                      "响应内容应该包含'TensorBoard'字样")
    
    def test_scalar_data_available(self):
        """测试标量数据是否可用"""
        # 检查标量数据API端点
        url = f"http://{self.test_host}:{self.test_port}/data/plugin/scalars/tags"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, "标量数据API应该可访问")
        
        # 验证标量标签存在
        data = response.json()
        self.assertIn(".", data, "应该包含标量数据")
        tags = data.get(".")
        expected_tags = [
            "training/loss", 
            "training/accuracy", 
            "validation/loss", 
            "validation/accuracy"
        ]
        for tag in expected_tags:
            self.assertIn(tag, tags, f"应该包含标量标签 '{tag}'")
    
    def test_histogram_data_available(self):
        """测试直方图数据是否可用"""
        # 检查直方图数据API端点
        url = f"http://{self.test_host}:{self.test_port}/data/plugin/histograms/tags"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, "直方图数据API应该可访问")
        
        # 验证直方图标签存在
        data = response.json()
        self.assertIn(".", data, "应该包含直方图数据")
        self.assertIn("model/weights", data.get("."), "应该包含权重直方图数据")
    
    def test_image_data_available(self):
        """测试图像数据是否可用"""
        # 检查图像数据API端点
        url = f"http://{self.test_host}:{self.test_port}/data/plugin/images/tags"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, "图像数据API应该可访问")
        
        # 验证图像标签存在
        data = response.json()
        self.assertIn(".", data, "应该包含图像数据")
        self.assertIn("example_images", data.get("."), "应该包含示例图像数据")
    
    def test_hparams_data_available(self):
        """测试超参数数据是否可用"""
        # 检查超参数数据API存在
        # 注意：TensorBoard HPARAMS插件的API结构可能会变化，首先尝试获取可用的插件
        url = f"http://{self.test_host}:{self.test_port}/data/plugins_listing"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, "插件列表API应该可访问")
        
        # 验证hparams插件存在于可用插件列表中
        plugins = response.json()
        self.assertIn("hparams", plugins, "hparams插件应该在可用插件列表中")
        
        # 超参数的具体API结构可能会随TensorBoard版本变化
        # 这里只验证插件是否可用，而不深入检查具体API端点
    
    def test_realtime_data_update(self):
        """测试实时数据更新功能"""
        # 在测试期间添加新的数据
        writer = SummaryWriter(self.test_log_dir)
        
        # 记录一些新的标量数据
        start_step = 100  # 从前面数据的末尾开始
        for i in range(10):
            step = start_step + i
            writer.add_scalar('training/loss_realtime', np.exp(-step/20), step)
            writer.add_scalar('training/accuracy_realtime', 1 - np.exp(-step/20), step)
        
        # 确保数据写入
        writer.flush()
        writer.close()
        
        # 等待TensorBoard处理新数据
        time.sleep(2)
        
        # 验证新数据是否可用
        url = f"http://{self.test_host}:{self.test_port}/data/plugin/scalars/tags"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        tags = data.get(".")
        self.assertIn("training/loss_realtime", tags, "新添加的实时数据标签应该可见")
        self.assertIn("training/accuracy_realtime", tags, "新添加的实时数据标签应该可见")

if __name__ == "__main__":
    unittest.main() 