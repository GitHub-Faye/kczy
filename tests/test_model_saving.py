#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试模型保存和加载功能
"""

import os
import sys
import unittest
import tempfile
import torch
import numpy as np
from pathlib import Path

# 确保可以导入src模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.vit import VisionTransformer
from src.utils.config import ViTConfig
from src.models.model_utils import (
    save_model,
    load_model,
    export_to_onnx,
    save_checkpoint,
    load_checkpoint,
    get_model_info
)
from src.models.optimizer_manager import OptimizerManager
from src.models.train import TrainingLoop, LossCalculator

class TestModelSaving(unittest.TestCase):
    """测试模型保存和加载功能"""

    def setUp(self):
        """初始化测试环境"""
        # 创建临时目录用于保存测试文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = Path(self.temp_dir.name)
        
        # 创建测试用模型
        self.model_config = ViTConfig.create_tiny(num_classes=10)
        self.model = VisionTransformer.from_config(self.model_config)
        
        # 创建一些测试输入
        self.test_input = torch.randn(
            2, 
            self.model_config.in_channels, 
            self.model_config.img_size, 
            self.model_config.img_size
        )
        
        # 初始化优化器状态字典
        self.optimizer_state = {
            'state': {},
            'param_groups': [
                {'lr': 0.001, 'momentum': 0.9, 'dampening': 0,
                 'weight_decay': 0.0001, 'nesterov': False}
            ]
        }
        
        # 训练历史记录
        self.history = {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [85.0, 87.5, 90.0]
        }
        
    def tearDown(self):
        """清理测试环境"""
        self.temp_dir.cleanup()
    
    def test_save_load_model(self):
        """测试保存和加载完整模型"""
        # 获取模型初始输出
        with torch.no_grad():
            self.model.eval()
            initial_output = self.model(self.test_input)
            
        # 保存模型
        save_path = self.save_path / "test_model.pt"
        save_model(self.model, str(save_path))
        
        # 确认文件已创建
        self.assertTrue(save_path.exists())
        self.assertTrue(Path(str(save_path).replace(".pt", "_config.json")).exists())
        
        # 加载模型
        loaded_model, _ = load_model(str(save_path))
        
        # 检查加载的模型与原始模型的输出是否一致
        with torch.no_grad():
            loaded_model.eval()
            loaded_output = loaded_model(self.test_input)
            
        # 验证输出一致
        self.assertTrue(torch.allclose(initial_output, loaded_output))
        
    def test_save_load_weights(self):
        """测试保存和加载模型权重"""
        # 获取模型初始输出
        with torch.no_grad():
            self.model.eval()
            initial_output = self.model(self.test_input)
            
        # 保存权重
        weights_path = self.save_path / "test_weights.pth"
        self.model.save_weights(str(weights_path))
        
        # 确认文件已创建
        self.assertTrue(weights_path.exists())
        
        # 创建新模型并加载权重
        new_model = VisionTransformer.from_config(self.model_config)
        new_model.load_weights(str(weights_path))
        
        # 检查加载权重后的模型输出是否与原始模型一致
        with torch.no_grad():
            new_model.eval()
            loaded_output = new_model(self.test_input)
            
        # 验证输出一致
        self.assertTrue(torch.allclose(initial_output, loaded_output))
    
    def test_save_load_checkpoint(self):
        """测试保存和加载训练检查点"""
        # 保存检查点
        checkpoint_path = self.save_path / "test_checkpoint.pt"
        save_checkpoint(
            model=self.model,
            optimizer_state=self.optimizer_state,
            file_path=str(checkpoint_path),
            epoch=5,
            train_history=self.history
        )
        
        # 确认文件已创建
        self.assertTrue(checkpoint_path.exists())
        
        # 创建新模型并加载检查点
        new_model = VisionTransformer.from_config(self.model_config)
        loaded_model, loaded_optimizer_state, loaded_epoch, loaded_history = load_checkpoint(
            str(checkpoint_path), new_model
        )
        
        # 验证加载的内容
        self.assertEqual(loaded_epoch, 5)
        self.assertEqual(loaded_history['loss'], self.history['loss'])
        self.assertEqual(loaded_history['accuracy'], self.history['accuracy'])
        self.assertEqual(
            loaded_optimizer_state['param_groups'][0]['lr'],
            self.optimizer_state['param_groups'][0]['lr']
        )
        
    def test_optimizer_manager_state(self):
        """测试使用OptimizerManager保存和加载优化器状态"""
        # 创建OptimizerManager
        optimizer_manager = OptimizerManager(
            optimizer_type='adamw',
            model=self.model,
            lr=0.001,
            weight_decay=0.01,
            scheduler_type='cosine',
            scheduler_params={'T_max': 10, 'eta_min': 0.0001}
        )
        
        # 记录初始学习率
        initial_lr = optimizer_manager.get_lr()
        
        # 模拟学习率调度
        for _ in range(3):
            optimizer_manager.scheduler_step()
        
        # 记录调度后的学习率
        scheduled_lr = optimizer_manager.get_lr()
        
        # 确认学习率已调度(变化)
        self.assertNotEqual(initial_lr, scheduled_lr)
        
        # 获取优化器状态
        optimizer_state = optimizer_manager.state_dict()
        
        # 保存检查点
        checkpoint_path = self.save_path / "optimizer_manager_checkpoint.pt"
        save_checkpoint(
            model=self.model,
            optimizer_state=optimizer_state,
            file_path=str(checkpoint_path),
            epoch=5
        )
        
        # 创建新模型和优化器管理器
        new_model = VisionTransformer.from_config(self.model_config)
        new_optimizer_manager = OptimizerManager(
            optimizer_type='adamw',
            model=new_model,
            lr=0.001,
            weight_decay=0.01,
            scheduler_type='cosine',
            scheduler_params={'T_max': 10, 'eta_min': 0.0001}
        )
        
        # 加载检查点
        _, loaded_optimizer_state, _, _ = load_checkpoint(
            str(checkpoint_path), new_model
        )
        
        # 将加载的优化器状态应用到新的优化器管理器
        new_optimizer_manager.load_state_dict(loaded_optimizer_state)
        
        # 检查学习率是否正确恢复
        self.assertAlmostEqual(new_optimizer_manager.get_lr(), scheduled_lr, places=6)
        
        # 模拟训练环境中的保存和恢复
        # 创建训练循环
        loss_calculator = LossCalculator('cross_entropy')
        training_loop = TrainingLoop(
            model=self.model,
            loss_calculator=loss_calculator,
            optimizer_manager=optimizer_manager,
            device=torch.device('cpu')
        )
        
        # 保存检查点
        train_checkpoint_path = self.save_path / "training_checkpoint.pt"
        if optimizer_manager is not None:
            optimizer_state = optimizer_manager.state_dict()
            save_checkpoint(
                model=self.model,
                optimizer_state=optimizer_state,
                file_path=str(train_checkpoint_path),
                epoch=5,
                train_history=self.history
            )
        
        # 创建新的训练环境
        new_training_loop = TrainingLoop(
            model=new_model,
            loss_calculator=loss_calculator,
            optimizer_manager=new_optimizer_manager,
            device=torch.device('cpu')
        )
        
        # 加载检查点
        loaded_model, loaded_opt_state, loaded_epoch, loaded_history = load_checkpoint(
            str(train_checkpoint_path), new_model
        )
        
        # 应用加载的优化器状态
        new_optimizer_manager.load_state_dict(loaded_opt_state)
        
        # 验证加载后的学习率
        self.assertAlmostEqual(new_optimizer_manager.get_lr(), scheduled_lr, places=6)

    def test_export_to_onnx(self):
        """测试导出到ONNX格式"""
        try:
            import onnx
        except ImportError:
            self.skipTest("ONNX库未安装，跳过ONNX导出测试")
            
        # 导出为ONNX
        onnx_path = self.save_path / "test_model.onnx"
        export_to_onnx(self.model, str(onnx_path))
        
        # 确认文件已创建
        self.assertTrue(onnx_path.exists())
        self.assertTrue(Path(str(onnx_path).replace(".onnx", "_config.json")).exists())
        
        # 检查文件大小确保不为空 (最小应该有几KB)
        self.assertGreater(os.path.getsize(onnx_path), 1000)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        # 保存模型
        model_path = self.save_path / "test_model_info.pt"
        save_model(self.model, str(model_path), optimizer_state=self.optimizer_state)
        
        # 获取模型信息
        info = get_model_info(str(model_path))
        
        # 验证信息
        self.assertEqual(info['model_type'], 'VisionTransformer')
        self.assertTrue(info['has_optimizer'])
        self.assertEqual(info['num_classes'], 10)  # 从我们的配置
        
    def test_save_with_optimizer(self):
        """测试保存包含优化器状态的模型"""
        # 保存带优化器的模型
        save_path = self.save_path / "test_model_with_optim.pt"
        save_model(self.model, str(save_path), optimizer_state=self.optimizer_state)
        
        # 加载模型和优化器状态
        loaded_model, loaded_optimizer_state = load_model(str(save_path))
        
        # 验证优化器状态加载正确
        self.assertIsNotNone(loaded_optimizer_state)
        self.assertEqual(
            loaded_optimizer_state['param_groups'][0]['lr'],
            self.optimizer_state['param_groups'][0]['lr']
        )


if __name__ == '__main__':
    unittest.main() 