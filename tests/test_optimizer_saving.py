#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试优化器状态保存和恢复功能
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
from src.models.optimizer_manager import OptimizerManager
from src.models.train import TrainingLoop, LossCalculator, BackpropManager
from src.models.model_utils import save_checkpoint, load_checkpoint
from torch.utils.data import TensorDataset, DataLoader

class TestOptimizerSaving(unittest.TestCase):
    """测试优化器状态保存和恢复功能"""

    def setUp(self):
        """初始化测试环境"""
        # 创建临时目录用于保存测试文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = Path(self.temp_dir.name)
        
        # 创建测试用模型
        self.model_config = ViTConfig.create_tiny(num_classes=10)
        self.model = VisionTransformer.from_config(self.model_config)
        
        # 创建一些测试数据
        self.batch_size = 2
        self.img_size = self.model_config.img_size
        self.in_channels = self.model_config.in_channels
        
        # 创建小型训练集
        inputs = torch.randn(10, self.in_channels, self.img_size, self.img_size)
        labels = torch.randint(0, 10, (10,))
        dataset = TensorDataset(inputs, labels)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size)
        
    def tearDown(self):
        """清理测试环境"""
        self.temp_dir.cleanup()
    
    def test_sgd_optimizer_saving(self):
        """测试SGD优化器状态保存和恢复"""
        optimizer_manager = OptimizerManager(
            optimizer_type='sgd',
            model=self.model,
            lr=0.01,
            weight_decay=0.0001,
            optimizer_params={'momentum': 0.9, 'nesterov': True}
        )
        
        # 验证优化器类型
        self.assertEqual(optimizer_manager.get_optimizer_name(), 'sgd')
        
        # 创建反向传播管理器
        backprop_manager = BackpropManager()
        
        # 创建训练循环
        loss_calculator = LossCalculator('cross_entropy')
        training_loop = TrainingLoop(
            model=self.model,
            loss_calculator=loss_calculator,
            optimizer_manager=optimizer_manager,
            backprop_manager=backprop_manager,
            device=torch.device('cpu')
        )
        
        # 进行几步训练
        history = training_loop.train(self.train_loader, num_epochs=2)
        
        # 记录训练后的学习率和部分参数
        post_train_lr = optimizer_manager.get_lr()
        post_train_params = list(self.model.parameters())[0].clone().detach()
        
        # 保存检查点
        checkpoint_path = self.save_path / "sgd_checkpoint.pt"
        optimizer_state = optimizer_manager.state_dict()
        save_checkpoint(
            model=self.model,
            optimizer_state=optimizer_state,
            file_path=str(checkpoint_path),
            epoch=2,
            train_history=history
        )
        
        # 创建新模型和优化器
        new_model = VisionTransformer.from_config(self.model_config)
        new_optimizer_manager = OptimizerManager(
            optimizer_type='sgd',
            model=new_model,
            lr=0.01,  # 使用相同的初始值
            weight_decay=0.0001,
            optimizer_params={'momentum': 0.9, 'nesterov': True}
        )
        
        # 加载检查点
        loaded_model, loaded_opt_state, loaded_epoch, _ = load_checkpoint(
            str(checkpoint_path), new_model
        )
        new_optimizer_manager.load_state_dict(loaded_opt_state)
        
        # 验证加载后的状态
        self.assertEqual(loaded_epoch, 2)
        self.assertAlmostEqual(new_optimizer_manager.get_lr(), post_train_lr)
        self.assertTrue(torch.allclose(
            list(new_model.parameters())[0], 
            post_train_params
        ))
        
    def test_adam_scheduler_saving(self):
        """测试Adam优化器和学习率调度器状态保存和恢复"""
        # 创建带调度器的优化器
        optimizer_manager = OptimizerManager(
            optimizer_type='adam',
            model=self.model,
            lr=0.001,
            weight_decay=0.0001,
            scheduler_type='step',
            scheduler_params={'step_size': 1, 'gamma': 0.5}
        )
        
        # 验证优化器和调度器
        self.assertEqual(optimizer_manager.get_optimizer_name(), 'adam')
        self.assertIsNotNone(optimizer_manager.scheduler)
        
        # 记录初始学习率
        initial_lr = optimizer_manager.get_lr()
        
        # 执行调度器步骤
        optimizer_manager.scheduler_step()
        
        # 验证学习率改变
        scheduled_lr = optimizer_manager.get_lr()
        self.assertNotEqual(initial_lr, scheduled_lr)
        self.assertAlmostEqual(scheduled_lr, initial_lr * 0.5)
        
        # 保存检查点
        checkpoint_path = self.save_path / "adam_scheduler_checkpoint.pt"
        save_checkpoint(
            model=self.model,
            optimizer_state=optimizer_manager.state_dict(),
            file_path=str(checkpoint_path),
            epoch=1
        )
        
        # 创建新的优化器和调度器
        new_model = VisionTransformer.from_config(self.model_config)
        new_optimizer_manager = OptimizerManager(
            optimizer_type='adam',
            model=new_model,
            lr=0.001,  # 使用相同的初始配置
            weight_decay=0.0001,
            scheduler_type='step',
            scheduler_params={'step_size': 1, 'gamma': 0.5}
        )
        
        # 加载检查点
        _, loaded_opt_state, _, _ = load_checkpoint(str(checkpoint_path), new_model)
        new_optimizer_manager.load_state_dict(loaded_opt_state)
        
        # 验证加载的学习率
        loaded_lr = new_optimizer_manager.get_lr()
        self.assertAlmostEqual(loaded_lr, scheduled_lr)
        
        # 再次运行调度器
        new_optimizer_manager.scheduler_step()
        
        # 验证学习率再次减半
        new_scheduled_lr = new_optimizer_manager.get_lr()
        self.assertAlmostEqual(new_scheduled_lr, scheduled_lr * 0.5)
        
    def test_optimizer_state_in_training_loop(self):
        """测试在完整训练循环中保存和恢复优化器状态"""
        # 创建优化器
        optimizer_manager = OptimizerManager(
            optimizer_type='adamw',
            model=self.model,
            lr=0.001,
            weight_decay=0.01,
            scheduler_type='cosine',
            scheduler_params={'T_max': 3, 'eta_min': 0.0001}
        )
        
        # 创建训练组件
        loss_calculator = LossCalculator('cross_entropy')
        backprop_manager = BackpropManager(grad_clip_value=1.0)
        
        # 创建训练循环
        training_loop = TrainingLoop(
            model=self.model,
            loss_calculator=loss_calculator,
            optimizer_manager=optimizer_manager,
            backprop_manager=backprop_manager,
            device=torch.device('cpu')
        )
        
        # 训练2个epoch
        history_part1 = training_loop.train(
            self.train_loader, 
            num_epochs=2,
            checkpoint_dir=str(self.save_path)
        )
        
        # 记录训练后的状态
        post_train_lr = optimizer_manager.get_lr()
        post_train_loss = history_part1['loss'][-1]
        
        # 保存检查点
        checkpoint_path = self.save_path / "training_checkpoint.pt"
        save_checkpoint(
            model=self.model,
            optimizer_state=optimizer_manager.state_dict(),
            file_path=str(checkpoint_path),
            epoch=2,
            train_history=history_part1
        )
        
        # 创建新模型和训练循环
        new_model = VisionTransformer.from_config(self.model_config)
        new_optimizer_manager = OptimizerManager(
            optimizer_type='adamw',
            model=new_model,
            lr=0.001,
            weight_decay=0.01,
            scheduler_type='cosine',
            scheduler_params={'T_max': 3, 'eta_min': 0.0001}
        )
        
        new_training_loop = TrainingLoop(
            model=new_model,
            loss_calculator=loss_calculator,
            optimizer_manager=new_optimizer_manager,
            backprop_manager=backprop_manager,
            device=torch.device('cpu')
        )
        
        # 加载检查点
        _, loaded_opt_state, loaded_epoch, loaded_history = load_checkpoint(
            str(checkpoint_path), new_model
        )
        
        # 应用优化器状态
        new_optimizer_manager.load_state_dict(loaded_opt_state)
        
        # 验证加载的状态
        self.assertEqual(loaded_epoch, 2)
        self.assertEqual(len(loaded_history['loss']), 2)
        self.assertAlmostEqual(new_optimizer_manager.get_lr(), post_train_lr)
        
        # 继续训练2个epoch
        history_part2 = new_training_loop.train(
            self.train_loader, 
            num_epochs=2,
            checkpoint_dir=str(self.save_path)
        )
        
        # 验证训练继续进行而不是重新开始
        # 由于随机初始化，我们只能验证学习率调度是否继续
        # 而不是精确的损失值
        post_continue_lr = new_optimizer_manager.get_lr()
        self.assertNotEqual(post_continue_lr, post_train_lr)
        
        # 最终测试，确认状态字典包含所有必要的优化器信息
        final_state = new_optimizer_manager.state_dict()
        self.assertIn('optimizer_type', final_state)
        self.assertIn('optimizer', final_state)
        self.assertIn('scheduler_type', final_state)
        self.assertIn('scheduler_params', final_state)
        self.assertIn('scheduler', final_state)


if __name__ == '__main__':
    unittest.main() 