#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型训练功能
验证训练循环、指标记录和检查点保存功能
"""
import os
import sys
import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vit import VisionTransformer
from src.models.train import TrainingLoop, LossCalculator, BackpropManager
from src.models.optimizer_manager import OptimizerManager
from src.utils.metrics_logger import MetricsLogger
from src.models.model_utils import save_checkpoint, load_checkpoint, save_model, load_model

class TestModelTraining(unittest.TestCase):
    """测试模型训练功能"""
    
    def setUp(self):
        """每个测试前的准备工作"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建小型测试数据集
        self.create_test_dataset()
        
        # 创建小型测试模型
        self.create_test_model()
        
        # 创建训练所需的组件
        self.setup_training_components()

    def tearDown(self):
        """每个测试后的清理工作"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)

    def create_test_dataset(self):
        """创建测试数据集"""
        # 创建小型随机数据集
        num_samples = 100
        img_size = 32
        num_classes = 5
        
        # 创建随机图像和标签
        images = torch.randn(num_samples, 3, img_size, img_size)
        labels = torch.randint(0, num_classes, (num_samples,))
        
        dataset = TensorDataset(images, labels)
        
        # 划分数据集
        train_size = 70
        val_size = 20
        test_size = 10
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=10)
        self.test_loader = DataLoader(test_dataset, batch_size=10)
        
        # 保存数据集属性
        self.img_size = img_size
        self.num_classes = num_classes

    def create_test_model(self):
        """创建测试模型"""
        # 创建小型 ViT 模型
        self.model = VisionTransformer(
            img_size=self.img_size,
            patch_size=4,
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=64,  # 小型嵌入维度
            depth=2,       # 只有2层
            num_heads=2,   # 只有2个头
            mlp_ratio=2,
            drop_rate=0.1
        )
        
        self.model = self.model.to(self.device)

    def setup_training_components(self):
        """设置训练组件"""
        # 创建损失计算器
        self.loss_calculator = LossCalculator(
            loss_type='cross_entropy'
        )
        
        # 创建优化器管理器
        self.optimizer_manager = OptimizerManager(
            optimizer_type='adam',
            model=self.model,
            lr=0.001,
            weight_decay=1e-5,
            scheduler_type=None
        )
        
        # 创建反向传播管理器
        self.backprop_manager = BackpropManager(
            grad_clip_norm=1.0
        )
        
        # 创建检查点和指标目录
        self.checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
        self.metrics_dir = os.path.join(self.test_dir, 'metrics')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 创建指标记录器
        self.metrics_logger = MetricsLogger(
            save_dir=self.metrics_dir,
            experiment_name='test_training',
            save_format='json',
            enable_tensorboard=False
        )
        
        # 创建训练循环
        self.training_loop = TrainingLoop(
            model=self.model,
            loss_calculator=self.loss_calculator,
            optimizer_manager=self.optimizer_manager,
            backprop_manager=self.backprop_manager,
            device=self.device,
            metrics_logger=self.metrics_logger
        )

    def test_train_epoch(self):
        """测试单个epoch的训练功能"""
        # 执行一个训练epoch
        train_metrics = self.training_loop.train_epoch(
            train_loader=self.train_loader,
            epoch=0
        )
        
        # 验证返回的指标
        self.assertIn('loss', train_metrics)
        self.assertIn('accuracy', train_metrics)
        
        # 验证指标值的合理性
        self.assertTrue(0 <= train_metrics['loss'] <= 10)  # 损失应该在合理范围内
        self.assertTrue(0 <= train_metrics['accuracy'] <= 100)  # 准确率以百分比表示，应该在[0,100]范围内

    def test_validation(self):
        """测试验证功能"""
        # 执行验证
        val_metrics = self.training_loop.validate(
            val_loader=self.val_loader
        )
        
        # 验证返回的指标
        self.assertIn('val_loss', val_metrics)
        self.assertIn('val_accuracy', val_metrics)
        
        # 验证指标值的合理性
        self.assertTrue(0 <= val_metrics['val_loss'] <= 10)
        self.assertTrue(0 <= val_metrics['val_accuracy'] <= 100)

    def test_train_multiple_epochs(self):
        """测试多个epoch的训练过程"""
        # 训练3个epochs
        num_epochs = 3
        
        training_history = self.training_loop.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=num_epochs,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_freq=1,
            log_interval=10  # 设置较大的值以减少日志输出
        )
        
        # 验证训练历史
        self.assertIn('train', training_history)
        self.assertIn('val', training_history)
        
        # 验证训练损失和准确率历史记录
        train_history = training_history['train']
        self.assertIn('loss', train_history)
        self.assertIn('accuracy', train_history)
        self.assertEqual(len(train_history['loss']), num_epochs)
        self.assertEqual(len(train_history['accuracy']), num_epochs)
        
        # 验证验证集损失和准确率历史记录
        val_history = training_history['val']
        self.assertIn('loss', val_history)
        self.assertIn('accuracy', val_history)
        self.assertEqual(len(val_history['loss']), num_epochs)
        self.assertEqual(len(val_history['accuracy']), num_epochs)
        
        # 验证检查点文件
        checkpoint_files = os.listdir(self.checkpoint_dir)
        self.assertTrue(len(checkpoint_files) > 0)
        
        # 验证指标文件
        metrics_files = os.listdir(self.metrics_dir)
        json_metrics_files = [f for f in metrics_files if f.endswith('.json')]
        self.assertTrue(len(json_metrics_files) > 0)

    def test_checkpoint_save_load(self):
        """测试检查点保存和加载功能"""
        # 训练1个epoch并保存检查点
        self.training_loop.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=1,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # 获取保存的检查点文件
        checkpoint_files = os.listdir(self.checkpoint_dir)
        self.assertTrue(len(checkpoint_files) > 0)
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_files[0])
        
        # 创建新模型并加载检查点
        new_model = VisionTransformer(
            img_size=self.img_size,
            patch_size=4,
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=64,
            depth=2,
            num_heads=2,
            mlp_ratio=2,
            drop_rate=0.1
        )
        
        new_model = new_model.to(self.device)
        
        # 验证模型状态不同
        original_state = self.model.state_dict()
        new_state = new_model.state_dict()
        
        # 随机选择一个参数比较
        param_name = list(original_state.keys())[0]
        self.assertFalse(torch.allclose(original_state[param_name], new_state[param_name]))
        
        # 加载检查点
        loaded_model, _, _, _ = load_checkpoint(
            file_path=checkpoint_path,
            model=new_model,
            device=self.device
        )
        
        # 验证加载后的模型状态与原始模型相同
        loaded_state = loaded_model.state_dict()
        for key in original_state:
            self.assertTrue(torch.allclose(original_state[key], loaded_state[key]))

    def test_model_save_load(self):
        """测试模型保存和加载功能"""
        # 保存模型
        model_path = os.path.join(self.test_dir, 'test_model.pth')
        
        metadata = {
            'test_name': 'model_save_load_test',
            'num_classes': self.num_classes,
            'img_size': self.img_size
        }
        
        save_model(self.model, model_path, metadata=metadata)
        
        # 确认文件存在
        self.assertTrue(os.path.exists(model_path))
        
        # 创建新模型实例
        new_model = VisionTransformer(
            img_size=self.img_size,
            patch_size=4,
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=64,
            depth=2,
            num_heads=2,
            mlp_ratio=2,
            drop_rate=0.1
        )
        
        # 加载模型
        loaded_model, loaded_metadata = load_model(
            file_path=model_path,
            model_class=None,  # 已经创建了模型实例
            device=self.device
        )
        
        # 验证元数据
        if loaded_metadata is not None:
            self.assertEqual(loaded_metadata.get('test_name'), metadata['test_name'])
            self.assertEqual(loaded_metadata.get('num_classes'), metadata['num_classes'])
        else:
            print("警告: 加载的模型没有元数据")
        
        # 验证模型状态
        original_state = self.model.state_dict()
        loaded_state = loaded_model.state_dict()
        
        for key in original_state:
            self.assertTrue(torch.allclose(original_state[key], loaded_state[key]))

    def test_early_stopping(self):
        """测试早停止功能"""
        # 创建一个具有早停止的训练循环
        max_epochs = 20  # 设置一个较大的最大epoch数
        patience = 2     # 设置较小的耐心值
        
        # 训练模型（应该早停止）
        training_history = self.training_loop.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=max_epochs,
            checkpoint_dir=self.checkpoint_dir,
            log_interval=10,
            early_stopping=True,
            early_stopping_patience=patience
        )
        
        # 验证训练轮数少于最大轮数（由于早停止）
        self.assertIn('train', training_history)
        train_history = training_history['train']
        self.assertIn('loss', train_history)
        self.assertLess(len(train_history['loss']), max_epochs)

    def test_evaluation(self):
        """测试模型评估功能"""
        # 首先训练模型几个epoch
        self.training_loop.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=2,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # 执行评估
        eval_results = self.training_loop.evaluate(
            test_loader=self.test_loader,
            num_classes=self.num_classes
        )
        
        # 验证评估结果
        self.assertIn('test_accuracy', eval_results)
        self.assertIn('precision', eval_results)
        self.assertIn('recall', eval_results)
        self.assertIn('f1_score', eval_results)
        self.assertIn('confusion_matrix', eval_results)
        
        # 验证结果值的合理性
        self.assertTrue(0 <= eval_results['test_accuracy'] <= 100)
        self.assertTrue(0 <= eval_results['precision'] <= 100)
        self.assertTrue(0 <= eval_results['recall'] <= 100)
        self.assertTrue(0 <= eval_results['f1_score'] <= 100)
        
        # 验证混淆矩阵的形状
        self.assertEqual(eval_results['confusion_matrix'].shape, (self.num_classes, self.num_classes))

if __name__ == '__main__':
    unittest.main() 