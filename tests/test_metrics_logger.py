import os
import unittest
import sys
import shutil
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.metrics_logger import MetricsLogger

class TestMetricsLogger(unittest.TestCase):
    """测试指标记录器功能"""
    
    def setUp(self):
        """每个测试前的设置"""
        # 创建临时目录用于保存指标
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"
        
    def tearDown(self):
        """每个测试后的清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """测试初始化"""
        metrics_logger = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name
        )
        
        # 检查实例属性
        self.assertEqual(metrics_logger.save_dir, self.temp_dir)
        self.assertEqual(metrics_logger.experiment_name, self.experiment_name)
        self.assertEqual(metrics_logger.save_format, 'csv')
        self.assertEqual(metrics_logger.save_freq, 1)
        
        # 检查指标文件路径
        expected_train_path = os.path.join(
            self.temp_dir, f"{self.experiment_name}_train_metrics.csv"
        )
        expected_eval_path = os.path.join(
            self.temp_dir, f"{self.experiment_name}_eval_metrics.csv"
        )
        
        self.assertEqual(metrics_logger.train_metrics_path, expected_train_path)
        self.assertEqual(metrics_logger.eval_metrics_path, expected_eval_path)
    
    def test_log_train_metrics(self):
        """测试记录训练指标"""
        metrics_logger = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name
        )
        
        # 模拟训练指标
        train_metrics = {
            'loss': 0.5,
            'accuracy': 90.0,
            'learning_rate': 0.001
        }
        
        # 记录指标
        metrics_logger.log_train_metrics(train_metrics, epoch=1)
        
        # 检查指标是否已保存
        self.assertTrue(os.path.exists(metrics_logger.train_metrics_path))
        
        # 检查记录的指标内容
        for metric_name, metric_value in train_metrics.items():
            self.assertIn(metric_name, metrics_logger.train_metrics)
            self.assertEqual(len(metrics_logger.train_metrics[metric_name]), 1)
            self.assertEqual(metrics_logger.train_metrics[metric_name][0]['epoch'], 1)
            self.assertEqual(metrics_logger.train_metrics[metric_name][0]['value'], metric_value)
    
    def test_log_eval_metrics(self):
        """测试记录评估指标"""
        metrics_logger = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name
        )
        
        # 模拟评估指标
        eval_metrics = {
            'val_loss': 0.4,
            'val_accuracy': 92.0
        }
        
        # 记录指标
        metrics_logger.log_eval_metrics(eval_metrics, epoch=1)
        
        # 检查指标是否已保存
        self.assertTrue(os.path.exists(metrics_logger.eval_metrics_path))
        
        # 检查记录的指标内容
        for metric_name, metric_value in eval_metrics.items():
            self.assertIn(metric_name, metrics_logger.eval_metrics)
            self.assertEqual(len(metrics_logger.eval_metrics[metric_name]), 1)
            self.assertEqual(metrics_logger.eval_metrics[metric_name][0]['epoch'], 1)
            self.assertEqual(metrics_logger.eval_metrics[metric_name][0]['value'], metric_value)
    
    def test_save_load_csv(self):
        """测试保存和加载CSV格式的指标"""
        metrics_logger = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            save_format='csv'
        )
        
        # 模拟多个epoch的训练指标
        for epoch in range(5):
            train_metrics = {
                'loss': 0.5 - epoch * 0.1,
                'accuracy': 80.0 + epoch * 2.0
            }
            metrics_logger.log_train_metrics(train_metrics, epoch)
        
        # 重新创建实例
        metrics_logger2 = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            save_format='csv'
        )
        
        # 加载指标
        loaded_metrics = metrics_logger2.load_train_metrics()
        
        # 检查加载的指标
        self.assertEqual(len(loaded_metrics), 2)  # loss和accuracy
        self.assertEqual(len(loaded_metrics['loss']), 5)  # 5个epoch
        self.assertEqual(len(loaded_metrics['accuracy']), 5)  # 5个epoch
    
    def test_save_load_json(self):
        """测试保存和加载JSON格式的指标"""
        metrics_logger = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            save_format='json'
        )
        
        # 模拟多个epoch的训练指标
        for epoch in range(5):
            train_metrics = {
                'loss': 0.5 - epoch * 0.1,
                'accuracy': 80.0 + epoch * 2.0
            }
            metrics_logger.log_train_metrics(train_metrics, epoch)
        
        # 重新创建实例
        metrics_logger2 = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            save_format='json'
        )
        
        # 加载指标
        loaded_metrics = metrics_logger2.load_train_metrics()
        
        # 检查加载的指标
        self.assertEqual(len(loaded_metrics), 2)  # loss和accuracy
        self.assertEqual(len(loaded_metrics['loss']), 5)  # 5个epoch
        self.assertEqual(len(loaded_metrics['accuracy']), 5)  # 5个epoch
    
    def test_plot_metric(self):
        """测试绘制指标曲线"""
        metrics_logger = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name
        )
        
        # 模拟多个epoch的训练和验证指标
        for epoch in range(10):
            train_metrics = {
                'loss': 1.0 - epoch * 0.1,
                'accuracy': 70.0 + epoch * 3.0
            }
            eval_metrics = {
                'val_loss': 0.9 - epoch * 0.08,
                'val_accuracy': 75.0 + epoch * 2.5
            }
            metrics_logger.log_train_metrics(train_metrics, epoch)
            metrics_logger.log_eval_metrics(eval_metrics, epoch)
        
        # 创建输出路径
        output_path = os.path.join(self.temp_dir, "loss_curve.png")
        
        # 绘制指标曲线
        metrics_logger.plot_metric('loss', output_path=output_path)
        
        # 检查图像是否已保存
        self.assertTrue(os.path.exists(output_path))
    
    def test_summary(self):
        """测试生成指标摘要"""
        metrics_logger = MetricsLogger(
            save_dir=self.temp_dir,
            experiment_name=self.experiment_name
        )
        
        # 模拟多个epoch的训练和验证指标
        for epoch in range(5):
            train_metrics = {
                'loss': 0.5 - epoch * 0.1,
                'accuracy': 80.0 + epoch * 2.0
            }
            eval_metrics = {
                'val_loss': 0.4 - epoch * 0.08,
                'val_accuracy': 85.0 + epoch * 1.5
            }
            metrics_logger.log_train_metrics(train_metrics, epoch)
            metrics_logger.log_eval_metrics(eval_metrics, epoch)
        
        # 获取简要摘要
        summary = metrics_logger.summary(detailed=False)
        
        # 检查摘要内容
        self.assertEqual(summary['experiment_name'], self.experiment_name)
        self.assertEqual(summary['train_metrics_count'], 2)  # loss和accuracy
        self.assertEqual(summary['eval_metrics_count'], 2)  # val_loss和val_accuracy
        
        # 获取详细摘要
        detailed_summary = metrics_logger.summary(detailed=True)
        
        # 检查详细摘要内容
        self.assertIn('train_metrics_stats', detailed_summary)
        self.assertIn('eval_metrics_stats', detailed_summary)
        self.assertIn('loss', detailed_summary['train_metrics_stats'])
        self.assertIn('accuracy', detailed_summary['train_metrics_stats'])
        self.assertIn('val_loss', detailed_summary['eval_metrics_stats'])
        self.assertIn('val_accuracy', detailed_summary['eval_metrics_stats'])

def simulate_training():
    """模拟训练过程，用于手动测试指标记录功能"""
    # 创建临时目录
    temp_dir = 'temp_metrics'
    os.makedirs(temp_dir, exist_ok=True)
    
    # 创建指标记录器
    metrics_logger = MetricsLogger(
        save_dir=temp_dir,
        experiment_name='simulation',
        save_format='csv'
    )
    
    # 模拟训练过程
    num_epochs = 50
    
    # 初始指标值
    loss = 2.0
    accuracy = 10.0
    val_loss = 1.8
    val_accuracy = 15.0
    
    # 模拟训练和评估
    for epoch in range(num_epochs):
        # 更新指标值（模拟训练过程中的变化）
        loss = max(0.1, loss - 0.04)
        accuracy = min(99.0, accuracy + 1.8)
        val_loss = max(0.15, val_loss - 0.035)
        val_accuracy = min(98.0, val_accuracy + 1.6)
        
        # 添加一些随机波动
        loss_noise = np.random.normal(0, 0.05)
        acc_noise = np.random.normal(0, 1.0)
        val_loss_noise = np.random.normal(0, 0.04)
        val_acc_noise = np.random.normal(0, 0.8)
        
        # 记录训练指标
        train_metrics = {
            'loss': loss + loss_noise,
            'accuracy': accuracy + acc_noise,
            'learning_rate': 0.001 * (0.9 ** (epoch // 10))
        }
        metrics_logger.log_train_metrics(train_metrics, epoch)
        
        # 记录评估指标
        eval_metrics = {
            'val_loss': val_loss + val_loss_noise,
            'val_accuracy': val_accuracy + val_acc_noise
        }
        metrics_logger.log_eval_metrics(eval_metrics, epoch)
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                 f"Loss: {train_metrics['loss']:.4f}, "
                 f"Acc: {train_metrics['accuracy']:.2f}%, "
                 f"Val Loss: {eval_metrics['val_loss']:.4f}, "
                 f"Val Acc: {eval_metrics['val_accuracy']:.2f}%")
    
    # 创建输出目录
    plots_dir = os.path.join(temp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制指标曲线
    metrics_logger.plot_metrics(['loss', 'accuracy'], output_dir=plots_dir)
    
    # 打印摘要
    summary = metrics_logger.summary(detailed=True)
    print("\n=== 指标摘要 ===")
    print(f"训练指标数量: {summary['train_metrics_count']}")
    print(f"评估指标数量: {summary['eval_metrics_count']}")
    print("\n训练指标统计:")
    for metric_name, stats in summary['train_metrics_stats'].items():
        print(f"  {metric_name}: Min={stats['min']:.4f}, Max={stats['max']:.4f}, Mean={stats['mean']:.4f}")
    print("\n评估指标统计:")
    for metric_name, stats in summary['eval_metrics_stats'].items():
        print(f"  {metric_name}: Min={stats['min']:.4f}, Max={stats['max']:.4f}, Mean={stats['mean']:.4f}")
    
    print(f"\n指标已保存到: {temp_dir}")
    print(f"指标曲线已保存到: {plots_dir}")

if __name__ == '__main__':
    # 运行单元测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # 手动测试指标记录功能（可选）
    print("\n=== 模拟训练过程，测试指标记录功能 ===")
    simulate_training() 