#!/usr/bin/env python
"""
测试脚本，用于验证绘制损失曲线的功能。
"""
import os
import sys
import logging

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.metrics_plots import plot_loss, plot_training_history
from src.utils.metrics_logger import MetricsLogger

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    测试绘制损失曲线功能。
    """
    logger.info("开始测试绘制损失曲线功能...")
    
    # 指定目录和文件路径
    metrics_dir = os.path.join('temp_metrics')
    plots_dir = os.path.join(metrics_dir, 'plots')
    train_metrics_path = os.path.join(metrics_dir, 'simulation_train_metrics.csv')
    eval_metrics_path = os.path.join(metrics_dir, 'simulation_eval_metrics.csv')
    
    # 确保输出目录存在
    os.makedirs(plots_dir, exist_ok=True)
    
    # 测试方法1：直接使用 plot_loss 函数（从CSV文件）
    logger.info("方法1：使用plot_loss从CSV文件生成曲线...")
    plot_loss(
        metrics_path=train_metrics_path,
        output_path=os.path.join(plots_dir, 'loss_curve_test1.png'),
        experiment_name='测试案例1',
        title='训练与验证损失曲线 - 直接从CSV文件'
    )
    
    # 测试方法2：使用 MetricsLogger 实例
    logger.info("方法2：使用MetricsLogger实例生成曲线...")
    metrics_logger = MetricsLogger(
        save_dir=metrics_dir,
        experiment_name='simulation',
        save_format='csv'
    )
    
    plot_loss(
        metrics_logger=metrics_logger,
        output_path=os.path.join(plots_dir, 'loss_curve_test2.png'),
        title='训练与验证损失曲线 - 使用MetricsLogger实例'
    )
    
    # 测试方法3：使用 plot_training_history 函数一次绘制多个指标
    logger.info("方法3：使用plot_training_history绘制多个指标...")
    plot_training_history(
        metrics_path=train_metrics_path,
        metrics=['loss', 'accuracy', 'learning_rate'],
        output_dir=plots_dir,
        experiment_name='多指标测试',
        save_format='png'
    )
    
    logger.info("测试完成！所有输出都已保存到 %s 目录", plots_dir)
    
if __name__ == "__main__":
    main() 