#!/usr/bin/env python
"""
测试增强的绘图保存功能，包括时间戳和元数据。
"""
import os
import sys
import json
import logging
from datetime import datetime

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.metrics_plots import plot_loss, plot_accuracy, plot_training_history
from src.utils.metrics_logger import MetricsLogger

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    测试增强的绘图保存功能。
    """
    logger.info("开始测试增强的绘图保存功能...")
    
    # 指定目录和文件路径
    metrics_dir = os.path.join('temp_metrics')
    plots_dir = os.path.join(metrics_dir, 'plots')
    train_metrics_path = os.path.join(metrics_dir, 'simulation_train_metrics.csv')
    
    # 确保输出目录存在
    os.makedirs(plots_dir, exist_ok=True)
    
    # 提供测试元数据
    test_metadata = {
        "model_version": "ViT-B/16-224",
        "training_parameters": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": 50,
            "optimizer": "Adam"
        },
        "dataset": "CIFAR-10",
        "generated_by": "test_plot_save.py",
        "timestamp": datetime.now().isoformat()
    }
    
    # 测试1：损失曲线保存 - 带时间戳
    logger.info("测试1：保存损失曲线 - 带时间戳")
    loss_plot_path = plot_loss(
        metrics_path=train_metrics_path,
        output_path=os.path.join(plots_dir, 'loss_curve_with_timestamp.png'),
        experiment_name='测试案例1',
        title='训练与验证损失曲线 - 带时间戳',
        add_timestamp=True
    )
    logger.info(f"损失曲线保存至：{loss_plot_path}")
    
    # 测试2：准确率曲线保存 - 带元数据
    logger.info("测试2：保存准确率曲线 - 带元数据")
    accuracy_plot_path = plot_accuracy(
        metrics_path=train_metrics_path,
        output_path=os.path.join(plots_dir, 'accuracy_curve_with_metadata.png'),
        experiment_name='测试案例2',
        title='训练与验证准确率曲线 - 带元数据',
        metadata=test_metadata
    )
    logger.info(f"准确率曲线保存至：{accuracy_plot_path}")
    
    # 检查元数据文件是否已创建
    metadata_file = f"{os.path.splitext(accuracy_plot_path)[0]}_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            saved_metadata = json.load(f)
        logger.info(f"元数据文件已创建: {metadata_file}")
        logger.info(f"保存的元数据: {saved_metadata}")
    else:
        logger.error(f"元数据文件未创建: {metadata_file}")
    
    # 测试3：多指标历史曲线 - 带时间戳和元数据
    logger.info("测试3：保存多指标历史曲线 - 带时间戳和元数据")
    result_paths = plot_training_history(
        metrics_path=train_metrics_path,
        metrics=['loss', 'accuracy', 'learning_rate'],
        output_dir=plots_dir,
        experiment_name='多指标测试',
        add_timestamp=True,
        metadata=test_metadata
    )
    
    logger.info(f"多指标历史曲线保存结果: {result_paths}")
    
    # 总结测试结果
    logger.info("测试完成！所有输出都已保存到 %s 目录", plots_dir)
    logger.info("检查以下功能是否正常工作:")
    logger.info("1. 是否能通过add_timestamp参数添加时间戳")
    logger.info("2. 是否能通过metadata参数添加元数据")
    logger.info("3. plot_training_history是否能正确返回保存的文件路径")
    
if __name__ == "__main__":
    main() 