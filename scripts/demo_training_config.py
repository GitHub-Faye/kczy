#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练参数配置演示脚本
展示如何创建、验证、打印和保存训练配置，以及如何设置TensorBoard
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_config import (
    create_training_config, validate_training_params, print_training_config,
    save_training_config, load_training_config, setup_tensorboard, 
    get_device_info, get_optimal_config_for_device
)
from src.utils.config import TrainingConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练参数配置演示')
    
    parser.add_argument('--profile', type=str, choices=['default', 'fast_dev', 'high_performance', 'auto'],
                        default='default', help='使用预定义配置文件')
    parser.add_argument('--config-file', type=str, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='output/config_file_test',
                        help='输出目录，用于保存配置文件和TensorBoard日志')
    
    # 一些常用训练参数，可以从命令行覆盖默认值
    parser.add_argument('--batch-size', type=int, help='批大小')
    parser.add_argument('--epochs', type=int, dest='num_epochs', help='训练轮数')
    parser.add_argument('--lr', type=float, dest='learning_rate', help='学习率')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], help='训练设备')
    
    # TensorBoard相关参数
    parser.add_argument('--enable-tensorboard', action='store_true', help='启用TensorBoard')
    parser.add_argument('--start-tensorboard', action='store_true', help='启动TensorBoard服务器')
    
    return parser.parse_args()

def demo_create_config():
    """演示创建不同类型的训练配置"""
    logger.info("=== 演示创建不同类型的训练配置 ===")
    
    # 创建默认配置
    default_config = TrainingConfig.create_default()
    logger.info(f"默认配置 - 批大小: {default_config.batch_size}, 学习率: {default_config.learning_rate}")
    
    # 创建快速开发配置
    fast_dev_config = TrainingConfig.create_fast_dev()
    logger.info(f"快速开发配置 - 批大小: {fast_dev_config.batch_size}, 学习率: {fast_dev_config.learning_rate}")
    
    # 创建高性能配置
    high_perf_config = TrainingConfig.create_high_performance()
    logger.info(f"高性能配置 - 批大小: {high_perf_config.batch_size}, 学习率: {high_perf_config.learning_rate}")
    
    # 获取设备信息
    device_info = get_device_info()
    logger.info(f"设备信息: {device_info}")
    
    # 获取设备优化配置
    optimal_config = get_optimal_config_for_device()
    logger.info(f"设备优化配置 - 批大小: {optimal_config.batch_size}, 设备: {optimal_config.device}")
    
    logger.info("")

def demo_config_from_args(args):
    """演示从命令行参数创建配置"""
    logger.info("=== 演示从命令行参数创建配置 ===")
    
    # 如果指定了auto配置，使用设备优化配置
    if args.profile == 'auto':
        logger.info("使用自动设备优化配置")
        config = get_optimal_config_for_device()
    else:
        # 从命令行参数创建配置
        config = create_training_config(args=args, profile=args.profile)
    
    # 验证参数
    validate_training_params(config)
    
    # 打印配置
    print_training_config(config)
    
    return config

def demo_save_load_config(config, output_dir):
    """演示保存和加载配置"""
    logger.info("=== 演示保存和加载配置 ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON格式
    json_path = os.path.join(output_dir, 'training_config.json')
    save_training_config(config, json_path)
    
    # 保存为YAML格式
    yaml_path = os.path.join(output_dir, 'training_config.yaml')
    save_training_config(config, yaml_path, format='yaml')
    
    # 加载JSON配置
    loaded_config_json = load_training_config(json_path)
    logger.info(f"从JSON加载的配置 - 批大小: {loaded_config_json.batch_size}, 学习率: {loaded_config_json.learning_rate}")
    
    # 加载YAML配置
    loaded_config_yaml = load_training_config(yaml_path)
    logger.info(f"从YAML加载的配置 - 批大小: {loaded_config_yaml.batch_size}, 学习率: {loaded_config_yaml.learning_rate}")
    
    logger.info("")
    
    return loaded_config_json

def demo_tensorboard(config, output_dir):
    """演示TensorBoard设置"""
    logger.info("=== 演示TensorBoard设置 ===")
    
    # 更新TensorBoard相关配置
    config.enable_tensorboard = True
    config.tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    config.tensorboard_port = 6006
    config.log_histograms = True
    config.log_images = True
    
    # 设置实验名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.metrics_experiment_name = f"demo_experiment_{timestamp}"
    
    # 设置TensorBoard
    writer = setup_tensorboard(config)
    
    if writer:
        # 记录一些示例数据
        for i in range(10):
            writer.add_scalar('demo/sin', (i+1)**0.5 * 0.1 * i, i)
            writer.add_scalar('demo/cos', (i+1)**0.5 * 0.1 * (10-i), i)
        
        logger.info(f"已记录示例数据到TensorBoard，日志目录: {config.tensorboard_dir}")
        
        # 关闭SummaryWriter
        writer.close()
    
    logger.info("")

def main():
    """主函数"""
    args = parse_args()
    
    logger.info("训练参数配置演示")
    logger.info("=" * 50)
    
    # 准备输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 演示创建不同类型的配置
    demo_create_config()
    
    # 从命令行参数创建配置
    config = demo_config_from_args(args)
    
    # 演示保存和加载配置
    config = demo_save_load_config(config, output_dir)
    
    # 如果启用了TensorBoard，演示TensorBoard设置
    if args.enable_tensorboard:
        config.start_tensorboard = args.start_tensorboard
        demo_tensorboard(config, output_dir)
    
    logger.info("演示完成，再见！")

if __name__ == "__main__":
    main() 