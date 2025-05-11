#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试命令行参数解析功能
"""

import sys
import os
import json

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils.cli import parse_args, load_config

def main():
    """测试CLI参数解析"""
    # 模拟不同的命令行参数
    test_cases = [
        # 基本参数
        ["--data-dir", "./data", "--anno-file", "./data/annotations.csv", "--num-classes", "10"],
        # 模型参数
        ["--model-type", "tiny", "--num-classes", "5", "--pretrained"],
        # 训练参数
        ["--epochs", "50", "--lr", "0.001", "--optimizer", "adamw", "--seed", "100"],
        # 日志参数
        ["--log-dir", "./custom_logs", "--experiment-name", "test_run", "--plot-metrics"],
        # 新增：超参数测试
        ["--lr", "0.0005", "--weight-decay", "1e-5", "--loss-type", "focal", 
         "--step-size", "20", "--gamma", "0.2"],
        # 新增：优化器特定参数测试
        ["--optimizer", "sgd", "--momentum", "0.95", "--nesterov"],
        # 新增：学习率调度器参数测试
        ["--scheduler", "cosine", "--t-max", "200", "--eta-min", "1e-6", "--min-lr", "1e-7"]
    ]
    
    print("=== 测试命令行参数解析 ===")
    for i, test_args in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1} ---")
        print(f"命令行参数: {' '.join(test_args)}")
        args = parse_args(test_args)
        # 将Namespace转换为字典，便于打印
        args_dict = vars(args)
        print(f"解析结果:")
        print(json.dumps(args_dict, indent=2, ensure_ascii=False))
    
    # 测试配置文件加载
    print("\n=== 测试配置文件加载 ===")
    config_path = os.path.join(os.path.dirname(__file__), "example_config.json")
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"配置文件内容:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        # 测试配置文件与命令行参数结合
        print("\n=== 测试配置文件与命令行参数结合 ===")
        # 命令行参数优先
        cmd_args = ["--config", config_path, "--lr", "0.0001", "--model-type", "base", 
                   "--loss-type", "focal", "--beta1", "0.85", "--beta2", "0.995"]
        print(f"命令行参数: {' '.join(cmd_args)}")
        args = parse_args(cmd_args)
        args_dict = vars(args)
        print(f"解析结果 (命令行参数优先):")
        print(json.dumps(args_dict, indent=2, ensure_ascii=False))
    else:
        print(f"警告: 配置文件不存在 - {config_path}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 