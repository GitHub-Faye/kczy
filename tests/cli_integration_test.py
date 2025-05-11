#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI与训练脚本集成测试
测试命令行参数是否能正确地与训练流程集成
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入必要的模块
from src.utils.cli import parse_args
from scripts.train import setup_environment, create_vit_config, create_training_config

def run_integration_test(test_name, args_list):
    """
    运行集成测试，测试CLI参数能否正确地与训练流程集成
    
    参数:
        test_name: 测试名称
        args_list: 命令行参数列表
    """
    print(f"\n执行集成测试: {test_name}")
    print(f"命令行参数: {' '.join(args_list)}")
    
    try:
        # 解析参数
        args = parse_args(args_list)
        print("✅ 命令行参数解析成功")
        
        # 设置环境
        setup_environment(args)
        print("✅ 环境设置成功")
        
        # 创建ViT模型配置
        vit_config = create_vit_config(args)
        print("✅ ViT模型配置创建成功")
        print(f"   - 模型类型: {args.model_type}")
        print(f"   - 嵌入维度: {vit_config.embed_dim}")
        print(f"   - 层数: {vit_config.depth}")
        print(f"   - 注意力头数: {vit_config.num_heads}")
        print(f"   - 分类数: {vit_config.num_classes}")
        
        # 创建训练配置
        training_config = create_training_config(args)
        print("✅ 训练配置创建成功")
        print(f"   - 优化器: {training_config.optimizer_type}")
        print(f"   - 学习率: {training_config.learning_rate}")
        print(f"   - 批量大小: {training_config.batch_size}")
        print(f"   - 轮数: {training_config.num_epochs}")
        print(f"   - 学习率调度器: {training_config.scheduler_type or 'None'}")
        
        # 保存配置到测试结果目录
        output_dir = os.path.join('tests', 'outputs', 'cli_tests', 'integration', test_name.replace(' ', '_'))
        os.makedirs(output_dir, exist_ok=True)
        
        vit_config_path = os.path.join(output_dir, 'vit_config.json')
        training_config_path = os.path.join(output_dir, 'training_config.json')
        
        vit_config.save(vit_config_path)
        training_config.save(training_config_path)
        
        # 保存参数到文件
        args_path = os.path.join(output_dir, 'parsed_args.json')
        with open(args_path, 'w', encoding='utf-8') as f:
            # 需要过滤掉不可序列化的对象
            args_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v 
                        for k, v in vars(args).items()}
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 测试结果已保存到: {output_dir}")
        return True, ""
    except Exception as e:
        error_message = f"❌ 测试失败: {str(e)}"
        print(error_message)
        return False, error_message

def main():
    """执行CLI集成测试"""
    output_dir = os.path.join('tests', 'outputs', 'cli_tests', 'integration')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取配置文件路径
    json_config_path = os.path.join('tests', 'outputs', 'cli_tests', 'extended_test_config.json')
    yaml_config_path = os.path.join('tests', 'outputs', 'cli_tests', 'extended_test_config.yaml')
    
    # 测试用例列表 [测试名称, 参数列表]
    test_cases = [
        # 基本参数测试
        (
            "基本训练设置", 
            ["--model-type", "tiny", "--num-classes", "10", "--epochs", "5", 
             "--output-dir", os.path.join(output_dir, "basic_test")]
        ),
        
        # 配置文件测试 - JSON
        (
            "JSON配置文件", 
            ["--config", json_config_path, "--output-dir", os.path.join(output_dir, "json_config_test")]
        ),
        
        # 配置文件测试 - YAML
        (
            "YAML配置文件", 
            ["--config", yaml_config_path, "--output-dir", os.path.join(output_dir, "yaml_config_test")]
        ),
        
        # 配置文件与参数混合测试
        (
            "配置文件与参数混合", 
            ["--config", json_config_path, "--model-type", "tiny", "--lr", "0.0001", 
             "--output-dir", os.path.join(output_dir, "mixed_config_test")]
        ),
        
        # 全特性测试
        (
            "全特性测试",
            ["--model-type", "base", "--num-classes", "20", "--img-size", "256", "--batch-size", "32",
             "--epochs", "10", "--lr", "0.0005", "--optimizer", "adamw", "--weight-decay", "1e-5",
             "--scheduler", "cosine", "--t-max", "10", "--min-lr", "1e-6",
             "--beta1", "0.9", "--beta2", "0.999", "--eps", "1e-8",
             "--data-dir", "./data/full_test", "--val-split", "0.2", "--num-workers", "4",
             "--use-augmentation", "--aug-rotate", "10", "--aug-translate", "0.1", "--aug-hflip",
             "--normalize", "--normalize-mean", "0.5,0.5,0.5", "--normalize-std", "0.5,0.5,0.5",
             "--early-stopping", "--patience", "5", "--grad-clip-value", "1.0",
             "--log-dir", os.path.join(output_dir, "full_test/logs"),
             "--checkpoint-dir", os.path.join(output_dir, "full_test/checkpoints"),
             "--output-dir", os.path.join(output_dir, "full_test"),
             "--experiment-name", "full_feature_test"]
        ),
        
        # 最小配置测试
        (
            "最小配置测试",
            ["--model-type", "tiny", "--num-classes", "2",
             "--output-dir", os.path.join(output_dir, "minimal_test")]
        ),
        
        # 边界条件测试
        (
            "边界条件测试",
            ["--model-type", "large", "--num-classes", "1000", "--img-size", "512",
             "--batch-size", "4", "--epochs", "1", "--lr", "0.00001",
             "--output-dir", os.path.join(output_dir, "edge_condition_test")]
        )
    ]
    
    # 运行测试
    results = {}
    
    for name, args in test_cases:
        success, error = run_integration_test(name, args)
        results[name] = {"success": success, "error": error}
    
    # 生成报告
    report_path = os.path.join(output_dir, "integration_test_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLI与训练脚本集成测试报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        passed = sum(1 for r in results.values() if r["success"])
        f.write(f"总测试数: {len(results)}\n")
        f.write(f"通过: {passed}\n")
        f.write(f"失败: {len(results) - passed}\n\n")
        
        f.write("详细测试结果:\n")
        f.write("-" * 40 + "\n")
        
        for name, result in results.items():
            status = "✅ 通过" if result["success"] else f"❌ 失败: {result['error']}"
            f.write(f"{name}: {status}\n")
    
    print(f"\n集成测试完成! 通过率: {passed}/{len(results)}")
    print(f"测试报告已保存到: {report_path}")
    
    # 返回成功代码如果所有测试都通过
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main()) 