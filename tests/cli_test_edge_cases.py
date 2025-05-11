#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
命令行参数解析功能的边界情况测试脚本
重点测试错误处理、特殊输入和边缘情况
"""

import sys
import os
import json
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils.cli import parse_args, load_config, create_parser, print_args_info

def test_edge_case(name, args_list, expected_exit=False, expected_output=None):
    """
    测试边界情况并输出结果到控制台和文件
    
    参数:
        name: 测试用例名称
        args_list: 命令行参数列表
        expected_exit: 是否预期会退出
        expected_output: 预期的输出中应包含的字符串
    
    返回:
        测试是否通过
    """
    print(f"\n执行测试: {name}")
    print(f"命令行参数: {' '.join(args_list)}")
    
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            args = parse_args(args_list)
        
        if expected_exit:
            print(f"❌ 测试失败: 预期退出但没有退出")
            return False
        
        print("✅ 命令行参数解析成功")
        print("\n解析结果:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        
        # 检查预期输出
        if expected_output:
            output = stdout.getvalue() + stderr.getvalue()
            if expected_output in output:
                print(f"✅ 预期输出存在: '{expected_output}'")
                return True
            else:
                print(f"❌ 测试失败: 预期输出不存在: '{expected_output}'")
                return False
        
        return True
    
    except SystemExit:
        output = stdout.getvalue() + stderr.getvalue()
        if expected_exit:
            print("✅ 预期的退出行为")
            if expected_output and expected_output in output:
                print(f"✅ 预期输出存在: '{expected_output}'")
                return True
            elif expected_output:
                print(f"❌ 测试失败: 预期输出不存在: '{expected_output}'")
                return False
            return True
        else:
            print(f"❌ 测试失败: 意外退出，输出: {output}")
            return False
    
    except Exception as e:
        if isinstance(e, FileNotFoundError) and "non_existent_file.json" in str(e) and expected_exit:
            print("✅ 预期的异常: 文件不存在")
            return True
        elif isinstance(e, json.JSONDecodeError) and expected_exit:
            print("✅ 预期的异常: 无效的JSON格式")
            return True
        else:
            print(f"❌ 测试失败: 出现异常: {str(e)}")
            return False

def main():
    """运行边界情况测试"""
    # 设置测试输出目录
    test_output_dir = os.path.join('tests', 'outputs', 'cli_tests')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 创建临时配置文件目录
    temp_config_dir = os.path.join(test_output_dir, 'temp_configs')
    os.makedirs(temp_config_dir, exist_ok=True)
    
    # 创建有效的配置文件
    valid_config = {
        "data-dir": "./data/valid",
        "batch-size": 16,
        "model-type": "small"
    }
    valid_config_path = os.path.join(temp_config_dir, 'valid.json')
    with open(valid_config_path, 'w') as f:
        json.dump(valid_config, f)
    
    # 创建无效的配置文件
    invalid_config = '{"data-dir": "./data/invalid", missing_quote: "value"}'
    invalid_config_path = os.path.join(temp_config_dir, 'invalid.json')
    with open(invalid_config_path, 'w') as f:
        f.write(invalid_config)
    
    # 测试用例列表
    test_cases = [
        # 基本测试
        ("默认参数", [], False),
        ("极大值参数", ["--epochs", "1000000"], False),
        
        # 选择参数测试
        ("无效的选择参数", ["--model-type", "gigantic"], True, "invalid choice"),
        ("无效的优化器选择", ["--optimizer", "invalid_optimizer"], True, "invalid choice"),
        ("无效的调度器选择", ["--scheduler", "invalid_scheduler"], True, "invalid choice"),
        
        # 配置文件测试
        ("不存在的配置文件", ["--config", "non_existent_file.json"], True),
        ("无效格式的JSON配置", ["--config", invalid_config_path], True),
        
        # 特殊参数格式测试
        ("无效的normalize-mean格式", ["--normalize-mean", "a,b,c"], False, "无法解析normalize-mean"),
        ("无效的normalize-std格式", ["--normalize-std", "x,y,z"], False, "无法解析normalize-std"),
        ("无效的milestones格式", ["--milestones", "10,twenty,30"], False, "无法解析milestones"),
        
        # 参数重复和冗余测试
        ("重复参数", ["--batch-size", "32", "--batch-size", "64"], False),
        ("冗余参数", ["--model-type", "base", "--model-type", "base"], False),
        
        # 参数组合和互相影响测试
        ("互相影响的布尔标志", ["--use-augmentation", "--aug-hflip", "--aug-vflip"], False),
        ("复杂参数组合-有效", ["--model-type", "tiny", "--epochs", "10", "--batch-size", "16", 
                      "--pretrained", "--early-stopping", "--patience", "5",
                      "--normalize-mean", "0.5,0.5,0.5", "--normalize-std", "0.2,0.2,0.2"], False),
        
        # 配置文件与命令行参数组合测试
        ("配置覆盖-1", ["--config", valid_config_path, "--batch-size", "32"], False),
        ("配置覆盖-2", ["--config", valid_config_path, "--model-type", "base"], False),
        
        # 帮助和错误处理测试
        ("帮助信息", ["--help"], True, "usage:"),
        ("未知参数", ["--unknown-parameter", "value"], True, "unrecognized arguments"),
        ("缺少必要参数值", ["--data-dir"], True, "expected one argument"),
        
        # 冲突参数测试
        ("冲突的参数组合", ["--oversampling", "--undersampling"], False),
    ]
    
    # 运行测试用例
    results = []
    for args in test_cases:
        name = args[0]
        args_list = args[1]
        expected_exit = args[2] if len(args) > 2 else False
        expected_output = args[3] if len(args) > 3 else None
        
        result = test_edge_case(name, args_list, expected_exit, expected_output)
        results.append((name, result))
    
    # 输出测试报告
    report_path = os.path.join(test_output_dir, 'edge_case_tests_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("命令行参数解析边界情况测试报告\n")
        f.write("=" * 80 + "\n\n")
        
        for name, result in results:
            status = "通过" if result else "失败"
            f.write(f"{name}: {status}\n")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        f.write(f"\n总测试数: {total}\n")
        f.write(f"通过: {passed}\n")
        f.write(f"失败: {total - passed}\n")
        f.write(f"通过率: {passed/total*100:.1f}%\n")
    
    print(f"\n测试报告已保存到: {report_path}")
    print(f"\n边界情况测试完成! 通过率: {passed/total*100:.1f}% ({passed}/{total})")
    
    # 返回成功码如果所有测试都通过
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main()) 