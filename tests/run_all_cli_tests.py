#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行所有CLI相关的测试脚本并生成综合报告
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def print_header(title):
    """打印格式化的标题"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_test(script_path, test_name):
    """运行指定的测试脚本并返回退出码"""
    print_header(f"运行测试: {test_name}")
    
    start_time = time.time()
    # 使用UTF-8编码避免中文编码问题
    process = subprocess.run([sys.executable, script_path], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True,
                            encoding='utf-8',
                            errors='replace')  # 使用'replace'处理无法解码的字符
    end_time = time.time()
    
    # 输出测试结果
    print(process.stdout)
    if process.stderr:
        print("错误输出:")
        print(process.stderr)
        
    duration = end_time - start_time
    status = "✅ 通过" if process.returncode == 0 else "❌ 失败"
    print(f"\n{status} - 用时: {duration:.2f}秒")
    
    return {
        "name": test_name,
        "status": status,
        "return_code": process.returncode,
        "duration": duration
    }

def save_report(results, output_dir):
    """将测试结果保存到报告文件中"""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"cli_tests_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("命令行参数解析测试综合报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 结果统计
        total = len(results)
        passed = sum(1 for r in results if r["return_code"] == 0)
        failed = total - passed
        
        f.write(f"总测试数: {total}\n")
        f.write(f"通过: {passed}\n")
        f.write(f"失败: {failed}\n")
        f.write(f"通过率: {passed/total*100:.1f}%\n\n")
        
        # 详细结果
        f.write("详细测试结果:\n")
        f.write("-" * 80 + "\n")
        for result in results:
            f.write(f"{result['status']} - {result['name']} (用时: {result['duration']:.2f}秒)\n")
    
    print(f"\n测试报告已保存到: {report_path}")
    return report_path

def main():
    """运行所有CLI测试脚本"""
    # 设置环境变量使用UTF-8编码
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # 创建测试输出目录
    output_dir = os.path.join('tests', 'outputs', 'cli_tests')
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试脚本列表
    test_scripts = [
        (os.path.join('tests', 'test_cli.py'), "单元测试 - 测试CLI参数解析"),
        (os.path.join('tests', 'cli_test_edge_cases.py'), "边界情况测试 - 测试错误处理和边缘情况"),
        (os.path.join('tests', 'cli_integration_test.py'), "集成测试 - 测试CLI与训练脚本集成"),
        (os.path.join('scripts', 'test_cli.py'), "基础功能测试 - 测试基本CLI功能"),
    ]
    
    # 运行所有测试
    print_header("开始运行所有CLI测试")
    
    print(f"总共 {len(test_scripts)} 个测试集")
    results = []
    
    for script_path, test_name in test_scripts:
        # 检查脚本是否存在
        if not os.path.exists(script_path):
            print(f"警告: 测试脚本不存在 - {script_path}")
            continue
            
        # 运行测试
        result = run_test(script_path, test_name)
        results.append(result)
    
    # 生成报告
    report_path = save_report(results, output_dir)
    
    # 输出最终结果
    passed = sum(1 for r in results if r["return_code"] == 0)
    total = len(results)
    
    print_header("测试结果总结")
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}% ({passed}/{total})")
    print(f"\n测试报告: {report_path}")
    
    # 返回成功码如果所有测试都通过
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main()) 