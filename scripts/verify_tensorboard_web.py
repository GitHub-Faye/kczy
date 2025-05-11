#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorBoard Web界面验证脚本
运行此脚本以验证TensorBoard Web界面功能
"""

import os
import sys
import logging
import unittest
import argparse

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入测试模块
sys.path.insert(0, os.path.join(project_root, 'tests'))
from test_tensorboard_web_interface import TestTensorBoardWebInterface

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TensorBoard Web界面验证工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="增加输出详细程度"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="仅运行测试，不生成测试报告"
    )
    
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./reports",
        help="测试报告输出目录"
    )
    
    return parser.parse_args()

def run_tests(args):
    """运行TensorBoard Web界面测试"""
    logger.info("开始验证TensorBoard Web界面...")
    
    # 设置测试运行器
    if args.verbose > 0:
        verbosity = args.verbose + 1  # unittest使用1作为基本详细级别
    else:
        verbosity = 1
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTensorBoardWebInterface)
    
    # 运行测试
    test_result = unittest.TextTestRunner(verbosity=verbosity).run(test_suite)
    
    # 生成测试报告
    if not args.test_only:
        try:
            import xmlrunner
            
            # 确保报告目录存在
            os.makedirs(args.report_dir, exist_ok=True)
            
            # 生成XML测试报告
            xml_report_file = os.path.join(args.report_dir, "tensorboard_web_test_results.xml")
            with open(xml_report_file, 'wb') as output:
                unittest.main(
                    testRunner=xmlrunner.XMLTestRunner(output=output),
                    testLoader=unittest.TestLoader().loadTestsFromTestCase(TestTensorBoardWebInterface),
                    exit=False
                )
            logger.info(f"XML测试报告已保存至: {xml_report_file}")
            
            # 生成HTML测试报告
            try:
                import HtmlTestRunner
                
                html_report_file = os.path.join(args.report_dir, "tensorboard_web_test_results.html")
                HtmlTestRunner.HTMLTestRunner(
                    output=args.report_dir,
                    report_name="tensorboard_web_test_results",
                    combine_reports=True,
                    report_title="TensorBoard Web界面验证测试报告"
                ).run(test_suite)
                logger.info(f"HTML测试报告已保存至: {html_report_file}")
            except ImportError:
                logger.warning("未找到HtmlTestRunner包，跳过HTML报告生成")
        
        except ImportError:
            logger.warning("未找到xmlrunner包，跳过XML报告生成")
    
    # 返回测试结果
    return test_result.wasSuccessful()

def main():
    """主函数"""
    args = parse_args()
    
    success = run_tests(args)
    
    if success:
        logger.info("TensorBoard Web界面验证成功！")
        print("\n" + "="*80)
        print("验证结果: 成功 ✓")
        print("TensorBoard Web界面可以正确显示所有类型的训练指标。")
        print("查看TensorBoard使用指南: docs/tensorboard_usage_guide.md")
        print("="*80 + "\n")
        return 0
    else:
        logger.error("TensorBoard Web界面验证失败！")
        print("\n" + "="*80)
        print("验证结果: 失败 ✗")
        print("TensorBoard Web界面测试未通过。请检查测试日志获取详细信息。")
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 