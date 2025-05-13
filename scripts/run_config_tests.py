#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行训练配置测试
"""

import os
import sys
import unittest
import logging

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tests():
    """运行训练配置测试"""
    # 导入测试模块
    import tests.test_train_config
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromModule(tests.test_train_config)
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # 输出测试结果摘要
    logger.info(f"运行了 {result.testsRun} 个测试")
    if result.wasSuccessful():
        logger.info("所有测试通过！✅")
    else:
        logger.error(f"测试失败: 错误 {len(result.errors)}，失败 {len(result.failures)} ❌")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 