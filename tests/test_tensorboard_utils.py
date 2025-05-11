#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorBoard工具模块的测试脚本
"""

import os
import sys
import unittest
import socket
import time

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.tensorboard_utils import (
    is_port_in_use,
    check_tensorboard_running,
    find_tensorboard_executable,
    start_tensorboard,
    stop_tensorboard
)

class TestTensorBoardUtils(unittest.TestCase):
    """TensorBoard工具函数测试类"""
    
    def test_is_port_in_use(self):
        """测试端口使用检查函数"""
        # 创建一个临时的socket服务器
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))  # 使用随机可用端口
        sock.listen(1)
        
        port = sock.getsockname()[1]
        
        # 端口应该被使用
        self.assertTrue(is_port_in_use(port, 'localhost'))
        
        # 关闭socket
        sock.close()
        
        # 等待端口释放
        time.sleep(0.1)
        
        # 端口应该不再被使用
        self.assertFalse(is_port_in_use(port, 'localhost'))
    
    def test_find_tensorboard_executable(self):
        """测试查找TensorBoard可执行文件函数"""
        try:
            tensorboard_cmd = find_tensorboard_executable()
            self.assertIsNotNone(tensorboard_cmd)
            
            # 应该是字符串
            self.assertIsInstance(tensorboard_cmd, str)
            
            # 字符串不应该为空
            self.assertTrue(len(tensorboard_cmd) > 0)
        except FileNotFoundError:
            # 如果找不到TensorBoard，则跳过测试
            self.skipTest("找不到TensorBoard可执行文件，跳过测试")
    
    def test_check_tensorboard_running(self):
        """测试检查TensorBoard运行状态函数"""
        # 创建一个临时的socket服务器模拟TensorBoard服务器
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))  # 使用随机可用端口
        sock.listen(1)
        
        port = sock.getsockname()[1]
        
        # 应该检测到服务运行
        self.assertTrue(check_tensorboard_running(port, 'localhost'))
        
        # 关闭socket
        sock.close()
        
        # 等待端口释放
        time.sleep(0.1)
        
        # 应该检测到服务未运行
        self.assertFalse(check_tensorboard_running(port, 'localhost'))
    
    def test_tensorboard_start_stop_background(self):
        """测试在后台启动和停止TensorBoard"""
        # 使用一个高端口号，避免冲突
        test_port = 19876
        # 使用一个临时目录作为日志目录
        test_dir = "test_tb_log_dir"
        
        # 确保测试目录存在
        os.makedirs(test_dir, exist_ok=True)
        
        # 确保端口没有被使用
        if is_port_in_use(test_port):
            self.skipTest(f"端口 {test_port} 已被使用，无法进行测试")
        
        try:
            # 尝试在后台启动TensorBoard
            try:
                process = start_tensorboard(
                    log_dir=test_dir,
                    port=test_port,
                    background=True,
                    timeout=20  # 增加等待启动的超时时间
                )
                
                # 如果启动失败，则跳过测试
                if not process:
                    self.skipTest("无法启动TensorBoard，可能是它不可用")
                
                # 验证TensorBoard正在运行
                self.assertTrue(check_tensorboard_running(test_port))
            except FileNotFoundError:
                self.skipTest("找不到TensorBoard可执行文件，跳过测试")
            
            # 尝试停止TensorBoard
            success = stop_tensorboard(process)
            self.assertTrue(success)
            
            # 给进程一些时间来终止
            time.sleep(1)
            
            # 验证TensorBoard已停止
            self.assertFalse(check_tensorboard_running(test_port))
        finally:
            # 清理临时目录
            import shutil
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main() 