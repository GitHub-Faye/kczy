#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单元测试脚本，用于测试test_visualization_with_model.py的功能。
"""

import os
import sys
import unittest
import torch
import tempfile
import shutil
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入测试目标
import scripts.test_visualization_with_model as viz_script
from src.models.vit import VisionTransformer

class TestVisualizationScript(unittest.TestCase):
    """测试可视化脚本的功能"""
    
    def setUp(self):
        """在每个测试之前设置环境"""
        # 创建临时输出目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建一个简单的ArgumentParser模拟对象
        class Args:
            def __init__(self, temp_dir):
                self.model = None
                self.model_type = 'tiny'
                self.num_classes = 10
                self.image = None
                self.random_image = True
                self.sample_dir = 'data/images'
                self.output_dir = temp_dir  # 使用外部传入的temp_dir
                self.format = 'png'
                self.dpi = 72
                self.prefix = 'test'
                self.add_timestamp = False
                self.mode = 'all'
                self.no_html = True
                self.compare = False
                self.seed = 42
                self.verbose = False
        
        self.args = Args(self.temp_dir)  # 传入temp_dir
        
        # 设置随机种子
        viz_script.set_seed(self.args.seed)
        
        # 获取设备
        self.device = viz_script.get_device()
    
    def tearDown(self):
        """在每个测试之后清理环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_argument_parsing(self):
        """测试参数解析功能"""
        # 模拟命令行参数
        sys.argv = ['test_visualization_with_model.py', 
                    '--model_type', 'tiny', 
                    '--random_image',
                    '--output_dir', self.temp_dir,
                    '--dpi', '72',
                    '--prefix', 'unittest']
        
        # 解析参数
        args = viz_script.parse_args()
        
        # 验证参数
        self.assertEqual(args.model_type, 'tiny')
        self.assertTrue(args.random_image)
        self.assertEqual(args.output_dir, self.temp_dir)
        self.assertEqual(args.dpi, 72)
        self.assertEqual(args.prefix, 'unittest')
    
    def test_model_creation(self):
        """测试模型创建功能"""
        # 创建模型
        model = viz_script.load_or_create_model(self.args, self.device)
        
        # 验证模型类型和结构
        self.assertIsInstance(model, VisionTransformer)
        
        # 检查模型是否可以进行前向传播
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = model(dummy_input)
        
        # 验证输出形状与预期的类别数量匹配
        self.assertEqual(output.shape[0], 1)  # 批次大小为1
        self.assertEqual(output.shape[1], self.args.num_classes)  # 输出类别数量
    
    def test_image_loading(self):
        """测试图像加载功能"""
        # 创建模型
        model = viz_script.load_or_create_model(self.args, self.device)
        
        # 加载随机图像
        img_tensor, original_img = viz_script.get_input_image(self.args, model, self.device)
        
        # 验证图像张量
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape[0], 1)  # 批次大小为1
        self.assertEqual(img_tensor.shape[1], 3)  # 3个颜色通道
        
        # 由于使用了随机图像，原始图像应为None
        self.assertIsNone(original_img)
    
    def test_output_path_generation(self):
        """测试输出路径生成功能"""
        # 生成输出路径
        output_path = viz_script.get_output_path(self.args, "test_component")
        
        # 验证路径格式
        expected_path = os.path.join(self.temp_dir, "test_test_component.png")
        self.assertEqual(output_path, expected_path)
        
        # 测试添加时间戳
        self.args.add_timestamp = True
        output_path_with_timestamp = viz_script.get_output_path(self.args, "test_component")
        
        # 验证路径包含时间戳
        self.assertTrue("test_test_component_" in output_path_with_timestamp)
        self.assertTrue(".png" in output_path_with_timestamp)
    
    def test_attention_visualization(self):
        """测试注意力可视化功能"""
        # 创建模型
        model = viz_script.load_or_create_model(self.args, self.device)
        
        # 创建测试图像
        img_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        
        # 测试注意力可视化
        results = viz_script.test_attention_visualization(self.args, model, img_tensor, None)
        
        # 验证生成的文件
        self.assertTrue('attention_heatmap' in results)
        self.assertTrue('all_attention_heads' in results)
        
        # 检查文件是否存在
        for _, file_path in results.items():
            self.assertTrue(os.path.exists(file_path))
    
    def test_model_structure_visualization(self):
        """测试模型结构可视化功能"""
        # 创建模型
        model = viz_script.load_or_create_model(self.args, self.device)
        
        # 测试模型结构可视化
        results = viz_script.test_model_structure_visualization(self.args, model)
        
        # 验证生成的文件
        self.assertTrue('model_structure' in results)
        self.assertTrue('encoder_block' in results)
        self.assertTrue('layer_weights' in results)
        
        # 检查文件是否存在
        for _, file_path in results.items():
            self.assertTrue(os.path.exists(file_path))
    
    def test_static_visualization(self):
        """测试静态可视化功能"""
        # 创建模型
        model = viz_script.load_or_create_model(self.args, self.device)
        
        # 创建测试图像
        img_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        
        # 测试静态可视化
        results = viz_script.test_static_visualization(self.args, model, img_tensor)
        
        # 验证生成的文件
        self.assertTrue('model_overview' in results)
        self.assertTrue('attention_analysis' in results)
        
        # 检查文件是否存在
        for _, file_path in results.items():
            self.assertTrue(os.path.exists(file_path))
    
    def test_complete_pipeline(self):
        """测试完整的可视化流程"""
        # 修改参数
        self.args.mode = 'all'
        self.args.compare = True
        
        # 调用主函数
        try:
            # 替换sys.argv，避免干扰单元测试
            original_argv = sys.argv
            sys.argv = ['test_visualization_with_model.py']
            
            # 通过修改parse_args函数的返回值来使用我们的测试参数
            original_parse_args = viz_script.parse_args
            viz_script.parse_args = lambda: self.args
            
            # 执行主函数
            viz_script.main()
            
            # 检查输出目录中是否有文件生成
            files = os.listdir(self.temp_dir)
            self.assertTrue(len(files) > 0)
            
        finally:
            # 恢复原始函数和参数
            sys.argv = original_argv
            viz_script.parse_args = original_parse_args

if __name__ == '__main__':
    unittest.main() 