"""
模型工具模块 - 提供用于模型保存、加载和转换的实用函数
"""

import os
import torch
from typing import Dict, Optional, Tuple, Union, Any
import json
import yaml
from pathlib import Path

from src.models.vit import VisionTransformer
from src.utils.config import ViTConfig


def save_model(model: torch.nn.Module, file_path: str, optimizer_state: Optional[Dict] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    保存模型到指定路径
    
    参数:
        model (torch.nn.Module): 要保存的模型
        file_path (str): 保存路径
        optimizer_state (Optional[Dict]): 优化器状态字典
        metadata (Optional[Dict[str, Any]]): 额外的元数据
        
    返回:
        None
    """
    if isinstance(model, VisionTransformer):
        # 使用VisionTransformer专用的保存方法
        model.save_model(file_path, save_optimizer=optimizer_state is not None, 
                         optimizer_state=optimizer_state)
    else:
        # 通用模型保存逻辑
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 构建保存字典
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': model.__class__.__name__,
            'version': '1.0.0'  # 版本信息
        }
        
        # 添加优化器状态（如果提供）
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        # 添加元数据（如果提供）
        if metadata is not None:
            checkpoint['metadata'] = metadata
            
        # 保存模型
        torch.save(checkpoint, file_path)
        print(f"模型已保存到: {file_path}")


def load_model(file_path: str, model_class: Optional[Any] = None, 
               device: Optional[Union[torch.device, str]] = None) -> Tuple[torch.nn.Module, Optional[Dict]]:
    """
    从文件加载模型
    
    参数:
        file_path (str): 模型文件路径
        model_class (Optional[Any]): 模型类（如果不是VisionTransformer）
        device (Optional[Union[torch.device, str]]): 设备（'cuda'、'cpu'或torch.device对象）
        
    返回:
        Tuple[torch.nn.Module, Optional[Dict]]: 模型和优化器状态（如果存在）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"模型文件不存在: {file_path}")
    
    # 确定设备
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    checkpoint = torch.load(file_path, map_location=device)
    
    # 检查是否是VisionTransformer模型
    model_type = checkpoint.get('model_type', '')
    if model_type == 'VisionTransformer':
        # 使用VisionTransformer的专用加载方法
        return VisionTransformer.load_model(file_path, device)
    
    # 通用模型加载逻辑
    if model_class is None:
        raise ValueError("对于非VisionTransformer模型，必须提供model_class参数")
    
    # 创建模型实例
    model = model_class()
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将模型移到指定设备
    model.to(device)
    
    # 获取优化器状态（如果存在）
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    
    return model, optimizer_state
    

def export_to_onnx(model: torch.nn.Module, file_path: str, input_shape: Optional[Tuple] = None,
                  dynamic_axes: Optional[Dict] = None, export_params: bool = True,
                  opset_version: int = 11) -> None:
    """
    将模型导出为ONNX格式
    
    参数:
        model (torch.nn.Module): 要导出的模型
        file_path (str): 输出文件路径
        input_shape (Optional[Tuple]): 输入形状
        dynamic_axes (Optional[Dict]): 动态轴信息
        export_params (bool): 是否导出参数
        opset_version (int): ONNX操作集版本
        
    返回:
        None
    """
    # 检查是否是VisionTransformer
    if isinstance(model, VisionTransformer):
        # 使用VisionTransformer的专用导出方法
        model.export_to_onnx(
            file_path=file_path,
            input_shape=input_shape,
            dynamic_axes=dynamic_axes,
            export_params=export_params,
            opset_version=opset_version
        )
    else:
        # 通用ONNX导出逻辑
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 设置模型为评估模式
        model.eval()
        
        # 确定输入形状
        if input_shape is None:
            raise ValueError("对于非VisionTransformer模型，必须提供input_shape参数")
            
        # 默认动态轴 - 使批次大小可变
        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            
        # 创建示例输入
        dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
        
        # 导出为ONNX
        torch.onnx.export(
            model,                        # 要导出的模型
            dummy_input,                  # 示例输入
            file_path,                    # 输出文件
            export_params=export_params,  # 存储训练过的参数权重
            opset_version=opset_version,  # ONNX版本
            input_names=['input'],        # 输入名称
            output_names=['output'],      # 输出名称
            dynamic_axes=dynamic_axes,    # 动态轴
            verbose=False
        )
            
        print(f"模型已导出为ONNX格式: {file_path}")


def save_checkpoint(model: torch.nn.Module, optimizer_state: Dict, file_path: str, 
                   epoch: int, train_history: Optional[Dict] = None, 
                   metadata: Optional[Dict] = None) -> None:
    """
    保存训练检查点
    
    参数:
        model (torch.nn.Module): 模型
        optimizer_state (Dict): 优化器状态
        file_path (str): 保存路径
        epoch (int): 当前训练轮次
        train_history (Optional[Dict]): 训练历史记录
        metadata (Optional[Dict]): 额外元数据
    
    返回:
        None
    """
    # 创建检查点目录
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # 构建检查点字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state
    }
    
    # 添加训练历史（如果提供）
    if train_history is not None:
        checkpoint['history'] = train_history
    
    # 添加元数据（如果提供）
    if metadata is not None:
        checkpoint['metadata'] = metadata
        
    # 添加模型类型
    checkpoint['model_type'] = model.__class__.__name__
    
    # 添加配置（如果是VisionTransformer）
    if isinstance(model, VisionTransformer):
        checkpoint['config'] = model.get_config().to_dict()
    
    # 保存检查点
    torch.save(checkpoint, file_path)
    print(f"检查点已保存到: {file_path}")
    
    
def load_checkpoint(file_path: str, model: torch.nn.Module, 
                   device: Optional[Union[torch.device, str]] = None) -> Tuple[torch.nn.Module, Dict, int, Optional[Dict]]:
    """
    加载训练检查点
    
    参数:
        file_path (str): 检查点文件路径
        model (torch.nn.Module): 要加载权重的模型
        device (Optional[Union[torch.device, str]]): 设备
        
    返回:
        Tuple[torch.nn.Module, Dict, int, Optional[Dict]]: 模型、优化器状态、轮次和训练历史
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"检查点文件不存在: {file_path}")
    
    # 确定设备
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # 加载检查点
    checkpoint = torch.load(file_path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将模型移到指定设备
    model.to(device)
    
    # 获取优化器状态
    optimizer_state = checkpoint['optimizer_state_dict']
    
    # 获取轮次
    epoch = checkpoint.get('epoch', 0)
    
    # 获取训练历史
    history = checkpoint.get('history', None)
    
    return model, optimizer_state, epoch, history


def get_model_info(file_path: str) -> Dict[str, Any]:
    """
    获取保存的模型文件信息
    
    参数:
        file_path (str): 模型文件路径
        
    返回:
        Dict[str, Any]: 模型信息
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"模型文件不存在: {file_path}")
    
    # 加载检查点
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    
    # 提取信息
    info = {
        'model_type': checkpoint.get('model_type', 'Unknown'),
        'version': checkpoint.get('version', 'Unknown'),
        'has_optimizer': 'optimizer_state_dict' in checkpoint,
        'has_history': 'history' in checkpoint,
        'epoch': checkpoint.get('epoch', None)
    }
    
    # 添加配置信息（如果存在）
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            # 对于ViT模型，添加关键参数
            info.update({
                'img_size': config.get('img_size', None),
                'patch_size': config.get('patch_size', None),
                'in_channels': config.get('in_channels', None),
                'num_classes': config.get('num_classes', None),
                'embed_dim': config.get('embed_dim', None),
                'depth': config.get('depth', None),
                'num_heads': config.get('num_heads', None),
            })
    
    # 添加元数据（如果存在）
    if 'metadata' in checkpoint:
        info['metadata'] = checkpoint['metadata']
    
    return info 