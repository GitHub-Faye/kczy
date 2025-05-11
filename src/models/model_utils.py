"""
模型工具模块 - 提供用于模型保存、加载和转换的实用函数
"""

import os
import torch
from typing import Dict, Optional, Tuple, Union, Any, List, Callable
import json
import yaml
from pathlib import Path
import time
import warnings
import numpy as np

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
                  opset_version: int = 11, simplify: bool = False, 
                  target_providers: Optional[List[str]] = None,
                  verify: bool = True, optimize: bool = False) -> str:
    """
    将模型导出为ONNX格式
    
    参数:
        model (torch.nn.Module): 要导出的模型
        file_path (str): 输出文件路径
        input_shape (Optional[Tuple]): 输入形状
        dynamic_axes (Optional[Dict]): 动态轴信息
        export_params (bool): 是否导出参数
        opset_version (int): ONNX操作集版本
        simplify (bool): 是否简化ONNX模型（需要安装onnx-simplifier）
        target_providers (Optional[List[str]]): 目标推理提供商列表（如 ['CPUExecutionProvider', 'CUDAExecutionProvider']）
        verify (bool): 是否验证导出的模型正确性
        optimize (bool): 是否优化ONNX模型（针对性能）
        
    返回:
        str: 导出的ONNX文件路径
    """
    # 检查是否是VisionTransformer
    if isinstance(model, VisionTransformer):
        # 使用VisionTransformer的专用导出方法
        model.export_to_onnx(
            file_path=file_path,
            input_shape=input_shape,
            dynamic_axes=dynamic_axes,
            export_params=export_params,
            opset_version=opset_version,
            simplify=simplify,
            target_providers=target_providers,
            verify=verify,
            optimize=optimize
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
        
        # 记录原始输出用于验证
        original_output = None
        if verify:
            with torch.no_grad():
                original_output = model(dummy_input).cpu().numpy()
        
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
        
        # 简化模型（如果需要）
        if simplify:
            try:
                import onnxsim
                model_path = simplify_onnx_model(file_path)
                print(f"模型已简化: {model_path}")
            except ImportError:
                warnings.warn("未找到onnx-simplifier包，跳过模型简化步骤")
        
        # 优化模型（如果需要）
        if optimize:
            model_path = optimize_onnx_model(file_path, target_providers)
            print(f"模型已优化: {model_path}")
            
        # 验证模型（如果需要）
        if verify and original_output is not None:
            verify_onnx_model(file_path, dummy_input.cpu().numpy(), original_output, target_providers)
    
    return file_path


def simplify_onnx_model(file_path: str) -> str:
    """
    简化ONNX模型以提高性能
    
    参数:
        file_path (str): ONNX模型文件路径
        
    返回:
        str: 简化后的ONNX模型文件路径
    """
    try:
        import onnx
        from onnxsim import simplify
        
        # 加载ONNX模型
        onnx_model = onnx.load(file_path)
        
        # 简化模型
        simplified_model, check = simplify(onnx_model)
        
        if not check:
            warnings.warn("简化模型失败，可能存在不支持的操作或结构")
            return file_path
            
        # 保存简化后的模型
        onnx.save(simplified_model, file_path)
        return file_path
        
    except ImportError:
        warnings.warn("未找到onnx或onnx-simplifier包，无法简化模型")
        return file_path


def optimize_onnx_model(file_path: str, target_providers: Optional[List[str]] = None) -> str:
    """
    优化ONNX模型以提高特定目标运行时的性能
    
    参数:
        file_path (str): ONNX模型文件路径
        target_providers (Optional[List[str]]): 目标推理提供商列表
            例如: ['CPUExecutionProvider', 'CUDAExecutionProvider']
            
    返回:
        str: 优化后的ONNX模型文件路径
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        # 设置默认提供商
        if target_providers is None:
            target_providers = ['CPUExecutionProvider']
            
        # 优化模型文件名
        opt_file_path = file_path.replace('.onnx', '_optimized.onnx')
        
        # 根据不同的提供商应用不同的优化
        if 'CUDAExecutionProvider' in target_providers:
            # GPU优化
            opt_options = optimizer.OptimizationOptions()
            opt_options.enable_gelu_approximation = True  # 使用GELU近似
            opt_options.enable_attention_fusion = True    # 启用注意力融合
            opt_options.enable_layer_norm_fusion = True   # 启用层规范化融合
            
            # 创建优化器并优化模型
            opt_model = optimizer.optimize_model(
                file_path,
                'cuda',
                opt_options
            )
            
            # 保存优化后的模型
            opt_model.save_model_to_file(opt_file_path)
        else:
            # CPU优化
            opt_options = optimizer.OptimizationOptions()
            opt_options.enable_gelu_approximation = True
            
            # 创建优化器并优化模型
            opt_model = optimizer.optimize_model(
                file_path,
                'cpu',
                opt_options
            )
            
            # 保存优化后的模型
            opt_model.save_model_to_file(opt_file_path)
            
        return opt_file_path
    
    except ImportError:
        warnings.warn("未找到onnx或onnxruntime.transformers包，无法优化模型")
        return file_path
    except Exception as e:
        warnings.warn(f"优化模型时出错: {str(e)}")
        return file_path


def verify_onnx_model(file_path: str, input_data: np.ndarray, 
                     expected_output: np.ndarray, 
                     providers: Optional[List[str]] = None,
                     rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    """
    验证ONNX模型的输出与预期输出是否一致
    
    参数:
        file_path (str): ONNX模型文件路径
        input_data (np.ndarray): 输入数据
        expected_output (np.ndarray): 预期输出
        providers (Optional[List[str]]): ONNX Runtime执行提供商
        rtol (float): 相对容差
        atol (float): 绝对容差
        
    返回:
        bool: 验证是否通过
    """
    try:
        import onnxruntime as ort
        
        # 设置默认提供商
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        # 创建ONNX Runtime会话
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(file_path, sess_options=session_options, providers=providers)
        
        # 获取输入名
        input_name = session.get_inputs()[0].name
        
        # 运行推理
        onnx_outputs = session.run(None, {input_name: input_data})
        
        # 比较输出
        is_close = np.allclose(onnx_outputs[0], expected_output, rtol=rtol, atol=atol)
        
        if is_close:
            print("✓ ONNX模型验证通过: 输出与PyTorch模型一致")
        else:
            max_diff = np.max(np.abs(onnx_outputs[0] - expected_output))
            print(f"× ONNX模型验证失败: 输出与PyTorch模型不一致，最大差异: {max_diff}")
            
        return is_close
    
    except ImportError:
        warnings.warn("未找到onnxruntime包，无法验证模型")
        return False
    except Exception as e:
        warnings.warn(f"验证模型时出错: {str(e)}")
        return False


def load_onnx_model(file_path: str, providers: Optional[List[str]] = None) -> Any:
    """
    加载ONNX模型用于推理
    
    参数:
        file_path (str): ONNX模型文件路径
        providers (Optional[List[str]]): ONNX Runtime执行提供商
        
    返回:
        Any: ONNX Runtime推理会话
    """
    try:
        import onnxruntime as ort
        
        # 设置默认提供商
        if providers is None:
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                
        # 创建ONNX Runtime会话
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(file_path, sess_options=session_options, providers=providers)
        
        return session
    
    except ImportError:
        raise ImportError("未找到onnxruntime包，无法加载ONNX模型")
    except Exception as e:
        raise RuntimeError(f"加载ONNX模型时出错: {str(e)}")


def onnx_inference(session: Any, input_data: np.ndarray) -> np.ndarray:
    """
    使用加载的ONNX模型进行推理
    
    参数:
        session (Any): ONNX Runtime推理会话
        input_data (np.ndarray): 输入数据
        
    返回:
        np.ndarray: 模型输出
    """
    try:
        # 获取输入名
        input_name = session.get_inputs()[0].name
        
        # 运行推理
        outputs = session.run(None, {input_name: input_data})
        
        return outputs[0]
    
    except Exception as e:
        raise RuntimeError(f"ONNX推理时出错: {str(e)}")


def get_onnx_model_info(file_path: str) -> Dict[str, Any]:
    """
    获取ONNX模型的信息
    
    参数:
        file_path (str): ONNX模型文件路径
        
    返回:
        Dict[str, Any]: 模型信息字典
    """
    try:
        import onnx
        
        # 加载ONNX模型
        model = onnx.load(file_path)
        
        # 提取信息
        info = {
            'model_path': file_path,
            'ir_version': model.ir_version,
            'producer_name': model.producer_name,
            'producer_version': model.producer_version,
            'domain': model.domain,
            'model_version': model.model_version,
            'doc_string': model.doc_string,
        }
        
        # 提取输入信息
        inputs = []
        for input in model.graph.input:
            input_info = {
                'name': input.name,
                'shape': [dim.dim_value for dim in input.type.tensor_type.shape.dim 
                         if hasattr(dim, 'dim_value') and dim.dim_value]
            }
            inputs.append(input_info)
        info['inputs'] = inputs
        
        # 提取输出信息
        outputs = []
        for output in model.graph.output:
            output_info = {
                'name': output.name,
                'shape': [dim.dim_value for dim in output.type.tensor_type.shape.dim 
                         if hasattr(dim, 'dim_value') and dim.dim_value]
            }
            outputs.append(output_info)
        info['outputs'] = outputs
        
        # 提取节点信息
        info['node_count'] = len(model.graph.node)
        op_types = {}
        for node in model.graph.node:
            op_type = node.op_type
            op_types[op_type] = op_types.get(op_type, 0) + 1
        info['operation_types'] = op_types
        
        return info
    
    except ImportError:
        warnings.warn("未找到onnx包，无法获取模型信息")
        return {'model_path': file_path, 'error': 'onnx package not found'}
    except Exception as e:
        warnings.warn(f"获取模型信息时出错: {str(e)}")
        return {'model_path': file_path, 'error': str(e)}


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
        'optimizer_state_dict': optimizer_state,
        'saved_at_timestamp': time.time(),  # 添加时间戳
    }
    
    # 添加训练历史（如果提供）
    if train_history is not None:
        checkpoint['history'] = train_history
    
    # 添加元数据（如果提供）
    if metadata is not None:
        checkpoint['metadata'] = metadata
        
    # 添加模型类型
    checkpoint['model_type'] = model.__class__.__name__
    
    # 优化器信息（从optimizer_state提取）
    if optimizer_state and isinstance(optimizer_state, dict):
        # 记录优化器类型和配置（如果存在）
        if 'optimizer_type' in optimizer_state:
            checkpoint['optimizer_info'] = {
                'type': optimizer_state['optimizer_type']
            }
            
            # 记录学习率调度器信息（如果存在）
            if 'scheduler_type' in optimizer_state:
                checkpoint['optimizer_info']['scheduler'] = {
                    'type': optimizer_state['scheduler_type']
                }
    
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
    
    # 打印加载的检查点信息
    print(f"加载检查点: epoch {epoch}")
    
    # 如果存在优化器信息，打印
    if 'optimizer_info' in checkpoint:
        opt_info = checkpoint['optimizer_info']
        print(f"优化器类型: {opt_info.get('type', 'unknown')}")
        if 'scheduler' in opt_info:
            print(f"调度器类型: {opt_info['scheduler'].get('type', 'unknown')}")
    
    # 如果存在时间戳，打印保存时间
    if 'saved_at_timestamp' in checkpoint:
        saved_time = checkpoint['saved_at_timestamp']
        saved_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(saved_time))
        print(f"检查点保存时间: {saved_time_str}")
    
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