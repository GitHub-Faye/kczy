"""
ONNX导出演示脚本 - 展示如何将Vision Transformer模型导出为ONNX格式并进行推理
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    VisionTransformer, 
    export_to_onnx, 
    load_onnx_model, 
    onnx_inference,
    get_onnx_model_info
)

def main():
    # 创建输出目录
    output_dir = Path("models/onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Vision Transformer ONNX导出演示 ===")
    
    # 创建模型 - 使用小型配置便于快速演示
    print("\n1. 创建Vision Transformer模型")
    model = VisionTransformer.create_tiny(num_classes=10)
    print(f"模型创建成功: {model.__class__.__name__}")
    
    # 打印模型信息
    print(f"模型尺寸: 图像大小={model.config.img_size}px, 类别数={model.config.num_classes}")
    print(f"嵌入维度: {model.config.embed_dim}, Transformer深度: {model.config.depth}")
    
    # 设置输出路径
    output_path = output_dir / "vit_tiny.onnx"
    
    # 导出为ONNX
    print("\n2. 导出模型为ONNX格式")
    try:
        onnx_path = export_to_onnx(
            model=model,
            file_path=str(output_path),
            input_shape=(1, 3, model.config.img_size, model.config.img_size),
            verify=True,
            simplify=True
        )
        print(f"模型导出成功: {onnx_path}")
        
        # 获取ONNX模型信息
        print("\n3. ONNX模型信息")
        try:
            info = get_onnx_model_info(onnx_path)
            print(f"模型IR版本: {info.get('ir_version', 'N/A')}")
            print(f"节点数量: {info.get('node_count', 'N/A')}")
            
            # 打印操作类型统计
            if 'operation_types' in info:
                print("\n操作类型统计:")
                for op_type, count in info['operation_types'].items():
                    print(f"  {op_type}: {count}")
                
            # 打印输入/输出信息
            if 'inputs' in info and info['inputs']:
                input_info = info['inputs'][0]
                print(f"\n输入名称: {input_info['name']}")
                print(f"输入形状: {input_info.get('shape', 'N/A')}")
                
            if 'outputs' in info and info['outputs']:
                output_info = info['outputs'][0]
                print(f"\n输出名称: {output_info['name']}")
                print(f"输出形状: {output_info.get('shape', 'N/A')}")
        except Exception as e:
            print(f"获取模型信息失败: {str(e)}")
        
        # 加载ONNX模型并进行推理
        print("\n4. ONNX模型推理")
        try:
            # 创建随机输入
            dummy_input = np.random.randn(1, 3, model.config.img_size, model.config.img_size).astype(np.float32)
            
            # 使用PyTorch模型进行推理
            model.eval()
            with torch.no_grad():
                torch_input = torch.tensor(dummy_input)
                torch_output = model(torch_input).numpy()
            
            # 使用ONNX模型进行推理
            onnx_session = load_onnx_model(onnx_path)
            onnx_output = onnx_inference(onnx_session, dummy_input)
            
            # 比较输出
            max_diff = np.max(np.abs(torch_output - onnx_output))
            print(f"PyTorch vs ONNX最大差异: {max_diff}")
            
            # 输出是否匹配
            is_close = np.allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-5)
            if is_close:
                print("√ 验证成功: PyTorch和ONNX输出匹配")
            else:
                print("× 验证失败: PyTorch和ONNX输出不匹配")
                
            # 性能测试
            print("\n5. 性能比较")
            import time
            
            # PyTorch性能
            torch_times = []
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    _ = model(torch_input)
                torch_times.append(time.time() - start)
                
            # ONNX性能
            onnx_times = []
            for _ in range(10):
                start = time.time()
                _ = onnx_inference(onnx_session, dummy_input)
                onnx_times.append(time.time() - start)
                
            print(f"PyTorch平均推理时间: {sum(torch_times)/len(torch_times)*1000:.2f} ms")
            print(f"ONNX平均推理时间: {sum(onnx_times)/len(onnx_times)*1000:.2f} ms")
            
        except Exception as e:
            print(f"ONNX推理失败: {str(e)}")
            
    except Exception as e:
        print(f"导出失败: {str(e)}")
    
    print("\n演示完成!")

if __name__ == "__main__":
    main() 