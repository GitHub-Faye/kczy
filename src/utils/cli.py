import argparse
import os
import sys
import json
import yaml
from typing import Dict, Any, Optional

def create_parser() -> argparse.ArgumentParser:
    """
    创建并配置命令行参数解析器
    
    返回:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="ViT模型分类器训练系统 - 用于训练视觉转换器模型并进行可视化分析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 通用参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="启用详细输出")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], 
                        default="auto", help="计算设备")

    # 数据集参数组
    dataset_group = parser.add_argument_group("数据集参数")
    dataset_group.add_argument("--data-dir", type=str, help="数据目录路径")
    dataset_group.add_argument("--anno-file", type=str, help="标注文件路径")
    dataset_group.add_argument("--img-size", type=int, default=224, help="图像尺寸")
    dataset_group.add_argument("--batch-size", type=int, default=32, help="批量大小")
    dataset_group.add_argument("--val-split", type=float, default=0.2, help="验证集划分比例")
    dataset_group.add_argument("--num-workers", type=int, default=4, help="数据加载器工作线程数")
    dataset_group.add_argument("--pin-memory", action="store_true", help="使用锁页内存")

    # 模型参数组
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument("--model-type", type=str, choices=["tiny", "small", "base", "large", "huge"], 
                        default="base", help="ViT模型类型")
    model_group.add_argument("--pretrained", action="store_true", help="使用预训练权重")
    model_group.add_argument("--pretrained-path", type=str, help="预训练权重路径")
    model_group.add_argument("--num-classes", type=int, help="分类类别数量")
    model_group.add_argument("--patch-size", type=int, default=16, help="补丁大小")
    model_group.add_argument("--embed-dim", type=int, help="嵌入向量的维度")
    model_group.add_argument("--depth", type=int, help="Transformer编码器块的数量")
    model_group.add_argument("--num-heads", type=int, help="注意力头的数量")

    # 训练参数组
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument("--epochs", type=int, default=100, help="训练轮数")
    train_group.add_argument("--lr", type=float, default=1e-3, help="学习率")
    train_group.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    train_group.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], 
                        default="adam", help="优化器类型")
    train_group.add_argument("--scheduler", type=str, 
                        choices=["step", "multistep", "exponential", "cosine", "plateau", "none"], 
                        default="cosine", help="学习率调度器类型")
    train_group.add_argument("--grad-clip-value", type=float, help="梯度裁剪阈值")
    train_group.add_argument("--grad-clip-norm", type=float, help="梯度范数裁剪阈值")
    train_group.add_argument("--early-stopping", action="store_true", help="启用早停策略")
    train_group.add_argument("--patience", type=int, default=10, help="早停策略的耐心值")
    train_group.add_argument("--use-mixed-precision", action="store_true", help="使用混合精度训练")

    # 日志和检查点参数组
    log_group = parser.add_argument_group("日志和检查点参数")
    log_group.add_argument("--log-dir", type=str, default="./logs", help="日志目录")
    log_group.add_argument("--log-freq", type=int, default=10, help="日志记录频率（每多少个batch记录一次）")
    log_group.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="检查点保存目录")
    log_group.add_argument("--checkpoint-freq", type=int, default=1, help="检查点保存频率（每多少个epoch保存一次）")
    log_group.add_argument("--metrics-dir", type=str, default="./metrics", help="指标保存目录")
    log_group.add_argument("--metrics-format", type=str, choices=["csv", "json"], 
                           default="csv", help="指标保存格式")
    log_group.add_argument("--plot-metrics", action="store_true", help="训练结束后绘制指标曲线")
    log_group.add_argument("--experiment-name", type=str, help="实验名称")
    
    return parser

def load_config(config_path: str) -> Dict[str, Any]:
    """
    从配置文件加载参数
    
    参数:
        config_path: 配置文件路径
        
    返回:
        Dict[str, Any]: 配置参数字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    _, ext = os.path.splitext(config_path)
    if ext.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif ext.lower() in ['.yml', '.yaml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}, 请使用 .json, .yml 或 .yaml")

def parse_args(args=None) -> argparse.Namespace:
    """
    解析命令行参数
    
    参数:
        args: 命令行参数列表，默认为None（使用sys.argv）
        
    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # 处理配置文件
    if parsed_args.config:
        config_dict = load_config(parsed_args.config)
        # 使用配置文件中的值作为默认值，但命令行参数优先
        for key, value in config_dict.items():
            # 将划线替换为下划线以匹配argparse命名风格
            key = key.replace('-', '_')
            if not hasattr(parsed_args, key) or getattr(parsed_args, key) is None:
                setattr(parsed_args, key, value)
    
    # 如果设备设置为auto，自动选择合适的设备
    if parsed_args.device == 'auto':
        import torch
        parsed_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return parsed_args

def print_args_info(args: argparse.Namespace) -> None:
    """
    打印参数信息，按组分类显示
    
    参数:
        args: 解析后的参数
    """
    # 转换为字典以便处理
    args_dict = vars(args)
    
    # 参数组分类
    general_args = [
        'config', 'output_dir', 'seed', 'verbose', 'device'
    ]
    
    dataset_args = [
        'data_dir', 'anno_file', 'img_size', 'batch_size', 'val_split',
        'num_workers', 'pin_memory'
    ]
    
    model_args = [
        'model_type', 'pretrained', 'pretrained_path', 'num_classes',
        'patch_size', 'embed_dim', 'depth', 'num_heads'
    ]
    
    train_args = [
        'epochs', 'lr', 'weight_decay', 'optimizer', 'scheduler',
        'grad_clip_value', 'grad_clip_norm', 'early_stopping', 'patience',
        'use_mixed_precision'
    ]
    
    log_args = [
        'log_dir', 'log_freq', 'checkpoint_dir', 'checkpoint_freq',
        'metrics_dir', 'metrics_format', 'plot_metrics', 'experiment_name'
    ]
    
    # 打印函数
    def print_section(title, arg_list):
        print(f"\n{title}")
        print("=" * len(title))
        for arg in arg_list:
            if arg in args_dict:
                value = args_dict[arg]
                # 不打印None值
                if value is not None:
                    print(f"{arg:20s}: {value}")
    
    # 打印各组参数
    print("\n参数信息:")
    print_section("通用参数", general_args)
    print_section("数据集参数", dataset_args)
    print_section("模型参数", model_args)
    print_section("训练参数", train_args)
    print_section("日志和检查点参数", log_args)
    
    # 打印设备信息
    if 'device' in args_dict:
        device = args_dict['device']
        if device == 'cuda':
            import torch
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            print(f"\n使用GPU: {device_name}")
        else:
            print(f"\n使用CPU进行计算")

def main():
    """CLI主函数示例"""
    args = parse_args()
    
    # 打印参数信息
    print_args_info(args)
    
    print("\n此脚本仅演示命令行参数解析功能")
    print("在实际应用中，您可以根据解析的参数调用相应的训练或评估函数")
    
if __name__ == "__main__":
    main() 