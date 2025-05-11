"""
指标绘图模块，用于可视化训练和评估指标。
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import logging
import datetime
import json
from typing import List, Optional, Union, Dict, Any, Tuple

from src.utils.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)

def save_plot(
    fig: plt.Figure,
    output_path: str,
    dpi: int = 300,
    save_format: str = 'png',
    add_timestamp: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    保存matplotlib图表到文件，带有额外功能如时间戳和元数据。

    参数:
        fig (plt.Figure): 要保存的matplotlib图表对象
        output_path (str): 输出图表的路径
        dpi (int): 图表DPI，默认为300
        save_format (str): 保存格式，默认为'png'
        add_timestamp (bool): 是否在文件名中添加时间戳，默认为False
        metadata (Optional[Dict[str, Any]]): 要添加到图表文件的元数据字典

    返回:
        str: 实际保存的文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理时间戳
    if add_timestamp:
        # 获取当前时间并格式化为字符串
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 分离文件名和扩展名
        base_path, ext = os.path.splitext(output_path)
        if not ext and save_format:
            ext = f".{save_format}"
        
        # 添加时间戳到文件名
        output_path = f"{base_path}_{timestamp}{ext}"
    
    # 保存图表
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=save_format)
    
    # 如果提供了元数据，保存为伴随的JSON文件
    if metadata:
        metadata_path = f"{os.path.splitext(output_path)[0]}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"图表元数据保存至: {metadata_path}")
    
    logger.info(f"图表保存至: {output_path}")
    return output_path

def plot_loss(
    metrics_path: Optional[str] = None, 
    output_path: Optional[str] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    experiment_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_format: str = 'png',
    dpi: int = 300,
    show_grid: bool = True,
    style: Optional[str] = 'seaborn-v0_8-darkgrid',
    add_timestamp: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    绘制训练过程中的损失曲线。

    参数:
        metrics_path (Optional[str]): 指标文件路径，如果为None则使用metrics_logger。
        output_path (Optional[str]): 输出图表的路径，如果为None则显示但不保存。
        metrics_logger (Optional[MetricsLogger]): 指标记录器实例，如果metrics_path为None，则使用此参数。
        experiment_name (Optional[str]): 实验名称，用于图表标题，如果为None则使用默认标题。
        figsize (Tuple[int, int]): 图表尺寸，默认为(10, 6)。
        title (Optional[str]): 图表标题，如果为None则使用默认标题。
        save_format (str): 保存格式，默认为'png'。
        dpi (int): 图表DPI，默认为300。
        show_grid (bool): 是否显示网格，默认为True。
        style (Optional[str]): Matplotlib样式，默认为'seaborn-v0_8-darkgrid'。
        add_timestamp (bool): 是否在保存的文件名中添加时间戳，默认为False。
        metadata (Optional[Dict[str, Any]]): 要添加到图表文件的元数据，如训练参数。

    返回:
        Optional[str]: 如果保存了图表，返回实际的保存路径；否则返回None
    """
    # 设置绘图样式
    if style:
        plt.style.use(style)
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    
    # 从指标文件或指标记录器加载数据
    train_loss_data = None
    val_loss_data = None
    
    if metrics_logger:
        # 使用提供的MetricsLogger实例
        logger.info("从MetricsLogger实例加载损失数据")
        if 'loss' in metrics_logger.train_metrics:
            train_loss_data = {
                'epochs': [entry['epoch'] for entry in metrics_logger.train_metrics['loss']],
                'values': [entry['value'] for entry in metrics_logger.train_metrics['loss']]
            }
        
        if 'val_loss' in metrics_logger.eval_metrics:
            val_loss_data = {
                'epochs': [entry['epoch'] for entry in metrics_logger.eval_metrics['val_loss']],
                'values': [entry['value'] for entry in metrics_logger.eval_metrics['val_loss']]
            }
    
    elif metrics_path:
        # 从指标文件加载
        logger.info(f"从文件加载损失数据: {metrics_path}")
        if not os.path.exists(metrics_path):
            logger.error(f"指标文件不存在: {metrics_path}")
            return None

        try:
            # 确定文件类型并加载数据
            if metrics_path.endswith('.csv'):
                df = pd.read_csv(metrics_path)
                
                # 提取训练损失数据
                train_loss_df = df[df['metric'] == 'loss']
                if not train_loss_df.empty:
                    train_loss_data = {
                        'epochs': train_loss_df['epoch'].tolist(),
                        'values': train_loss_df['value'].tolist()
                    }
                
                # 提取验证损失数据
                val_loss_df = df[df['metric'] == 'val_loss']
                if not val_loss_df.empty:
                    val_loss_data = {
                        'epochs': val_loss_df['epoch'].tolist(),
                        'values': val_loss_df['value'].tolist()
                    }
            
            elif metrics_path.endswith('.json'):
                # 创建临时MetricsLogger来加载JSON数据
                temp_logger = MetricsLogger(save_dir=os.path.dirname(metrics_path),
                                          experiment_name=os.path.basename(metrics_path).split('_')[0],
                                          save_format='json')
                if 'loss' in temp_logger.train_metrics:
                    train_loss_data = {
                        'epochs': [entry['epoch'] for entry in temp_logger.train_metrics['loss']],
                        'values': [entry['value'] for entry in temp_logger.train_metrics['loss']]
                    }
                
                if 'val_loss' in temp_logger.eval_metrics:
                    val_loss_data = {
                        'epochs': [entry['epoch'] for entry in temp_logger.eval_metrics['val_loss']],
                        'values': [entry['value'] for entry in temp_logger.eval_metrics['val_loss']]
                    }
            else:
                logger.error(f"不支持的文件格式: {metrics_path}")
                return None
        except Exception as e:
            logger.error(f"加载指标数据时出错: {e}")
            return None
    else:
        logger.error("必须提供metrics_path或metrics_logger参数之一")
        return None
    
    # 绘制训练损失曲线
    if train_loss_data:
        plt.plot(train_loss_data['epochs'], train_loss_data['values'], 'b-', label='Training Loss')
    else:
        logger.warning("未找到训练损失数据")
    
    # 绘制验证损失曲线
    if val_loss_data:
        plt.plot(val_loss_data['epochs'], val_loss_data['values'], 'r-', label='Validation Loss')
    
    # 设置图表标题和标签
    if title:
        plt.title(title)
    elif experiment_name:
        plt.title(f'Training and Validation Loss - {experiment_name}')
    else:
        plt.title('Training and Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if show_grid:
        plt.grid(True)
    
    # 保存或显示图表
    if output_path:
        # 使用通用的保存函数
        return save_plot(
            fig=fig,
            output_path=output_path,
            dpi=dpi,
            save_format=save_format,
            add_timestamp=add_timestamp,
            metadata=metadata
        )
    else:
        plt.show()
        return None
    
    plt.close()

def plot_accuracy(
    metrics_path: Optional[str] = None, 
    output_path: Optional[str] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    experiment_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_format: str = 'png',
    dpi: int = 300,
    show_grid: bool = True,
    style: Optional[str] = 'seaborn-v0_8-darkgrid',
    y_lim: Optional[Tuple[float, float]] = None,
    use_percentage: bool = True,
    add_timestamp: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    绘制训练过程中的准确率曲线。

    参数:
        metrics_path (Optional[str]): 指标文件路径，如果为None则使用metrics_logger。
        output_path (Optional[str]): 输出图表的路径，如果为None则显示但不保存。
        metrics_logger (Optional[MetricsLogger]): 指标记录器实例，如果metrics_path为None，则使用此参数。
        experiment_name (Optional[str]): 实验名称，用于图表标题，如果为None则使用默认标题。
        figsize (Tuple[int, int]): 图表尺寸，默认为(10, 6)。
        title (Optional[str]): 图表标题，如果为None则使用默认标题。
        save_format (str): 保存格式，默认为'png'。
        dpi (int): 图表DPI，默认为300。
        show_grid (bool): 是否显示网格，默认为True。
        style (Optional[str]): Matplotlib样式，默认为'seaborn-v0_8-darkgrid'。
        y_lim (Optional[Tuple[float, float]]): Y轴范围，如(0, 100)显示0-100%的准确率范围。
        use_percentage (bool): 是否将准确率值显示为百分比，默认为True。
        add_timestamp (bool): 是否在保存的文件名中添加时间戳，默认为False。
        metadata (Optional[Dict[str, Any]]): 要添加到图表文件的元数据，如训练参数。

    返回:
        Optional[str]: 如果保存了图表，返回实际的保存路径；否则返回None
    """
    # 设置绘图样式
    if style:
        plt.style.use(style)
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    
    # 从指标文件或指标记录器加载数据
    train_accuracy_data = None
    val_accuracy_data = None
    
    if metrics_logger:
        # 使用提供的MetricsLogger实例
        logger.info("从MetricsLogger实例加载准确率数据")
        if 'accuracy' in metrics_logger.train_metrics:
            train_accuracy_data = {
                'epochs': [entry['epoch'] for entry in metrics_logger.train_metrics['accuracy']],
                'values': [entry['value'] for entry in metrics_logger.train_metrics['accuracy']]
            }
        
        if 'val_accuracy' in metrics_logger.eval_metrics:
            val_accuracy_data = {
                'epochs': [entry['epoch'] for entry in metrics_logger.eval_metrics['val_accuracy']],
                'values': [entry['value'] for entry in metrics_logger.eval_metrics['val_accuracy']]
            }
    
    elif metrics_path:
        # 从指标文件加载
        logger.info(f"从文件加载准确率数据: {metrics_path}")
        if not os.path.exists(metrics_path):
            logger.error(f"指标文件不存在: {metrics_path}")
            return None

        try:
            # 确定文件类型并加载数据
            if metrics_path.endswith('.csv'):
                df = pd.read_csv(metrics_path)
                
                # 提取训练准确率数据
                train_accuracy_df = df[df['metric'] == 'accuracy']
                if not train_accuracy_df.empty:
                    train_accuracy_data = {
                        'epochs': train_accuracy_df['epoch'].tolist(),
                        'values': train_accuracy_df['value'].tolist()
                    }
                
                # 提取验证准确率数据
                val_accuracy_df = df[df['metric'] == 'val_accuracy']
                if not val_accuracy_df.empty:
                    val_accuracy_data = {
                        'epochs': val_accuracy_df['epoch'].tolist(),
                        'values': val_accuracy_df['value'].tolist()
                    }
            
            elif metrics_path.endswith('.json'):
                # 创建临时MetricsLogger来加载JSON数据
                temp_logger = MetricsLogger(save_dir=os.path.dirname(metrics_path),
                                          experiment_name=os.path.basename(metrics_path).split('_')[0],
                                          save_format='json')
                if 'accuracy' in temp_logger.train_metrics:
                    train_accuracy_data = {
                        'epochs': [entry['epoch'] for entry in temp_logger.train_metrics['accuracy']],
                        'values': [entry['value'] for entry in temp_logger.train_metrics['accuracy']]
                    }
                
                if 'val_accuracy' in temp_logger.eval_metrics:
                    val_accuracy_data = {
                        'epochs': [entry['epoch'] for entry in temp_logger.eval_metrics['val_accuracy']],
                        'values': [entry['value'] for entry in temp_logger.eval_metrics['val_accuracy']]
                    }
            else:
                logger.error(f"不支持的文件格式: {metrics_path}")
                return None
        except Exception as e:
            logger.error(f"加载指标数据时出错: {e}")
            return None
    else:
        logger.error("必须提供metrics_path或metrics_logger参数之一")
        return None
    
    # 绘制训练准确率曲线
    if train_accuracy_data:
        plt.plot(train_accuracy_data['epochs'], train_accuracy_data['values'], 'b-', label='Training Accuracy')
    else:
        logger.warning("未找到训练准确率数据")
    
    # 绘制验证准确率曲线
    if val_accuracy_data:
        plt.plot(val_accuracy_data['epochs'], val_accuracy_data['values'], 'r-', label='Validation Accuracy')
    
    # 设置图表标题和标签
    if title:
        plt.title(title)
    elif experiment_name:
        plt.title(f'Training and Validation Accuracy - {experiment_name}')
    else:
        plt.title('Training and Validation Accuracy')
    
    plt.xlabel('Epoch')
    
    # 设置Y轴标签，根据use_percentage设置
    if use_percentage:
        plt.ylabel('Accuracy (%)')
    else:
        plt.ylabel('Accuracy')
    
    # 设置Y轴范围
    if y_lim:
        plt.ylim(y_lim)
    
    plt.legend()
    
    if show_grid:
        plt.grid(True)
    
    # 保存或显示图表
    if output_path:
        # 使用通用的保存函数
        return save_plot(
            fig=fig,
            output_path=output_path,
            dpi=dpi,
            save_format=save_format,
            add_timestamp=add_timestamp,
            metadata=metadata
        )
    else:
        plt.show()
        return None
    
    plt.close()

def plot_training_history(
    metrics_path: Optional[str] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    metrics: List[str] = None,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_format: str = 'png',
    dpi: int = 300,
    show_grid: bool = True,
    style: Optional[str] = 'seaborn-v0_8-darkgrid',
    add_timestamp: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    绘制多种训练指标的历史曲线，包含损失和准确率等。

    参数:
        metrics_path (Optional[str]): 指标文件路径。
        metrics_logger (Optional[MetricsLogger]): 指标记录器实例，如果metrics_path为None，则使用此参数。
        metrics (List[str]): 要绘制的指标名称列表，如['loss', 'accuracy', 'learning_rate']。
        output_dir (Optional[str]): 输出目录路径，如果为None则显示但不保存。
        experiment_name (Optional[str]): 实验名称，用于图表标题。
        figsize (Tuple[int, int]): 图表尺寸，默认为(10, 6)。
        save_format (str): 保存格式，默认为'png'。
        dpi (int): 图表DPI，默认为300。
        show_grid (bool): 是否显示网格，默认为True。
        style (Optional[str]): Matplotlib样式，默认为'seaborn-v0_8-darkgrid'。
        add_timestamp (bool): 是否在保存的文件名中添加时间戳，默认为False。
        metadata (Optional[Dict[str, Any]]): 要添加到图表文件的元数据，如训练参数。

    返回:
        Dict[str, str]: 保存的文件路径字典，键为指标名称，值为文件路径
    """
    # 如果未指定指标，则默认绘制loss和accuracy
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    # 尝试加载数据
    logger_to_use = None
    
    if metrics_logger:
        # 使用提供的MetricsLogger实例
        logger_to_use = metrics_logger
    elif metrics_path:
        # 从指标文件加载
        logger.info(f"从文件加载数据: {metrics_path}")
        if not os.path.exists(metrics_path):
            logger.error(f"指标文件不存在: {metrics_path}")
            return {}

        try:
            # 确定文件类型并加载数据
            if metrics_path.endswith('.csv'):
                # 创建临时MetricsLogger从CSV文件加载
                logger_to_use = MetricsLogger(
                    save_dir=os.path.dirname(metrics_path),
                    experiment_name=experiment_name or 'temp_experiment',
                    save_format='csv'
                )
                # 手动加载CSV文件
                if os.path.exists(metrics_path):
                    try:
                        df = pd.read_csv(metrics_path)
                        # 处理训练指标
                        for metric in df['metric'].unique():
                            metric_df = df[df['metric'] == metric]
                            for _, row in metric_df.iterrows():
                                if 'val_' in metric:
                                    logger_to_use.log_eval_metric(
                                        metric_name=metric.replace('val_', ''),
                                        value=row['value'],
                                        epoch=row['epoch']
                                    )
                                else:
                                    logger_to_use.log_train_metric(
                                        metric_name=metric,
                                        value=row['value'],
                                        epoch=row['epoch']
                                    )
                    except Exception as e:
                        logger.error(f"加载CSV文件时出错: {e}")
                        
            elif metrics_path.endswith('.json'):
                # 创建临时MetricsLogger从JSON文件加载
                logger_to_use = MetricsLogger(
                    save_dir=os.path.dirname(metrics_path),
                    experiment_name=experiment_name or os.path.basename(metrics_path).split('_')[0],
                    save_format='json'
                )
                # 如果需要，这里可以添加手动加载JSON文件的代码
            else:
                logger.error(f"不支持的文件格式: {metrics_path}")
                return {}
        except Exception as e:
            logger.error(f"加载指标数据时出错: {e}")
            return {}
    
    # 使用临时记录器或提供的记录器
    if not logger_to_use:
        logger.error("必须提供有效的metrics_path或metrics_logger参数")
        return {}
    
    # 如果指定了输出目录，确保它存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个指标并绘制
    results = {}
    for metric in metrics:
        if metric == 'loss':
            # 使用专用的损失绘图函数
            if output_dir:
                output_path = os.path.join(output_dir, f"{metric}_curve.{save_format}")
            else:
                output_path = None
            
            results[metric] = plot_loss(
                metrics_logger=logger_to_use,
                output_path=output_path,
                experiment_name=experiment_name,
                figsize=figsize,
                title=f"{experiment_name + ' - ' if experiment_name else ''}Loss Over Time",
                save_format=save_format,
                dpi=dpi,
                show_grid=show_grid,
                style=style,
                add_timestamp=add_timestamp,
                metadata=metadata
            )
        elif metric == 'accuracy':
            # 使用专用的准确率绘图函数
            if output_dir:
                output_path = os.path.join(output_dir, f"{metric}_curve.{save_format}")
            else:
                output_path = None
            
            results[metric] = plot_accuracy(
                metrics_logger=logger_to_use,
                output_path=output_path,
                experiment_name=experiment_name,
                figsize=figsize,
                title=f"{experiment_name + ' - ' if experiment_name else ''}Accuracy Over Time",
                save_format=save_format,
                dpi=dpi,
                show_grid=show_grid,
                style=style,
                add_timestamp=add_timestamp,
                metadata=metadata
            )
        else:
            # 对其他指标，使用通用的绘图方法
            # 设置绘图样式
            if style:
                plt.style.use(style)
            
            # 创建图表
            fig = plt.figure(figsize=figsize)
            
            # 提取训练指标
            epochs = []
            values = []
            if metric in logger_to_use.train_metrics:
                epochs = [entry['epoch'] for entry in logger_to_use.train_metrics[metric]]
                values = [entry['value'] for entry in logger_to_use.train_metrics[metric]]
                plt.plot(epochs, values, 'b-', label=f'Training {metric.capitalize()}')
            
            # 提取验证指标（如果存在）
            val_metric = f'val_{metric}'
            if val_metric in logger_to_use.eval_metrics:
                val_epochs = [entry['epoch'] for entry in logger_to_use.eval_metrics[val_metric]]
                val_values = [entry['value'] for entry in logger_to_use.eval_metrics[val_metric]]
                plt.plot(val_epochs, val_values, 'r-', label=f'Validation {metric.capitalize()}')
            
            # 设置图表标题和标签
            plt.title(f"{experiment_name + ' - ' if experiment_name else ''}{metric.capitalize()} Over Time")
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            
            if show_grid:
                plt.grid(True)
            
            # 保存或显示图表
            if output_dir:
                output_path = os.path.join(output_dir, f"{metric}_curve.{save_format}")
                results[metric] = save_plot(
                    fig=fig,
                    output_path=output_path,
                    dpi=dpi,
                    save_format=save_format,
                    add_timestamp=add_timestamp,
                    metadata=metadata
                )
            else:
                plt.show()
            
            plt.close()
    
    logger.info(f"已绘制所有指定指标: {', '.join(metrics)}")
    return results

if __name__ == "__main__":
    # 示例用法
    plots_dir = os.path.join('temp_metrics', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制损失曲线
    plot_loss(
        metrics_path=os.path.join('temp_metrics', 'simulation_train_metrics.csv'),
        output_path=os.path.join(plots_dir, 'loss_curve_demo.png'),
        experiment_name='Simulation'
    )
    
    # 绘制准确率曲线
    plot_accuracy(
        metrics_path=os.path.join('temp_metrics', 'simulation_train_metrics.csv'),
        output_path=os.path.join(plots_dir, 'accuracy_curve_demo.png'),
        experiment_name='Simulation'
    )
    
    # 绘制多个指标曲线
    plot_training_history(
        metrics_path=os.path.join('temp_metrics', 'simulation_train_metrics.csv'),
        metrics=['loss', 'accuracy'],
        output_dir=plots_dir,
        experiment_name='Simulation'
    ) 