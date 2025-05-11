"""
指标绘图模块，用于可视化训练和评估指标。
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import List, Optional, Union, Dict, Any, Tuple

from src.utils.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)

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
    style: Optional[str] = 'seaborn-v0_8-darkgrid'
) -> None:
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

    返回:
        None
    """
    # 设置绘图样式
    if style:
        plt.style.use(style)
    
    # 创建图表
    plt.figure(figsize=figsize)
    
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
            return

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
                return
        except Exception as e:
            logger.error(f"加载指标数据时出错: {e}")
            return
    else:
        logger.error("必须提供metrics_path或metrics_logger参数之一")
        return
    
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
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存图表
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=save_format)
        logger.info(f"损失曲线图保存至: {output_path}")
    else:
        plt.show()
    
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
    style: Optional[str] = 'seaborn-v0_8-darkgrid'
) -> None:
    """
    绘制多个训练指标的历史曲线。

    参数:
        metrics_path (Optional[str]): 指标文件路径，如果为None则使用metrics_logger。
        metrics_logger (Optional[MetricsLogger]): 指标记录器实例，如果metrics_path为None，则使用此参数。
        metrics (List[str]): 要绘制的指标名称列表，如为None则绘制所有可用指标。
        output_dir (Optional[str]): 输出目录，如果为None则显示但不保存。
        experiment_name (Optional[str]): 实验名称，用于图表标题。
        figsize (Tuple[int, int]): 图表尺寸，默认为(10, 6)。
        save_format (str): 保存格式，默认为'png'。
        dpi (int): 图表DPI，默认为300。
        show_grid (bool): 是否显示网格，默认为True。
        style (Optional[str]): Matplotlib样式，默认为'seaborn-v0_8-darkgrid'。

    返回:
        None
    """
    # 如果未指定指标，则默认绘制loss和accuracy
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    # 如果提供了metrics_path，则创建临时MetricsLogger
    temp_logger = None
    if metrics_path and not metrics_logger:
        try:
            # 确定文件类型并加载数据
            if metrics_path.endswith('.csv'):
                # 从metrics_path推断保存目录和实验名称
                save_dir = os.path.dirname(metrics_path)
                exp_name = os.path.basename(metrics_path).split('_')[0]
                if 'train' in metrics_path:
                    temp_logger = MetricsLogger(save_dir=save_dir, 
                                              experiment_name=exp_name,
                                              save_format='csv')
            elif metrics_path.endswith('.json'):
                save_dir = os.path.dirname(metrics_path)
                exp_name = os.path.basename(metrics_path).split('_')[0]
                temp_logger = MetricsLogger(save_dir=save_dir, 
                                          experiment_name=exp_name,
                                          save_format='json')
            else:
                logger.error(f"不支持的文件格式: {metrics_path}")
                return
        except Exception as e:
            logger.error(f"加载指标数据时出错: {e}")
            return
    
    # 使用临时记录器或提供的记录器
    logger_to_use = metrics_logger or temp_logger
    
    if not logger_to_use:
        logger.error("必须提供有效的metrics_path或metrics_logger参数")
        return
    
    # 如果指定了输出目录，确保它存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个指标并绘制
    for metric in metrics:
        if metric == 'loss':
            # 使用专门的plot_loss函数
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"{metric}_curve.{save_format}")
            
            plot_loss(
                metrics_logger=logger_to_use,
                output_path=output_path,
                experiment_name=experiment_name,
                figsize=figsize,
                save_format=save_format,
                dpi=dpi,
                show_grid=show_grid,
                style=style
            )
        else:
            # 使用MetricsLogger的plot_metric方法绘制其他指标
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"{metric}_curve.{save_format}")
            
            logger_to_use.plot_metric(metric, output_path)
    
    logger.info(f"已绘制所有指定指标: {', '.join(metrics)}")

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
    
    # 绘制多个指标曲线
    plot_training_history(
        metrics_path=os.path.join('temp_metrics', 'simulation_train_metrics.csv'),
        metrics=['loss', 'accuracy'],
        output_dir=plots_dir,
        experiment_name='Simulation'
    ) 