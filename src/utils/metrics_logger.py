import os
import json
import csv
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb  # 替换 torch.utils.tensorboard 为 wandb

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    指标记录器，用于记录、保存和加载训练/评估指标。
    
    参数:
        save_dir (str): 指标保存目录
        experiment_name (str): 实验名称，用于生成文件名
        save_format (str): 保存格式，'csv' 或 'json'
        save_freq (int): 保存频率（每多少个epoch保存一次）
        enable_wandb (bool): 是否启用Weights & Biases记录
        wandb_project (str): W&B项目名称
        wandb_config (Dict): W&B配置参数
    """
    def __init__(
        self,
        save_dir: str = 'metrics',
        experiment_name: Optional[str] = None,
        save_format: str = 'csv',
        save_freq: int = 1,
        enable_wandb: bool = False,
        wandb_project: str = 'vit_training',
        wandb_config: Optional[Dict] = None
    ):
        self.save_dir = save_dir
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.save_format = save_format.lower()
        self.save_freq = save_freq
        self.enable_wandb = enable_wandb
        self.wandb_project = wandb_project
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 指标存储结构
        self.train_metrics = defaultdict(list)  # 每个指标为键，值为包含epoch和value的字典列表
        self.eval_metrics = defaultdict(list)
        
        # 指标文件路径
        self.train_metrics_path = os.path.join(
            self.save_dir, 
            f"{self.experiment_name}_train_metrics.{self.save_format}"
        )
        self.eval_metrics_path = os.path.join(
            self.save_dir, 
            f"{self.experiment_name}_eval_metrics.{self.save_format}"
        )
        
        # 初始化Weights & Biases
        self.wandb_initialized = False
        if self.enable_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.experiment_name,
                config=wandb_config or {},
            )
            self.wandb_initialized = True
            logger.info(f"已启用Weights & Biases，项目：{self.wandb_project}，实验名称：{self.experiment_name}")
        
        # 读取已有的指标（如果存在）
        self._load_existing_metrics()
        
        logger.info(f"MetricsLogger初始化完成。实验名称：{self.experiment_name}，保存格式：{self.save_format}")
    
    def _load_existing_metrics(self):
        """
        加载已存在的指标文件（如果有）。
        """
        if os.path.exists(self.train_metrics_path):
            self.train_metrics = self.load_train_metrics()
            logger.info(f"加载已有训练指标：{self.train_metrics_path}")
        
        if os.path.exists(self.eval_metrics_path):
            self.eval_metrics = self.load_eval_metrics()
            logger.info(f"加载已有评估指标：{self.eval_metrics_path}")
    
    def log_train_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        记录训练指标。
        
        参数:
            metrics (Dict[str, float]): 训练指标字典，键为指标名，值为指标值
            epoch (int): 当前训练轮次
        """
        # 记录每个指标
        for metric_name, metric_value in metrics.items():
            self.train_metrics[metric_name].append({
                'epoch': epoch,
                'value': metric_value,
                'timestamp': datetime.now().isoformat()
            })
            
            # 记录到Weights & Biases
            if self.enable_wandb and self.wandb_initialized:
                wandb.log({f"train/{metric_name}": metric_value}, step=epoch)
        
        # 按保存频率保存指标
        if epoch % self.save_freq == 0:
            self._save_metrics(self.train_metrics, self.train_metrics_path)
            logger.debug(f"保存训练指标到：{self.train_metrics_path}")
        
        return self
    
    def log_eval_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        记录评估指标。
        
        参数:
            metrics (Dict[str, float]): 评估指标字典，键为指标名，值为指标值
            epoch (int): 当前训练轮次
        """
        # 记录每个指标
        for metric_name, metric_value in metrics.items():
            self.eval_metrics[metric_name].append({
                'epoch': epoch,
                'value': metric_value,
                'timestamp': datetime.now().isoformat()
            })
            
            # 记录到Weights & Biases
            if self.enable_wandb and self.wandb_initialized:
                # 评估阶段，检查当前wandb步骤，避免使用较小的步骤值
                if wandb.run and hasattr(wandb.run, 'step'):
                    current_step = wandb.run.step
                    log_step = max(epoch, current_step + 1)  # 确保步骤大于当前步骤
                else:
                    log_step = epoch
                
                wandb.log({f"val/{metric_name}": metric_value}, step=log_step)
        
        # 按保存频率保存指标
        if epoch % self.save_freq == 0:
            self._save_metrics(self.eval_metrics, self.eval_metrics_path)
            logger.debug(f"保存评估指标到：{self.eval_metrics_path}")
        
        return self
    
    def log_histogram(self, name: str, values: np.ndarray, epoch: int, tag: str = "model"):
        """
        记录直方图数据到Weights & Biases。
        
        参数:
            name (str): 指标名称
            values (np.ndarray): 值数组
            epoch (int): 当前轮次
            tag (str): 标签类别，如'model'、'gradients'等
        """
        if self.enable_wandb and self.wandb_initialized:
            wandb.log({f"{tag}/{name}": wandb.Histogram(values)}, step=epoch)
        return self
    
    def log_image(self, name: str, image: np.ndarray, epoch: int, tag: str = "images"):
        """
        记录图像到Weights & Biases。
        
        参数:
            name (str): 图像名称
            image (np.ndarray): 图像数据
            epoch (int): 当前轮次
            tag (str): 标签类别
        """
        if self.enable_wandb and self.wandb_initialized:
            wandb.log({f"{tag}/{name}": wandb.Image(image)}, step=epoch)
        return self
    
    def log_figure(self, name: str, figure: plt.Figure, epoch: int, tag: str = "plots"):
        """
        记录matplotlib图表到Weights & Biases。
        
        参数:
            name (str): 图表名称
            figure (plt.Figure): matplotlib图表
            epoch (int): 当前轮次
            tag (str): 标签类别
        """
        if self.enable_wandb and self.wandb_initialized:
            wandb.log({f"{tag}/{name}": wandb.Image(figure)}, step=epoch)
        return self
    
    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        记录超参数到Weights & Biases。
        
        参数:
            hparams (Dict[str, Any]): 超参数字典
            metrics (Dict[str, float]): 与超参数关联的指标结果
        """
        if self.enable_wandb and self.wandb_initialized:
            wandb.config.update(hparams, allow_val_change=True)  # 允许值改变
            wandb.log(metrics)
        return self
    
    def close(self):
        """
        关闭Weights & Biases连接。应在训练结束时调用。
        """
        if self.enable_wandb and self.wandb_initialized:
            wandb.finish()
            logger.info("Weights & Biases连接已关闭")
    
    def _save_metrics(self, metrics: Dict[str, List[Dict]], path: str):
        """
        将指标保存到文件。
        
        参数:
            metrics (Dict): 要保存的指标字典
            path (str): 保存路径
        """
        if self.save_format == 'json':
            self._save_json(metrics, path)
        else:  # 默认为CSV
            self._save_csv(metrics, path)
    
    def _save_json(self, metrics: Dict[str, List[Dict]], path: str):
        """
        将指标保存为JSON格式。
        
        参数:
            metrics (Dict): 要保存的指标字典
            path (str): 保存路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
    
    def _save_csv(self, metrics: Dict[str, List[Dict]], path: str):
        """
        将指标保存为CSV格式。
        
        参数:
            metrics (Dict): 要保存的指标字典
            path (str): 保存路径
        """
        # 将指标转换为pandas DataFrame
        data = []
        for metric_name, values in metrics.items():
            for entry in values:
                data.append({
                    'metric': metric_name,
                    'epoch': entry['epoch'],
                    'value': entry['value'],
                    'timestamp': entry['timestamp']
                })
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
    
    def load_train_metrics(self) -> Dict[str, List[Dict]]:
        """
        加载训练指标。
        
        返回:
            Dict[str, List[Dict]]: 加载的训练指标
        """
        return self._load_metrics(self.train_metrics_path)
    
    def load_eval_metrics(self) -> Dict[str, List[Dict]]:
        """
        加载评估指标。
        
        返回:
            Dict[str, List[Dict]]: 加载的评估指标
        """
        return self._load_metrics(self.eval_metrics_path)
    
    def _load_metrics(self, path: str) -> Dict[str, List[Dict]]:
        """
        从文件加载指标。
        
        参数:
            path (str): 指标文件路径
        
        返回:
            Dict: 加载的指标字典
        """
        if self.save_format == 'json':
            return self._load_json(path)
        else:  # 默认为CSV
            return self._load_csv(path)
    
    def _load_json(self, path: str) -> Dict[str, List[Dict]]:
        """
        从JSON文件加载指标。
        
        参数:
            path (str): JSON文件路径
        
        返回:
            Dict: 加载的指标字典
        """
        with open(path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # 转换为defaultdict
        result = defaultdict(list)
        for metric_name, values in metrics.items():
            result[metric_name] = values
        
        return result
    
    def _load_csv(self, path: str) -> Dict[str, List[Dict]]:
        """
        从CSV文件加载指标。
        
        参数:
            path (str): CSV文件路径
        
        返回:
            Dict: 加载的指标字典
        """
        result = defaultdict(list)
        
        try:
            df = pd.read_csv(path)
            for metric_name in df['metric'].unique():
                metric_df = df[df['metric'] == metric_name]
                for _, row in metric_df.iterrows():
                    result[metric_name].append({
                        'epoch': int(row['epoch']),
                        'value': float(row['value']),
                        'timestamp': row['timestamp']
                    })
        except Exception as e:
            logger.error(f"加载CSV指标文件出错：{e}")
        
        return result
    
    def plot_metric(self, metric_name: str, output_path: Optional[str] = None) -> None:
        """
        绘制单个指标的曲线图。
        
        参数:
            metric_name (str): 指标名称
            output_path (Optional[str]): 输出路径，如果为None则显示图像不保存
        """
        plt.figure(figsize=(10, 6))
        
        # 训练指标
        if metric_name in self.train_metrics and self.train_metrics[metric_name]:
            epochs = [entry['epoch'] for entry in self.train_metrics[metric_name]]
            values = [entry['value'] for entry in self.train_metrics[metric_name]]
            plt.plot(epochs, values, 'b-', label=f'Train {metric_name}')
        
        # 验证指标（通常是val_+metric_name）
        val_metric = f"val_{metric_name}"
        if val_metric in self.eval_metrics and self.eval_metrics[val_metric]:
            epochs = [entry['epoch'] for entry in self.eval_metrics[val_metric]]
            values = [entry['value'] for entry in self.eval_metrics[val_metric]]
            plt.plot(epochs, values, 'r-', label=f'Validation {metric_name}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs. Epoch')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"指标曲线已保存到：{output_path}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_metrics(self, metric_names: List[str], output_dir: Optional[str] = None) -> None:
        """
        绘制多个指标的曲线图。
        
        参数:
            metric_names (List[str]): 指标名称列表
            output_dir (Optional[str]): 输出目录，如果为None则显示图像不保存
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for metric_name in metric_names:
            if output_dir:
                output_path = os.path.join(output_dir, f"{metric_name}_curve.png")
            else:
                output_path = None
            
            self.plot_metric(metric_name, output_path)
    
    def summary(self, detailed: bool = False) -> Dict[str, Any]:
        """
        生成指标摘要。
        
        参数:
            detailed (bool): 是否生成详细摘要
        
        返回:
            Dict: 摘要字典
        """
        result = {
            'experiment_name': self.experiment_name,
            'train_metrics_count': len(self.train_metrics),
            'eval_metrics_count': len(self.eval_metrics),
        }
        
        if detailed:
            # 计算训练指标统计信息
            train_stats = {}
            for metric_name, entries in self.train_metrics.items():
                if entries:
                    values = [entry['value'] for entry in entries]
                    train_stats[metric_name] = {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
            
            # 计算评估指标统计信息
            eval_stats = {}
            for metric_name, entries in self.eval_metrics.items():
                if entries:
                    values = [entry['value'] for entry in entries]
                    eval_stats[metric_name] = {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
            
            result['train_metrics_stats'] = train_stats
            result['eval_metrics_stats'] = eval_stats
        
        return result
    
    def get_best_epoch(self, metric_name: str, mode: str = 'min') -> Dict[str, Any]:
        """
        获取指定指标的最佳轮次。
        
        参数:
            metric_name (str): 指标名称
            mode (str): 'min'表示值越小越好，'max'表示值越大越好
        
        返回:
            Dict: 包含最佳轮次和对应指标值的字典
        """
        metrics_dict = self.eval_metrics if metric_name in self.eval_metrics else self.train_metrics
        
        if metric_name not in metrics_dict or not metrics_dict[metric_name]:
            return {'epoch': -1, 'value': None}
        
        entries = metrics_dict[metric_name]
        if mode == 'min':
            best_entry = min(entries, key=lambda x: x['value'])
        else:  # mode == 'max'
            best_entry = max(entries, key=lambda x: x['value'])
        
        return {
            'epoch': best_entry['epoch'],
            'value': best_entry['value']
        }
    
    def export_metrics(self, output_path: str, format: str = 'csv') -> None:
        """
        将所有指标导出到单个文件。
        
        参数:
            output_path (str): 输出文件路径
            format (str): 导出格式，'csv'或'json'
        """
        # 合并训练和评估指标
        all_metrics = {}
        for metric_name, entries in self.train_metrics.items():
            all_metrics[f"train_{metric_name}"] = entries
        
        for metric_name, entries in self.eval_metrics.items():
            all_metrics[metric_name] = entries
        
        # 导出指标
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2)
        else:  # 默认为CSV
            # 将指标转换为pandas DataFrame
            data = []
            for metric_name, values in all_metrics.items():
                for entry in values:
                    data.append({
                        'metric': metric_name,
                        'epoch': entry['epoch'],
                        'value': entry['value'],
                        'timestamp': entry['timestamp']
                    })
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
        
        logger.info(f"已将所有指标导出到：{output_path}")
    
    def visualize_all_metrics(self, output_dir: str) -> None:
        """
        可视化所有指标并保存图表。
        
        参数:
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有训练指标名称
        train_metric_names = list(self.train_metrics.keys())
        
        # 绘制所有训练指标
        if train_metric_names:
            self.plot_metrics(train_metric_names, output_dir)
        
        # 如果使用W&B，上传所有图表
        if self.enable_wandb and self.wandb_initialized:
            for file in os.listdir(output_dir):
                if file.endswith('.png') or file.endswith('.jpg'):
                    image_path = os.path.join(output_dir, file)
                    wandb.log({f"summary_plots/{file}": wandb.Image(image_path)})
            
            # 记录摘要统计信息
            summary_stats = self.summary()
            for key, value in summary_stats.items():
                if isinstance(value, (int, float)):
                    wandb.run.summary[key] = value 