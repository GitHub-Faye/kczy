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

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    指标记录器，用于记录、保存和加载训练/评估指标。
    
    参数:
        save_dir (str): 指标保存目录
        experiment_name (str): 实验名称，用于生成文件名
        save_format (str): 保存格式，'csv' 或 'json'
        save_freq (int): 保存频率（每多少个epoch保存一次）
    """
    def __init__(
        self,
        save_dir: str = 'metrics',
        experiment_name: Optional[str] = None,
        save_format: str = 'csv',
        save_freq: int = 1
    ):
        self.save_dir = save_dir
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.save_format = save_format.lower()
        self.save_freq = save_freq
        
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
        
        # 按保存频率保存指标
        if epoch % self.save_freq == 0:
            self._save_metrics(self.eval_metrics, self.eval_metrics_path)
            logger.debug(f"保存评估指标到：{self.eval_metrics_path}")
        
        return self
    
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
            Dict: 训练指标字典
        """
        if not os.path.exists(self.train_metrics_path):
            return defaultdict(list)
        
        return self._load_metrics(self.train_metrics_path)
    
    def load_eval_metrics(self) -> Dict[str, List[Dict]]:
        """
        加载评估指标。
        
        返回:
            Dict: 评估指标字典
        """
        if not os.path.exists(self.eval_metrics_path):
            return defaultdict(list)
        
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
        可视化所有指标，并生成汇总报告。
        
        参数:
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有独特的指标名称（不包括val_前缀）
        all_metric_names = set()
        for name in self.train_metrics.keys():
            all_metric_names.add(name)
        
        for name in self.eval_metrics.keys():
            if name.startswith('val_'):
                base_name = name[4:]  # 移除'val_'前缀
                all_metric_names.add(base_name)
        
        # 为每个指标绘制曲线
        for metric_name in all_metric_names:
            output_path = os.path.join(output_dir, f"{metric_name}_curve.png")
            self.plot_metric(metric_name, output_path)
        
        # 生成汇总报告
        summary = self.summary(detailed=True)
        report_path = os.path.join(output_dir, "metrics_summary.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # 生成markdown格式的汇总报告
        md_report_path = os.path.join(output_dir, "metrics_summary.md")
        with open(md_report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 实验指标汇总报告\n\n")
            f.write(f"## 实验名称: {summary['experiment_name']}\n\n")
            
            f.write("## 训练指标\n\n")
            f.write("| 指标名称 | 最小值 | 最大值 | 平均值 | 标准差 |\n")
            f.write("|---------|-------|-------|-------|-------|\n")
            for metric_name, stats in summary.get('train_metrics_stats', {}).items():
                f.write(f"| {metric_name} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['mean']:.4f} | {stats['std']:.4f} |\n")
            
            f.write("\n## 评估指标\n\n")
            f.write("| 指标名称 | 最小值 | 最大值 | 平均值 | 标准差 |\n")
            f.write("|---------|-------|-------|-------|-------|\n")
            for metric_name, stats in summary.get('eval_metrics_stats', {}).items():
                f.write(f"| {metric_name} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['mean']:.4f} | {stats['std']:.4f} |\n")
            
            f.write("\n## 最佳模型\n\n")
            for metric_name in summary.get('eval_metrics_stats', {}):
                if metric_name.startswith('val_'):
                    mode = 'min' if 'loss' in metric_name else 'max'
                    best = self.get_best_epoch(metric_name, mode)
                    f.write(f"- **{metric_name}**: 轮次 {best['epoch']}, 值 {best['value']:.4f} ({mode}模式)\n")
        
        logger.info(f"已生成指标汇总报告：{md_report_path}") 