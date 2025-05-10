import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.models.vit import VisionTransformer
from src.utils.config import TrainingConfig
from src.models.optimizer_manager import OptimizerManager
from src.utils.metrics_logger import MetricsLogger
from src.models.model_utils import save_checkpoint, save_model

class LossCalculator:
    """
    损失计算器，提供多种损失函数选择
    
    参数:
        loss_type (str): 损失函数类型，支持 'cross_entropy', 'mse', 'bce', 'focal'
        class_weights (Optional[torch.Tensor]): 类别权重，用于加权损失计算
        gamma (float): Focal Loss的gamma参数
        reduction (str): 损失计算方式，'mean', 'sum', 或 'none'
    """
    def __init__(
        self, 
        loss_type: str = 'cross_entropy',
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        self.loss_type = loss_type.lower()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction
        
        # 初始化损失函数
        if self.loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        elif self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(weight=class_weights, reduction=reduction)
        elif self.loss_type == 'focal':
            # Focal Loss需要在forward中特殊实现
            self.loss_fn = None
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}. 支持的类型: 'cross_entropy', 'mse', 'bce', 'focal'")
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        参数:
            inputs (torch.Tensor): 模型输出的预测值 (B, C, ...)
            targets (torch.Tensor): 目标值 (B, ...)
            
        返回:
            torch.Tensor: 计算得到的Focal Loss
        """
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.class_weights, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        参数:
            outputs (torch.Tensor): 模型输出
            targets (torch.Tensor): 目标值
            
        返回:
            torch.Tensor: 计算得到的损失值
        """
        if self.loss_type == 'focal':
            return self.focal_loss(outputs, targets)
        else:
            return self.loss_fn(outputs, targets)
    
    def get_loss_name(self) -> str:
        """获取损失函数名称"""
        return self.loss_type

class BackpropManager:
    """
    反向传播管理器，处理梯度计算、缩放和裁剪
    
    参数:
        grad_clip_value (Optional[float]): 梯度裁剪的阈值，None表示不裁剪
        grad_clip_norm (Optional[float]): 梯度范数裁剪的阈值，None表示不裁剪
        grad_scaler (Optional[torch.cuda.amp.GradScaler]): 用于混合精度训练的梯度缩放器
    """
    def __init__(
        self,
        grad_clip_value: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        grad_scaler: Optional[object] = None
    ):
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.grad_scaler = grad_scaler
        
    def compute_gradients(
        self, 
        loss: torch.Tensor, 
        model: nn.Module,
        optimizer: Union[optim.Optimizer, OptimizerManager],
        retain_graph: bool = False
    ) -> None:
        """
        计算损失的梯度
        
        参数:
            loss (torch.Tensor): 损失值
            model (nn.Module): 需要计算梯度的模型
            optimizer (Union[optim.Optimizer, OptimizerManager]): 优化器或优化器管理器
            retain_graph (bool): 是否保留计算图
        """
        # 处理OptimizerManager
        if isinstance(optimizer, OptimizerManager):
            optimizer_obj = optimizer.optimizer
        else:
            optimizer_obj = optimizer
        
        # 清除之前的梯度
        optimizer_obj.zero_grad()
        
        # 使用混合精度训练时的梯度计算
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            # 标准梯度计算
            loss.backward(retain_graph=retain_graph)
    
    def apply_gradient_clipping(self, model: nn.Module) -> None:
        """
        应用梯度裁剪
        
        参数:
            model (nn.Module): 需要裁剪梯度的模型
        """
        if self.grad_clip_value is not None:
            # 按值裁剪梯度
            torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip_value)
            
        if self.grad_clip_norm is not None:
            # 按范数裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
    
    def optimizer_step(self, optimizer: Union[optim.Optimizer, OptimizerManager]) -> None:
        """
        执行优化器步骤，更新模型参数
        
        参数:
            optimizer (Union[optim.Optimizer, OptimizerManager]): 优化器或优化器管理器
        """
        # 处理OptimizerManager
        if isinstance(optimizer, OptimizerManager):
            optimizer_obj = optimizer.optimizer
        else:
            optimizer_obj = optimizer
            
        if self.grad_scaler is not None:
            # 混合精度训练中的参数更新
            self.grad_scaler.step(optimizer_obj)
            self.grad_scaler.update()
        else:
            # 标准参数更新
            optimizer_obj.step()
    
    def backward_and_update(
        self, 
        loss: torch.Tensor, 
        model: nn.Module, 
        optimizer: Union[optim.Optimizer, OptimizerManager],
        retain_graph: bool = False
    ) -> None:
        """
        执行完整的反向传播和参数更新过程
        
        参数:
            loss (torch.Tensor): 损失值
            model (nn.Module): 模型
            optimizer (Union[optim.Optimizer, OptimizerManager]): 优化器或优化器管理器
            retain_graph (bool): 是否保留计算图
        """
        # 计算梯度
        self.compute_gradients(loss, model, optimizer, retain_graph)
        
        # 应用梯度裁剪
        self.apply_gradient_clipping(model)
        
        # 更新模型参数
        self.optimizer_step(optimizer)

class TrainingLoop:
    """
    训练循环，整合损失计算、反向传播和优化器步骤
    
    参数:
        model (nn.Module): 要训练的模型
        loss_calculator (LossCalculator): 损失计算器
        optimizer_manager (Optional[OptimizerManager]): 优化器管理器，仅做评估时可为None
        backprop_manager (Optional[BackpropManager]): 反向传播管理器，仅做评估时可为None
        device (torch.device): 运行设备
        metrics_logger (Optional[MetricsLogger]): 指标记录器，用于记录训练和评估指标
    """
    def __init__(
        self,
        model: nn.Module,
        loss_calculator: LossCalculator,
        optimizer_manager: Optional[OptimizerManager] = None,
        backprop_manager: Optional[BackpropManager] = None,
        device: torch.device = torch.device('cpu'),
        metrics_logger: Optional[MetricsLogger] = None
    ):
        self.model = model
        self.loss_calculator = loss_calculator
        self.optimizer_manager = optimizer_manager
        self.backprop_manager = backprop_manager
        self.device = device
        self.metrics_logger = metrics_logger
        
        # 将模型移至指定设备
        self.model.to(self.device)
    
    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: TrainingConfig,
        class_weights: Optional[torch.Tensor] = None
    ) -> 'TrainingLoop':
        """
        从训练配置创建训练循环
        
        参数:
            model (nn.Module): 要训练的模型
            config (TrainingConfig): 训练配置
            class_weights (Optional[torch.Tensor]): 类别权重，用于加权损失计算
            
        返回:
            TrainingLoop: 训练循环实例
        """
        # 设置随机种子
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
        
        # 确定设备
        device = torch.device(config.device)
        
        # 创建损失计算器
        loss_calculator = LossCalculator(
            loss_type=config.loss_type,
            class_weights=class_weights
        )
        
        # 创建优化器管理器
        optimizer_manager = OptimizerManager(
            optimizer_type=config.optimizer_type,
            model=model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            scheduler_type=config.scheduler_type,
            scheduler_params=config.scheduler_params,
            **config.optimizer_params
        )
        
        # 创建梯度缩放器（用于混合精度训练）
        grad_scaler = None
        if config.use_mixed_precision and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                grad_scaler = GradScaler()
            except ImportError:
                print("警告：当前PyTorch版本不支持混合精度训练，将使用全精度训练")
                
        # 创建反向传播管理器
        backprop_manager = BackpropManager(
            grad_clip_value=config.grad_clip_value,
            grad_clip_norm=config.grad_clip_norm,
            grad_scaler=grad_scaler
        )
        
        # 创建指标记录器
        metrics_logger = MetricsLogger(
            save_dir=config.metrics_dir,
            experiment_name=config.metrics_experiment_name,
            save_format=config.metrics_format,
            save_freq=config.metrics_save_freq
        )
        
        # 创建训练循环
        return cls(
            model=model,
            loss_calculator=loss_calculator,
            optimizer_manager=optimizer_manager,
            backprop_manager=backprop_manager,
            device=device,
            metrics_logger=metrics_logger
        )

    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个完整的epoch
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            epoch (int): 当前epoch
            
        返回:
            Dict[str, float]: 训练指标
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失
            loss = self.loss_calculator(outputs, targets)
            
            # 如果有优化器和反向传播管理器，执行反向传播和参数更新
            if self.optimizer_manager is not None and self.backprop_manager is not None:
                self.backprop_manager.backward_and_update(
                    loss, self.model, self.optimizer_manager
                )
            
            # 计算准确率（如果是分类任务）
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 累加损失
            total_loss += loss.item()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        # 学习率调度（如果有）
        learning_rate = 0.0
        if self.optimizer_manager is not None:
            self.optimizer_manager.scheduler_step()
            learning_rate = self.optimizer_manager.get_lr()
        
        # 记录训练指标
        train_metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': learning_rate
        }
        
        # 如果有指标记录器，记录训练指标
        if self.metrics_logger is not None:
            self.metrics_logger.log_train_metrics(train_metrics, epoch)
        
        return train_metrics
    
    def validate(
        self, 
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        参数:
            val_loader (DataLoader): 验证数据加载器
            
        返回:
            Dict[str, float]: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.loss_calculator(outputs, targets)
                
                # 计算准确率（如果是分类任务）
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 累加损失
                total_loss += loss.item()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
        log_interval: int = 1,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 1
    ) -> Dict[str, List[float]]:
        """
        完整训练过程
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (Optional[DataLoader]): 验证数据加载器
            num_epochs (int): 训练的总epoch数
            checkpoint_path (Optional[str]): 检查点保存路径
            log_interval (int): 日志记录间隔
            early_stopping (bool): 是否使用早停策略
            early_stopping_patience (int): 早停策略的耐心值
            checkpoint_dir (Optional[str]): 检查点保存目录，如果提供，将根据epoch保存多个检查点
            checkpoint_freq (int): 检查点保存频率（每多少个epoch保存一次）
            
        返回:
            Dict[str, List[float]]: 训练历史记录
        """
        # 确保模型处于训练模式
        if self.optimizer_manager is None or self.backprop_manager is None:
            raise ValueError("训练模式下必须提供optimizer_manager和backprop_manager")
            
        history = {
            'loss': [],
            'accuracy': [],
            'learning_rate': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # 早停策略相关变量
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 记录训练指标
            history['loss'].append(train_metrics['loss'])
            history['accuracy'].append(train_metrics['accuracy'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            # 验证
            if val_loader:
                val_metrics = self.validate(val_loader)
                
                # 记录验证指标
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])
                
                # 记录验证指标到指标记录器
                if self.metrics_logger is not None:
                    self.metrics_logger.log_eval_metrics(val_metrics, epoch)
                
                # 早停策略
                if early_stopping:
                    val_loss = val_metrics['val_loss']
                    val_accuracy = val_metrics['val_accuracy']
                    
                    # 检查是否是最佳模型
                    is_best = False
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        is_best = True
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        is_best = True
                    
                    if is_best:
                        patience_counter = 0
                        best_model_state = self.model.state_dict()
                    else:
                        patience_counter += 1
                        
                    # 如果连续early_stopping_patience个epoch没有改善，则提前停止训练
                    if patience_counter >= early_stopping_patience:
                        print(f"早停: 在{patience_counter}个epoch中验证指标没有改善，停止训练")
                        # 恢复最佳模型
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        break
            
            # 记录日志
            if (epoch + 1) % log_interval == 0:
                log_str = f"Epoch [{epoch+1}/{num_epochs}] "
                log_str += f"Loss: {train_metrics['loss']:.4f} "
                log_str += f"Acc: {train_metrics['accuracy']:.2f}% "
                
                if val_loader:
                    log_str += f"Val Loss: {val_metrics['val_loss']:.4f} "
                    log_str += f"Val Acc: {val_metrics['val_accuracy']:.2f}% "
                
                log_str += f"LR: {train_metrics['learning_rate']:.6f}"
                print(log_str)
            
            # 保存检查点
            if checkpoint_path:
                # 使用新的保存检查点功能
                if self.optimizer_manager is not None:
                    optimizer_state = self.optimizer_manager.state_dict()
                    save_checkpoint(
                        model=self.model,
                        optimizer_state=optimizer_state,
                        file_path=checkpoint_path,
                        epoch=epoch + 1,
                        train_history=history,
                        metadata={
                            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'device': str(self.device)
                        }
                    )
                else:
                    # 如果没有优化器管理器，只保存模型状态
                    save_model(
                        model=self.model,
                        file_path=checkpoint_path,
                        metadata={
                            'epoch': epoch + 1,
                            'history': history,
                            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'device': str(self.device)
                        }
                    )
            
            # 如果提供了检查点目录，每checkpoint_freq个epoch保存一个检查点
            if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                
                # 使用新的保存检查点功能
                if self.optimizer_manager is not None:
                    optimizer_state = self.optimizer_manager.state_dict()
                    save_checkpoint(
                        model=self.model,
                        optimizer_state=optimizer_state,
                        file_path=checkpoint_file,
                        epoch=epoch + 1,
                        train_history=history,
                        metadata={
                            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'device': str(self.device)
                        }
                    )
                else:
                    # 如果没有优化器管理器，只保存模型状态
                    save_model(
                        model=self.model,
                        file_path=checkpoint_file,
                        metadata={
                            'epoch': epoch + 1,
                            'history': history,
                            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'device': str(self.device)
                        }
                    )
                print(f"检查点已保存到: {checkpoint_file}")
        
        # 训练结束后，如果有指标记录器，自动绘制指标曲线
        if self.metrics_logger is not None and hasattr(self.metrics_logger, 'plot_metrics'):
            # 确定要绘制的指标
            metrics_to_plot = ['loss', 'accuracy']
            # 尝试绘制指标曲线
            try:
                output_dir = checkpoint_dir if checkpoint_dir else 'metrics_plots'
                self.metrics_logger.plot_metrics(metrics_to_plot, output_dir=output_dir)
                print(f"指标曲线已保存到: {output_dir}")
            except Exception as e:
                print(f"绘制指标曲线时出错: {str(e)}")
        
        return history

    def evaluate(
        self, 
        test_loader: DataLoader,
        num_classes: Optional[int] = None,
        average: str = 'weighted',
        visualize_confusion_matrix: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        在测试集上评估模型，并计算更多的评价指标
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            num_classes (Optional[int]): 类别数量，用于计算评价指标
            average (str): 多分类评价指标的平均方式，可选'micro', 'macro', 'weighted'
            visualize_confusion_matrix (bool): 是否可视化混淆矩阵
            output_path (Optional[str]): 混淆矩阵图像保存路径
            
        返回:
            Dict[str, Union[float, np.ndarray, str]]: 测试指标
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.loss_calculator(outputs, targets)
                
                # 计算准确率（如果是分类任务）
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 累加损失
                total_loss += loss.item()
                
                # 记录真实标签和预测标签
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 检测类别数量
        if num_classes is None:
            num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        
        # 是否为二分类问题
        is_binary = num_classes == 2
        
        # 计算更多的评价指标
        if is_binary:
            # 二分类问题的评价指标
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        else:
            # 多分类问题的评价指标
            precision = precision_score(y_true, y_pred, average=average, zero_division=0)
            recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        class_report = classification_report(y_true, y_pred, labels=range(num_classes))
        
        # 可视化混淆矩阵
        if visualize_confusion_matrix:
            self._plot_confusion_matrix(conf_matrix, num_classes, output_path)
        
        # 整理评估指标
        eval_metrics = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'precision': precision * 100.0,  # 转换为百分比
            'recall': recall * 100.0,        # 转换为百分比
            'f1_score': f1 * 100.0,          # 转换为百分比
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        # 如果有指标记录器，记录评估指标
        if self.metrics_logger is not None:
            # 创建一个指标副本，不包含复杂对象（如numpy数组和字符串）
            metrics_for_logging = {k: v for k, v in eval_metrics.items()
                                  if not isinstance(v, (np.ndarray, str))}
            self.metrics_logger.log_eval_metrics(metrics_for_logging, 0)  # 使用epoch=0表示最终测试
        
        return eval_metrics
    
    def _plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        num_classes: int,
        output_path: Optional[str] = None
    ) -> None:
        """
        绘制混淆矩阵
        
        参数:
            conf_matrix (np.ndarray): 混淆矩阵
            num_classes (int): 类别数量
            output_path (Optional[str]): 图像保存路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=range(num_classes),
            yticklabels=range(num_classes)
        )
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到 {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    @classmethod
    def evaluate_model(
        cls,
        test_loader: DataLoader,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        loss_type: str = 'cross_entropy',
        device: Optional[torch.device] = None,
        num_classes: Optional[int] = None,
        average: str = 'weighted',
        visualize_confusion_matrix: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        加载和评估模型的静态方法
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            model (nn.Module): 要评估的模型
            checkpoint_path (Optional[str]): 模型检查点路径，如果提供则加载模型权重
            loss_type (str): 损失函数类型
            device (Optional[torch.device]): 运行设备，如果为None则自动选择
            num_classes (Optional[int]): 类别数量
            average (str): 多分类评价指标的平均方式
            visualize_confusion_matrix (bool): 是否可视化混淆矩阵
            output_path (Optional[str]): 混淆矩阵图像保存路径
            
        返回:
            Dict[str, Union[float, np.ndarray, str]]: 测试指标
        """
        # 设置设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型移至设备
        model = model.to(device)
        
        # 如果提供了检查点路径，加载模型权重
        if checkpoint_path:
            print(f"加载模型检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建损失计算器
        loss_calculator = LossCalculator(loss_type=loss_type)
        
        # 创建指标记录器
        metrics_logger = MetricsLogger(save_dir='metrics', experiment_name='model_evaluation')
        
        # 创建空的优化器管理器和反向传播管理器（评估时不需要）
        optimizer_manager = None
        backprop_manager = None
        
        # 创建训练循环实例
        training_loop = cls(
            model=model,
            loss_calculator=loss_calculator,
            optimizer_manager=optimizer_manager,
            backprop_manager=backprop_manager,
            device=device,
            metrics_logger=metrics_logger
        )
        
        # 评估模型
        eval_metrics = training_loop.evaluate(
            test_loader=test_loader,
            num_classes=num_classes,
            average=average,
            visualize_confusion_matrix=visualize_confusion_matrix,
            output_path=output_path
        )
        
        # 打印评估结果
        print("\n" + "=" * 50)
        print("模型评估结果:")
        print("=" * 50)
        print(f"测试损失: {eval_metrics['test_loss']:.4f}")
        print(f"测试准确率: {eval_metrics['test_accuracy']:.2f}%")
        print(f"精确率: {eval_metrics['precision']:.2f}%")
        print(f"召回率: {eval_metrics['recall']:.2f}%")
        print(f"F1分数: {eval_metrics['f1_score']:.2f}%")
        print("\n分类报告:")
        print(eval_metrics['classification_report'])
        print("=" * 50)
        
        return eval_metrics 